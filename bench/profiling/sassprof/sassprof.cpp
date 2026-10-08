// sassprof: CUPTI injection library that collects SASS-patching metrics (per-instruction executed
// counts, thread-level counts, divergence, global/shared/local memory efficiency) for every kernel
// executed between cudaProfilerStart() and cudaProfilerStop(), plus one line per kernel launch
// (API callback: name, grid, block) to normalise the counts per launch. (CUPTI activity tracing is
// not compatible with SASS patching: CUPTI_ERROR_NOT_COMPATIBLE.)
//
// It does not use the hardware performance-monitor counters, so it works while DCGM holds them
// (Nsight Compute, CUPTI PM sampling and PC sampling do not). Counts are exact; kernel durations
// under patching are meaningless and are not reported.
//
// Use:  CUDA_INJECTION64_PATH=/path/libsassprof.so SASSPROF_OUT=DIR NVPROF_ID=sassprof gmx mdrun ... -resetstep N
//       (NVPROF_ID makes GROMACS call cudaProfilerStart() at the counter reset and cudaProfilerStop() at the end.)
// Env:  SASSPROF_OUT      output directory (created; required)
//       SASSPROF_METRICS  comma-separated SASS metric names (default: a set covering ncu's source counters)
//       SASSPROF_AFTER    ';'-separated substrings of kernel names. CUPTI 2025.3 (CUDA 13.0) collects SASS
//                         metrics only for the first function launched after cuptiSassMetricsEnable() (and
//                         nothing after the first flush; enabling inside the target's own launch callback
//                         collects nothing). So collection is armed at cudaProfilerStart() and enabled when
//                         the most recent launches match this sequence (checked at the launch's API exit);
//                         the next launch is the profiled kernel: one kernel per run, accumulated over all its
//                         launches until cudaProfilerStop(). Without SASSPROF_AFTER: enable at
//                         cudaProfilerStart() (profiles the first kernel launched after it).
//       SASSPROF_LAZY     lazy patching (default 1)
// Output: DIR/sass_metrics.tsv  cubin_crc, function_index, function, pc_offset, metric, value
//         DIR/launches.tsv      api (runtime/driver), callback id, function, grid xyz, block xyz
//         DIR/cubins/<crc>.cubin every module referenced by the metrics (for disassembly with line info)
//         DIR/sassprof.log
//
// Part of bench/profiling (GROMACS fork profiling tools); not used by GROMACS itself.

#include <cuda.h>
#include <cupti.h>
#include <cuda_runtime_api.h>
#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <cupti_sass_metrics.h>

#include <sys/stat.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace
{

std::mutex                                          g_mutex;
std::string                                         g_outDir;
FILE*                                               g_log = nullptr;
std::vector<std::string>                            g_metricNames;
std::map<uint64_t, std::string>                     g_metricIdToName;
std::map<uint32_t, std::vector<char>>               g_cubins; // crc -> module image
std::set<uint32_t>                                  g_referencedCrcs;
bool                                                g_configured = false;
bool                                                g_enabled    = false;
bool                                                g_armed      = false;
std::vector<std::string>                            g_after;   // launch-name pattern that precedes the target
std::vector<std::string>                            g_recent;  // most recent launch names (driver API)
bool                                                g_pending      = false; // pattern matched, enable at next non-launch API exit
int                                                 g_runtimeDepth = 0;     // nesting of runtime API calls
int                                                 g_logLaunches  = 0;     // log this many launches after enabling
CUcontext                                           g_ctx        = nullptr;
CUpti_SubscriberHandle                              g_subscriber;
// (crc, function index, function name, pc, metric id) -> accumulated value
std::map<std::tuple<uint32_t, uint32_t, std::string, uint32_t, uint64_t>, uint64_t> g_values;
std::vector<std::string>                            g_launchLines;

void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void logf(const char* fmt, ...)
{
    if (!g_log)
    {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    fflush(g_log);
}

#define CUPTI_CHECK(call)                                                       \
    do                                                                          \
    {                                                                           \
        CUptiResult _st = (call);                                               \
        if (_st != CUPTI_SUCCESS)                                               \
        {                                                                       \
            const char* _msg = nullptr;                                         \
            cuptiGetResultString(_st, &_msg);                                   \
            logf("CUPTI error %d (%s) at %s:%d: %s\n", int(_st), _msg ? _msg : "?", \
                 __FILE__, __LINE__, #call);                                    \
            return false;                                                       \
        }                                                                       \
    } while (0)

const char* kDefaultMetrics =
        "smsp__sass_inst_executed,smsp__sass_thread_inst_executed,smsp__sass_thread_inst_executed_pred_on,"
        "smsp__sass_branch_targets_threads_divergent,smsp__sass_branch_targets_threads_uniform,"
        "smsp__sass_sectors_mem_global,smsp__sass_sectors_mem_global_ideal,smsp__sass_l1tex_tags_mem_global,"
        "smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared,smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared_ideal,"
        "smsp__sass_sectors_mem_local";

// ---- SASS metrics ------------------------------------------------------------------------------
bool configure(CUcontext ctx)
{
    if (g_configured)
    {
        return true;
    }
    CUdevice dev;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS)
    {
        CUcontext prev = nullptr;
        cuCtxPushCurrent(ctx);
        cuCtxGetDevice(&dev);
        cuCtxPopCurrent(&prev);
    }
    CUpti_Profiler_Initialize_Params initParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
    CUPTI_CHECK(cuptiProfilerInitialize(&initParams));

    CUpti_Device_GetChipName_Params chip = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    chip.deviceIndex                     = static_cast<size_t>(dev);
    CUPTI_CHECK(cuptiDeviceGetChipName(&chip));

    std::vector<CUpti_SassMetrics_Config> configs;
    for (const auto& name : g_metricNames)
    {
        CUpti_SassMetrics_GetProperties_Params p = { CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE };
        p.pChipName                              = chip.pChipName;
        p.pMetricName                            = name.c_str();
        CUPTI_CHECK(cuptiSassMetricsGetProperties(&p));
        CUpti_SassMetrics_Config c;
        c.metricId          = p.metric.metricId;
        c.outputGranularity = CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_GPU;
        configs.push_back(c);
        g_metricIdToName[c.metricId] = name;
    }
    CUpti_SassMetricsSetConfig_Params sc = { CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE };
    sc.pConfigs                          = configs.data();
    sc.numOfMetricConfig                 = configs.size();
    sc.deviceIndex                       = static_cast<uint32_t>(dev);
    CUPTI_CHECK(cuptiSassMetricsSetConfig(&sc));
    g_configured = true;
    logf("configured %zu metrics on device %d (%s)\n", configs.size(), int(dev), chip.pChipName);
    return true;
}

bool enable(CUcontext ctx)
{
    if (g_enabled || !configure(ctx))
    {
        return g_enabled;
    }
    CUpti_SassMetricsEnable_Params ep = { CUpti_SassMetricsEnable_Params_STRUCT_SIZE };
    ep.ctx                            = ctx;
    ep.enableLazyPatching             = getenv("SASSPROF_LAZY") ? uint8_t(atoi(getenv("SASSPROF_LAZY"))) : 1;
    CUPTI_CHECK(cuptiSassMetricsEnable(&ep));
    g_enabled     = true;
    g_ctx         = ctx;
    g_logLaunches = 4;
    logf("enabled SASS patching on ctx %p\n", static_cast<void*>(ctx));
    return true;
}

bool flush(CUcontext ctx)
{
    CUpti_SassMetricsGetDataProperties_Params dp = { CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE };
    dp.ctx                                       = ctx;
    CUPTI_CHECK(cuptiSassMetricsGetDataProperties(&dp));
    logf("flush: %zu patched-instruction records, %zu instances\n", dp.numOfPatchedInstructionRecords,
         dp.numOfInstances);
    if (dp.numOfPatchedInstructionRecords == 0 || dp.numOfInstances == 0)
    {
        return true;
    }
    CUpti_SassMetricsFlushData_Params fp = { CUpti_SassMetricsFlushData_Params_STRUCT_SIZE };
    fp.ctx                               = ctx;
    fp.numOfPatchedInstructionRecords    = dp.numOfPatchedInstructionRecords;
    fp.numOfInstances                    = dp.numOfInstances;
    std::vector<CUpti_SassMetrics_Data> data(dp.numOfPatchedInstructionRecords);
    std::vector<CUpti_SassMetrics_InstanceValue> values(dp.numOfPatchedInstructionRecords * dp.numOfInstances);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i].pInstanceValues = &values[i * dp.numOfInstances];
    }
    fp.pMetricsData = data.data();
    CUPTI_CHECK(cuptiSassMetricsFlushData(&fp));
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto& d : data)
    {
        std::string fn = d.functionName ? d.functionName : "?";
        g_referencedCrcs.insert(d.cubinCrc);
        for (size_t k = 0; k < dp.numOfInstances; ++k)
        {
            const auto& v = d.pInstanceValues[k];
            g_values[std::make_tuple(d.cubinCrc, d.functionIndex, fn, d.pcOffset, v.metricId)] += v.value;
        }
    }
    return true;
}

bool disable(CUcontext ctx)
{
    if (!g_enabled)
    {
        return true;
    }
    cuCtxSynchronize();
    flush(ctx);
    CUpti_SassMetricsDisable_Params dp = { CUpti_SassMetricsDisable_Params_STRUCT_SIZE };
    dp.ctx                             = ctx;
    CUPTI_CHECK(cuptiSassMetricsDisable(&dp));
    logf("disabled; dropped records %zu\n", dp.numOfDroppedRecords);
    g_enabled = false;
    return true;
}

void writeOutput()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    std::string path = g_outDir + "/sass_metrics.tsv";
    if (FILE* f = fopen(path.c_str(), "w"))
    {
        fprintf(f, "cubin_crc\tfunction_index\tfunction\tpc_offset\tmetric\tvalue\n");
        for (const auto& [key, value] : g_values)
        {
            const auto& [crc, fidx, fn, pc, mid] = key;
            fprintf(f, "%u\t%u\t%s\t%u\t%s\t%llu\n", crc, fidx, fn.c_str(), pc, g_metricIdToName[mid].c_str(),
                    static_cast<unsigned long long>(value));
        }
        fclose(f);
    }
    path = g_outDir + "/launches.tsv";
    if (FILE* f = fopen(path.c_str(), "w"))
    {
        fprintf(f, "api\tcbid\tfunction\tgrid_x\tgrid_y\tgrid_z\tblock_x\tblock_y\tblock_z\n");
        for (const auto& l : g_launchLines)
        {
            fprintf(f, "%s\n", l.c_str());
        }
        fclose(f);
    }
    std::string cubinDir = g_outDir + "/cubins";
    mkdir(cubinDir.c_str(), 0755);
    for (uint32_t crc : g_referencedCrcs)
    {
        auto it = g_cubins.find(crc);
        if (it == g_cubins.end())
        {
            logf("cubin %u not captured\n", crc);
            continue;
        }
        std::string p = cubinDir + "/" + std::to_string(crc) + ".cubin";
        if (FILE* f = fopen(p.c_str(), "wb"))
        {
            fwrite(it->second.data(), 1, it->second.size(), f);
            fclose(f);
        }
    }
    logf("wrote %zu values, %zu launches, %zu cubins\n", g_values.size(), g_launchLines.size(),
         g_referencedCrcs.size());
}

void CUPTIAPI callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata)
{
    if (domain == CUPTI_CB_DOMAIN_RESOURCE)
    {
        const auto* rd = static_cast<const CUpti_ResourceData*>(cbdata);
        if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED)
        {
            const auto* md = static_cast<const CUpti_ModuleResourceData*>(rd->resourceDescriptor);
            if (md && md->pCubin && md->cubinSize > 0)
            {
                CUpti_GetCubinCrcParams cp = { CUpti_GetCubinCrcParamsSize };
                cp.cubinSize               = md->cubinSize;
                cp.cubin                   = md->pCubin;
                if (cuptiGetCubinCrc(&cp) == CUPTI_SUCCESS)
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    g_cubins[cp.cubinCrc].assign(md->pCubin, md->pCubin + md->cubinSize);
                }
            }
        }
        else if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)
        {
            // nothing: collection starts at cudaProfilerStart()
        }
        else if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING)
        {
            if (g_enabled && rd->context == g_ctx)
            {
                disable(rd->context);
                writeOutput();
            }
        }
        return;
    }
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API || domain == CUPTI_CB_DOMAIN_RUNTIME_API)
    {
        const auto* ci = static_cast<const CUpti_CallbackData*>(cbdata);
        bool isLaunch = false;
        if (domain == CUPTI_CB_DOMAIN_DRIVER_API)
        {
            isLaunch = cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz
                       || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz;
        }
        else
        {
            isLaunch = cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000
                       || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060
                       || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060
                       || cbid == CUPTI_RUNTIME_TRACE_CBID___cudaLaunchKernel_v13000
                       || cbid == CUPTI_RUNTIME_TRACE_CBID___cudaLaunchKernel_ptsz_v13000;
            g_runtimeDepth += (ci->callbackSite == CUPTI_API_ENTER) ? 1 : -1;
        }
        if (g_pending && ci->callbackSite == CUPTI_API_EXIT && g_runtimeDepth == 0)
        {
            CUcontext ctx = ci->context;
            if (!ctx)
            {
                cuCtxGetCurrent(&ctx);
            }
            logf("enabling at exit of %s API call %u (%s)\n", domain == CUPTI_CB_DOMAIN_DRIVER_API ? "driver" : "runtime",
                 unsigned(cbid), ci->functionName ? ci->functionName : "?");
            g_pending = false;
            // Patching only takes effect when no kernel is in flight (enabling with queued work collects nothing).
            cuCtxSynchronize();
            enable(ctx);
        }
        if (isLaunch)
        {
            const char* sym = ci->symbolName ? ci->symbolName : "?";
            if (domain == CUPTI_CB_DOMAIN_DRIVER_API && ci->callbackSite == CUPTI_API_EXIT && !g_after.empty())
            {
                g_recent.push_back(sym);
                if (g_recent.size() > g_after.size())
                {
                    g_recent.erase(g_recent.begin());
                }
                bool match = g_armed && !g_enabled && g_recent.size() == g_after.size();
                for (size_t i = 0; match && i < g_after.size(); ++i)
                {
                    match = std::strstr(g_recent[i].c_str(), g_after[i].c_str()) != nullptr;
                }
                if (match)
                {
                    // Enable at the exit of this launch: of the driver launch itself when it was called
                    // directly (cuFFT), else of the enclosing runtime launch call (runtime depth 0).
                    logf("pattern matched at launch of %s\n", sym);
                    g_pending = true;
                    g_armed   = false;
                    if (g_runtimeDepth == 0)
                    {
                        CUcontext ctx = ci->context;
                        if (!ctx)
                        {
                            cuCtxGetCurrent(&ctx);
                        }
                        logf("enabling at exit of driver launch\n");
                        g_pending = false;
                        cuCtxSynchronize();
                        enable(ctx);
                    }
                }
            }
            if (g_enabled && ci->callbackSite == CUPTI_API_ENTER && domain == CUPTI_CB_DOMAIN_DRIVER_API && g_logLaunches > 0)
            {
                --g_logLaunches;
                logf("  launch after enable: %s\n", sym);
            }
            if (g_enabled && ci->callbackSite == CUPTI_API_ENTER)
            {
                unsigned g[3] = { 0, 0, 0 }, b[3] = { 0, 0, 0 };
                if (domain == CUPTI_CB_DOMAIN_DRIVER_API
                    && (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz))
                {
                    const auto* p = static_cast<const cuLaunchKernel_params*>(ci->functionParams);
                    g[0] = p->gridDimX; g[1] = p->gridDimY; g[2] = p->gridDimZ;
                    b[0] = p->blockDimX; b[1] = p->blockDimY; b[2] = p->blockDimZ;
                }
                else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API
                         && (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
                             || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000))
                {
                    const auto* p = static_cast<const cudaLaunchKernel_v7000_params*>(ci->functionParams);
                    g[0] = p->gridDim.x; g[1] = p->gridDim.y; g[2] = p->gridDim.z;
                    b[0] = p->blockDim.x; b[1] = p->blockDim.y; b[2] = p->blockDim.z;
                }
                std::ostringstream os;
                os << (domain == CUPTI_CB_DOMAIN_DRIVER_API ? "driver" : "runtime") << '\t' << cbid << '\t'
                   << (ci->symbolName ? ci->symbolName : "?") << '\t' << g[0] << '\t' << g[1] << '\t' << g[2]
                   << '\t' << b[0] << '\t' << b[1] << '\t' << b[2];
                std::lock_guard<std::mutex> lock(g_mutex);
                g_launchLines.push_back(os.str());
            }
            return;
        }
        if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000 && ci->callbackSite == CUPTI_API_EXIT)
        {
            CUcontext ctx = ci->context;
            if (!ctx)
            {
                cuCtxGetCurrent(&ctx);
            }
            logf("cudaProfilerStart\n");
            if (g_after.empty())
            {
                enable(ctx);
            }
            else
            {
                g_armed = true; // configure + enable later (SetConfig long before Enable collects nothing)
            }
        }
        else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000 && ci->callbackSite == CUPTI_API_ENTER)
        {
            logf("cudaProfilerStop\n");
            CUcontext ctx = g_ctx;
            disable(ctx);
            writeOutput();
        }
    }
}

bool init()
{
    const char* out = getenv("SASSPROF_OUT");
    if (!out)
    {
        fprintf(stderr, "sassprof: SASSPROF_OUT not set, doing nothing\n");
        return false;
    }
    g_outDir = out;
    mkdir(g_outDir.c_str(), 0755);
    g_log = fopen((g_outDir + "/sassprof.log").c_str(), "a");
    if (const char* a = getenv("SASSPROF_AFTER"))
    {
        std::stringstream as(a);
        std::string       pat;
        while (std::getline(as, pat, ';'))
        {
            if (!pat.empty())
            {
                g_after.push_back(pat);
            }
        }
    }
    const char* m = getenv("SASSPROF_METRICS");
    std::stringstream ss(m && *m ? m : kDefaultMetrics);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty())
        {
            g_metricNames.push_back(item);
        }
    }
    CUPTI_CHECK(cuptiSubscribe(&g_subscriber, reinterpret_cast<CUpti_CallbackFunc>(callback), nullptr));
    CUPTI_CHECK(cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED));
    CUPTI_CHECK(cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
    CUPTI_CHECK(cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                                    CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING));
    CUPTI_CHECK(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CHECK(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
    logf("sassprof initialised, %zu metrics\n", g_metricNames.size());
    return true;
}

} // namespace

extern "C" int InitializeInjection(void)
{
    init();
    return 1;
}
