"""Records (summary.json rows) for the GPU-kernel session: SASS-metric kernel profiles, NB-kernel
replay timings and force deviations, variant instruction counts.

    bench/profiling/gpu_records.py RESULTS_DIR      -> RESULTS_DIR/raw/records/gpu.json

Every row: source (path under RESULTS_DIR), experiment, config, step_type, metric, unit, value, n, note,
overhead. summarize.py collect merges raw/records/*.json into summary.json / summary.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sassprof"))
import sassanalyze  # noqa: E402

REGIONS = Path(__file__).resolve().parent / "sassprof" / "regions-nbnxm-cuda.toml"


def kernel_rows(res: Path) -> list[dict]:
    out = []
    p = res / "raw" / "sass" / "analysis-iso-2100.json"
    if not p.exists():
        return out
    a = json.loads(p.read_text())
    for target, r in a.items():
        if "error" in r:
            continue
        base = {"source": f"raw/sass/{target}/ + raw/sass/analysis-iso-2100.json", "experiment": "sass-metrics",
                "config": f"L build (SASS == P), production command, plain-step variant of {target}",
                "step_type": "per launch (400-step window, plain-step launches)",
                "overhead": "SASS patching: counts exact, time not measured under patching",
                "note": r["function"][:120]}
        pl = r["per_launch"]
        rows = [("warp_instructions_per_launch", "warp inst", pl.get("smsp__sass_inst_executed")),
                ("thread_instructions_per_launch", "thread inst", pl.get("smsp__sass_thread_inst_executed")),
                ("simt_efficiency", "fraction", r["simt_efficiency"]),
                ("predicated_on_efficiency", "fraction", r["predicated_on_efficiency"]),
                ("branch_divergent_fraction", "fraction", r["branch_divergent_fraction"]),
                ("global_sector_efficiency", "fraction", r["global_sector_efficiency"]),
                ("shared_wavefront_efficiency", "fraction", r["shared_wavefront_efficiency"]),
                ("global_sectors_per_launch", "sectors", pl.get("smsp__sass_sectors_mem_global")),
                ("local_sectors_per_launch", "sectors", r["local_sectors_per_launch"]),
                ("launches_in_window", "launches", r["launches"])]
        for k, v in r["class_per_launch"].items():
            rows.append((f"warp_inst_class_{k}_per_launch", "warp inst", v))
        if "timing" in r:
            t = r["timing"]
            rows.append(("isolated_duration_2100MHz", "us", t["duration_us"]))
            rows.append(("ipc_per_sm_isolated_2100MHz", "warp inst/cycle/SM", t["ipc_per_sm"]))
            for k, v in t["pipe_utilisation"].items():
                rows.append((f"pipe_util_lower_bound_{k}", "fraction (model)", v))
            for k, v in t["l1_utilisation"].items():
                rows.append((f"l1_util_lower_bound_{k}", "fraction (model)", v))
        for metric, unit, value in rows:
            if value is None:
                continue
            out.append({**base, "metric": f"{target}:{metric}", "unit": unit, "value": value, "n": r["launches"]})
        if target == "nb_f":
            for reg, rv in r["regions"].items():
                out.append({**base, "metric": f"nb_f:region_share:{reg}", "unit": "fraction of warp inst",
                            "value": rv["share"], "n": r["launches"],
                            "note": f"{rv['warp_inst']:.4g} warp inst/launch, SIMT {rv['simt_efficiency']}; regions "
                                    "bench/profiling/sassprof/regions-nbnxm-cuda.toml"})
    return out


def variant_rows(res: Path) -> list[dict]:
    out = []
    names = {"13": "exp3 fsw factored + Ewald Horner", "1": "exp1 fsw factored", "2": "exp2 Ewald Horner",
             "4": "exp7 fsw + Horner + LJ fetch hoisted", "9": "exp0 at 12 blocks/SM", "6": "ko8 no force switch",
             "7": "ko16 no Ewald correction"}
    for d in sorted((res / "raw" / "sass-variants").glob("v*/nb_f")):
        v = d.parent.name[1:]
        try:
            r = sassanalyze.analyse_kernel(d, sassanalyze.load_regions(REGIONS))
        except Exception as e:  # noqa: BLE001
            print("skip", d, e)
            continue
        if "error" in r:
            continue
        out.append({"source": str(d.relative_to(res)), "experiment": "sass-metrics-nb-variants",
                    "config": f"GMX_EXP_NB_VARIANT={v} ({names.get(v, '?')}), THROWAWAY build 9194ba626fe54212",
                    "step_type": "per launch", "metric": "nb_f:warp_instructions_per_launch", "unit": "warp inst",
                    "value": r["per_launch"]["smsp__sass_inst_executed"], "n": r["launches"],
                    "overhead": "SASS patching; throwaway", "note": r["function"][:100]})
    return out


def replay_rows(res: Path) -> list[dict]:
    out = []
    p = res / "raw" / "replay" / "summary.json"
    if not p.exists():
        return out
    for s in json.loads(p.read_text()):
        d = Path(s["dir"]).name
        for v, x in s["variants"].items():
            base = {"source": f"raw/replay/{d}/replay.txt", "experiment": "nb-replay",
                    "config": f"{v}; SM clock {d} (median {s['sm_clock_mhz_median']} MHz); THROWAWAY build "
                              "9194ba626fe54212",
                    "step_type": "NB force kernel, isolated replay on live data (8 points x 30 reps)",
                    "overhead": "isolated (stream synchronised), CUDA events; throwaway", "n": s["replay_points"]}
            out.append({**base, "metric": "nb_kernel_median_us", "unit": "us", "value": x["median_us"]})
            out.append({**base, "metric": "nb_kernel_ratio_to_prod", "unit": "ratio", "value": x["ratio_to_prod_median"],
                        "min": x["ratio_min"], "max": x["ratio_max"]})
            out.append({**base, "metric": "force_rel_rms_vs_prod", "unit": "-", "value": x["f_rel_rms_median"],
                        "max": x["f_rel_rms_max"]})
    return out


def main():
    res = Path(sys.argv[1])
    rows = kernel_rows(res) + variant_rows(res) + replay_rows(res)
    (res / "raw" / "records").mkdir(parents=True, exist_ok=True)
    (res / "raw" / "records" / "gpu.json").write_text(json.dumps(rows, indent=1, default=str))
    print(f"{len(rows)} records -> {res / 'raw' / 'records' / 'gpu.json'}")


if __name__ == "__main__":
    main()
