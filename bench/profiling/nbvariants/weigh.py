"""Weight the per-copy body-size change of NB kernel variants by the measured execution count of each body copy.

    weigh.py NCU_SOURCE_SASS.csv   (in a directory with v0.sass, v1.sass, ... from compile.sh)
The CSV is `ncu --import REP --page source --print-source sass --csv -k regex:nbnxn_kernel_ElecEw_VdwLJFsw_F`.
The i-slot test count (8.23 M per launch) is MAS1's; mask-test deltas per variant are set in the loop below.
"""
import csv, re, sys
# measured exec counts of the 16 body copies (MUFU.RSQ, address order) and of i-slot tests, from ncu
rows=list(csv.reader(open(sys.argv[1]))); h=rows[1]; ix={k:i for i,k in enumerate(h)}
rsq=[]; tot=0
for r in rows[2:]:
    if len(r)<len(h): continue
    try: n=float(r[ix['Instructions Executed']] or 0)
    except ValueError: continue
    tot+=n
    if 'MUFU.RSQ' in r[ix['Source']]: rsq.append(n)
def bodies(f):
    L=[l.strip() for l in open(f)]; op=[re.sub(r"^@!?U?P\w+\s+","",l).split(" ")[0] for l in L]
    out=[]
    for i,o in enumerate(op):
        if o.startswith("MUFU.RSQ"):
            a=i
            while not (op[a]=="BRA" and L[a].startswith("@")): a-=1
            b=i
            while op[b]!="BSYNC": b+=1
            out.append(b-a-1)
    return out, len(L)
b0,n0=bodies("v0.sass")
print(f"production: {tot/1e6:.1f} M warp instr/launch, {sum(rsq)/1e6:.3f} M bodies, {len(rsq)} copies")
islot=8.23e6
for v,dtest in (("v1",-2),("v2",0),("v3",0),("v123",-2)):
    bv,nv=bodies(v+".sass")
    d=sum((x-y)*e for x,y,e in zip(bv,b0,rsq))
    dt=dtest*islot
    print(f"{v:5s}: static {nv-n0:+5d}; bodies {d/1e6:+6.1f} M; mask tests {dt/1e6:+6.1f} M; total {100*(d+dt)/tot:+5.1f}% of executed warp instructions")
