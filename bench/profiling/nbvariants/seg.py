"""Segment NB kernel SASS (from compile.sh) into pair-interaction bodies and i-cluster mask tests.

    seg.py v0.sass v1.sass ...
"""
import sys, re
for f in sys.argv[1:]:
    L=[l.strip() for l in open(f)]
    op=[re.sub(r"^@!?U?P\w+\s+","",l).split(" ")[0] for l in L]
    rsq=[i for i,o in enumerate(op) if o.startswith("MUFU.RSQ")]
    bodies=[]; tests=[]
    for r in rsq:
        a=r
        while not (op[a]=="BRA" and L[a].startswith("@")): a-=1   # divergent if (r2<rc) branch
        b=r
        while op[b]!="BSYNC": b+=1
        bodies.append(b-a-1)
    # i-slot test pattern: instructions from a BSYNC to the next LDS.128 (next i-iteration) when that span is short
    i=0
    for k,o in enumerate(op):
        if o=="BSYNC":
            j=k+1
            while j<len(op) and op[j] not in ("LDS.128","BSYNC","SHFL.DOWN") and j-k<12: j+=1
            if j<len(op) and op[j]=="LDS.128": tests.append(j-k-1)
    from collections import Counter
    print(f"{f}: {len(L)} instr; body sizes {Counter(bodies).most_common(4)}; bsync->next LDS.128 spans {Counter(tests).most_common(4)}")
