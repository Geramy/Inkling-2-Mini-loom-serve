#!/usr/bin/env bash
# usage: diag.sh <port>   (local) prints registration state and compile diagnostics of the first model
BOX=${BOX:-user@gpu-box}
ssh "$BOX" "curl -s localhost:${1:?port}/v1/status | python3 -c '
import sys,json
d=json.load(sys.stdin)
for m in d[\"data\"]:
    l=m[\"loom\"]; print(l[\"registration\"], l.get(\"error\",{}).get(\"message\",\"\")[:600]); c=l.get(\"compilation\",{})
    print({k:c.get(k) for k in (\"commands\",\"kernels\",\"reused_kernels\",\"errors\",\"stage_duration_sum_ns\")}, \"prep_ns\", l.get(\"preparation_duration_ns\"))
    seen=set()
    for x in c.get(\"diagnostics\",[]):
        k=(x[\"source\"].split(\"/\")[-1],x[\"line\"],x[\"message\"][:140])
        if k in seen: continue
        seen.add(k); print(\" \",k[0]+\":\"+str(k[1]),x[\"code\"],x[\"message\"][:300])
'"
