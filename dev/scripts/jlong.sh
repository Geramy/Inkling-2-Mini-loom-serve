#!/usr/bin/env bash
# usage: jlong.sh <arch> <port> "<visits>"   (local) 1332 token prompt at the given visits: chunks 444x3, 512+512+308, 512+512+276 then 32 single tokens; prints ab and l_out rows
cd "$(dirname "$0")/.."
ONES=$(printf '1,%.0s' $(seq 1 31))1
for v in $3; do
  INKLING_PREFILL=512 bash scripts/jrun.sh "$1" "$2" all,$v Q8 900 2>&1 | head -1 | cut -c1-100
  for c in 444,444,444 512,512,308 512,512,276,$ONES; do
    echo "== visit $v chunks $(echo $c | cut -c1-24)"
    TMO=${TMO:-300} TAIL=40 bash scripts/jchecklong.sh "$2" "$v" --only ab,k,v,y,l_out --chunks $c | grep -E "^ab|^k |^v |^y |^l_out|RESULT|Error|rror" | cut -c1-100
  done
done
