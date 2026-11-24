"""usage: greedy_check.py URL MODEL OUT.json   Greedy 64 token outputs for the oracle prompts A, B, C and for 8k and 32k character documents."""
import json
import os
import sys

sys.path.insert(0, os.environ.get("SKILLS_BENCH", os.path.expanduser("~/inkling/loom-skills/skills/loom-benchmark/scripts")))
from bench_common import doc, stream, thinking_extra

url, model, out = sys.argv[1:4]
P = os.path.expanduser("~/inkling/M30/oracle/prompts")
ESSAY = "Write a long, detailed essay discussing this documentation.\n\n"
LONG = os.environ.get("GREEDY_LONG", "1") == "1"
cases = {"A": open(f"{P}/A.txt").read(), "B": open(f"{P}/B.txt").read(), "C": open(f"{P}/C_user.txt").read(),
         "L8k": ESSAY + doc(8000), "L32k": ESSAY + doc(32000)}
res = {}
for k, text in cases.items():
    if k.startswith("L") and not LONG:
        continue
    s = stream(url, model, [{"role": "user", "content": text}], 64, extra=thinking_extra("off"), timeout=900)
    res[k] = dict(text=s["text"], tokens=s["tokens"], prompt_tokens=s["prompt_tokens"])
    print(k, s["prompt_tokens"], s["tokens"], flush=True)
json.dump(res, open(out, "w"), indent=1)
