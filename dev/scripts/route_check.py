#!/usr/bin/env python3
# usage: route_check.py <port> <model id> <out.json>   Sends oracle prompts A, B, C (greedy, 64 new tokens) and stores the replies; compare two files with: route_check.py --diff a.json b.json
import http.client, json, os, sys

def ask(port, model, text):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    body = {"model": model, "max_tokens": 64, "temperature": 0, "messages": [{"role": "user", "content": text}]}
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read())
    m = d["choices"][0]["message"]
    return {"usage": d["usage"], "finish": d["choices"][0]["finish_reason"], "reasoning": m.get("reasoning_content") or "", "content": m.get("content") or ""}

if sys.argv[1] == "--diff":
    a, b = (json.load(open(p)) for p in sys.argv[2:4])
    bad = [k for k in a if a[k] != b[k]]
    print("DIFF" if bad else "IDENTICAL", bad)
    sys.exit(1 if bad else 0)
port, model, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle", "prompts")
res = {n: ask(port, model, open(os.path.join(here, f), encoding="utf-8").read().strip()) for n, f in (("A", "A.txt"), ("B", "B.txt"), ("C", "C_user.txt"))}
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items():
    print(k, v["usage"], v["finish"], repr((v["reasoning"] + v["content"])[:70]))
