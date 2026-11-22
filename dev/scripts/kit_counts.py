"""Kit simulate token counts against a live loom-serve /v1/chat/completions/input_tokens for the 44 golden chat cases.
usage: LOOM_KIT=kit python kit_counts.py <pkg dir with llm_config.json + tokenizer.json> <url> <golden.json>"""
import json, os, sys, urllib.error, urllib.request

sys.path.insert(0, os.path.abspath(os.environ["LOOM_KIT"]))
import chat_probe
from tokenizers import Tokenizer

pkg, url, golden = sys.argv[1:4]
chat = json.load(open(os.path.join(pkg, "llm_config.json")))["chat"]
tok = Tokenizer.from_file(os.path.join(pkg, "tokenizer.json"))


def post(body):
    req = urllib.request.Request(url + "/v1/chat/completions/input_tokens", json.dumps(body, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:120]


bad = 0
for case in json.load(open(golden, encoding="utf-8")):
    body = dict(case["body"], model="inkling-2-mini")
    kw = {"reasoning_effort": body["reasoning_effort"]} if "reasoning_effort" in body else {}
    try:
        text = chat_probe.simulate(chat, body["messages"], body.get("tools"), gen=True, **kw)
        n = len(tok.encode(text, add_special_tokens=False).ids)
    except chat_probe.ProbeError as e:
        text, n = None, str(e)[:60]
    status, reply = post(body)
    if text is None:
        ok = status == 400
        detail = f"kit raises, server {status}"
    elif status == 400:
        ok = case["expect"] == "reject"
        detail = f"kit {n} tokens, server rejects ({'expected: Loom policy' if ok else 'UNEXPECTED'})"
    else:
        ok = status == 200 and reply["input_tokens"] == n
        detail = f"kit {n} server {reply['input_tokens'] if status == 200 else status}"
    bad += not ok
    print("PASS" if ok else "FAIL", case["name"], detail)
print(f"{'FAIL' if bad else 'PASS'}: {bad} failing of 44")
sys.exit(1 if bad else 0)
