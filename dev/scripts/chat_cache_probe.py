"""usage: chat_cache_probe.py [--tokens N] [--effort E] [--turns T] [--max-tokens M] [--modes m1,m2,...] [--out F] URL
Turn 1 (system N tokens + user) then follow-up turns that extend the same conversation. Prints TTFT, prompt, cached and completion tokens per turn.
Modes: <stream|json>-<echo|plain|noreason> : echo sends reasoning_content back, plain sends content only (reasoning dropped), noreason is plain with effort none."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatlib import cached, chat, get

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--tokens", type=int, default=4000)
ap.add_argument("--effort", default="low")
ap.add_argument("--turns", type=int, default=2)
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--modes", default="json-echo,json-plain,stream-echo,stream-plain")
ap.add_argument("--out", default="")
ap.add_argument("--temperature", type=float, default=0.0)
a = ap.parse_args()

from chat_cache_probe_text import system_text


def snap(url):
    out = {}
    try:
        st = get(url, "/v1/status")
        out["status"] = st
    except Exception as e:
        out["status"] = str(e)
    return out


rows = []
for mode in a.modes.split(","):
    st, how = mode.split("-")
    nonce = f"{int(time.time() * 1000) % 10**9}"
    effort = "none" if how == "noreason" else a.effort
    msgs = [{"role": "system", "content": system_text(nonce, a.tokens)}, {"role": "user", "content": "Summarize the notes about glaciers in two sentences."}]
    follow = ["Now give the three most important numbers from your summary.", "Which of the topics has the smallest margin?", "Thank you. Repeat your first answer in one line."]
    for turn in range(a.turns):
        r = chat(a.url, msgs, stream=(st == "stream"), temperature=a.temperature, max_tokens=a.max_tokens, reasoning_effort=effort)
        pt, ct, comp = cached(r)
        row = {"mode": mode, "turn": turn + 1, "status": r["status"], "ttft": round(r["ttft"], 3), "total": round(r["total"], 3), "prompt": pt, "cached": ct,
               "completion": comp, "finish": r["finish"], "reasoning_chars": len(r["reasoning"]), "content_chars": len(r["content"]), "content": r["content"][:80], "error": r["error"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if r["status"] != 200:
            break
        m = {"role": "assistant", "content": r["content"]}
        if how == "echo" and r["reasoning"]:
            m["reasoning_content"] = r["reasoning"]
        msgs = msgs + [m, {"role": "user", "content": follow[turn % len(follow)]}]
print("STATUS", json.dumps(snap(a.url))[:1500])
try:
    print("METRICS", [l for l in str(get(a.url, "/metrics")).splitlines() if ("cache" in l or "prefix" in l or "session" in l or "checkpoint" in l) and not l.startswith("#")][:40])
except Exception as e:
    print("metrics error", e)
if a.out:
    open(a.out, "a").write("\n".join(json.dumps(r) for r in rows) + "\n")
