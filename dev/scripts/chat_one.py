"""usage: chat_one.py URL REQUEST.json [--stream]   sends one chat request and prints status, usage, reasoning, content and calls."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatlib import chat

url, path = sys.argv[1], sys.argv[2]
req = json.load(open(path))
msgs = req.pop("messages")
tools = req.pop("tools", None)
r = chat(url, msgs, stream="--stream" in sys.argv, tools=tools, **req)
print(json.dumps({k: r[k] for k in ("status", "ttft", "total", "usage", "finish", "error", "tool_calls")}, ensure_ascii=False))
print("MESSAGE:", json.dumps(r.get("message"), ensure_ascii=False)[:600])
print("REASONING:", r["reasoning"][:int(os.environ.get("CUT", "1500"))])
print("CONTENT:", r["content"][:1500])
