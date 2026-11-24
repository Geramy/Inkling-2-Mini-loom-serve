"""usage: chat_obs.py URL [grep]   prints the text-topic observation fields of /metrics (and optionally the lines that match grep)."""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatlib import get

url = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else ""
txt = get(url, "/metrics")
for l in str(txt).splitlines():
    if l.startswith("#") or "loom_serve_model_observation" not in l:
        continue
    m = re.search(r'topic="([^"]*)".*field="([^"]*)",unit="([^"]*)"\} (\S+)', l)
    if m and (not pat or re.search(pat, l)):
        print(f"{m.group(1)}.{m.group(2)} [{m.group(3)}] = {m.group(4)}")
