import sys, json
from gguf import GGUFReader
r = GGUFReader(sys.argv[1], "r")
def s(k):
    f = r.fields[k]; return bytes(f.parts[f.data[0]]).decode("utf-8")
open(sys.argv[2], "w", encoding="utf-8", newline="\n").write(s("tokenizer.chat_template"))
f = r.fields["tokenizer.ggml.tokens"]; ty = r.fields["tokenizer.ggml.token_type"]
out = []
for i, idx in enumerate(f.data):
    t = int(ty.parts[ty.data[i]][0])
    if t != 1:
        out.append((i, t, bytes(f.parts[idx]).decode("utf-8", "replace")))
json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False)
print(len(out), "non-normal tokens")
