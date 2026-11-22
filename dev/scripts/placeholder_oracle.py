# usage: placeholder_oracle.py <port> <gguf> [bank]   checks the placeholder answer against a host replay of the kernel
import json, struct, sys, urllib.request
import numpy as np
from gguf import GGUFReader

port, gguf = sys.argv[1], sys.argv[2]
bank = len(sys.argv) > 3 and sys.argv[3] == "bank"
r = GGUFReader(gguf, "r")
T = {t.name: t for t in r.tensors}

def raw(name):
    return np.asarray(T[name].data).view(np.uint8).reshape(-1)

def post(path, doc):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(doc).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

def bias_value():
    if bank:
        names = json.load(open("bank_tensors.json"))
        value = 0
        for n in names:
            value ^= int(raw(n)[-1])
        return value
    norm_word = struct.unpack("<I", raw("output_norm.weight")[-4:].tobytes())[0]
    return norm_word ^ int(raw("token_embd.weight")[-1]) ^ int(raw("blk.0.ffn_gate_up_exps.weight")[-1])

def expected(prompt_ids, max_tokens, eos=200006):
    bias = bias_value()
    m32 = 0xFFFFFFFF
    h = k = 0
    history = []
    out = []
    pending = list(prompt_ids)
    while True:
        for t in pending:
            p = len(history)
            h = ((5 * h + t + bias) & m32) % 97
            k = ((7 * k + h + p) & m32) % 101
            history.append(k)
        end = len(history)
        pred = eos if end % 11 == 0 else 64 + ((h + sum(history)) & m32) % 26
        out.append(pred)
        if pred == eos or len(out) >= max_tokens:
            return out
        pending = [pred]

effort = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
text = effort + "<|message_user|><|content_text|>Hello<|end_message|>"
ids = post("/tokenize", {"model": "inkling-2-mini", "content": text, "add_special": False, "parse_special": True})["tokens"]
want = expected(ids, 40)
reply = post("/v1/chat/completions", {"model": "inkling-2-mini", "max_tokens": 40, "messages": [{"role": "user", "content": "Hello"}]})
got = reply["choices"][0]["message"]["content"]
exp_text = "".join(chr(97 + (t - 64)) for t in want if t != 200006)
print("prompt tokens", len(ids), "usage", reply["usage"])
print("got     ", got)
print("expected", exp_text)
print("MATCH" if got == exp_text else "MISMATCH")
