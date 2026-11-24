"""Request option check on a server built with scripts/chat_policy_kernel.py: the generated text spells the 24-byte option record that bind_request stored
({f32 temperature, f32 top_p, i32 top_k, i32 flags, i64 seed}, flags bit 0 explicit seed, bit 1 tool calls forbidden), plus accept and reject cases.
usage: test_options.py <port>   exit code 1 on any FAIL"""
import json, struct, sys, urllib.error, urllib.request

PORT = sys.argv[1]
MODEL = "inkling-2-mini"


def post(raw):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", raw if isinstance(raw, bytes) else json.dumps(raw).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as reply:
            return reply.status, json.load(reply)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def record(extra):
    body = {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "Hello"}], **extra}
    status, reply = post(body)
    if status != 200:
        return status, str(reply)[:160]
    text = reply["choices"][0]["message"]["content"]
    nibbles = [ord(c) - 97 for c in text]
    data = bytes(nibbles[i] << 4 | nibbles[i + 1] for i in range(0, 48, 2))
    return 200, struct.unpack("<ffiiq", data)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(("PASS" if ok else "FAIL"), name, detail)


if "--defaults" in sys.argv:
    deployed = (f32(0.6), f32(0.95), 20, 0)
    for name, extra, want in [("deployment defaults", {}, deployed), ("null inherits", {"temperature": None, "top_k": None}, deployed),
                              ("request overrides one option", {"temperature": 1.0}, (1.0, deployed[1], 20, 0)), ("request overrides top_k", {"top_k": 0}, (deployed[0], deployed[1], 0, 0))]:
        status, got = record(extra)
        check(name, status == 200 and got[:4] == want, str(got))
    print(f"{'FAIL' if failures else 'PASS'}: {failures} failing checks")
    sys.exit(1 if failures else 0)
status, got = record({})
check("defaults", status == 200 and got[:4] == (1.0, 1.0, 0, 0), str(got))
default_seed_a = got[4] if status == 200 else None
status, got = record({})
check("implicit seeds differ per request", status == 200 and got[4] != default_seed_a, f"{default_seed_a} then {got[4] if status == 200 else got}")
for name, extra, want in [
    ("all options", {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 42}, (f32(0.7), f32(0.9), 40, 1, 42)),
    ("greedy", {"temperature": 0}, (0.0, 1.0, 0, 0, None)),
    ("negative seed", {"seed": -5}, (1.0, 1.0, 0, 1, -5)),
    ("large seed", {"seed": 9007199254740993}, (1.0, 1.0, 0, 1, 9007199254740993)),
    ("integers as floats", {"temperature": 2, "top_p": 1}, (2.0, 1.0, 0, 0, None)),
    ("small top_p", {"top_p": 0.001}, (1.0, f32(0.001), 0, 0, None)),
    ("tool_choice none", {"tool_choice": "none"}, (1.0, 1.0, 0, 2, None)),
    ("tool_choice none with seed", {"tool_choice": "none", "seed": 7}, (1.0, 1.0, 0, 3, 7)),
    ("tool_choice auto", {"tool_choice": "auto"}, (1.0, 1.0, 0, 0, None)),
    ("null options", {"temperature": None, "top_p": None, "top_k": None, "seed": None, "tool_choice": None}, (1.0, 1.0, 0, 0, None)),
    ("default values of unsupported fields", {"n": 1, "stop": None, "logprobs": False, "frequency_penalty": 0, "presence_penalty": 0, "repetition_penalty": 1, "min_p": 0, "response_format": None, "chat_template_kwargs": None}, (1.0, 1.0, 0, 0, None)),
    ("template kwargs object", {"chat_template_kwargs": {"enable_thinking": False}}, (1.0, 1.0, 0, 0, None)),
    ("template kwargs unknown key is ignored", {"chat_template_kwargs": {"foo": 1, "preserve_thinking": True}}, (1.0, 1.0, 0, 0, None)),
    ("unknown extension fields", {"user": "u", "metadata": {"a": 1}, "store": False, "parallel_tool_calls": True}, (1.0, 1.0, 0, 0, None)),
]:
    status, got = record(extra)
    ok = status == 200 and all(w is None or g == w for g, w in zip(got, want))
    check(name, ok, str(got))
for name, extra in [
    ("negative temperature", {"temperature": -1}), ("string temperature", {"temperature": "hot"}), ("infinite temperature", {"temperature": 1e999}),
    ("zero top_p", {"top_p": 0}), ("top_p above one", {"top_p": 1.5}), ("negative top_k", {"top_k": -1}), ("fractional top_k", {"top_k": 1.5}), ("huge top_k", {"top_k": 3000000000}),
    ("fractional seed", {"seed": 1.5}), ("string seed", {"seed": "7"}), ("n above one", {"n": 2}), ("stop", {"stop": ["x"]}), ("logprobs", {"logprobs": True}),
    ("penalty", {"presence_penalty": 0.5}), ("repetition penalty", {"repetition_penalty": 1.1}), ("min_p", {"min_p": 0.1}), ("response_format", {"response_format": {"type": "json_object"}}),
    ("template kwargs string", {"chat_template_kwargs": "x"}), ("logit_bias", {"logit_bias": {"1": 5}}),
]:
    status, got = record(extra)
    check(f"rejects {name}", status == 400, f"status {status} {got if status != 400 else ''}")
status, reply = post(b'{"model":"inkling-2-mini","messages":[{"role":"user","content":"x"}],"temperature":1,"temperature":2}')
check("rejects duplicate temperature", status == 400, f"status {status}")
print(f"{'FAIL' if failures else 'PASS'}: {failures} failing checks")
sys.exit(1 if failures else 0)
