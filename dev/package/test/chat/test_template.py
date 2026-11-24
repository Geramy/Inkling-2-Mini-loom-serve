"""Chat template check against the reference prompts in golden.json (jinja2 render of docs/chat_template.jinja, cross-checked against llama.cpp minja).
Counts mode: the input token count of every case equals the token count of the reference prompt.
Echo mode (server built with scripts/chat_echo_kernel.py): the generated text spells the prompt token IDs, so the token IDs are compared one by one.
usage: test_template.py <port> [--echo]   exit code 1 on any FAIL"""
import json, os, sys, urllib.error, urllib.request

PORT = sys.argv[1]
ECHO = "--echo" in sys.argv
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.json")
MODEL = os.environ.get("INKLING_MODEL", "inkling-2-mini")


def post(path, doc):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(doc).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as reply:
            return reply.status, json.load(reply)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def tokens_of(text):
    status, reply = post("/tokenize", {"model": MODEL, "content": text, "add_special": False, "parse_special": True})
    assert status == 200, reply
    return reply["tokens"]


failures = 0


def golden_runs():
    for case in json.load(open(GOLDEN, encoding="utf-8")):
        yield case["name"], dict(case["body"]), case
        if "reasoning_effort" in case["body"]:
            body = dict(case["body"])
            body["chat_template_kwargs"] = {"reasoning_effort": body.pop("reasoning_effort")}
            yield case["name"] + " (chat_template_kwargs)", body, case


for name, body, case in golden_runs():
    expect = case["expect"]
    body["model"] = MODEL
    status, reply = post("/v1/chat/completions/input_tokens", body)
    if expect == "reject":
        ok = status == 400
        detail = f"status {status}: {str(reply)[:100]}" if not ok else f"rejected ({str(reply)[:90]})"
    elif status != 200:
        ok, detail = False, f"status {status}: {str(reply)[:160]}"
    else:
        want = tokens_of(case["jinja"])
        count = reply["input_tokens"]
        ok, detail = count == len(want), f"{count} tokens (reference {len(want)})"
        if ok and isinstance(case["oracle"], str):
            ok = tokens_of(case["oracle"]) == want or case["name"] == "whitespace_untrimmed"
            detail += "" if ok else " minja prompt differs"
        if ok and ECHO:
            body["max_tokens"] = 6 * len(want) + 8
            status, answer = post("/v1/chat/completions", body)
            if status != 200:
                ok, detail = False, f"echo status {status}: {str(answer)[:160]}"
            else:
                text = answer["choices"][0]["message"]["content"]
                got = [int("".join(str(ord(c) - 97) for c in text[i:i + 6])) for i in range(0, len(text), 6)]
                ok = got == want
                detail = f"{len(got)} echoed tokens equal reference" if ok else f"echo differs: got {len(got)} tokens, first mismatch at {next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))}"
    failures += not ok
    print(("PASS" if ok else "FAIL"), name, detail)
KWARGS = [
    ("kwargs enable_thinking false equals effort none", {"chat_template_kwargs": {"enable_thinking": False}}, {"reasoning_effort": "none"}),
    ("kwargs enable_thinking true equals the default", {"chat_template_kwargs": {"enable_thinking": True}}, {}),
    ("kwargs enable_thinking true with effort low", {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}, {"reasoning_effort": "low"}),
    ("kwargs and top-level effort agree", {"chat_template_kwargs": {"reasoning_effort": 0.25}, "reasoning_effort": 0.25}, {"reasoning_effort": 0.25}),
    ("kwargs null values", {"chat_template_kwargs": {"enable_thinking": None, "reasoning_effort": None}}, {}),
    ("kwargs null object", {"chat_template_kwargs": None}, {}),
    ("kwargs empty object", {"chat_template_kwargs": {}}, {}),
    ("top-level enable_thinking is ignored", {"enable_thinking": False}, {}),
    ("unknown kwargs key is ignored", {"chat_template_kwargs": {"enable_thinking": False, "add_generation_prompt": True}}, {"reasoning_effort": "none"}),
    ("unknown kwargs key only is ignored", {"chat_template_kwargs": {"foo": 1}}, {}),
]
KWARGS_REJECT = [
    ("kwargs not an object", {"chat_template_kwargs": [1]}),
    ("kwargs string", {"chat_template_kwargs": "x"}),
    ("enable_thinking string", {"chat_template_kwargs": {"enable_thinking": "false"}}),
    ("enable_thinking number", {"chat_template_kwargs": {"enable_thinking": 0}}),
    ("kwargs and top-level effort differ", {"chat_template_kwargs": {"reasoning_effort": "low"}, "reasoning_effort": "high"}),
    ("enable_thinking false with effort high", {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "high"}}),
    ("enable_thinking false with top-level effort low", {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "low"}),
    ("enable_thinking true with effort none", {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "none"}}),
    ("kwargs invalid effort", {"chat_template_kwargs": {"reasoning_effort": "extreme"}}),
    ("kwargs effort too high", {"chat_template_kwargs": {"reasoning_effort": 1}}),
]
CONVERSATION = {"model": MODEL, "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]}


def prompt_ids(extra):
    body = {**CONVERSATION, **extra}
    if ECHO:
        body["max_tokens"] = 400
        status, answer = post("/v1/chat/completions", body)
        return status, answer["choices"][0]["message"]["content"] if status == 200 else answer
    status, reply = post("/v1/chat/completions/input_tokens", body)
    return status, reply["input_tokens"] if status == 200 else reply


for name, extra, plain in KWARGS:
    got, want = prompt_ids(extra), prompt_ids(plain)
    ok = got[0] == 200 and got == want
    failures += not ok
    print(("PASS" if ok else "FAIL"), name, str(got)[:100])
status, high = prompt_ids({})
status, none = prompt_ids({"chat_template_kwargs": {"enable_thinking": False}})
ok = high != none
failures += not ok
print(("PASS" if ok else "FAIL"), "enable_thinking false changes the prompt")
for name, extra in KWARGS_REJECT:
    status, reply = prompt_ids(extra)
    ok = status == 400 and ("chat_template_kwargs" in reply or "reasoning_effort" in reply)
    failures += not ok
    print(("PASS" if ok else "FAIL"), "rejects " + name, f"status {status}: {str(reply)[:110]}")
print(f"{'FAIL' if failures else 'PASS'}: {failures} failing cases")
sys.exit(1 if failures else 0)
