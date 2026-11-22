"""Output channel check on a server built with scripts/chat_script_kernel.py (scenarios.py): content, reasoning and tool calls for JSON and
streamed responses, truncation, and reuse of the retained state when the client echoes the assistant message.
usage: test_output.py <port>   exit code 1 on any FAIL"""
import json, os, sys, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import SCENARIOS

PORT = sys.argv[1]
MODEL = "inkling-2-mini"


def request(doc, stream=False):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(doc).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as reply:
            if not stream:
                return reply.status, json.load(reply)
            events = [json.loads(line[6:]) for line in reply.read().decode().splitlines() if line.startswith("data: {")]
            return reply.status, events
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def prompt(k):
    return {"model": MODEL, "messages": [{"role": "user", "content": "Hello" + " Hello" * k}]}


def parse_message(message):
    calls = [(c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls") or []]
    return message.get("content") or "", message.get("reasoning_content") or message.get("reasoning") or "", calls


failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(("PASS" if ok else "FAIL"), name, detail)


for k, (name, _segments, max_tokens, content, reasoning, calls, finish) in enumerate(SCENARIOS):
    doc = dict(prompt(k), max_tokens=max_tokens)
    status, reply = request(doc)
    if finish == "error":
        check(f"{name} json", status >= 400, f"status {status}")
        status, events = request(dict(doc, stream=True), stream=True)
        check(f"{name} stream", status >= 400 or any("error" in e for e in events) if isinstance(events, list) else status >= 400, f"status {status}")
        continue
    if status != 200:
        check(f"{name} json", False, f"status {status}: {str(reply)[:200]}")
        continue
    choice = reply["choices"][0]
    got = parse_message(choice["message"])
    ok = got == (content, reasoning, calls) and choice["finish_reason"] == finish
    check(f"{name} json", ok, "" if ok else f"got {got} finish {choice['finish_reason']}")
    status, events = request(dict(doc, stream=True), stream=True)
    text, thought, streamed_calls, last_finish = "", "", {}, None
    for event in events:
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            text += delta.get("content") or ""
            thought += delta.get("reasoning_content") or delta.get("reasoning") or ""
            for call in delta.get("tool_calls") or []:
                entry = streamed_calls.setdefault(call["index"], [None, ""])
                entry[0] = (call.get("function") or {}).get("name") or entry[0]
                entry[1] += (call.get("function") or {}).get("arguments") or ""
            last_finish = choice.get("finish_reason") or last_finish
    streamed = (text, thought, [tuple(v) for _, v in sorted(streamed_calls.items())])
    ok = streamed == (content, reasoning, calls) and last_finish == finish
    check(f"{name} stream", ok, "" if ok else f"got {streamed} finish {last_finish}")

# The client echoes the assistant turn of think_then_text: the retained state is reused, a client that drops the reasoning replays.
k = 1
status, first = request(dict(prompt(k), max_tokens=64))
first_usage = first["usage"]
history = prompt(k)["messages"] + [{"role": "assistant", "content": "answer", "reasoning_content": "think"}, {"role": "user", "content": "Next"}]
status, second = request({"model": MODEL, "messages": history, "max_tokens": 4})
cached = second["usage"]["prompt_tokens_details"]["cached_tokens"] if status == 200 else -1
check("reuse echoed assistant turn", cached >= first_usage["prompt_tokens"], f"cached {cached} of first prompt {first_usage['prompt_tokens']} and {first_usage['completion_tokens']} generated")
status, first = request(dict(prompt(k), max_tokens=64))
history = prompt(k)["messages"] + [{"role": "assistant", "content": "answer"}, {"role": "user", "content": "Next"}]
status, third = request({"model": MODEL, "messages": history, "max_tokens": 4})
cached = third["usage"]["prompt_tokens_details"]["cached_tokens"] if status == 200 else -1
check("dropped reasoning replays after the prompt", 0 <= cached <= first["usage"]["prompt_tokens"], f"cached {cached}")
print(f"{'FAIL' if failures else 'PASS'}: {failures} failing checks")
sys.exit(1 if failures else 0)
