"""usage: chat_tool_probe.py [--only a,b] [--out F] [--effort E] [--max-tokens N] URL
OpenAI-compatible tool flows against the chat endpoint. Every case prints one JSON line {case, ok, status, note, ...}.
ok is judged on the protocol (status, finish_reason, call shape), not on the model's wording. Cases that depend on the model are marked model=true."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatlib import cached, chat

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--only", default="")
ap.add_argument("--out", default="")
ap.add_argument("--effort", default="low")
ap.add_argument("--max-tokens", type=int, default=2500)
a = ap.parse_args()

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "description": "Search the web.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "filters": {"type": "object", "properties": {"lang": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}}}, "required": ["query"]}}}
NOTE = {"type": "function", "function": {"name": "save_note", "description": "Save a note with the given text.", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "text": {"type": "string"}}, "required": ["title", "text"]}}}
STEP = {"type": "function", "function": {"name": "next_step", "description": "Returns the next step number and a value. Call it again until it says done.", "parameters": {"type": "object", "properties": {"after": {"type": "integer"}}, "required": ["after"]}}}
NOARGS = {"type": "function", "function": {"name": "get_time", "description": "Current time.", "parameters": {"type": "object", "properties": {}}}}
TOOLS = [WEATHER, SEARCH]
RESULTS = []


def kw(**extra):
    d = {"temperature": 0, "max_tokens": a.max_tokens, "reasoning_effort": a.effort}
    d.update(extra)
    return d


def call_msg(calls, content=None, **extra):
    m = {"role": "assistant", "content": content, "tool_calls": [{"id": cid, "type": "function", "function": {"name": n, "arguments": args}} for cid, n, args in calls]}
    m.update(extra)
    return m


def report(case, ok, r=None, note="", model=False, **extra):
    row = {"case": case, "ok": bool(ok), "status": r["status"] if r else None, "model": model, "note": note}
    if r:
        row["finish"] = r.get("finish")
        row["prompt_cached"] = cached(r)
        if r.get("error"):
            row["error"] = (r["error"].get("error", r["error"]) if isinstance(r["error"], dict) else r["error"])
    row.update(extra)
    RESULTS.append(row)
    print(json.dumps(row, ensure_ascii=False)[:900], flush=True)


def valid_json(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def flow_first(stream):
    r = chat(a.url, [{"role": "user", "content": "What is the weather in Paris for 3 days? Use the tool."}], stream=stream, tools=TOOLS, **kw())
    calls = r["tool_calls"]
    ok = r["status"] == 200 and r["finish"] == "tool_calls" and len(calls) >= 1 and calls[0]["name"] == "get_weather" and valid_json(calls[0]["arguments"]) is not None
    report("basic_call_" + ("stream" if stream else "json"), ok, r, f"calls={[(c['name'], c['arguments']) for c in calls]} content={r['content'][:60]!r}", model=True)
    return r


def check_stream_shape():
    r = chat(a.url, [{"role": "user", "content": "What is the weather in Paris for 3 days? Use the tool."}], stream=True, tools=TOOLS, **kw())
    frames = [c for c in r["raw"] if c.get("choices")]
    tcs = [(c["choices"][0]["delta"].get("tool_calls") or []) for c in frames]
    flat = [t for l in tcs for t in l]
    names = [t["function"].get("name") for t in flat if t.get("function", {}).get("name")]
    ids = [t.get("id") for t in flat if t.get("id")]
    idx = sorted({t.get("index") for t in flat})
    fin = [c["choices"][0].get("finish_reason") for c in frames if c["choices"][0].get("finish_reason")]
    first = flat[0] if flat else {}
    ok = (r["status"] == 200 and len(names) == len(idx) and len(ids) == len(idx) and fin == ["tool_calls"] and first.get("type") == "function"
          and valid_json("".join(t.get("function", {}).get("arguments") or "" for t in flat if t.get("index") == idx[0])) is not None if flat else False)
    role_first = frames[0]["choices"][0]["delta"].get("role") if frames else None
    report("stream_tool_delta_shape", ok, r, f"index={idx} ids={ids} names={names} finish={fin} first_type={first.get('type')} frames_with_calls={sum(1 for l in tcs if l)} first_role={role_first} usage={r['usage']}", model=True)


def history(extra_tool=None, args_as="string", tool_content="Sunny, 21 C", with_name=False, with_id=True, content=None, reasoning=None, asst_extra=None):
    args = {"city": "Paris", "days": 3}
    a_val = json.dumps(args) if args_as == "string" else args
    m = call_msg([("call_abc123", "get_weather", a_val)], content=content)
    if reasoning:
        m["reasoning_content"] = reasoning
    m.update(asst_extra or {})
    t = {"role": "tool", "content": tool_content}
    if with_id:
        t["tool_call_id"] = "call_abc123"
    if with_name:
        t["name"] = "get_weather"
    return [{"role": "user", "content": "What is the weather in Paris for 3 days? Use the tool."}, m, t]


def final_answer(case, msgs, model=True, tools=TOOLS, expect_status=200, **extra):
    any_finish = extra.pop("model_ok", False)
    r = chat(a.url, msgs, stream=extra.pop("stream", False), tools=tools, **kw(**extra))
    ok = r["status"] == expect_status and (expect_status != 200 or any_finish or (r["finish"] in ("stop",) and (r["content"] or r["reasoning"])))
    report(case, ok, r, f"content={r['content'][:100]!r}", model=model)
    return r


def loop_case(case, with_name, n_rounds, echo_reasoning=False, stream=False):
    msgs = [{"role": "user", "content": "Call next_step with after=0, then keep calling next_step with the returned step number until it reports done. Then tell me the last step number."}]
    rounds = 0
    ids = 0
    cache_rows = []
    last = None
    while rounds < n_rounds + 4:
        r = chat(a.url, msgs, stream=stream, tools=[STEP], **kw(max_tokens=1500))
        last = r
        cache_rows.append(cached(r))
        if r["status"] != 200:
            report(case, False, r, f"round {rounds} failed", model=False)
            return
        if not r["tool_calls"]:
            break
        calls = r["tool_calls"]
        am = {"role": "assistant", "content": r["content"] or None, "tool_calls": [{"id": c["id"] or f"call_{ids + i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for i, c in enumerate(calls)]}
        if echo_reasoning and r["reasoning"]:
            am["reasoning_content"] = r["reasoning"]
        msgs.append(am)
        for i, c in enumerate(calls):
            args = valid_json(c["arguments"]) or {}
            after = args.get("after", 0) if isinstance(args, dict) else 0
            nxt = after + 1
            res = json.dumps({"step": nxt, "value": nxt * nxt, "done": nxt >= n_rounds})
            tm = {"role": "tool", "tool_call_id": c["id"] or f"call_{ids + i}", "content": res}
            if with_name:
                tm["name"] = c["name"]
            msgs.append(tm)
        ids += len(calls)
        rounds += 1
    ok = last["status"] == 200 and not last["tool_calls"] and rounds >= n_rounds and str(n_rounds) in (last["content"] or "")
    report(case, ok, last, f"rounds={rounds} final={last['content'][:80]!r} cache(prompt,cached,completion)={cache_rows}", model=True)


def want(name):
    return not a.only or name in a.only.split(",")


def main():
    if want("basic"):
        flow_first(False)
        flow_first(True)
        check_stream_shape()
    if want("results"):
        final_answer("tool_result_name_and_id", history(with_name=True))
        final_answer("tool_result_id_only", history(with_name=False))
        final_answer("tool_result_name_only", history(with_name=True, with_id=False))
        final_answer("tool_result_args_object_runtime_limit", history(with_name=True, args_as="object"), expect_status=400)
        final_answer("tool_result_id_only_args_object_runtime_limit", history(with_name=False, args_as="object"), expect_status=400)
        final_answer("assistant_content_empty_string", history(with_name=True, content=""))
        final_answer("assistant_content_omitted", [m if m["role"] != "assistant" else {k: v for k, v in m.items() if k != "content"} for m in history(with_name=True)])
        final_answer("assistant_with_name_and_calls", history(with_name=True, asst_extra={"name": "bot"}))
        final_answer("assistant_reasoning_replay", history(with_name=True, reasoning="I should call the weather tool for Paris."))
        final_answer("tool_content_parts", history(with_name=True, tool_content=[{"type": "text", "text": "Sunny, 21 C"}]))
        final_answer("tool_content_json_object", history(with_name=True, tool_content={"temp": 21}), expect_status=400)
        final_answer("tool_content_empty", history(with_name=True, tool_content=""))
        final_answer("tool_msg_extra_fields", [dict(m, **({"is_error": False} if m["role"] == "tool" else {})) for m in history(with_name=True)])
        final_answer("tool_call_without_type_runtime_limit", [m if m["role"] != "assistant" else dict(m, tool_calls=[{k: v for k, v in c.items() if k != "type"} for c in m["tool_calls"]]) for m in history(with_name=True)], expect_status=400)
        final_answer("tool_call_empty_arguments", [m if m["role"] != "assistant" else dict(m, tool_calls=[dict(c, function={"name": "get_weather", "arguments": ""}) for c in m["tool_calls"]]) for m in history(with_name=True)], model_ok=True)
        final_answer("tool_call_arguments_non_object", [m if m["role"] != "assistant" else dict(m, tool_calls=[dict(c, function={"name": "get_weather", "arguments": "[1]"}) for c in m["tool_calls"]]) for m in history(with_name=True)], expect_status=400)
    if want("parallel"):
        msgs = [{"role": "user", "content": "Get the weather in Paris and in Rome with two tool calls in the same turn."}]
        r = chat(a.url, msgs, tools=TOOLS, **kw())
        two = r["status"] == 200 and r["finish"] == "tool_calls" and len(r["tool_calls"]) >= 2
        ids = [c["id"] for c in r["tool_calls"]]
        report("parallel_calls_json", two and len(set(ids)) == len(ids), r, f"calls={[(c['id'], c['name'], c['arguments']) for c in r['tool_calls']]}", model=True)
        r2 = chat(a.url, msgs, stream=True, tools=TOOLS, **kw())
        report("parallel_calls_stream", r2["status"] == 200 and r2["finish"] == "tool_calls" and len(r2["tool_calls"]) >= 2 and [c["frames"] for c in r2["tool_calls"]] and all(c["names"] == 1 for c in r2["tool_calls"]), r2,
               f"calls={[(c['id'], c['name'], c['arguments'], c['frames']) for c in r2['tool_calls']]}", model=True)
        if two:
            hist = msgs + [call_msg([(c["id"], c["name"], c["arguments"]) for c in r["tool_calls"]])]
            for c in r["tool_calls"]:
                city = (valid_json(c["arguments"]) or {}).get("city", "?")
                hist.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": f"{city}: 20 C"})
            final_answer("parallel_results_named", hist)
            hist2 = [m if m["role"] != "tool" else {k: v for k, v in m.items() if k != "name"} for m in hist]
            final_answer("parallel_results_id_only", hist2)
    if want("args"):
        r = chat(a.url, [{"role": "user", "content": "Search for the Japanese words 東京 ラーメン with filters lang ja and tags [\"food\", \"café\"]. Use the search tool."}], tools=TOOLS, **kw())
        c = r["tool_calls"][0] if r["tool_calls"] else None
        args = valid_json(c["arguments"]) if c else None
        report("unicode_nested_args", bool(args) and "東京" in c["arguments"] and isinstance(args.get("filters"), dict), r, f"args={c['arguments'][:200] if c else None}", model=True)
        long_text = "The quick brown fox jumps over the lazy dog near the river bank."
        r = chat(a.url, [{"role": "user", "content": f"Save a note titled 'long' whose text is this paragraph repeated 40 times, separated by spaces: {long_text}"}], tools=[NOTE], **kw(max_tokens=4000, reasoning_effort="none"))
        c = r["tool_calls"][0] if r["tool_calls"] else None
        args = valid_json(c["arguments"]) if c else None
        report("long_arguments", bool(args) and len(args.get("text", "")) > 2500, r, f"len={len(args.get('text', '')) if args else None}", model=True)
        r = chat(a.url, [{"role": "user", "content": "What time is it? Use the tool."}], tools=[NOARGS], **kw())
        c = r["tool_calls"][0] if r["tool_calls"] else None
        report("no_arguments_tool", bool(c) and valid_json(c["arguments"]) == {}, r, f"args={c['arguments'] if c else None}", model=True)
    if want("choice"):
        u = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
        r = chat(a.url, u, tools=TOOLS, **kw(tool_choice="none"))
        report("tool_choice_none", r["status"] == 200 and not r["tool_calls"] and r["finish"] == "stop", r, f"content={r['content'][:80]!r}", model=True)
        r = chat(a.url, [{"role": "user", "content": "Say hello."}], tools=TOOLS, **kw(tool_choice="required"))
        report("tool_choice_required", r["status"] == 200 and len(r["tool_calls"]) >= 1, r, f"calls={len(r['tool_calls'])} content={r['content'][:60]!r}", model=True)
        r = chat(a.url, [{"role": "user", "content": "Say hello."}], tools=TOOLS, **kw(tool_choice={"type": "function", "function": {"name": "search"}}))
        report("tool_choice_named", r["status"] == 200 and len(r["tool_calls"]) >= 1 and r["tool_calls"][0]["name"] == "search", r, f"calls={[c['name'] for c in r['tool_calls']]}", model=True)
        r = chat(a.url, u, tools=TOOLS, **kw(tool_choice="auto", parallel_tool_calls=False))
        report("tool_choice_auto_parallel_false", r["status"] == 200 and r["finish"] == "tool_calls" and len(r["tool_calls"]) == 1, r, "", model=True)
    if want("shapes"):
        u = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
        for label, tools in [("tool_without_type", [{"function": WEATHER["function"]}]), ("flat_tool", [WEATHER["function"]]),
                             ("tool_strict_flag", [{"type": "function", "function": dict(WEATHER["function"], strict=True)}]),
                             ("tool_params_null", [{"type": "function", "function": {"name": "get_time", "description": "t", "parameters": None}}]),
                             ("tool_description_null", [{"type": "function", "function": {"name": "get_time", "description": None, "parameters": {"type": "object", "properties": {}}}}]),
                             ("tool_no_description", [{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}])]:
            r = chat(a.url, u, tools=tools, **kw(max_tokens=200))
            report(label, r["status"] == 200, r, "")
        r = chat(a.url, u, tools=[], **kw(max_tokens=200))
        report("tools_empty_list", r["status"] == 200, r, "")
        for label, role in [("developer_role", "developer"), ("system_role", "system")]:
            r = chat(a.url, [{"role": role, "content": "Be terse."}] + u, tools=TOOLS, **kw(max_tokens=200))
            report(label, r["status"] == 200, r, "")
        r = chat(a.url, u + [{"role": "system", "content": "Reminder: be terse."}], **kw(max_tokens=200))
        report("system_after_user", r["status"] == 200, r, "")
        r = chat(a.url, [{"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "there"}]}], **kw(max_tokens=100))
        report("user_content_parts", r["status"] == 200, r, "")
        r = chat(a.url, [{"role": "user", "content": "Hi", "name": "alice"}], **kw(max_tokens=100))
        report("user_name_field", r["status"] == 200, r, "")
        r = chat(a.url, [{"role": "function", "name": "get_weather", "content": "x"}, {"role": "user", "content": "Hi"}], **kw(max_tokens=100))
        report("legacy_function_role_rejected", r["status"] == 400, r, "legacy role is not supported", model=False)
    if want("effort"):
        u = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
        for label, extra in [("effort_none", {"reasoning_effort": "none"}), ("effort_high", {"reasoning_effort": "high"}), ("effort_number", {"reasoning_effort": 0.5}),
                             ("kwargs_enable_thinking_false", {"reasoning_effort": None, "chat_template_kwargs": {"enable_thinking": False}}),
                             ("kwargs_effort_low", {"reasoning_effort": None, "chat_template_kwargs": {"reasoning_effort": "low"}}),
                             ("top_enable_thinking_false", {"enable_thinking": False}),
                             ("openai_reasoning_object", {"reasoning": {"effort": "low"}}),
                             ("kwargs_unknown_key_ignored", {"chat_template_kwargs": {"preserve_thinking": True}})]:
            k = kw(max_tokens=1500)
            k.update(extra)
            if k.get("reasoning_effort", 1) is None:
                k.pop("reasoning_effort")
            r = chat(a.url, u, tools=TOOLS, **k)
            report(label, r["status"] == 200 and r["finish"] in ("tool_calls", "stop", "length"), r, f"reasoning_chars={len(r['reasoning'])} calls={[c['name'] for c in r['tool_calls']]}", model=True)
    if want("loop"):
        loop_case("loop6_named", True, 6)
        loop_case("loop6_id_only", False, 6)
        loop_case("loop6_id_only_echo_reasoning", False, 6, echo_reasoning=True)
        loop_case("loop6_named_stream", True, 6, stream=True)
    ok = sum(1 for r in RESULTS if r["ok"])
    print(f"SUMMARY {ok}/{len(RESULTS)} ok; failing: {[r['case'] for r in RESULTS if not r['ok']]}")
    if a.out:
        open(a.out, "a").write("\n".join(json.dumps(r, ensure_ascii=False) for r in RESULTS) + "\n")


main()
