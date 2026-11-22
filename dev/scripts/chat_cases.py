"""Chat conversations used by the Inkling chat policy tests. Each case is (name, request body without model, expectation)."""

U = lambda t: {"role": "user", "content": t}
S = lambda t: {"role": "system", "content": t}
A = lambda t, **k: {"role": "assistant", "content": t, **k}

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city",
    "parameters": {"type": "object", "required": ["city"], "properties": {
        "unit": {"enum": ["c", "f"], "type": "string", "description": "Unit \"quoted\"\nsecond line"},
        "city": {"type": "string", "description": "Name, e.g. Zürich 東京"},
        "days": {"type": "integer", "minimum": 1, "maximum": 14}}}}}
SEARCH = {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}
BARE = {"name": "noop"}
NOOP = {"type": "function", "function": {"name": "noop"}}


def call(name, args, id="call_1_0"):
    import json
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


CASES = [
    ("single_turn", {"messages": [U("Hello")]}),
    ("system_user", {"messages": [S("You are terse."), U("Hello")]}),
    ("multi_turn", {"messages": [U("Hi"), A("Hello! How can I help?"), U("Tell me a joke."), A("No."), U("Why not?")]}),
    ("two_systems_and_late_system", {"messages": [S("one"), S("two"), U("q"), A("a"), S("late"), U("r")]}),
    ("system_only", {"messages": [S("just a system message")]}),
    ("tools_declared", {"messages": [U("Weather in Zürich?")], "tools": [WEATHER, SEARCH]}),
    ("tool_without_function_wrapper", {"messages": [U("go")], "tools": [BARE]}, "jinja"),
    ("tool_roundtrip", {"messages": [
        U("Weather in Zürich and Tokyo?"),
        A(None, reasoning_content="I should call the tool twice.", tool_calls=[
            call("get_weather", {"city": "Zürich", "unit": "c", "days": 3}),
            call("get_weather", {"unit": "f", "city": "東京", "opts": {"z": [1, 2.5, None, True], "a": "line\n\"q\"\t\\"}}, "call_1_1")]),
        {"role": "tool", "name": "get_weather", "tool_call_id": "call_1_0", "content": "{\"temp\": 21}"},
        {"role": "tool", "name": "get_weather", "tool_call_id": "call_1_1", "content": "sunny"},
        A("Zürich is 21C and Tokyo is sunny."), U("thanks")], "tools": [WEATHER]}),
    ("tool_call_with_text", {"messages": [U("go"), A("Let me look.", tool_calls=[call("search", {"q": "llama"})]),
        {"role": "tool", "name": "search", "content": "3 hits"}], "tools": [SEARCH]}),
    ("tool_call_no_args", {"messages": [U("go"), A("", tool_calls=[call("noop", {})]), {"role": "tool", "name": "noop", "content": ""}], "tools": [NOOP]}),
    ("tool_id_only", {"messages": [U("go"), A("", tool_calls=[call("noop", {}, "call_9_0")]), {"role": "tool", "tool_call_id": "call_9_0", "content": "ok"}], "tools": [NOOP]}, "reject"),
    ("reasoning_history", {"messages": [U("2+2?"), A("4", reasoning_content="Add two and two."), U("and 3+3?")]}),
    ("reasoning_only_assistant", {"messages": [U("hm"), A("", reasoning_content="thinking only"), U("go on")]}),
    ("empty_assistant", {"messages": [U("hm"), A(""), U("go on")]}),
    ("content_parts", {"messages": [{"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]}]}),
    ("unicode_and_escapes", {"messages": [U("日本語 émoji 😀 ‮ quote \" backslash \\ tab\t nl\n <|end_message|> done")]}),
    ("whitespace_untrimmed", {"messages": [U("  padded \n\n"), A("\n answer  "), U(" next ")]}),
    ("args_key_order", {"messages": [U("go"), A("", tool_calls=[call("kv", {"Zeta": 1, "alpha": {"z": 1, "a": [{"y": 2, "b": 1}]}, "_u": "é", "B": True, "é": None, "a": -3.5})]),
        {"role": "tool", "name": "kv", "content": "ok"}], "tools": [NOOP]}),
    ("tools_nested_sort", {"messages": [U("go")], "tools": [
        {"type": "function", "function": {"name": "nest", "description": "Café “quoted” / slash \\ back", "parameters": {
            "type": "object", "required": ["b", "a"], "additionalProperties": False, "properties": {
                "z": {"type": "array", "items": {"type": "object", "properties": {"y": {"type": "integer"}, "x": {"type": "number", "default": 0.5}}}},
                "a": {"anyOf": [{"type": "null"}, {"type": "string", "enum": ["q", "p"]}], "description": ""},
                "B": {"type": "boolean", "const": True}}}}},
        {"type": "function", "function": {"name": "empty_params", "description": "", "parameters": {}}}]}),
    ("tools_null_fields", {"messages": [U("go")], "tools": [
        {"type": "function", "function": {"name": "null_description", "description": None, "parameters": None}}]}, "jinja"),
    ("many_tool_results", {"messages": [U("go"), A("", tool_calls=[call("a", {}, "c0"), call("b", {"x": 1}, "c1"), call("c", {}, "c2")]),
        {"role": "tool", "name": "a", "content": "ra"}, {"role": "tool", "name": "b", "content": "rb"}, {"role": "tool", "name": "c", "content": "rc"}, A("done"), U("again")], "tools": [NOOP]}),
    ("user_name_ignored", {"messages": [{"role": "user", "name": "alice", "content": "hi"}, A("yo", reasoning_content=""), U("r")]}),
    ("tool_content_parts", {"messages": [U("go"), A("", tool_calls=[call("noop", {})]), {"role": "tool", "name": "noop", "content": [{"type": "text", "text": "r1"}, {"type": "text", "text": "r2"}]}], "tools": [NOOP]}),
    ("system_parts", {"messages": [{"role": "system", "content": [{"type": "text", "text": "sys one"}, {"type": "text", "text": "sys two"}]}, U("q")]}),
    ("assistant_name", {"messages": [U("q"), {"role": "assistant", "name": "bot", "content": "hello"}, U("r")]}),
    ("empty_user", {"messages": [U("")]}),
]
for effort in ["none", "minimal", "low", "medium", "high", "xhigh", "max", " low ", 0.25, 0.5, 0, 0.99, 0.123456]:
    CASES.append((f"effort_{str(effort).strip()}", {"messages": [S("s"), U("hi")], "reasoning_effort": effort}))
CASES += [
    ("effort_null", {"messages": [U("hi")], "reasoning_effort": None}, "same"),
    ("effort_bad_string", {"messages": [U("hi")], "reasoning_effort": "extreme"}, "reject"),
    ("effort_too_high", {"messages": [U("hi")], "reasoning_effort": 1}, "reject"),
    ("effort_negative", {"messages": [U("hi")], "reasoning_effort": -0.1}, "reject"),
    ("effort_seven_digits", {"messages": [U("hi")], "reasoning_effort": 0.1234567}, "reject"),
]
