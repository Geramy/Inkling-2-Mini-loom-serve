"""Adds the policy cases (deliberate differences from the template, docs/CHAT_POLICY.md section 2) to package/test/chat/golden.json.
The reference of a policy case is the template's own render of the equivalent message list (jinja2, no oracle engine): a tool message without name gets an id that
no call carries, a developer message is a system message, the name of an assistant message with calls is dropped, the reasoning field is reasoning_content,
tool_choice none removes the tools.
usage: chat_golden_policy.py <chat_template.jinja> <golden.json> [out.json]   (needs jinja2; rewrites golden.json when out is not given)"""
import copy
import json
import re
import sys

import jinja2
from chat_cases import A, S, U, WEATHER, SEARCH, NOOP, call

source = open(sys.argv[1], encoding="utf-8").read()
source = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", source)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
template = env.from_string(source)


def equivalent(body):
    body = copy.deepcopy(body)
    for m in body["messages"]:
        if m["role"] == "tool" and not m.get("name"):
            m["tool_call_id"] = "unresolved"
        if m["role"] == "developer":
            m["role"] = "system"
        if m["role"] == "assistant" and m.get("tool_calls"):
            m.pop("name", None)
        if m["role"] == "assistant" and "reasoning" in m:
            m["reasoning_content"] = m.pop("reasoning")
        for c in m.get("tool_calls") or []:
            c["function"]["arguments"] = json.loads(c["function"]["arguments"])
    choice, tools = body.get("tool_choice"), body.get("tools")
    if choice == "none":
        tools = None
    elif tools and (choice == "required" or isinstance(choice, dict)):
        if isinstance(choice, dict):
            tools = [t for t in tools if (t.get("function") or t).get("name") == (choice.get("function") or choice).get("name")]
        body["messages"].append({"role": "system", "content": "The next assistant message must be a call of a declared tool. Do not answer in text."})
    kwargs = {"messages": body["messages"], "tools": tools}
    if "reasoning_effort" in body:
        kwargs["reasoning_effort"] = body["reasoning_effort"]
    return template.render(**kwargs)


TOOL = {"role": "tool", "content": "ok"}
POLICY = [
    ("tool_id_only", {"messages": [U("go"), A("", tool_calls=[call("noop", {}, "call_9_0")]), {"role": "tool", "tool_call_id": "call_9_0", "content": "ok"}], "tools": [NOOP]}),
    ("tool_id_only_parallel", {"messages": [U("go"), A("", reasoning_content="Two lookups.", tool_calls=[call("noop", {}, "call_9_0"), call("search", {"q": "x"}, "call_9_1")]),
                                            {"role": "tool", "tool_call_id": "call_9_0", "content": "a"}, {"role": "tool", "tool_call_id": "call_9_1", "content": "b"},
                                            A("done"), U("again")], "tools": [NOOP, SEARCH]}),
    ("tool_id_only_and_named_mixed", {"messages": [U("go"), A("", tool_calls=[call("noop", {}, "c0"), call("search", {"q": "x"}, "c1")]),
                                                   {"role": "tool", "tool_call_id": "c0", "content": "a"}, {"role": "tool", "name": "search", "tool_call_id": "c1", "content": "b"}], "tools": [NOOP, SEARCH]}),
    ("developer_role", {"messages": [{"role": "developer", "content": "Be terse."}, U("hi")]}),
    ("developer_role_with_tools", {"messages": [{"role": "developer", "content": "Be terse."}, U("Weather?")], "tools": [WEATHER], "reasoning_effort": "low"}),
    ("assistant_name_with_calls", {"messages": [U("go"), A("", name="bot", tool_calls=[call("noop", {}, "c0")]), {"role": "tool", "name": "noop", "content": "ok"}], "tools": [NOOP]}),
    ("assistant_name_equals_call", {"messages": [U("go"), A("Checking.", name="noop", tool_calls=[call("noop", {}, "c0")]), {"role": "tool", "name": "noop", "content": "ok"}], "tools": [NOOP]}),
    ("reasoning_field_alias", {"messages": [U("2+2?"), A("4", reasoning="Add two and two."), U("and 3+3?")]}),
    ("reasoning_field_alias_with_calls", {"messages": [U("go"), A("", reasoning="Need the tool.", tool_calls=[call("noop", {}, "c0")]), {"role": "tool", "name": "noop", "content": "ok"}], "tools": [NOOP]}),
    ("tool_choice_none_drops_tools", {"messages": [U("Weather in Zürich?")], "tools": [WEATHER, SEARCH], "tool_choice": "none"}),
    ("tool_choice_none_with_history", {"messages": [U("go"), A("", tool_calls=[call("noop", {}, "c0")]), {"role": "tool", "name": "noop", "content": "ok"}], "tools": [NOOP], "tool_choice": "none"}),
    ("tool_choice_auto_keeps_tools", {"messages": [U("Weather in Zürich?")], "tools": [WEATHER], "tool_choice": "auto"}),
    ("tool_choice_required_demands_call", {"messages": [U("Weather in Zürich?")], "tools": [WEATHER, SEARCH], "tool_choice": "required"}),
    ("tool_choice_named_keeps_one_tool", {"messages": [U("Weather in Zürich?")], "tools": [WEATHER, SEARCH], "tool_choice": {"type": "function", "function": {"name": "search"}}}),
    ("tool_choice_required_after_tool_result", {"messages": [U("go"), A("", tool_calls=[call("noop", {}, "c0")]), {"role": "tool", "name": "noop", "content": "ok"}], "tools": [NOOP], "tool_choice": "required", "reasoning_effort": "low"}),
    ("kwargs_unknown_key_ignored", {"messages": [U("hi")], "chat_template_kwargs": {"preserve_thinking": True, "enable_thinking": False}, "reasoning_effort": None}),
]

golden = json.load(open(sys.argv[2], encoding="utf-8"))
names = {n for n, _ in POLICY}
golden = [g for g in golden if g["name"] not in names]
for name, body in POLICY:
    if name == "kwargs_unknown_key_ignored":
        entry = {"name": name, "body": {"messages": body["messages"], "chat_template_kwargs": body["chat_template_kwargs"]},
                 "expect": "policy", "jinja": template.render(messages=body["messages"], tools=None, reasoning_effort="none"), "oracle": None, "engines_agree": True}
    else:
        entry = {"name": name, "body": body, "expect": "policy", "jinja": equivalent(body), "oracle": None, "engines_agree": True}
    golden.append(entry)
json.dump(golden, open(sys.argv[3] if len(sys.argv) > 3 else sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(POLICY), "policy cases,", len(golden), "cases in total")
