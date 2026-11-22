"""Writes package/test/chat/golden.json: the exact prompt for every chat case from two references.
oracle = llama-server /apply-template (minja, the llama.cpp reference engine); jinja = Python jinja2 with the Hugging Face tojson.
usage: chat_golden.py <oracle port> <chat_template.jinja> <out.json>   (needs jinja2)"""
import json, re, sys, urllib.request
import jinja2
from chat_cases import CASES

port, template_path, out = sys.argv[1:4]
source = open(template_path, encoding="utf-8").read()
source = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", source)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class Raised(Exception):
    pass


def raise_exception(message):
    raise Raised(message)


env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
template = env.from_string(source)


def python_render(body):
    messages = json.loads(json.dumps(body["messages"]))
    for m in messages:
        for c in m.get("tool_calls") or []:
            c["function"]["arguments"] = json.loads(c["function"]["arguments"])
    kwargs = {"messages": messages, "tools": body.get("tools")}
    if "reasoning_effort" in body:
        kwargs["reasoning_effort"] = body["reasoning_effort"]
    return template.render(**kwargs)


def oracle_render(body):
    body = dict(body)
    if "reasoning_effort" in body:
        body["chat_template_kwargs"] = {"reasoning_effort": body.pop("reasoning_effort")}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/apply-template", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req))["prompt"]
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:200]}


result = []
for case in CASES:
    name, body = case[0], case[1]
    entry = {"name": name, "body": body, "expect": case[2] if len(case) > 2 else "match"}
    try:
        entry["jinja"] = python_render(body)
    except Exception as e:
        entry["jinja"] = {"error": str(e)[:200]}
    entry["oracle"] = oracle_render(body)
    both_error = isinstance(entry["jinja"], dict) and isinstance(entry["oracle"], dict)
    same = entry["jinja"] == entry["oracle"] or both_error or entry["expect"] in ("jinja", "reject")
    entry["engines_agree"] = same
    result.append(entry)
    flag = "agree" if same else "DIFFER"
    print(f"{name:32s} {flag}", "" if same else f"\n  jinja : {entry['jinja']!r}\n  oracle: {entry['oracle']!r}")
json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
