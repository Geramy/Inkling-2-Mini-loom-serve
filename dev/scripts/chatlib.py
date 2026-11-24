"""Small OpenAI-compatible client (urllib only) for the chat cache and tool probes."""
import json
import time
import urllib.error
import urllib.request


def model_id(url):
    return json.load(urllib.request.urlopen(url + "/v1/models", timeout=10))["data"][0]["id"]


def post(url, path, body, timeout=900):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw}


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=30) as r:
        raw = r.read().decode("utf8", "replace")
    try:
        return json.loads(raw)
    except Exception:
        return raw


def chat(url, messages, stream=False, tools=None, model=None, **kw):
    """Returns dict: status, ttft (first content/reasoning/tool delta, or total when not streaming), total, usage, content, reasoning,
    tool_calls (list of {id,name,arguments}), finish, raw (non-stream body or list of chunks), error."""
    body = {"model": model or model_id(url), "messages": messages}
    if tools is not None:
        body["tools"] = tools
    body.update(kw)
    t0 = time.time()
    out = {"status": 0, "ttft": None, "usage": None, "content": "", "reasoning": "", "tool_calls": [], "finish": None, "error": None}
    if not stream:
        status, js = post(url, "/v1/chat/completions", body)
        out["status"] = status
        out["total"] = out["ttft"] = time.time() - t0
        out["raw"] = js
        if status != 200:
            out["error"] = js
            return out
        ch = js["choices"][0]
        m = ch["message"]
        out["content"] = m.get("content") or ""
        out["reasoning"] = m.get("reasoning_content") or ""
        out["finish"] = ch["finish_reason"]
        out["usage"] = js.get("usage")
        out["message"] = m
        for c in m.get("tool_calls") or []:
            out["tool_calls"].append({"id": c.get("id"), "name": c["function"]["name"], "arguments": c["function"]["arguments"]})
        return out
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    chunks = []
    calls = {}
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            out["status"] = r.status
            for line in r:
                line = line.decode("utf8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                js = json.loads(data)
                chunks.append(js)
                if js.get("usage"):
                    out["usage"] = js["usage"]
                for ch in js.get("choices") or []:
                    d = ch.get("delta") or {}
                    got = False
                    if d.get("content"):
                        out["content"] += d["content"]; got = True
                    if d.get("reasoning_content"):
                        out["reasoning"] += d["reasoning_content"]; got = True
                    for tc in d.get("tool_calls") or []:
                        got = True
                        e = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": "", "frames": 0, "names": 0})
                        e["frames"] += 1
                        if tc.get("id"):
                            e["id"] = tc["id"]
                        f = tc.get("function") or {}
                        if f.get("name"):
                            e["name"] += f["name"]; e["names"] += 1
                        e["arguments"] += f.get("arguments") or ""
                    if got and out["ttft"] is None:
                        out["ttft"] = time.time() - t0
                    if ch.get("finish_reason"):
                        out["finish"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        out["status"] = e.code
        raw = e.read().decode("utf8", "replace")
        try:
            out["error"] = json.loads(raw)
        except Exception:
            out["error"] = raw
        return out
    out["total"] = time.time() - t0
    if out["ttft"] is None:
        out["ttft"] = out["total"]
    out["raw"] = chunks
    out["tool_calls"] = [calls[k] for k in sorted(calls)]
    return out


def cached(out):
    u = out.get("usage") or {}
    return (u.get("prompt_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens"), u.get("completion_tokens"))
