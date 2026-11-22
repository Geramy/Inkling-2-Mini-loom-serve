"""Scripted generations for the output-channel tests. Scenario k is selected by a prompt of 16 + k tokens ("Hello" followed by k times " Hello").
A segment is a marker name or literal text of printable ASCII without spaces (token id = byte - 33, the single-byte pieces of the tokenizer).
python scenarios.py > scenarios.json   writes the kernel script spec for scripts/chat_script_kernel.py"""
import json

MARKERS = {"MM": 200001, "TH": 200008, "TX": 200004, "EM": 200010, "INV": 200049, "EOS": 200006}


def ids(segments):
    out = []
    for segment in segments:
        out += [MARKERS[segment]] if segment in MARKERS else [ord(c) - 33 for c in segment]
    return out


CALL_F = ["MM", "f", "INV", '{"name":"f","args":{}}', "EM"]
CALL_G = ["MM", "g", "INV", '{"name":"g","args":{"k":[1,2]}}', "EM"]

# name, segments, max_tokens, expected content, reasoning, calls [(name, arguments)], finish_reason
SCENARIOS = [
    ("plain_text", ["MM", "TX", "Hi_there", "EM"], 64, "Hi_there", "", [], "stop"),
    ("think_then_text", ["MM", "TH", "think", "EM", "MM", "TX", "answer", "EM"], 64, "answer", "think", [], "stop"),
    ("tool_only", ["MM", "get_x", "INV", '{"name":"get_x","args":{"b":1,"a":"z"}}', "EM"], 64, "", "", [("get_x", '{"b":1,"a":"z"}')], "tool_calls"),
    ("text_and_two_tools", ["MM", "TX", "Look.", "EM"] + CALL_F + CALL_G, 96, "Look.", "", [("f", "{}"), ("g", '{"k":[1,2]}')], "tool_calls"),
    ("think_text_tool", ["MM", "TH", "hm", "EM", "MM", "TX", "ok", "EM"] + CALL_F, 96, "ok", "hm", [("f", "{}")], "tool_calls"),
    ("raw_text_without_markers", ["plain"], 64, "plain", "", [], "stop"),
    ("truncated_text", ["MM", "TX", "abcdefgh", "EM"], 5, "abc", "", [], "length"),
    ("truncated_tool", ["MM", "f", "INV", '{"name":"f","args":{"a":1}}', "EM"], 12, "", "", [], "length"),
    ("empty_generation", [], 64, "", "", [], "stop"),
    ("tool_after_truncated_text", ["MM", "TX", "ab", "EM"] + CALL_F, 11, "ab", "", [], "length"),
    ("call_name_mismatch", ["MM", "f", "INV", '{"name":"g","args":{}}', "EM"], 64, None, None, None, "error"),
    ("two_text_messages", ["MM", "TX", "one", "EM", "MM", "TX", "two", "EM"], 64, "onetwo", "", [], "stop"),
    ("whitespace_free_unicode", ["MM", "TX", "caf~", "EM"], 64, "caf~", "", [], "stop"),
]

if __name__ == "__main__":
    print(json.dumps({"base": 16, "scenarios": [ids(s[1]) for s in SCENARIOS]}))
