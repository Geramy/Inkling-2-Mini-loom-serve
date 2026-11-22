"""Marks the Inkling message and content tokens as not special so that the chat response decoder keeps them in the text.
The end-of-turn token <|content_model_end_sampling|> and the text boundary tokens stay special and are never decoded.
usage: chat_tokenizer.py <tokenizer.json in> <tokenizer.json out>"""
import json, sys

STRUCTURE = ["<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>", "<|content_text|>", "<|content_image|>", "<|content_thinking|>", "<|end_message|>", "<|content_tool_error|>", "<|content_xml|>", "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>"]
doc = json.load(open(sys.argv[1], encoding="utf-8"))
count = 0
for token in doc["added_tokens"]:
    if token["content"] in STRUCTURE:
        token["special"] = False
        count += 1
assert count == len(STRUCTURE), count
json.dump(doc, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
