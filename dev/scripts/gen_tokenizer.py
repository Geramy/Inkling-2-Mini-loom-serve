import json, sys
from gguf import GGUFReader, GGUFValueType

REGEX = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
         r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
         r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")

def strings(field):
    return [bytes(field.parts[i]).decode("utf-8") for i in field.data]

def main(gguf_path, out_path):
    r = GGUFReader(gguf_path, "r")
    tokens = strings(r.fields["tokenizer.ggml.tokens"])
    merges = strings(r.fields["tokenizer.ggml.merges"])
    types = [int(r.fields["tokenizer.ggml.token_type"].parts[i][0]) for i in r.fields["tokenizer.ggml.token_type"].data]
    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
             for i, (t, ty) in enumerate(zip(tokens, types)) if ty == 3]
    doc = {
        "version": "1.0",
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": REGEX}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": {t: i for i, t in enumerate(tokens)},
                  "merges": [m.split(" ") for m in merges]},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    print(f"tokens={len(tokens)} merges={len(merges)} added={len(added)} types={sorted(set(types))}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
