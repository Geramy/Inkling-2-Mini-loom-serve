#!/usr/bin/env python3
"""Builds the published layout of the Inkling 2 Mini Loom package from package/. Nothing in the output is edited by hand.

usage: publish_build.py --cfg CFG.json --skills SKILLS_DIR [--gguf-dir DIR] [--package package] [--out publish/unsloth-Inkling-2-Mini-loom]

Layout of the output tree
  model.json        card: services.chat, variants ud-q4-k-xl and ud-q8-k-xl, presets chat, chat-r9700, chat-q8, chat-q8-r9700, origins.checkpoint
  llm_config.json   build facts for the skills (written by make_llm_config.py, needs --cfg and --skills)
  weights.loom      weight-name stub, data of program/prepare.loom
  tokenizer.json    GGUF tokenizer, the 12 structure tokens are not special
  program/          the Loom sources, same relative paths as package/ below one directory:
    prepare.loom storage.loom config.loom card_defaults.loom   stage declaration and storage
    control/          chat service glue (chat, JSON, sampler, identity)
    dispatch/         per-visit schedule (targets, spans, layer, model, hooks_*)
    kernel/{dense,attn,experts,mix}/   kernels
    lib/{inkling,vendor,media_stub.loom}   motif libraries and vendored files
  LICENSE NOTICE .gitattributes
Dropped from package/: capture service, variants and presets, capture-only sources, test/, devices/, the old model.json and card.json.
The old card.json is the source of the service configuration: --verify (default) proves that nothing the programs read was lost.
"""
import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REPO = "unsloth/Inkling-2-Mini-GGUF"
REVISION = "ea0c79b2ecd73669e774c8519b79bfabfddbc027"
MODEL_ID = "thinkingmachines/Inkling-2-Mini"
CHECKPOINT_DIR = "Mainline_Support"
STRUCTURE = ["<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>", "<|content_text|>", "<|content_image|>", "<|content_thinking|>", "<|end_message|>", "<|content_tool_error|>", "<|content_xml|>", "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>"]

PRESET_CONTEXT = {"chat": 131072, "chat-r9700": 131072, "chat-q8": 32768, "chat-q8-r9700": 32768}
TRANSIENT_BYTES = {"chat": 8000000000, "chat-r9700": 8000000000, "chat-q8": 3600000000, "chat-q8-r9700": 3600000000}
PREFILL_CAPACITY = {"chat": 2048, "chat-r9700": 2048, "chat-q8": 512, "chat-q8-r9700": 512}
DELIBERATE = ("execution.context_capacity", "execution.prefill_capacity", "memory.transient_bytes", "memory.workload")

PRESETS = {
    "chat": ("interactive-q4", "ud-q4-k-xl"),
    "chat-r9700": ("interactive-q4", "ud-q4-k-xl"),
    "chat-q8": ("interactive-q8", "ud-q8-k-xl"),
    "chat-q8-r9700": ("interactive-q8", "ud-q8-k-xl"),
}

LFS = ["*.7z", "*.arrow", "*.bin", "*.bz2", "*.ckpt", "*.ftz", "*.gz", "*.h5", "*.joblib", "*.lfs.*", "*.mlmodel", "*.model", "*.msgpack", "*.npy", "*.npz", "*.onnx", "*.ot", "*.parquet", "*.pb", "*.pickle", "*.pkl", "*.pt", "*.pth", "*.rar", "*.safetensors", "saved_model/**/*", "*.tar.*", "*.tar", "*.tflite", "*.tgz", "*.wasm", "*.xz", "*.zip", "*.zst", "*tfevents*"]

LICENSE = """MIT License

Copyright (c) 2026 Geramy Loveless

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def prefixed(x):
    if isinstance(x, str):
        return "program/" + x if x.endswith(".loom") else x
    if isinstance(x, list):
        return [prefixed(i) for i in x]
    return x


def gguf_name(path):
    return os.path.basename(path)


def build_card(old):
    src = old["programs"]["source"]
    inputs = []
    for p in src["inputs"]:
        if p == "prepare.loom":
            inputs.append({"path": "program/prepare.loom", "data": ["weights.loom", "llm_config.json"]})
        else:
            inputs.append(prefixed(p))
    card = {
        "schema_version": 1,
        "model": {"id": MODEL_ID, "name": old["model"]["name"], "description": old["model"]["description"], "license": "mit"},
        "programs": {"source": {"inputs": inputs, "libraries": prefixed(src["libraries"])}},
        "services": {"chat": copy.deepcopy(old["services"]["chat"])},
        "variants": {},
        "presets": {},
        "origins": {"checkpoint": {"kind": "hf", "repository": REPO, "revision": REVISION}},
    }
    for name, variant in old["variants"].items():
        svc = variant["services"]["chat"]
        files = []
        for f in svc["parameters"]["model"]["files"]:
            base = gguf_name(f["path"])
            files.append({"path": "parameter/" + base, "format": f["format"], "origin": f"@checkpoint/{CHECKPOINT_DIR}/{base[:-len('.gguf')]}.gguf_file"})
        card["variants"][name] = {"services": {"chat": {
            "programs": ["source"],
            "config": copy.deepcopy(svc["config"]),
            "parameters": {"model": {"files": files}},
            "tokenizers": {"text": "tokenizer.json"},
        }}}
    for new, (olds, variant) in PRESETS.items():
        p = copy.deepcopy(old["presets"][olds])
        assert p["variant"] == variant and p["service"] == "chat"
        p["execution"]["context_capacity"] = PRESET_CONTEXT[new]
        p["execution"]["prefill_capacity"] = PREFILL_CAPACITY[new]
        p["memory"]["transient_bytes"] = TRANSIENT_BYTES[new]
        p["memory"]["workload"] = f"4 rows, {PRESET_CONTEXT[new]} context, {PREFILL_CAPACITY[new]} prefill tokens, 4 checkpoints; measured: 19.9 GB + 39.4 KB per context position, non-weight memory 7.3 GB at 131072 with prefill 2048 and 3.6 GB at 32768 with prefill 512 (docs/CONTEXT_TABLE.md)"
        card["presets"][new] = p
    return card


def verify(old, card):
    """The service configuration the programs read must equal the old card's chat service, variant by variant."""
    for name, variant in old["variants"].items():
        a, b = variant["services"]["chat"], card["variants"][name]["services"]["chat"]
        assert a["config"] == b["config"], (name, a["config"], b["config"])
        assert a["programs"] == b["programs"]
        assert [gguf_name(f["path"]) for f in a["parameters"]["model"]["files"]] == [gguf_name(f["path"]) for f in b["parameters"]["model"]["files"]]
    assert card["services"]["chat"] == old["services"]["chat"]
    src, new = old["programs"]["source"], card["programs"]["source"]
    strip = lambda l: [x["path"] if isinstance(x, dict) else x for x in l]
    assert [("program/" + x) for x in src["inputs"]] == strip(new["inputs"])
    assert [("program/" + x) for x in src["libraries"]] == new["libraries"]
    for new_name, (olds, variant) in PRESETS.items():
        a, b = copy.deepcopy(old["presets"][olds]), copy.deepcopy(card["presets"][new_name])
        assert b["execution"].pop("context_capacity") == PRESET_CONTEXT[new_name]
        assert b["execution"].pop("prefill_capacity") == PREFILL_CAPACITY[new_name]
        assert b["memory"].pop("transient_bytes") == TRANSIENT_BYTES[new_name]
        b["memory"].pop("workload")
        a["execution"].pop("context_capacity"); a["execution"].pop("prefill_capacity"); a["memory"].pop("transient_bytes"); a["memory"].pop("workload")
        assert a == b, new_name
    print("verify: chat service config, variants and programs equal package/card.json; presets equal it except " + ", ".join(DELIBERATE))


def tokenizer_ok(path):
    if not os.path.isfile(path):
        return False
    doc = json.load(open(path, encoding="utf-8"))
    flags = {t["content"]: t["special"] for t in doc["added_tokens"]}
    return all(flags.get(s) is False for s in STRUCTURE)


def ensure_tokenizer(package, gguf_dir):
    path = os.path.join(package, "tokenizer.json")
    if tokenizer_ok(path):
        return path
    gguf = os.path.join(gguf_dir, "Inkling-2-Mini-UD-Q4_K_XL.gguf")
    raw = path + ".raw"
    subprocess.run([sys.executable, os.path.join(HERE, "gen_tokenizer.py"), gguf, raw], check=True)
    subprocess.run([sys.executable, os.path.join(HERE, "chat_tokenizer.py"), raw, path], check=True)
    os.remove(raw)
    assert tokenizer_ok(path)
    return path


def notice(package, shipped):
    """VENDORED.md reduced to the files that are shipped: a table row of a file that is not shipped is dropped, and a table without rows with its section."""
    text = open(os.path.join(package, "lib", "VENDORED.md"), encoding="utf-8").read()
    blocks = []
    for block in re.split(r"(?m)^(?=## )", text):
        lines = block.splitlines()
        rows = [l for l in lines if re.match(r"\| `lib/", l)]
        keep = [l for l in rows if re.match(r"\| `([^`]+)`", l).group(1) in shipped]
        if rows and not keep:
            continue
        blocks.append("\n".join(l for l in lines if l not in rows or l in keep) + "\n")
    text = "".join(blocks).replace("`lib/", "`program/lib/")
    text = "\n".join(l for l in text.splitlines() if not l.startswith("- Pending:"))
    head = ("NOTICE\n\nThe files under program/lib/ and the dense kernel motifs derive from the projects below (MIT). Their notices apply to those files. "
            "The model weights are not part of this repository: they are the Unsloth GGUF files of thinkingmachines/Inkling-2-Mini (Apache-2.0), fetched from the origin named in model.json.\n\n")
    return head + text.replace("# Vendored files", "Vendored files", 1) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=os.path.join(HERE, "..", "package"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "publish", "unsloth-Inkling-2-Mini-loom"))
    ap.add_argument("--cfg", required=True, help="cfg.json of the skills' inspect_gguf.py for the Q4_K_XL file")
    ap.add_argument("--skills", required=True, help="loom-skills checkout (read only)")
    ap.add_argument("--gguf-dir", default=os.environ.get("INKLING_MODELS", os.path.expanduser("~/models/inkling2-mini/Mainline_Support")))
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    package, out = os.path.abspath(a.package), os.path.abspath(a.out)

    old = json.load(open(os.path.join(package, "card.json"), encoding="utf-8"))
    card = build_card(old)
    if not a.no_verify:
        verify(old, card)

    tok = ensure_tokenizer(package, a.gguf_dir)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "program"))
    src = old["programs"]["source"]
    files = src["inputs"] + src["libraries"]
    for rel in sorted(set(files)):
        dst = os.path.join(out, "program", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(package, rel), dst)
    shutil.copyfile(os.path.join(package, "weights.loom"), os.path.join(out, "weights.loom"))
    shutil.copyfile(tok, os.path.join(out, "tokenizer.json"))
    with open(os.path.join(out, "model.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(card, f, indent=1)
        f.write("\n")
    with open(os.path.join(out, "LICENSE"), "w", newline="\n") as f:
        f.write(LICENSE)
    with open(os.path.join(out, "NOTICE"), "w", encoding="utf-8", newline="\n") as f:
        f.write(notice(package, set(src["libraries"])))
    with open(os.path.join(out, ".gitattributes"), "w", newline="\n") as f:
        f.write("".join(f"{e} filter=lfs diff=lfs merge=lfs -text\n" for e in LFS))

    named = {e if isinstance(e, str) else e["path"] for e in card["programs"]["source"]["inputs"] + card["programs"]["source"]["libraries"]}
    present = {os.path.relpath(os.path.join(r, f), out).replace(os.sep, "/") for r, _, fs in os.walk(os.path.join(out, "program")) for f in fs if f.endswith(".loom")}
    assert present == named, ("not named by model.json", sorted(present - named), "named but missing", sorted(named - present))
    import make_llm_config
    make_llm_config.write_llm_config(out, a.cfg, a.skills, a.gguf_dir)
    print("wrote", out, sum(len(f) for _, _, f in os.walk(out)), "files")


if __name__ == "__main__":
    main()
