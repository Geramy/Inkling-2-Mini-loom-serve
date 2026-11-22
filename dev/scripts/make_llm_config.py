#!/usr/bin/env python3
"""Writes llm_config.json of the published package and, with --work, a work dir in the shape the loom-skills scripts expect.

usage: make_llm_config.py --publish DIR --cfg CFG.json --skills SKILLS_DIR [--gguf-dir DIR] [--chat CHAT_BLOCK.json] [--work W] [--quant q4|q8]

llm_config.json is what the skills' catalog.py build() writes for a hand-written package: the cfg.json of inspect_gguf.py without tensors, plus
model_id, catalog{variant, preset, draft_kinds, memory}, sdk, generator and resolved. catalog.py itself is not used: it keeps one variant and one preset
and rewrites the card, but our card carries four presets. The helpers kit_source.generator_stamp and kb_choices.exempt_block are the skills' own.
The chat block is kit/chat_block.json (chat kit, task T3; override with --chat). Without that file "chat" is null.

W layout: pkg/ (publish tree, parameter/ links to the local GGUF files), cfg.json, gguf/, bench/, catalog/{catalog.json, model.json, program/}.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHAT = os.path.join(HERE, "..", "kit", "chat_block.json")
REPO = "unsloth/Inkling-2-Mini-GGUF"
REVISION = "ea0c79b2ecd73669e774c8519b79bfabfddbc027"
TOKENIZER_REPO = "thinkingmachines/Inkling-2-Mini"
CHECKPOINT_DIR = "Mainline_Support"
KV_BYTES_PER_POSITION = 39400
WORKSPACE_GIB = 2.14
GLOBAL_VISITS = [5, 11, 17, 23, 29, 35, 41, 47]
QUANTS = {"q4": ("ud-q4-k-xl", "chat", "Inkling-2-Mini-UD-Q4_K_XL.gguf"), "q8": ("ud-q8-k-xl", "chat-q8", "Inkling-2-Mini-UD-Q8_K_XL.gguf")}


def load_helpers(skills):
    starter = os.path.join(skills, "skills", "loom-port-llm", "starter")
    sys.path.insert(0, starter)
    os.environ.setdefault("LOOM_ROOT", os.path.expanduser("~/loom"))
    import kit_source
    import kb_choices
    return kit_source, kb_choices


def sdk_pin():
    # loom-install.json has no hrx_system_sha for a PyPI wheel: take the skills pin when the wheel hash is the same
    root = os.environ.get("LOOM_ROOT", os.path.expanduser("~/loom"))
    doc = json.load(open(os.path.join(root, "loom-install.json"), encoding="utf-8"))
    pin = {k: doc[k] for k in ("hrx_loom", "source", "wheel_sha256", "hrx_system_sha")}
    if pin["hrx_system_sha"] is None:
        import kit_source
        if kit_source.SDK_PIN["wheel_sha256"] == pin["wheel_sha256"]:
            pin["hrx_system_sha"] = kit_source.SDK_PIN["hrx_system_sha"]
    return pin


def memory_block(card, quant):
    variant, preset, _ = QUANTS[quant]
    return {
        "resident_weights_bytes": card["presets"][preset]["memory"]["weights_bytes"],
        "kv_bytes_per_position": KV_BYTES_PER_POSITION,
        "workspace_gib": WORKSPACE_GIB,
        "over_cap": False,
        "note": "resident_weights_bytes is the tensor data of the GGUF file; kv_bytes_per_position and workspace_gib are fitted to the used memory measured at 4 rows on both GPUs (19.9 GB + 39.4 KB per context position, docs/CONTEXT_TABLE.md; re-measured at 131072 with prefill_capacity 2048: 24.9 GB used, inside the fit); the global KV alone is 32768 bytes per position",
    }


def chat_block(path):
    return json.load(open(path, encoding="utf-8")) if path and os.path.exists(path) else None


def build_config(cfg, card, skills, quant="q4", chat=None):
    kit_source, kb_choices = load_helpers(skills)
    variant, preset, gguf = QUANTS[quant]
    out = {k: v for k, v in cfg.items() if k != "tensors"}
    out["gguf"] = gguf
    out["kv_layers"] = len(GLOBAL_VISITS)
    out["loop"] = {"start": 6, "end": 18, "count": 3, "visits": 48, "global_visits": GLOBAL_VISITS}
    out["chat"] = chat
    out["model_id"] = card["model"]["id"]
    out["catalog"] = {"variant": variant, "preset": preset, "draft_kinds": [], "memory": memory_block(card, quant)}
    out["sdk"] = sdk_pin()
    out["generator"] = kit_source.generator_stamp()
    out["resolved"] = kb_choices.exempt_block("catalog package: hand-written programs, no generator runs")
    return out


def write_llm_config(publish, cfg_path, skills, gguf_dir=None, quant="q4", chat_path=DEFAULT_CHAT):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    card = json.load(open(os.path.join(publish, "model.json"), encoding="utf-8"))
    config = build_config(cfg, card, skills, quant, chat_block(chat_path))
    with open(os.path.join(publish, "llm_config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(config, f, indent=1)
        f.write("\n")
    return config


def catalog_json(card, quant):
    variant, preset, gguf = QUANTS[quant]
    return {
        "source": {"repo": REPO, "revision": REVISION, "file": f"{CHECKPOINT_DIR}/{gguf[:-len('.gguf')]}.gguf_file"},
        "variant": variant,
        "preset": preset,
        "context": card["presets"][preset]["execution"]["context_capacity"],
        "prefill_capacity": card["presets"][preset]["execution"]["prefill_capacity"],
        "memory": memory_block(card, quant),
        "sdk": sdk_pin(),
        "tokenizer": TOKENIZER_REPO,
    }


def make_work(publish, work, cfg_path, gguf_dir, quant):
    card = json.load(open(os.path.join(publish, "model.json"), encoding="utf-8"))
    for d in ("pkg", "gguf", "bench", "catalog"):
        shutil.rmtree(os.path.join(work, d), ignore_errors=True)
        os.makedirs(os.path.join(work, d))
    os.makedirs(os.path.join(work, "pkg", "parameter"))
    for _, _, name in QUANTS.values():
        os.symlink(os.path.join(gguf_dir, name), os.path.join(work, "gguf", name))
        os.symlink(os.path.join(gguf_dir, name), os.path.join(work, "pkg", "parameter", name))
    skip = shutil.ignore_patterns("model.json.lock", "__pycache__")
    shutil.copytree(publish, os.path.join(work, "pkg"), ignore=skip, dirs_exist_ok=True)
    shutil.copytree(os.path.join(publish, "program"), os.path.join(work, "catalog", "program"))
    shutil.copyfile(os.path.join(publish, "model.json"), os.path.join(work, "catalog", "model.json"))
    json.dump(catalog_json(card, quant), open(os.path.join(work, "catalog", "catalog.json"), "w"), indent=1)
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg["gguf"] = os.path.join(work, "gguf", QUANTS["q4"][2])
    json.dump(cfg, open(os.path.join(work, "cfg.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--skills", required=True)
    ap.add_argument("--gguf-dir", default=os.environ.get("INKLING_MODELS", os.path.expanduser("~/models/inkling2-mini/Mainline_Support")))
    ap.add_argument("--chat", default=DEFAULT_CHAT)
    ap.add_argument("--work")
    ap.add_argument("--quant", choices=sorted(QUANTS), default="q4")
    a = ap.parse_args()
    publish = os.path.abspath(a.publish)
    write_llm_config(publish, a.cfg, a.skills, a.gguf_dir, a.quant, a.chat)
    print("wrote", os.path.join(publish, "llm_config.json"))
    if a.work:
        make_work(publish, os.path.abspath(a.work), a.cfg, os.path.abspath(a.gguf_dir), a.quant)
        print("work dir", os.path.abspath(a.work))


if __name__ == "__main__":
    main()
