#!/usr/bin/env python3
"""Writes a skills-schema roofline.json ({"metrics": [{metric, ceiling, target, stretch}]}) from our bench/<gpu>/roofline.json, so perf.py can print the campaign verdict.
usage: roofline_campaign_view.py IN OUT [checkpoint, default ud-q4-k-xl]"""
import json
import sys

d = json.load(open(sys.argv[1]))
ck = d["checkpoints"][sys.argv[3] if len(sys.argv) > 3 else "ud-q4-k-xl"]["metrics"]
names = {"decode": "decode_tps", "decode_2k": "decode_tps_2k", "decode_8k": "decode_tps_8k", "decode_32k": "decode_tps_32k",
         "prefill_1k": "prefill_1000", "prefill_2k": "prefill_2000", "prefill_8k": "prefill_8000", "prefill_16k": "prefill_16000",
         "prefill_32k": "prefill_32000", "streams4": "c4_steady"}
out = {"note": "view of roofline.json (inkling-roofline/1) in the skills' metric names; made by scripts/roofline_campaign_view.py",
       "metrics": [{"metric": names[k], **{x: v[x] for x in ("ceiling", "target", "stretch")}} for k, v in ck.items()]}
json.dump(out, open(sys.argv[2], "w"), indent=1)
