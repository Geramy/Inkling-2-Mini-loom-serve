import sys, numpy as np
from gguf import GGUFReader
r = GGUFReader(sys.argv[1], "r")
by = {t.name: t for t in r.tensors}
for n in ["blk.0.ffn_gate_up_exps.scale", "blk.0.ffn_down_exps.scale", "blk.19.ffn_gate_up_exps.scale", "blk.0.exp_probs_b.bias", "blk.6.loop_msg_query.weight", "blk.0.attn_k_conv.weight", "blk.11.attn_post_conv.weight", "blk.0.attn_rel_bias.weight", "blk.0.attn_q_clip.weight", "blk.0.attn_rel_clip.weight", "blk.1.attn_res_query.weight", "output_res_query.weight", "blk.0.ffn_res_query.weight", "token_embd_norm.weight"]:
    t = by[n]; d = np.asarray(t.data, dtype=np.float32).reshape(-1)
    print(n, t.shape.tolist(), "min %.4g max %.4g mean %.4g std %.4g" % (d.min(), d.max(), d.mean(), d.std()), d[:6].tolist())
t = by["blk.0.attn_k_conv.weight"]; print("k_conv[ch0]", np.asarray(t.data).reshape(1024, 4)[0].tolist(), "shape", t.data.shape)
t = by["blk.0.ffn_gate_up_exps.weight"]; raw = np.asarray(t.data); print("nvfp4 raw", raw.dtype, raw.shape, raw[0,0,:36].tolist() if raw.ndim==3 else raw[:36].tolist())
