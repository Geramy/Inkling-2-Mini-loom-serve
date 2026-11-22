# usage: gen_bank_probe.py <gguf> <kernel.loom> <out_kernel.loom> <out_tensors.json>
# Rewrites the placeholder kernel so that one fold launch per checkpoint tensor xors that tensor's last byte into an accumulator.
# The whole checkpoint then becomes one bound parameter bank. The accumulator seeds the placeholder recurrence.
import json, re, sys
from gguf import GGUFReader

gguf, src, out, out_json = sys.argv[1:5]
r = GGUFReader(gguf, "r")
tensors = sorted(r.tensors, key=lambda t: t.name)
json.dump([t.name for t in tensors], open(out_json, "w"))
text = open(src).read()
text = text.replace("%norm: view<2304xi32>, %embedding: view<492106752xi8>, %experts: view<382205952xi8>", "%acc: buffer")
text = re.sub(r"    %norm_word = view\.load.*?%bias_value = scalar\.xori %mixed, %experts_word : i32\n",
              lambda m: "    %acc_view = buffer.view %acc[%origin] : buffer -> view<1xi32>\n    %bias_value = view.load %acc_view[0] : view<1xi32> -> i32\n", text, flags=re.S)
kernels = '''
kernel.def target(@target) @zero() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%acc: buffer) {
  %origin = index.constant 0 : offset
  %view = buffer.view %acc[%origin] : buffer -> view<1xi32>
  %zero = scalar.constant 0 : i32
  view.store %zero, %view[0] : i32, view<1xi32>
  kernel.return
}

kernel.def target(@target) @fold() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%length: index, %tensor: buffer, %acc: buffer) {
  %origin = index.constant 0 : offset
  %c1_index = index.constant 1 : index
  %last_index = index.sub %length, %c1_index : index
  %last = index.cast %last_index : index to offset
  %byte = buffer.load.i8.u %tensor[%last]
  %view = buffer.view %acc[%origin] : buffer -> view<1xi32>
  %previous = view.load %view[0] : view<1xi32> -> i32
  %next = scalar.xori %previous, %byte : i32
  view.store %next, %view[0] : i32, view<1xi32>
  kernel.return
}

'''
text = text.replace("command.def public @advance()", kernels + "command.def public @advance()")
lines = ["  %acc_bytes = index.constant 4 : offset", "  %acc = buffer.alloca<global> align(256) %acc_bytes : buffer",
         "  kernel.launch @zero(%acc) : (buffer)"]
for i, t in enumerate(tensors):
    n = int(t.n_bytes)
    lines.append(f'  %p{i} = command.parameter %weights, "{t.name}" : view<{n}xi8>')
    lines.append(f"  %n{i} = index.constant {n} : index")
    lines.append(f"  kernel.launch @fold(%n{i}, %p{i}, %acc) : (index, view<{n}xi8>, buffer)")
lines.append("  kernel.launch @consume(%acc, %residual, %metadata, %origins, %arena, %tokens) : (buffer, buffer, buffer, buffer, buffer, buffer)")
text = re.sub(r'  %norm = command\.parameter.*?  kernel\.launch @consume\([^\n]*\n', lambda m: "\n".join(lines) + "\n", text, flags=re.S)
open(out, "w").write(text)
print("tensors", len(tensors))
