"""Rewrites the skeleton placeholder kernel into a scripted kernel for output-channel tests.
The scenario is selected by the prompt length L (base + k selects scenario k); generation step j emits scenarios[k][j], then EOS.
usage: chat_script_kernel.py <kernel.loom in> <kernel.loom out> <scenarios.json>   json: {"base": 16, "scenarios": [[token ids], ...]}"""
import json, sys

text = open(sys.argv[1], encoding="utf-8").read()
spec = json.load(open(sys.argv[3]))
text = text.replace("    view.store %final_h, %write_state[0] : i32, view<1xi32>\n", "    %carry = scf.select %fresh, %length32, %initial_h : i32\n    view.store %carry, %write_state[0] : i32, view<1xi32>\n")
text = text.replace("    view.store %final_h, %hidden[%span] : i32, view<[%spans]xi32>\n", "    view.store %carry, %hidden[%span] : i32, view<[%spans]xi32>\n")
start = text.index("      %sum = scf.for %position = [%c0 to %end step %c1]")
end = text.index("      %result = buffer.view %outputs[%origin]")
lines = [
    "      %always = scalar.constant true : i1",
    "      %hidden = buffer.view %residual[%origin] : buffer -> view<[%spans]xi32>",
    "      %prompt = view.load %hidden[%span] : view<[%spans]xi32> -> i32",
    "      %step = scalar.subi %end32, %prompt : i32",
    f"      %base = scalar.constant {spec['base']} : i32",
    "      %scenario = scalar.subi %prompt, %base : i32",
    "      %eos = view.load %header[1] : view<4xi32> -> i32",
    "      %p0 = scf.select %always, %eos, %eos : i32",
]
n = 0
for k, ids in enumerate(spec["scenarios"]):
    lines.append(f"      %scenario_{k} = scalar.constant {k} : i32")
    lines.append(f"      %in_{k} = scalar.cmpi eq, %scenario, %scenario_{k} : i32")
    for j, token in enumerate(ids):
        lines += [f"      %step_{k}_{j} = scalar.constant {j} : i32", f"      %token_{k}_{j} = scalar.constant {token} : i32",
                  f"      %is_{k}_{j} = scalar.cmpi eq, %step, %step_{k}_{j} : i32", f"      %hit_{k}_{j} = scalar.andi %in_{k}, %is_{k}_{j} : i1",
                  f"      %p{n + 1} = scf.select %hit_{k}_{j}, %token_{k}_{j}, %p{n} : i32"]
        n += 1
lines.append(f"      %prediction = scf.select %always, %p{n}, %p{n} : i32")
text = text[:start] + "\n".join(lines) + "\n" + text[end:]
open(sys.argv[2], "w", encoding="utf-8").write(text)
