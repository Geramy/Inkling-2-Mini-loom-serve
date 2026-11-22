"""Device sampler check: determinism, greedy, top-k, top-p, temperature and the tool-open exclusion, with chi-square tests on synthetic logits.
Draws are independent through the position counter; every draw reads logit row zero (stride 0) of a vocabulary of 200058 F32 values.
usage: test_sampler.py <device uri> [<device uri> ...]   (box, with loom-test on the path)   exit code 1 on any FAIL"""
import json, math, os, subprocess, sys, tempfile
import numpy as np
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROL = os.path.join(HERE, "..", "..", "control")
VOCAB = 200058
IDS = [3, 77, 1000, 50000, 120000, 199999, 200049, 200057]
PROBS = [0.30, 0.20, 0.15, 0.12, 0.10, 0.07, 0.04, 0.02]
N = 20000
SEEDS = [1234567890123, 987654321]

# name, temperature, top_p, top_k, flags
SPECS = [
    ("full", 1.0, 1.0, 0, 0),
    ("hot", 2.0, 1.0, 0, 0),
    ("cold", 0.5, 1.0, 0, 0),
    ("top_k3", 1.0, 1.0, 3, 0),
    ("top_p", 1.0, 0.55, 0, 0),
    ("top_k_p", 1.0, 0.8, 5, 0),
    ("greedy", 0.0, 1.0, 0, 0),
    ("single", 1.0, 1.0, 1, 0),
    ("no_tool", 1.0, 1.0, 0, 2),
]
# Rows with top_k 200058 keep every token but take the template path: the plain rows must draw the same tokens.
SLOW = [("full_slow", 1.0, 1.0, 200058, 0), ("hot_slow", 2.0, 1.0, 200058, 0), ("cold_slow", 0.5, 1.0, 200058, 0), ("no_tool_slow", 1.0, 1.0, 200058, 2)]
PAIRS = [("full", "full_slow"), ("hot", "hot_slow"), ("cold", "cold_slow"), ("no_tool", "no_tool_slow")]


def expected(temperature, top_p, top_k, flags):
    """Probability of each of the eight tokens under the documented order: temperature, top-k, nucleus (smallest prefix with mass >= top_p), exclusion."""
    logits = np.log(PROBS)
    if temperature == 0.0 or top_k == 1:
        p = np.zeros(8)
        p[int(np.argmax(logits))] = 1.0
        return p
    alive = np.ones(8, bool)
    if flags & 2:
        alive[IDS.index(200049)] = False
    weights = np.where(alive, np.exp((logits - logits.max()) / temperature), 0.0)
    order = np.argsort(-weights, kind="stable")
    if top_k:
        weights[order[top_k:]] = 0.0
    if top_p < 1.0:
        total = weights.sum()
        keep, mass = [], 0.0
        for i in order:
            if weights[i] == 0.0:
                continue
            keep.append(i)
            mass += weights[i]
            if mass >= top_p * total:
                break
        mask = np.zeros(8, bool)
        mask[keep] = True
        weights = np.where(mask, weights, 0.0)
    return weights / weights.sum()


def tail_mass(temperature, top_p, top_k, flags):
    """Probability mass of the 200050 background tokens (logit -30) when no top-k or nucleus cut removes them."""
    if temperature == 0.0 or top_k or top_p < 1.0:
        return 0.0
    logits = np.log(PROBS)
    alive = np.ones(8, bool)
    if flags & 2:
        alive[IDS.index(200049)] = False
    main = np.where(alive, np.exp((logits - logits.max()) / temperature), 0.0).sum()
    tail = (VOCAB - 8) * math.exp((-30.0 - logits.max()) / temperature)
    return tail / (main + tail)


def logit_chain():
    lines = ["      %v0 = scalar.constant -30.0 : f32"]
    for k, (token, p) in enumerate(zip(IDS, PROBS)):
        lines += [f"      %id_{k} = scalar.constant {token} : i32", f"      %is_{k} = scalar.cmpi eq, %id, %id_{k} : i32",
                  f"      %l_{k} = scalar.constant {math.log(p):.9f} : f32", f"      %v{k + 1} = scf.select %is_{k}, %l_{k}, %v{k} : f32"]
    return "\n".join(lines), f"%v{len(IDS)}"


def case_text(name, temperature, top_p, top_k, flags):
    tensors = f"tensor<{N}xi32>"
    scratch = f"tensor<{5 * 128 * N}xi32>"
    epoch = f"tensor<{8 + 27 * N}xi32>"
    text = [f"check.case public @{name} {{", f"  %rows = check.literal value({N}) : index", "  %stride = check.literal value(0) : index",
            f"  %temperature = check.literal value({temperature}) : f32", f"  %top_p = check.literal value({top_p}) : f32",
            f"  %top_k = check.literal value({top_k}) : i32", f"  %flags = check.literal value({flags}) : i32",
            f"  %seed_a = check.literal value({SEEDS[0]}) : i64", f"  %seed_c = check.literal value({SEEDS[1]}) : i64",
            f"  %logits = check.generate.fill value(0.0) : tensor<{VOCAB}xf32>", f"  %epoch = check.generate.fill value(0) : {epoch}",
            f"  %scratch = check.generate.fill value(0) : {scratch}",
            f"  %out_a = check.generate.fill value(-1) : {tensors}", f"  %out_b = check.generate.fill value(-1) : {tensors}", f"  %out_c = check.generate.fill value(-1) : {tensors}",
            f"  kernel.launch @test_logits(%logits) : (tensor<{VOCAB}xf32>)",
            f"  kernel.launch @test_identities[%rows](%rows, %epoch) : [index](index, {epoch})"]
    for label, seed in (("a", "a"), ("b", "a"), ("c", "c")):
        text += [f"  kernel.launch @test_policy(%epoch, %rows, %temperature, %top_p, %top_k, %flags, %seed_{seed}) : ({epoch}, index, f32, f32, i32, i32, i64)"]
        for kernel in ("inkling_sample_stats", "inkling_sample_draw"):
            text += [f"  kernel.launch @{kernel}[%rows](%rows, %stride, %epoch, %logits, %scratch) : [index](index, index, {epoch}, tensor<{VOCAB}xf32>, {scratch})"]
        text += [f"  kernel.launch @inkling_sample_rows[%rows](%rows, %stride, %epoch, %logits, %out_{label}, %scratch) : [index](index, index, {epoch}, tensor<{VOCAB}xf32>, {tensors}, {scratch})"]
    text += [f"  check.expect.equal actual(%out_a) expected(%out_b) : {tensors}"]
    for label in "abc":
        text.append(f'  check.file.write.npy value(%out_{label}) path("{name}_{label}.npy") mode(always) : {tensors}')
    text += ["  check.return", "}", ""]
    return "\n".join(text)


def module_text():
    chain, last = logit_chain()
    head = f"""config.decl @inkling.packet_capacity : %value: index where [range(%value, 2, 2147483646)]

func.decl @inkling_packet_identity_origin(%capacity: index) -> (offset)

func.decl @inkling_packet_policy_origin(%capacity: index) -> (offset)

kernel.decl @inkling_sample_stats(%rows: index) launch(%rows_argument: index, %logit_stride: index, %epoch: buffer, %logits: buffer, %scratch: buffer)

kernel.decl @inkling_sample_draw(%rows: index) launch(%rows_argument: index, %logit_stride: index, %epoch: buffer, %logits: buffer, %scratch: buffer)

kernel.decl @inkling_sample_rows(%rows: index) launch(%rows_argument: index, %logit_stride: index, %epoch: buffer, %logits: buffer, %outputs: buffer, %scratch: buffer)

kernel.def @test_logits() {{
  %c1 = index.constant 1 : index
  %threads = index.constant 256 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%threads, %c1, %c1) : index
}} launch(%logits: buffer) {{
  %origin = index.constant 0 : offset
  %columns = index.constant {VOCAB} : index
  %values = buffer.view %logits[%origin] : buffer -> view<{VOCAB}xf32>
  %thread = kernel.workitem.id<x> : index
  %c0 = index.constant 0 : index
  %threads = index.constant 256 : index
  scf.for %base = [%c0 to %columns step %threads] {{
    %column = index.add %base, %thread : index
    %live = index.cmp ult, %column, %columns : index
    scf.if %live {{
      %id = index.cast %column : index to i32
{chain}
      view.store {last}, %values[%column] : f32, view<{VOCAB}xf32>
    }}
  }}
  kernel.return
}}

kernel.def @test_policy() {{
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
}} launch(%epoch: buffer, %rows: index, %temperature: f32, %top_p: f32, %top_k: i32, %flags: i32, %seed: i64) {{
  %origin = index.constant 0 : offset
  %capacity = config.get @inkling.packet_capacity : index
  %policy_origin = func.call @inkling_packet_policy_origin(%capacity) : (index) -> (offset)
  %seed_relative = index.constant 16 : offset
  %seed_origin = index.add %policy_origin, %seed_relative : offset
  %header = buffer.view %epoch[%origin] : buffer -> view<3xi32>
  %floats = buffer.view %epoch[%policy_origin] : buffer -> view<6xf32>
  %words = buffer.view %epoch[%policy_origin] : buffer -> view<6xi32>
  %seeds = buffer.view %epoch[%seed_origin] : buffer -> view<1xi64>
  %count = index.cast %rows : index to i32
  view.store %count, %header[2] : i32, view<3xi32>
  view.store %temperature, %floats[0] : f32, view<6xf32>
  view.store %top_p, %floats[1] : f32, view<6xf32>
  view.store %top_k, %words[2] : i32, view<6xi32>
  view.store %flags, %words[3] : i32, view<6xi32>
  view.store %seed, %seeds[0] : i64, view<1xi64>
  kernel.return
}}

kernel.def @test_identities(%rows_value: index) {{
  %rows = index.assume %rows_value [range(%rows_value, 1, 2147483647)] : index
  %c256 = index.constant 256 : index
  %c1 = index.constant 1 : index
  %groups_up = index.add %rows, %c256 : index
  %groups_down = index.sub %groups_up, %c1 : index
  %groups = index.div %groups_down, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_value: index, %epoch: buffer) {{
  %capacity = config.get @inkling.packet_capacity : index
  %identity_origin = func.call @inkling_packet_identity_origin(%capacity) : (index) -> (offset)
  %rows = index.assume %rows_value [range(%rows_value, 1, 2147483647)] : index
  %records = buffer.view %epoch[%identity_origin] : buffer -> view<[%capacity]x2xi32>
  %group = kernel.workgroup.id<x> : index
  %thread = kernel.workitem.id<x> : index
  %c256 = index.constant 256 : index
  %base = index.mul %group, %c256 : index
  %row_value = index.add %base, %thread : index
  %live = index.cmp ult, %row_value, %rows : index
  scf.if %live {{
    %row = index.assume %row_value [ge(%row_value, 0), lt(%row_value, %capacity)] : index
    %c0_i32 = scalar.constant 0 : i32
    %position = index.cast %row : index to i32
    view.store %c0_i32, %records[%row, 0] : i32, view<[%capacity]x2xi32>
    view.store %position, %records[%row, 1] : i32, view<[%capacity]x2xi32>
  }}
  kernel.return
}}

"""
    return head + "\n".join(case_text(*spec) for spec in SPECS + SLOW)


def run(device, work):
    loom = os.path.join(work, "sampler_test.loom")
    open(loom, "w").write(module_text())
    linked = os.path.join(work, "sampler_all.loombc")
    vendor = os.path.join(HERE, "vendor")
    subprocess.run(["loom-link", loom, os.path.join(CONTROL, "chat_sampler.loom"), os.path.join(HERE, "..", "..", "lib", "vendor", "packet.loom"), os.path.join(vendor, "sampling.loom"), os.path.join(vendor, "random.loom"), "--to=bc", f"--output={linked}"], check=True)
    out = os.path.join(work, "out")
    os.makedirs(out, exist_ok=True)
    result = subprocess.run(["loom-test", linked, f"--device={device}", f"--file-output-root={out}", f"--config=inkling.packet_capacity={N}"], capture_output=True, text=True)
    report = json.loads(result.stdout) if result.stdout.startswith("{") else {}
    return out, report, result


failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(("PASS" if ok else "FAIL"), name, detail)


results = {}
for device in sys.argv[1:]:
    work = tempfile.mkdtemp(prefix="sampler_")
    out, report, process = run(device, work)
    label = device.split("//")[-1][:12]
    check(f"{label} loom-test", process.returncode == 0 and report.get("failed_sample_count", 1) == 0, f"samples {report.get('sample_count')} failed {report.get('failed_sample_count')} {process.stderr[-300:] if process.returncode else ''}")
    if process.returncode:
        continue
    for name, temperature, top_p, top_k, flags in SPECS:
        a, b, c = (np.load(os.path.join(out, f"{name}_{x}.npy")) for x in "abc")
        results[(device, name)] = a
        check(f"{label} {name} determinism", np.array_equal(a, b), "same seed twice gives equal tokens")
        p = expected(temperature, top_p, top_k, flags)
        counts = np.array([(a == t).sum() for t in IDS])
        outside = int(len(a) - counts.sum())
        tail = N * tail_mass(temperature, top_p, top_k, flags)
        inside_ok = (outside == 0) if tail < 5 else (0.6 * tail <= outside <= 1.3 * tail)
        check(f"{label} {name} support", inside_ok and all(counts[i] == 0 for i in range(8) if p[i] == 0), f"outside {outside} (expected about {tail:.0f}), forbidden hits {[int(counts[i]) for i in range(8) if p[i] == 0]}")
        live = p > 0
        counts = counts
        if live.sum() > 1:
            inside = int(counts.sum())
            statistic = float((((counts[live] - inside * p[live]) ** 2) / (inside * p[live])).sum())
            pvalue = float(chi2.sf(statistic, live.sum() - 1))
            check(f"{label} {name} chi-square", pvalue > 0.001, f"chi2 {statistic:.2f} df {live.sum() - 1} p {pvalue:.3f}")
            changed = float((a != c).mean())
            check(f"{label} {name} seeds differ", changed > 0.3, f"{changed:.2f} of draws change with the seed")
        else:
            check(f"{label} {name} deterministic choice", counts[int(np.argmax(p))] == N, f"{counts.tolist()}")
    for fast, slow in PAIRS:
        a = np.load(os.path.join(out, f"{fast}_a.npy"))
        s = np.load(os.path.join(out, f"{slow}_a.npy"))
        check(f"{label} {fast} equals template path", np.array_equal(a, s), f"{int((a != s).sum())} of {N} tokens differ from {slow}")
if len(sys.argv) > 2:
    first = sys.argv[1]
    for name, *_ in SPECS:
        same = float(np.mean(results[(first, name)] == results[(sys.argv[2], name)])) if (first, name) in results and (sys.argv[2], name) in results else -1
        print("INFO", name, f"same-seed tokens equal across the two GPUs: {same:.4f}")
print(f"{'FAIL' if failures else 'PASS'}: {failures} failing checks")
sys.exit(1 if failures else 0)
