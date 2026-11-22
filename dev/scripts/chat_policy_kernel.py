"""Turns a test package (scripts/chat_build_test_pkg.sh) into a policy echo package: encode_epoch appends the first four 24-byte option records of the
control state (written by bind_request) to the metadata packet and the kernel emits them as 48 hex nibbles (letter a + nibble, high nibble first).
usage: chat_policy_kernel.py <package dir>"""
import sys

pkg = sys.argv[1]
kernel_path, control_path = pkg + "/kernel.loom", pkg + "/control.loom"
text = open(kernel_path, encoding="utf-8").read()
text = text.replace("    view.store %final_h, %write_state[0] : i32, view<1xi32>\n", "    %carry = scf.select %fresh, %length32, %initial_h : i32\n    view.store %carry, %write_state[0] : i32, view<1xi32>\n")
text = text.replace("    view.store %final_h, %hidden[%span] : i32, view<[%spans]xi32>\n", "    view.store %carry, %hidden[%span] : i32, view<[%spans]xi32>\n")
start = text.index("      %sum = scf.for %position = [%c0 to %end step %c1]")
end = text.index("      %result = buffer.view %outputs[%origin]")
new = """      %hidden = buffer.view %residual[%origin] : buffer -> view<[%spans]xi32>
      %prompt = view.load %hidden[%span] : view<[%spans]xi32> -> i32
      %step = scalar.subi %end32, %prompt : i32
      %c48_i32 = scalar.constant 48 : i32
      %c2_i32_b = scalar.constant 2 : i32
      %c15_i32 = scalar.constant 15 : i32
      %c4_i32 = scalar.constant 4 : i32
      %c64_i32 = scalar.constant 64 : i32
      %c24_i64 = scalar.constant 24 : i64
      %c36_i64 = scalar.constant 36 : i64
      %in_range = scalar.cmpi ult, %step, %c48_i32 : i32
      %byte_index = scalar.divui %step, %c2_i32_b : i32
      %low_nibble = scalar.remui %step, %c2_i32_b : i32
      %count64 = scalar.extui %count32 : i32 to i64
      %row64 = scalar.extui %row32 : i32 to i64
      %table_bytes = scalar.muli %count64, %c36_i64 : i64
      %row_bytes = scalar.muli %row64, %c24_i64 : i64
      %byte64 = scalar.extui %byte_index : i32 to i64
      %relative = scalar.addi %table_bytes, %row_bytes : i64
      %with_byte = scalar.addi %relative, %byte64 : i64
      %with_header = scalar.addi %with_byte, %c16_i64 : i64
      %load_bound = scalar.assume %with_header [range(%with_header, 0, 4294967295)] : i64
      %load_offset = index.cast %load_bound : i64 to offset
      %byte = buffer.load.i8.u %metadata[%load_offset]
      %shifted = scalar.shrui %byte, %c4_i32 : i32
      %masked = scalar.andi %byte, %c15_i32 : i32
      %is_low = scalar.cmpi eq, %low_nibble, %c1_i32 : i32
      %nibble = scf.select %is_low, %masked, %shifted : i32
      %letter = scalar.addi %nibble, %c64_i32 : i32
      %eos = view.load %header[1] : view<4xi32> -> i32
      %prediction = scf.select %in_range, %letter, %eos : i32
"""
new = new.replace("%c16_i64", "%c16_i64")
new = "      %c16_i64 = scalar.constant 16 : i64\n" + new
text = text[:start] + new + text[end:]
open(kernel_path, "w", encoding="utf-8").write(text)

control = open(control_path, encoding="utf-8").read()
old = "  %metadata_length = scalar.addi %header_length, %plan_length : i64\n"
assert old in control
control = control.replace(old, """  %policy_origin = index.constant 128 : offset
  %policy_bytes = index.constant 96 : offset
  %policy_target = index.add %header_bytes, %plan_bytes : offset
  buffer.copy %control_state[%policy_origin], %metadata[%policy_target], %policy_bytes
  %base_length = scalar.addi %header_length, %plan_length : i64
  %policy_length = scalar.constant 96 : i64
  %metadata_length = scalar.addi %base_length, %policy_length : i64
""")
open(control_path, "w", encoding="utf-8").write(control)
