"""Rewrites the skeleton placeholder kernel into an echo kernel: the first L generated tokens spell the prompt token IDs as six letters
each (letter a + digit), then EOS. L is the prompt length, so the prompt must fit one prefill span.
usage: chat_echo_kernel.py <kernel.loom in> <kernel.loom out>"""
import re, sys

text = open(sys.argv[1], encoding="utf-8").read()
text = text.replace("view.store %next_k, %cache[0] : i32, view<1xi32>", "view.store %token, %cache[0] : i32, view<1xi32>")
text = text.replace("    view.store %final_h, %write_state[0] : i32, view<1xi32>\n", "    %carry = scf.select %fresh, %length32, %initial_h : i32\n    view.store %carry, %write_state[0] : i32, view<1xi32>\n")
text = text.replace("    view.store %final_h, %hidden[%span] : i32, view<[%spans]xi32>\n", "    view.store %carry, %hidden[%span] : i32, view<[%spans]xi32>\n")
start = text.index("      %sum = scf.for %position = [%c0 to %end step %c1]")
end = text.index("      %result = buffer.view %outputs[%origin]")
new = """      %hidden = buffer.view %residual[%origin] : buffer -> view<[%spans]xi32>
      %prompt = view.load %hidden[%span] : view<[%spans]xi32> -> i32
      %step = scalar.subi %end32, %prompt : i32
      %c6_i32 = scalar.constant 6 : i32
      %c5_i32 = scalar.constant 5 : i32
      %c10_i32 = scalar.constant 10 : i32
      %c64_i32 = scalar.constant 64 : i32
      %token_index = scalar.divui %step, %c6_i32 : i32
      %digit_position = scalar.remui %step, %c6_i32 : i32
      %in_range = scalar.cmpi ult, %token_index, %prompt : i32
      %safe_index = scf.select %in_range, %token_index, %c0_i32 : i32
      %safe_bound = scalar.assume %safe_index [range(%safe_index, 0, 2147483646)] : i32
      %safe_value = index.cast %safe_bound : i32 to index
      %safe_position = index.assume %safe_value [range(%safe_value, 0, 2147483646)] : index
      %c5 = index.constant 5 : index
      %offset = func.call @history_offset(%origins, %row, %safe_position, %blocks) : (buffer, index, index, index) -> (offset)
      %cache = buffer.view %arena[%offset] : buffer -> view<1xi32>
      %value = view.load %cache[0] : view<1xi32> -> i32
      %remaining = scalar.subi %c5_i32, %digit_position : i32
      %divisor = scf.for %k = [%c0 to %c5 step %c1](%power = %c1_i32 : i32) -> (i32) {
        %k_word = index.cast %k : index to i32
        %scale = scalar.cmpi ult, %k_word, %remaining : i32
        %scaled = scalar.muli %power, %c10_i32 : i32
        %next = scf.select %scale, %scaled, %power : i32
        scf.yield %next : i32
      }
      %quotient = scalar.divui %value, %divisor : i32
      %digit = scalar.remui %quotient, %c10_i32 : i32
      %letter = scalar.addi %digit, %c64_i32 : i32
      %eos = view.load %header[1] : view<4xi32> -> i32
      %prediction = scf.select %in_range, %letter, %eos : i32
"""
text = text[:start] + new + text[end:]
text = text.replace("      %c4 = index.constant 4 : index\n      %context = config.get @inkling.context : index\n      %last_position = index.sub %context, %c1 : index\n      %last_block = index.div %last_position, %c4 : index\n      %blocks = index.add %last_block, %c1 : index\n      %sum", "XX")
open(sys.argv[2], "w", encoding="utf-8").write(text)
