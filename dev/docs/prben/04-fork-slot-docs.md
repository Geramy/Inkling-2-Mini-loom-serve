# PR 4: document the fork slot contract and prefix cache limits (docs only)

Branch: loom-serve-fork-slot-docs (https://github.com/Geramy/hrx-system/tree/loom-serve-fork-slot-docs)
Commit: 28181ee5, author Geramy Loveless <geramy.loveless@amd.com>, base main 5fbb30ba
Patch: 04-fork-slot-docs.patch
Diff stat: model-programs.md +17, http-profiles.md 3 lines, model-documents.md 4 lines (3 files, 22 insertions, 3 deletions)

## Title
[Loom] Document the recurrent fork contract and prefix cache limits

## Body
A text program that updates recurrent state in place must handle the first span after a fork. The runtime gives a forked row a private writer slot and keeps the shared slot as an immutable reader (`state.h`, `recurrent.anchor`), and it does not copy the slot ("a branch needs a private recurrent writer without copying the entire KV prefix"). This was visible only in the fixture README and the tiny-text `encode_state`.

We hit it in a real package (Inkling, hybrid ring and conv state): the attention kernels used the write slot only, so after every prefix cache capture and hit the first span read stale state. Greedy output differed from a cold run as early as the 4th token (cache on against cache off identical in 4 of 24 prompts; after the program copied the reader into the writer: 24 of 24, and 48 of 48 repeats). Our notes had assumed "a fork copies the slot". The fix is on the program side, so this PR is documentation.

Changes:
- `model-programs.md`: the `encode_state` record (four little-endian 64-bit words: row, reader offset, writer offset, consumed positions; from `loom_serve_text_publish_state_bindings` in `model.c`), when reader and writer differ, and what a program must do.
- `http-profiles.md`: a prefix hit ends at a retained boundary; the tokens after it replay, at most about one prefill window.
- `model-documents.md`: `session_cache_capacity` serves requests that name a session; `prefix_cache_capacity` endpoints are shared by all requests and each holds one recurrent slot.

Please check the claims against the code. I derived them from `model.c` (record layout), `state.c` (anchor release) and `service.c` (capture and extension). The "about one prefill window" bound is from our measurements on 0.3.0 (hits at multiples of `prefill_capacity`); I did not re-measure on main, where the near-end extension may shorten it.

Documentation only. Not run: the docs site build (`site_test.py`, MkDocs). No commands or links were added.
