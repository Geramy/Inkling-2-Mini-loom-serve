# Lane PRBEN: loom-serve PR and issue drafts for ROCm/hrx-system

Nothing here is opened upstream. Fork: https://github.com/Geramy/hrx-system (branches pushed to the fork only). Base: ROCm/hrx-system main 5fbb30ba. Work clone: C:\Users\gloveles\AppData\Local\Temp\hrx-pr-work\hrx-system. Build tree on the box: ~/inkling/optPRBEN (CMake, clang).

| # | Item | Branch | Commit | Files |
|---|---|---|---|---|
| 1 | Tool args as JSON object | loom-serve-tool-args-object | 6b174210 | 01-tool-args-object.md/.patch |
| 2 | Tool call type default | loom-serve-tool-call-type-default | 8f4f6b56 | 02-tool-call-type-default.md/.patch |
| 3 | Tool name from tool_call_id (design proposal) | loom-serve-tool-name-from-call-id | 4bd399d5 | 03-tool-name-from-call-id.md/.patch |
| 4 | Fork slot contract and cache docs | loom-serve-fork-slot-docs | 28181ee5 | 04-fork-slot-docs.md/.patch |
| 5 | Issue: prompt-end prefix capture | none | | 05-issue-prefix-cache-prompt-end.md |
| 6 | Issue: tool_choice enforcement | none | | 06-issue-tool-choice-enforcement.md |
| 7 | Issue: generated-token checkpoints | none | | 07-issue-generated-token-checkpoints.md |

Repo conventions seen: CONTRIBUTING.md (a PR is evidence plus a concrete implementation; the maintainers may rework or split it; state problem, impact, reproducer, design and verification); commit titles `[Loom] Verb ...`; tests in `tool_policy_test.py` style (Python driving `text_request_check`).

## Opened (2026-11-24)
- PR 1 https://github.com/ROCm/hrx-system/pull/3551
- PR 2 https://github.com/ROCm/hrx-system/pull/3552
- PR 4 https://github.com/ROCm/hrx-system/pull/3553
- PR 3 https://github.com/ROCm/hrx-system/pull/3554
- Issue 05 https://github.com/ROCm/hrx-system/issues/3555
- Issue 06 https://github.com/ROCm/hrx-system/issues/3556
- Issue 07 https://github.com/ROCm/hrx-system/issues/3557
