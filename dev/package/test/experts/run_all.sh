#!/usr/bin/env bash
# Runs all expert tests on one GPU. Usage: run_all.sh gfx1201|gfx1151 (on the GPU box, venv active, ~/inkling/inkD/package/test/experts).
# Correctness runs take a shared GPU lock inside runner.py. Exit code 1 if any test fails.
set -u
arch=${1:?arch}
rc=0
python test_gate_up.py "$arch" || rc=1
python test_oracle.py "$arch" || rc=1
ROWS=5 PATHS=portable python test_oracle.py "$arch" 0 9 19 27 39 20 21 22 23 || rc=1
ROWS=3,9 PATHS=portable python test_oracle.py "$arch" 0 19 || rc=1
python test_prefill.py "$arch" 37 308 512 || rc=1
exit $rc
