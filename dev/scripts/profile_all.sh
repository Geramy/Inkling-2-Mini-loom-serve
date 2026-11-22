#!/usr/bin/env bash
# usage: profile_all.sh <r9700|strix> [sessions...]   (GPU box) Runs the profiled sessions one by one, each under the exclusive GPU lock.
G=${1:?gpu}; shift
S=${*:-base dshort_p dshort_d d8k_p d8k_d d32k_p d32k_d pf2k pf8k pf32k c4_p c4_d}
for s in $S; do flock -x ~/hrx-pr1/gpu.lock bash "$(dirname "$0")/profile_run.sh" $G $s; done
