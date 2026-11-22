#!/usr/bin/env bash
# usage: ab_abba.sh <gpu> <variant name> <metrics> [reps]   (GPU box) ABBA of mat-<gpu> (A) against var-<gpu>-<variant> (B), each session under the exclusive GPU lock.
G=${1:?gpu}; X=${2:?variant}; MET=${3:?metrics}; REPS=${4:-3}
V=${INKV:-$HOME/inkling/inkV}; H=$(cd "$(dirname "$0")" && pwd)
for s in A B B A; do
  d=$V/mat-$G; [ $s = B ] && d=$V/var-$G-$X
  flock -x ~/hrx-pr1/gpu.lock bash "$H/ab_run.sh" $G "$d" "$X.$s.$(date +%s)" "$MET" "$REPS"
done
