#!/usr/bin/env bash
# usage: locked.sh <command...>   (box) exclusive GPU lock; retries when the command ends with status 3 (a foreign server runs)
for i in $(seq 40); do
  flock -x ~/hrx-pr1/gpu.lock "$@"
  rc=$?
  [ $rc -ne 3 ] && exit $rc
  sleep 20
done
exit 3
