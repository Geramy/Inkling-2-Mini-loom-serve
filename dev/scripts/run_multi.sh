#!/bin/bash
# usage: run_multi.sh <session_cache> <prefix_cache> <probe args...>   (box) sessions on the R9700 under the caller's flock
cd ~/inkling/optCHAT || exit 1
SC=$1; PC=$2; shift 2
python3 scripts/chat_session.py --mat mat-r9700 --port 18512 --log multi.log --set session_cache_capacity=$SC --set prefix_cache_capacity=$PC -- bash -c "python3 scripts/chat_multi_probe.py @URL@ $* | cut -c1-230; python3 scripts/chat_obs.py @URL@ 'pool|cache'"
