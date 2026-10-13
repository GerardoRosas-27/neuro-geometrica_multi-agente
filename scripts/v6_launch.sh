#!/bin/sh
# Launch v6 DEV then CONFIRM (nohup) ONLY when the v5 run has finished.
# Pre-registration: docs/preregistro_v6.md. Do not edit constants here.
# Usage: scripts/v6_launch.sh            (refuses if v5 or v6 is running)
cd "$(dirname "$0")/.." || exit 1
BIN=./target/release/run_v6
if pgrep -f run_predictive_field_v5 >/dev/null; then
  echo "v5 still running (run_predictive_field_v5 alive): not launching v6"; exit 1
fi
# v5 must also be COMPLETE (a reboot can kill it half-way). Read-only check.
V5=${V5_DIR:-/workspace/ngm-v5}/artifacts/v5/confirm
n5=$(ls "$V5"/seeds/*.json 2>/dev/null | wc -l)
if [ "${V6_IGNORE_V5:-0}" != 1 ] && { [ "$n5" -lt 16 ] || [ ! -f "$V5/metrics.csv" ]; }; then
  echo "v5 CONFIRM not complete ($n5/16 seeds): not launching v6 (V6_IGNORE_V5=1 to override)"; exit 1
fi
if pgrep -f "run_v6 run --phase" >/dev/null; then
  echo "v6 already running"; exit 0
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "worktree dirty: commit first (SHA is recorded per seed)"; exit 1
fi
PATH=$HOME/.cargo/bin:$PATH cargo build --release --bin run_v6 || exit 1
nohup sh -c "$BIN run --phase dev >> /tmp/v6dev.log 2>&1 && \
  $BIN run --phase confirm >> /tmp/v6conf.log 2>&1" >/dev/null 2>&1 &
echo "v6 launched (pid $!) sha=$(git rev-parse --short HEAD); logs /tmp/v6dev.log /tmp/v6conf.log"
