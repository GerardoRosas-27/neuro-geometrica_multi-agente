#!/bin/sh
# Infra only: (re)launch v5 DEV then CONFIRM if not already running.
# Finished seeds are cached (artifacts/v5/<phase>/seeds), partial seeds
# resume bit-exactly from artifacts/v5/<phase>/train_ckpt. Safe to call
# repeatedly (e.g. from cron @reboot / every few minutes).
cd "$(dirname "$0")/.." || exit 1
BIN=./target/release/run_predictive_field_v5
if pgrep -f "run_predictive_field_v5 run --phase" >/dev/null; then
  echo "v5 already running"; exit 0
fi
[ -x "$BIN" ] || cargo build --release --bin run_predictive_field_v5 || exit 1
nohup sh -c "$BIN run --phase dev >> /tmp/v5dev.log 2>&1 && \
  $BIN run --phase confirm >> /tmp/v5conf.log 2>&1" >/dev/null 2>&1 &
echo "v5 launched (pid $!)"
