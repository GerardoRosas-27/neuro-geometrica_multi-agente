#!/bin/sh
# Infra only: relaunch v6 after a crash/reboot. Finished seeds are cached
# (artifacts/v6/<phase>/seeds), partial arms resume bit-exactly from
# artifacts/v6/<phase>/train_ckpt. Safe to call repeatedly.
cd "$(dirname "$0")/.." || exit 1
n=$(ls artifacts/v6/confirm/seeds/*.json 2>/dev/null | wc -l)
if [ "$n" -ge 16 ] && [ -f artifacts/v6/confirm/metrics.csv ]; then
  echo "v6 DEV+CONFIRM complete"; exit 0
fi
exec "$(dirname "$0")/v6_launch.sh"
