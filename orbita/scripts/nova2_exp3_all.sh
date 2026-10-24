#!/usr/bin/env bash
# DEV luego CONFIRM (mismo binario). Reanudable: las seeds con .done se saltan.
set -euo pipefail
cd "$(dirname "$0")/../.."
PHASE=dev JOBS=${JOBS:-7} bash orbita/scripts/nova2_exp3.sh
PHASE=confirm JOBS=${JOBS:-7} bash orbita/scripts/nova2_exp3.sh
