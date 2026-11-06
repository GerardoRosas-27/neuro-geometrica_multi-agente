#!/usr/bin/env bash
# Chequeo de salud (cron cada hora): último evento, edad, disco, GPU.
RUN=${RUN:?}; L=$RUN/stream_log.jsonl
echo "== $(date -Iseconds)"; tail -n 3 $L 2>/dev/null
[ -f $L ] && echo "edad log: $(( $(date +%s) - $(stat -c %Y $L) )) s (alerta si > 3x duración típica de shard)"
grep -c divergencia $L 2>/dev/null | sed 's/^/divergencias: /'; grep -q abortado $L 2>/dev/null && echo "ALERTA: abortado"
df -h $RUN | tail -1; command -v rocm-smi >/dev/null && rocm-smi --showuse --showmemuse | grep -E "GPU|%" | head -6
pgrep -f run_stream.sh >/dev/null && echo "supervisor vivo" || echo "ALERTA: supervisor NO corre"
