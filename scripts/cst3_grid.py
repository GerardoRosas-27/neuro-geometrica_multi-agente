#!/usr/bin/env python3
"""uso: cst3_grid.py JOBS 'name|seed|flags' ...  (líneas también por stdin). Sólo semillas de ajuste (smoke)."""
import sys,subprocess,os
from concurrent.futures import ThreadPoolExecutor
J=int(sys.argv[1]); L=[l for l in sys.argv[2:]] or [l.strip() for l in sys.stdin if l.strip()]
def run(l):
    n,s,f=l.split('|'); subprocess.run(['orbita/scripts/cst3_tune.sh',n,s,f])
with ThreadPoolExecutor(J) as ex: list(ex.map(run,L))
