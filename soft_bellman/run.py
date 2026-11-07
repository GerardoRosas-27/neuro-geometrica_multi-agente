#!/usr/bin/env python3
"""Orchestrate Rust jobs; Python performs NO training."""
import argparse, concurrent.futures, hashlib, json, os, pathlib, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('phase',choices=['DEV','CONFIRM']);p.add_argument('--workers',type=int,default=4);p.add_argument('--output',default='docs/soft_bellman_v1/raw');a=p.parse_args()
assert 1<=a.workers<=4
out=ROOT/a.output/a.phase;out.mkdir(parents=True,exist_ok=True)
assert not list(out.glob('*/metrics.csv')), 'Refusing to overwrite completed experiment'
source=ROOT/'soft_bellman/src/main.rs'
fingerprint=hashlib.sha256(source.read_bytes()).hexdigest()
freeze=ROOT/'docs/soft_bellman_v1/FROZEN.json'
if a.phase=='CONFIRM':
    assert freeze.exists(), 'Freeze implementation before CONFIRM'
    assert json.loads(freeze.read_text())['source_sha256']==fingerprint
start=time.time();base=0xA400 if a.phase=='DEV' else 0xB400
meta={'phase':a.phase,'workers':a.workers,'seed_start':base,'seed_end':base+15,'source_sha256':fingerprint,'start_epoch':start,'command':__import__('sys').argv,'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),'system':os.uname().sysname+' '+os.uname().machine,'nice':os.getpriority(os.PRIO_PROCESS,0)}
(out/'run.json').write_text(json.dumps(meta,indent=2))
def job(seed):
    d=out/f'{seed:04X}';d.mkdir(exist_ok=True)
    with (d/'stdout.log').open('w') as log:
        subprocess.run([str(ROOT/'soft_bellman/target/release/soft-bellman-v1'),'train',str(seed),a.phase,str(d)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2700)
    return seed
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
    for seed in ex.map(job,range(base,base+16)):
        print(f'{a.phase} {seed:04X} completed elapsed={time.time()-start:.2f}s',flush=True)
        assert time.time()-start<2700, 'Global 45-minute bound'
meta['elapsed_seconds']=time.time()-start;meta['completed']=16
(out/'run.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta),flush=True)
