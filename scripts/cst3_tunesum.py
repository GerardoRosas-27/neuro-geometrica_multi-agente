import json,glob,sys
pat=sys.argv[1] if len(sys.argv)>1 else '*'
for f in sorted(glob.glob(f'artifacts/constelacion_v3/tune/{pat}/smoke/metrics_*.json')):
  d=json.load(open(f));n=f.split('/')[3]
  for a,v in d['arms'].items():
    g=lambda x: float('nan') if x is None else x
    print(n,d['seed'],a,'new %.3f known %.3f'%(g(v['held']['new_med_log']),g(v['held']['known_med_log'])),'pre %.3f'%g(v['pre_known']['known_med_log']),'acc %.2f/%.2f'%(v['held']['known_acc16'],v['held']['new_acc16']),'wp %.3g'%g(v['wp_rms']),'col',v['collapse']['collapsed'],'e33',[round(x,2) for x in v['e33']['acc']],'orb %.2f triv %.2f'%(d['orbita']['new_med_log'],d['trivial']['new_med_log']))
