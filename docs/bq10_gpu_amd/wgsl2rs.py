# Transpila compute_phi_* (WGSL) a Rust f32, mecánicamente, para inferencia CPU idéntica al shader.
import re,sys
def ext(path,name):
    s=open(path,encoding='utf-8').read()
    i=s.index('fn '+name); j=s.index('\n}\n',i)+3
    f=s[i:j]
    f=f.replace('fn %s(x: f32, y: f32, idx: u32) -> f32'%name,'#[allow(unused_parens,clippy::all)]\npub fn %s(x: f32, y: f32, idx: u32) -> f32'%name)
    f=re.sub(r'\b(\d+)u\b',r'\1u32',f)
    f=re.sub(r'\bf32\(',r'f32c(',f)
    return f
out='// AUTOGENERADO por docs/bq10_gpu_amd/wgsl2rs.py desde los shaders WGSL (x50: rama; x20: zip)\nuse super::w::*;\n'
out+=ext(sys.argv[1],'compute_phi_1024')+'\n'+ext(sys.argv[2],'compute_phi_512')
open(sys.argv[3],'w').write(out)
