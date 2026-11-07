#!/usr/bin/env python3
"""Append audit-only tests to an unchanged copy of the frozen Rust source."""
import pathlib, subprocess, tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/'soft_bellman_simplicial/src/main.rs'
EXTRA=r"""
#[cfg(test)]mod orientation_audit {use super::*;
#[test]fn hodge_edge_orientation_covariance(){
 let mut r=Rng::new(321);let m=Mesh::generate(&mut r,[2,1,1],0.06);let mut flipped=m.clone();
 let signs:Vec<f64>=(0..m.edges.len()).map(|i|if i%3==0{-1.0}else{1.0}).collect();
 for(i,e)in flipped.edges.iter_mut().enumerate(){if signs[i]<0.0{e.swap(0,1);}}
 for b in &mut flipped.b2{for(e,s)in b{*s*=signs[*e];}}
 assert!(flipped.boundary_checks());
 let(m0,m1,m2)=m.hodge_weights(&TEACHER);let h:Vec<_>=(0..m.edges.len()).map(|_|r.normal()).collect();let sh:Vec<_>=h.iter().zip(&signs).map(|(h,s)|h*s).collect();
 let a=m.hodge1(&h,&m0,&m1,&m2);let b=flipped.hodge1(&sh,&m0,&m1,&m2);
 for i in 0..a.len(){assert!((b[i]-signs[i]*a[i]).abs()<1e-9);}
}
#[test]fn gauge_exact_form_has_no_face_curl(){
 let mut r=Rng::new(321);let m=Mesh::generate(&mut r,[2,1,1],0.06);let potential:Vec<_>=(0..m.p.len()).map(|_|r.normal()).collect();
 let a:Vec<_>=m.edges.iter().map(|e|potential[e[1]]-potential[e[0]]).collect();
 for b in &m.b2{let curl:f64=b.iter().map(|&(e,s)|s*a[e]).sum();assert!(curl.abs()<1e-12);}
}
#[test]fn symmetric_transport_and_positive_face_density(){
 let mut r=Rng::new(321);let m=Mesh::generate(&mut r,[2,1,1],0.06);let g=Graph::new(&m,false,false);let c=costs(&m,&g,&TEACHER,Kind::Full,false).0;
 for(i,e)in g.edges.iter().enumerate(){let j=*g.out[e.dst].iter().find(|&&j|g.edges[j].dst==e.src).unwrap();assert!((c[i]-c[j]).abs()<1e-12);assert!(c[i]>0.0);}
}
}
"""
with tempfile.TemporaryDirectory(prefix='simplicial-audit-') as d:
 p=pathlib.Path(d)/'audit.rs';p.write_text(SOURCE.read_text()+EXTRA)
 exe=pathlib.Path(d)/'audit'
 subprocess.run(['rustc','--edition=2021','--test','-C','opt-level=2',str(p),'-o',str(exe)],check=True)
 subprocess.run([str(exe),"--test-threads=2"],check=True)
