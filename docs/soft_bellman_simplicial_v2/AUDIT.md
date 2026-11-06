# Final audit
- Frozen Rust source SHA matches pre-CONFIRM commit; no training configuration changes after DEV.
- Exact DEV/CONFIRM16+16 seed sets;128 learned variants total,32 matched seed packages.
- Every raw log has epochs0..120, all4 training losses improve for32/32 packages.
-1120 rows=32seeds×5splits×7models, all reported numeric losses/metrics finite.
-11 base debug/release tests and3 supplementary orientation/gauge/symmetry checks (14 unique tests) pass.
- Replay all4 models B400 identical after excluding clocks.
- Refinement/control comparisons use identical geometry/true costs and original vertex starts. No continuous-optimality or Hodge/Regge-learning claim.
- Changed-law negative and significant remaining discretization gap explicitly reported.
- All changes restricted to soft_bellman_simplicial/ and docs/soft_bellman_simplicial_v2/; v1 and existing app unchanged.
