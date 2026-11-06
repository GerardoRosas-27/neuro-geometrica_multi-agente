# Final evidence audit
- 512 aggregated rows = 32 seeds × 4 splits × 4 models.
- Exact DEV0xA400–A40F and CONFIRM0xB400–B40F ranges present; no missing seeds.
- Each of32 logs contains epochs0..160, and final field training NLL < initial NLL for32/32.
- No NaN/Infinity in reported losses or parameters; all field route successes100% on evaluated connected maps.
- Training source SHA256 equals the pre-CONFIRM freeze.
- Debug tests9/9 and optimized release tests9/9; `tests.log`, `tests_release.log`.
- One complete B400 replay matches all non-timing output exactly.
- All modifications confined to `soft_bellman/` and `docs/soft_bellman_v1/`.
- No root app tests/benchmarks/training run, no production integration claimed.
