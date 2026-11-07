# Development notes before DEV
- Rust numeric literal syntax corrected before compiling.
- First topology suite:10/11 passed; Regge fixture had zero interior hinges after random tetrahedron blocking, so an assertion requiring a hinge was inappropriate. Replaced with a complete cube fixture (6 tetrahedra) having exactly one interior body diagonal; boundary hinges excluded. No metric/config/dataset tuning or experiment started before fixing tests.
