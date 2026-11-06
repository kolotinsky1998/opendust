# Non-uniform Ar+/Ar presheath

`run.py` demonstrates the required two-stage workflow:

1. equilibrate noninteracting Ar+ test ions with the quadratic field and
   Phelps null collisions, then save the cylindrical compensating background;
2. load that background and run the interacting OpenDustJAX simulation.

The preparation and production configurations must use identical geometry,
particle count, time step, seed, softening, plasma parameters, and cross-section
tables. The production step count may differ. `dt_s` is the outer ion-ion-force
step; `CollisionSubcyclingConfig()` automatically resolves collision substeps
whose null-collision candidate probability remains at or below 0.1. Charging
uses a still finer grid when required.

The preparation writes `presheath_background.npz/json`. The NPZ file contains
`density_m3`, `ion_potential_V`, the negative compensating `potential_V`, the
`r_m`/`z_m` grid, and the convergence history.
