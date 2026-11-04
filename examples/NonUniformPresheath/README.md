# Non-uniform Ar+/Ar presheath

`run.py` demonstrates the required two-stage workflow:

1. equilibrate noninteracting Ar+ test ions with the quadratic field and
   Phelps null collisions, then save the cylindrical compensating background;
2. load that background and run the interacting OpenDustJAX simulation.

The preparation and production configurations must use identical geometry,
particle count, time step, seed, softening, plasma parameters, and cross-section
tables. The production step count may differ. Choose `dt_s` so the constructor's
null-collision candidate probability check remains at or below 0.1.

The preparation writes `presheath_background.npz/json`. The NPZ file contains
`density_m3`, `ion_potential_V`, the negative compensating `potential_V`, the
`r_m`/`z_m` grid, and the convergence history.
