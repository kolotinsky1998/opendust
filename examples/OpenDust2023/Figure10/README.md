# OpenDust 2023 — Figure 10

Ion drag and self-consistent charge versus collision frequency.

Production parameters are copied from the corresponding legacy example on
`origin/main`. Each case uses one NVIDIA GPU and the CUDA force backend.

```bash
sbatch submit.sbatch
```

The thirteen case result(s), aggregate CSV, PNG, and PDF are written under
`results/`. The last successful array task builds the aggregate artifacts.
For a short smoke run, invoke `run.py` directly on a GPU with overrides such
as `--particles 1024 --steps 10 --background-samples 256 --output-dir /tmp/odj-smoke`.

