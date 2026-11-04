#!/bin/bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXAMPLE_DIR"
PYTHON_EXECUTABLE="${OPENDUST_PYTHON:-$(command -v python3)}"
if ! "$PYTHON_EXECUTABLE" -c 'import opendust_jax' >/dev/null 2>&1; then
  echo "Python $PYTHON_EXECUTABLE cannot import opendust_jax. Activate opendust-jax first." >&2
  exit 1
fi

submitted_jobs=()
cleanup_partial_submission() {
  if ((${#submitted_jobs[@]})); then
    echo "Submission failed; cancelling partial job graph: ${submitted_jobs[*]}" >&2
    scancel "${submitted_jobs[@]}" || true
  fi
}
trap cleanup_partial_submission ERR

export_spec="ALL,OPENDUST_EXAMPLE_DIR=$EXAMPLE_DIR,OPENDUST_PYTHON=$PYTHON_EXECUTABLE"
submission="$(sbatch --parsable --chdir="$EXAMPLE_DIR" --export="$export_spec" "$EXAMPLE_DIR/slurm_constant.sh")"
constant_job="${submission%%;*}"
submitted_jobs+=("$constant_job")
submission="$(sbatch --parsable --chdir="$EXAMPLE_DIR" --export="$export_spec" --dependency="afterok:$constant_job" "$EXAMPLE_DIR/slurm_branch.sh" linear)"
linear_job="${submission%%;*}"
submitted_jobs+=("$linear_job")
submission="$(sbatch --parsable --chdir="$EXAMPLE_DIR" --export="$export_spec" --dependency="afterok:$constant_job" "$EXAMPLE_DIR/slurm_branch.sh" quadratic)"
quadratic_job="${submission%%;*}"
submitted_jobs+=("$quadratic_job")
submission="$(sbatch --parsable --chdir="$EXAMPLE_DIR" --export="$export_spec" --dependency="afterok:$linear_job:$quadratic_job" "$EXAMPLE_DIR/slurm_postprocess.sh")"
post_job="${submission%%;*}"
submitted_jobs+=("$post_job")
trap - ERR
echo "constant=$constant_job linear=$linear_job quadratic=$quadratic_job postprocess=$post_job"
