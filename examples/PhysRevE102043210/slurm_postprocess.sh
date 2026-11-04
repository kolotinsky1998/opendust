#!/bin/bash
#SBATCH --job-name=od-figures
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$EXAMPLE_DIR/postprocess.py" --config "$EXAMPLE_DIR/article_config.json"
