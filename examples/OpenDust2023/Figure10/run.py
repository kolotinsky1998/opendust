#!/usr/bin/env python3
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import run_case_cli
from config import CASE_COUNT, build_case

if __name__ == "__main__":
    run_case_cli(HERE, CASE_COUNT, "OpenDust 2023 Figure 10 collision sweep", build_case)
