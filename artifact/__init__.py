"""Portable entry points for the FeasPG multistage stochastic programming code."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'real_data_hydro_experiments'))
