"""Makes ``import bama_deep`` work when pytest collects this folder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
