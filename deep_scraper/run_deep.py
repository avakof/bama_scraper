#!/usr/bin/env python3
"""Launcher so the deep scraper runs without any install step.

Puts this folder on sys.path (for ``bama_deep``) and relies on the existing
editable install of ``bama_scraper`` for the shared normalization/fetch layer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bama_deep.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
