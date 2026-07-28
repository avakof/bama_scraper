"""Longitudinal monitoring of Bama.ir vehicle listings.

Observed disappearance is a fact; a sale is an inference. That distinction is
maintained throughout this package -- in the schema, the state machine, the
scoring layer and the reports.
"""

from .config import MONITOR_VERSION

__version__ = MONITOR_VERSION
__all__ = ["MONITOR_VERSION"]
