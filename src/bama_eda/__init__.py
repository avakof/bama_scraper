"""Exploratory data analysis for the Bama monitoring database.

A separate package on purpose: analysis code has different failure modes from
collection code, and mixing them makes it too easy for an analytical convenience
to change what gets scraped.

Nothing here writes to the mother database.
"""

from .models import ANALYSIS_VERSION

__all__ = ["ANALYSIS_VERSION"]
