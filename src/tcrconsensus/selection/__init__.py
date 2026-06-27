"""Method selection: rule-based scenario to method mapping + tiered execution."""

from .selector import select_methods
from .tiered import execute_tiered, split_methods_by_tier, detect_divergent_tcrs, DEFAULT_TIERS

__all__ = [
    "select_methods",
    "execute_tiered",
    "split_methods_by_tier",
    "detect_divergent_tcrs",
    "DEFAULT_TIERS",
]
