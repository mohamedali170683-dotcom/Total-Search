"""Calculators for engagement scores and unified metrics."""

from src.calculators.proxy_scores import (
    InstagramEngagementCalculator,
    TikTokEngagementCalculator,
    # Backward-compatible aliases (deprecated)
    InstagramProxyCalculator,
    TikTokProxyCalculator,
)
from src.calculators.unified_score import UnifiedScoreCalculator

__all__ = [
    # New class names (preferred)
    "TikTokEngagementCalculator",
    "InstagramEngagementCalculator",
    # Deprecated aliases
    "TikTokProxyCalculator",
    "InstagramProxyCalculator",
    "UnifiedScoreCalculator",
]
