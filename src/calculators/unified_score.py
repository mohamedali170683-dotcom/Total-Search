"""Unified score calculator for cross-platform keyword analysis."""

import logging
import math
from typing import Any

from src.models.keyword import (
    Platform,
    PlatformMetrics,
    PlatformScore,
    TrendDirection,
    UnifiedKeywordData,
)

logger = logging.getLogger(__name__)


class UnifiedScoreCalculator:
    """
    Creates a single demand score across all platforms.

    Approach:
    - Normalize each platform's score to 0-100 scale using logarithmic scaling
    - Weight platforms based on relevance (configurable)
    - Calculate weighted average
    - Determine overall trend direction
    """

    DEFAULT_WEIGHTS = {
        Platform.GOOGLE: 0.25,
        Platform.YOUTUBE: 0.18,
        Platform.AMAZON: 0.18,
        Platform.TIKTOK: 0.13,
        Platform.INSTAGRAM: 0.13,
        Platform.PINTEREST: 0.13,
    }

    # Normalization parameters (calibrated for typical search volumes)
    # Using log scale to handle the massive range (10 to 10,000,000+)
    LOG_BASE = 10
    MIN_VOLUME_FOR_SCORE = 10  # Minimum volume to register a score
    MAX_VOLUME_REFERENCE = 10_000_000  # Reference for max score (100)

    def __init__(self, weights: dict[Platform, float] | None = None):
        """
        Initialize calculator with optional custom weights.

        Args:
            weights: Custom platform weights (should sum to 1.0)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Validate that weights sum to approximately 1.0."""
        total = sum(self.weights.values())
        if not 0.99 <= total <= 1.01:
            logger.warning(f"Platform weights sum to {total}, normalizing...")
            for platform in self.weights:
                self.weights[platform] /= total

    def calculate(
        self,
        keyword_data: UnifiedKeywordData,
        weights: dict[Platform, float] | None = None,
    ) -> UnifiedKeywordData:
        """
        Calculate unified score and update keyword data.

        Args:
            keyword_data: UnifiedKeywordData with platform metrics
            weights: Optional custom weights for this calculation

        Returns:
            Updated UnifiedKeywordData with unified metrics
        """
        active_weights = weights or self.weights
        platform_scores: list[PlatformScore] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for platform, metrics in keyword_data.platforms_dict.items():
            if metrics is None:
                continue

            raw_volume = metrics.effective_volume
            if raw_volume < self.MIN_VOLUME_FOR_SCORE:
                continue

            normalized = self._normalize_volume(raw_volume)
            weight = active_weights.get(platform, 0)
            weighted_score = normalized * weight

            platform_scores.append(
                PlatformScore(
                    platform=platform,
                    raw_volume=raw_volume,
                    normalized_score=normalized,
                    weight=weight,
                    weighted_score=weighted_score,
                )
            )

            weighted_sum += weighted_score
            total_weight += weight

        # Calculate final unified score
        if total_weight > 0:
            # Adjust for missing platforms
            unified_score = int(weighted_sum / total_weight)
        else:
            unified_score = 0

        # Determine cross-platform trend
        cross_platform_trend = self.determine_cross_platform_trend(keyword_data)

        # Identify best platform
        best_platform = self.identify_best_platform(platform_scores)

        # Update keyword data
        keyword_data.unified_demand_score = unified_score
        keyword_data.cross_platform_trend = cross_platform_trend
        keyword_data.best_platform = best_platform
        keyword_data.platform_scores = platform_scores

        return keyword_data

    def _normalize_volume(self, volume: int) -> float:
        """
        Normalize a raw volume to 0-100 scale using logarithmic scaling.

        This handles the massive range of search volumes gracefully:
        - 10 → ~0
        - 100 → ~14
        - 1,000 → ~29
        - 10,000 → ~43
        - 100,000 → ~57
        - 1,000,000 → ~71
        - 10,000,000 → 100
        """
        if volume < self.MIN_VOLUME_FOR_SCORE:
            return 0.0

        # Log scale normalization
        log_volume = math.log(volume, self.LOG_BASE)
        log_max = math.log(self.MAX_VOLUME_REFERENCE, self.LOG_BASE)
        log_min = math.log(self.MIN_VOLUME_FOR_SCORE, self.LOG_BASE)

        normalized = ((log_volume - log_min) / (log_max - log_min)) * 100

        # Clamp to 0-100
        return max(0.0, min(100.0, normalized))

    def determine_cross_platform_trend(
        self,
        keyword_data: UnifiedKeywordData,
    ) -> TrendDirection:
        """
        Determine overall trend direction across all platforms.

        Uses weighted voting based on platform confidence and data availability.
        """
        trend_votes: dict[TrendDirection, float] = {
            TrendDirection.GROWING: 0.0,
            TrendDirection.STABLE: 0.0,
            TrendDirection.DECLINING: 0.0,
        }

        for platform, metrics in keyword_data.platforms_dict.items():
            if metrics is None or metrics.trend is None:
                continue

            # Weight by platform importance and confidence
            base_weight = self.weights.get(platform, 0.1)
            confidence_multiplier = {
                "high": 1.0,
                "medium": 0.7,
                "proxy": 0.5,
            }.get(metrics.confidence.value, 0.5)

            vote_weight = base_weight * confidence_multiplier
            trend_votes[metrics.trend] += vote_weight

        # Return trend with highest weighted votes
        if not any(trend_votes.values()):
            return TrendDirection.STABLE

        return max(trend_votes, key=trend_votes.get)

    def identify_best_platform(
        self,
        platform_scores: list[PlatformScore],
    ) -> Platform | None:
        """
        Identify the platform with highest relative demand.

        Compares normalized scores (not raw volumes) to account for
        different scale across platforms.
        """
        if not platform_scores:
            return None

        # Find platform with highest normalized score
        best = max(platform_scores, key=lambda ps: ps.normalized_score)
        return best.platform

    def get_platform_breakdown(
        self,
        keyword_data: UnifiedKeywordData,
    ) -> dict[str, Any]:
        """
        Get detailed breakdown of scores by platform.

        Useful for debugging and understanding score composition.
        """
        breakdown = {
            "keyword": keyword_data.keyword,
            "unified_score": keyword_data.unified_demand_score,
            "cross_platform_trend": keyword_data.cross_platform_trend.value,
            "best_platform": keyword_data.best_platform.value if keyword_data.best_platform else None,
            "platforms": {},
        }

        if keyword_data.platform_scores:
            for ps in keyword_data.platform_scores:
                breakdown["platforms"][ps.platform.value] = {
                    "raw_volume": ps.raw_volume,
                    "normalized_score": round(ps.normalized_score, 2),
                    "weight": ps.weight,
                    "weighted_contribution": round(ps.weighted_score, 2),
                }

        return breakdown


class WeightPresets:
    """Predefined weight configurations for different use cases."""

    # Balanced weights for general analysis
    BALANCED = {
        Platform.GOOGLE: 0.22,
        Platform.YOUTUBE: 0.17,
        Platform.AMAZON: 0.17,
        Platform.TIKTOK: 0.15,
        Platform.INSTAGRAM: 0.14,
        Platform.PINTEREST: 0.15,
    }

    # E-commerce focused (Amazon + Pinterest weighted higher)
    ECOMMERCE = {
        Platform.GOOGLE: 0.15,
        Platform.YOUTUBE: 0.08,
        Platform.AMAZON: 0.40,
        Platform.TIKTOK: 0.10,
        Platform.INSTAGRAM: 0.07,
        Platform.PINTEREST: 0.20,
    }

    # Content/Influencer focused (social platforms weighted higher)
    CONTENT = {
        Platform.GOOGLE: 0.12,
        Platform.YOUTUBE: 0.20,
        Platform.AMAZON: 0.08,
        Platform.TIKTOK: 0.25,
        Platform.INSTAGRAM: 0.20,
        Platform.PINTEREST: 0.15,
    }

    # SEO focused (Google weighted higher)
    SEO = {
        Platform.GOOGLE: 0.45,
        Platform.YOUTUBE: 0.18,
        Platform.AMAZON: 0.10,
        Platform.TIKTOK: 0.10,
        Platform.INSTAGRAM: 0.07,
        Platform.PINTEREST: 0.10,
    }

    # Video content focused
    VIDEO = {
        Platform.GOOGLE: 0.12,
        Platform.YOUTUBE: 0.40,
        Platform.AMAZON: 0.08,
        Platform.TIKTOK: 0.25,
        Platform.INSTAGRAM: 0.05,
        Platform.PINTEREST: 0.10,
    }

    # Lifestyle/Visual (Pinterest + Instagram heavy - beauty, fashion, home, food)
    LIFESTYLE = {
        Platform.GOOGLE: 0.15,
        Platform.YOUTUBE: 0.12,
        Platform.AMAZON: 0.13,
        Platform.TIKTOK: 0.15,
        Platform.INSTAGRAM: 0.20,
        Platform.PINTEREST: 0.25,
    }

    @classmethod
    def get_preset(cls, name: str) -> dict[Platform, float]:
        """Get a weight preset by name."""
        presets = {
            "balanced": cls.BALANCED,
            "ecommerce": cls.ECOMMERCE,
            "content": cls.CONTENT,
            "seo": cls.SEO,
            "video": cls.VIDEO,
            "lifestyle": cls.LIFESTYLE,
        }
        return presets.get(name.lower(), cls.BALANCED)

    @classmethod
    def list_presets(cls) -> dict[str, str]:
        """List available presets with descriptions."""
        return {
            "balanced": "General analysis across all platforms",
            "ecommerce": "Amazon + Pinterest weighted for shopping intent",
            "content": "Social platforms weighted for content creators",
            "seo": "Google weighted for organic search focus",
            "video": "YouTube + TikTok weighted for video content",
            "lifestyle": "Pinterest + Instagram for visual/lifestyle brands",
        }
