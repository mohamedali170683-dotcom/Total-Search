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
        Platform.GOOGLE: 0.30,
        Platform.AMAZON: 0.22,
        Platform.YOUTUBE: 0.15,
        Platform.INSTAGRAM: 0.14,
        Platform.TIKTOK: 0.11,
        Platform.PINTEREST: 0.08,
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
    """
    Research-backed weight configurations for different audience profiles.

    Data sources:
    - NP Digital Web Summit 2025: daily search volumes across platforms
    - Jungle Scout Consumer Trends 2025: where consumers start product searches
    - Claneo State of Search 2025: platform usage by age group
    - SparkToro/Datos Q2 2025: search market share
    - McKinsey State of Beauty 2025: TikTok Shop category data
    - Adobe Gen Z Search Study 2024: Gen Z search behavior
    """

    # General: average adult, cross-industry
    GENERAL = {
        Platform.GOOGLE: 0.30,
        Platform.AMAZON: 0.22,
        Platform.YOUTUBE: 0.15,
        Platform.INSTAGRAM: 0.14,
        Platform.TIKTOK: 0.11,
        Platform.PINTEREST: 0.08,
    }

    # E-commerce: purchase-focused brands
    ECOMMERCE = {
        Platform.AMAZON: 0.32,
        Platform.GOOGLE: 0.25,
        Platform.PINTEREST: 0.14,
        Platform.INSTAGRAM: 0.12,
        Platform.YOUTUBE: 0.10,
        Platform.TIKTOK: 0.07,
    }

    # Beauty & Lifestyle: beauty, fashion, food, home
    BEAUTY = {
        Platform.TIKTOK: 0.25,
        Platform.INSTAGRAM: 0.22,
        Platform.GOOGLE: 0.18,
        Platform.AMAZON: 0.15,
        Platform.PINTEREST: 0.12,
        Platform.YOUTUBE: 0.08,
    }

    # Gen Z Audience: ages 16-27
    GEN_Z = {
        Platform.TIKTOK: 0.26,
        Platform.INSTAGRAM: 0.22,
        Platform.YOUTUBE: 0.18,
        Platform.GOOGLE: 0.18,
        Platform.PINTEREST: 0.08,
        Platform.AMAZON: 0.08,
    }

    # B2B / Technology: tech, SaaS, professional services
    B2B_TECH = {
        Platform.GOOGLE: 0.40,
        Platform.YOUTUBE: 0.22,
        Platform.AMAZON: 0.18,
        Platform.INSTAGRAM: 0.08,
        Platform.TIKTOK: 0.07,
        Platform.PINTEREST: 0.05,
    }

    # Video & Content: entertainment, education, creators
    VIDEO_CONTENT = {
        Platform.YOUTUBE: 0.30,
        Platform.TIKTOK: 0.25,
        Platform.INSTAGRAM: 0.18,
        Platform.GOOGLE: 0.15,
        Platform.PINTEREST: 0.07,
        Platform.AMAZON: 0.05,
    }

    @classmethod
    def get_preset(cls, name: str) -> dict[Platform, float]:
        """Get a weight preset by name."""
        presets = {
            "general": cls.GENERAL,
            "ecommerce": cls.ECOMMERCE,
            "beauty": cls.BEAUTY,
            "gen_z": cls.GEN_Z,
            "b2b_tech": cls.B2B_TECH,
            "video_content": cls.VIDEO_CONTENT,
        }
        return presets.get(name.lower(), cls.GENERAL)

    @classmethod
    def list_presets(cls) -> dict[str, str]:
        """List available presets with descriptions."""
        return {
            "general": "Average adult, cross-industry (NP Digital 2025, Jungle Scout 2025)",
            "ecommerce": "Purchase-focused brands (Amazon 56-74% product search start)",
            "beauty": "Beauty, fashion, food, home (80% TikTok Shop sales = beauty)",
            "gen_z": "Ages 16-27 (64% use TikTok as search engine)",
            "b2b_tech": "Tech, SaaS, professional services (Google-dominant)",
            "video_content": "Entertainment, education, creators (YouTube + TikTok)",
        }
