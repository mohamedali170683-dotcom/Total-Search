"""Unit tests for unified score calculator."""

import pytest

from src.calculators.unified_score import UnifiedScoreCalculator, WeightPresets
from src.models.keyword import (
    Confidence,
    GoogleMetrics,
    Platform,
    TrendDirection,
    UnifiedKeywordData,
    YouTubeMetrics,
)


class TestUnifiedScoreCalculator:
    """Tests for UnifiedScoreCalculator."""

    def test_calculate_with_all_platforms(self, sample_unified_keyword_data):
        """Test unified score calculation with all platforms."""
        calculator = UnifiedScoreCalculator()
        result = calculator.calculate(sample_unified_keyword_data)

        assert result.unified_demand_score > 0
        assert result.unified_demand_score <= 100
        assert result.cross_platform_trend is not None
        assert result.best_platform is not None
        assert result.platform_scores is not None
        assert len(result.platform_scores) == 5

    def test_calculate_with_single_platform(self, sample_google_metrics):
        """Test unified score calculation with single platform."""
        data = UnifiedKeywordData(
            keyword="test",
            google=sample_google_metrics,
        )

        calculator = UnifiedScoreCalculator()
        result = calculator.calculate(data)

        assert result.unified_demand_score > 0
        assert result.best_platform == Platform.GOOGLE
        assert len(result.platform_scores) == 1

    def test_calculate_with_no_data(self):
        """Test unified score calculation with no platform data."""
        data = UnifiedKeywordData(keyword="test")

        calculator = UnifiedScoreCalculator()
        result = calculator.calculate(data)

        assert result.unified_demand_score == 0
        assert result.platform_scores == []

    def test_normalize_volume_log_scale(self):
        """Test that volume normalization uses log scale correctly."""
        calculator = UnifiedScoreCalculator()

        # Test various volumes
        score_100 = calculator._normalize_volume(100)
        score_1000 = calculator._normalize_volume(1000)
        score_10000 = calculator._normalize_volume(10000)
        score_1000000 = calculator._normalize_volume(1000000)

        # Scores should increase but not linearly
        assert score_1000 > score_100
        assert score_10000 > score_1000
        assert score_1000000 > score_10000

        # Log scale means 10x increase doesn't mean 10x score
        assert score_1000 < score_100 * 3  # Not linear

    def test_normalize_volume_bounds(self):
        """Test that normalized volume stays within bounds."""
        calculator = UnifiedScoreCalculator()

        # Very small volume
        assert calculator._normalize_volume(1) == 0
        assert calculator._normalize_volume(5) == 0

        # Very large volume
        score_huge = calculator._normalize_volume(100000000)
        assert score_huge <= 100

    def test_cross_platform_trend_growing(self):
        """Test cross-platform trend detection for growing keywords."""
        data = UnifiedKeywordData(
            keyword="test",
            google=GoogleMetrics(
                search_volume=10000,
                trend=TrendDirection.GROWING,
                confidence=Confidence.HIGH,
                source="dataforseo_google",
            ),
            youtube=YouTubeMetrics(
                search_volume=5000,
                trend=TrendDirection.GROWING,
                confidence=Confidence.HIGH,
                source="dataforseo_youtube",
            ),
        )

        calculator = UnifiedScoreCalculator()
        trend = calculator.determine_cross_platform_trend(data)

        assert trend == TrendDirection.GROWING

    def test_cross_platform_trend_mixed(self):
        """Test cross-platform trend with mixed signals."""
        data = UnifiedKeywordData(
            keyword="test",
            google=GoogleMetrics(
                search_volume=10000,
                trend=TrendDirection.GROWING,
                confidence=Confidence.HIGH,
                source="dataforseo_google",
            ),
            youtube=YouTubeMetrics(
                search_volume=5000,
                trend=TrendDirection.DECLINING,
                confidence=Confidence.HIGH,
                source="dataforseo_youtube",
            ),
        )

        calculator = UnifiedScoreCalculator()
        trend = calculator.determine_cross_platform_trend(data)

        # Should weight by platform importance - Google has higher default weight
        assert trend in [TrendDirection.GROWING, TrendDirection.STABLE]

    def test_identify_best_platform(self, sample_unified_keyword_data):
        """Test best platform identification."""
        calculator = UnifiedScoreCalculator()
        calculator.calculate(sample_unified_keyword_data)

        best = calculator.identify_best_platform(sample_unified_keyword_data.platform_scores)
        assert best is not None
        assert best in Platform

    def test_custom_weights(self, sample_unified_keyword_data):
        """Test unified score with custom weights."""
        custom_weights = {
            Platform.GOOGLE: 0.8,
            Platform.YOUTUBE: 0.05,
            Platform.AMAZON: 0.05,
            Platform.TIKTOK: 0.05,
            Platform.INSTAGRAM: 0.05,
        }

        calculator = UnifiedScoreCalculator(weights=custom_weights)
        result = calculator.calculate(sample_unified_keyword_data)

        # With Google heavily weighted, Google's contribution should dominate
        google_score = next(
            ps for ps in result.platform_scores if ps.platform == Platform.GOOGLE
        )
        assert google_score.weight == 0.8

    def test_weight_validation(self):
        """Test that weights are normalized if they don't sum to 1."""
        bad_weights = {
            Platform.GOOGLE: 0.5,
            Platform.YOUTUBE: 0.5,
            Platform.AMAZON: 0.5,  # Sum = 1.5
        }

        calculator = UnifiedScoreCalculator(weights=bad_weights)
        total = sum(calculator.weights.values())

        # Should be normalized to ~1.0
        assert 0.99 <= total <= 1.01

    def test_get_platform_breakdown(self, sample_unified_keyword_data):
        """Test platform breakdown output."""
        calculator = UnifiedScoreCalculator()
        calculator.calculate(sample_unified_keyword_data)

        breakdown = calculator.get_platform_breakdown(sample_unified_keyword_data)

        assert "keyword" in breakdown
        assert "unified_score" in breakdown
        assert "platforms" in breakdown
        assert "google" in breakdown["platforms"]
        assert "raw_volume" in breakdown["platforms"]["google"]
        assert "normalized_score" in breakdown["platforms"]["google"]


class TestWeightPresets:
    """Tests for weight presets."""

    def test_balanced_weights_sum_to_one(self):
        """Test that balanced weights sum to 1."""
        total = sum(WeightPresets.BALANCED.values())
        assert abs(total - 1.0) < 0.01

    def test_ecommerce_weights_favor_amazon(self):
        """Test that ecommerce preset favors Amazon."""
        assert WeightPresets.ECOMMERCE[Platform.AMAZON] > WeightPresets.ECOMMERCE[Platform.GOOGLE]
        assert WeightPresets.ECOMMERCE[Platform.AMAZON] > 0.4

    def test_seo_weights_favor_google(self):
        """Test that SEO preset favors Google."""
        assert WeightPresets.SEO[Platform.GOOGLE] > WeightPresets.SEO[Platform.AMAZON]
        assert WeightPresets.SEO[Platform.GOOGLE] >= 0.5

    def test_content_weights_favor_social(self):
        """Test that content preset favors social platforms."""
        social_weight = WeightPresets.CONTENT[Platform.TIKTOK] + WeightPresets.CONTENT[Platform.INSTAGRAM]
        search_weight = WeightPresets.CONTENT[Platform.GOOGLE]
        assert social_weight > search_weight

    def test_get_preset_valid(self):
        """Test getting valid presets."""
        assert WeightPresets.get_preset("balanced") == WeightPresets.BALANCED
        assert WeightPresets.get_preset("ECOMMERCE") == WeightPresets.ECOMMERCE
        assert WeightPresets.get_preset("seo") == WeightPresets.SEO

    def test_get_preset_invalid(self):
        """Test that invalid preset returns balanced."""
        assert WeightPresets.get_preset("invalid") == WeightPresets.BALANCED
        assert WeightPresets.get_preset("") == WeightPresets.BALANCED
