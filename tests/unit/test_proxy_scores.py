"""Unit tests for proxy score calculators."""

import pytest

from src.calculators.proxy_scores import InstagramProxyCalculator, TikTokProxyCalculator
from src.models.keyword import Confidence, TrendDirection


class TestTikTokProxyCalculator:
    """Tests for TikTokProxyCalculator."""

    def test_calculate_basic(self, mock_tiktok_raw_data):
        """Test basic proxy score calculation."""
        calculator = TikTokProxyCalculator()
        metrics = calculator.calculate(mock_tiktok_raw_data)

        assert metrics.proxy_score is not None
        assert metrics.proxy_score > 0
        assert metrics.confidence == Confidence.PROXY
        assert metrics.source == "apify_tiktok"
        assert metrics.hashtag_views == 5000000
        assert metrics.video_count == 100

    def test_calculate_empty_data(self):
        """Test calculation with empty data."""
        calculator = TikTokProxyCalculator()
        metrics = calculator.calculate({"stats": {}, "videos": []})

        assert metrics.proxy_score == 0
        assert metrics.confidence == Confidence.PROXY

    def test_trend_velocity_growing(self):
        """Test trend velocity detection for growing content."""
        calculator = TikTokProxyCalculator()

        # Create videos with increasing engagement over time
        videos = [
            {
                "createTime": 1700000000 + i * 86400,
                "diggCount": 1000 if i < 50 else 3000,  # Recent videos have 3x engagement
                "commentCount": 100,
                "shareCount": 50,
            }
            for i in range(100)
        ]

        velocity = calculator.calculate_trend_velocity(videos)
        assert velocity > 1.0  # Growing

    def test_trend_velocity_declining(self):
        """Test trend velocity detection for declining content."""
        calculator = TikTokProxyCalculator()

        # Create videos with decreasing engagement over time
        videos = [
            {
                "createTime": 1700000000 + i * 86400,
                "diggCount": 3000 if i < 50 else 1000,  # Recent videos have less engagement
                "commentCount": 100,
                "shareCount": 50,
            }
            for i in range(100)
        ]

        velocity = calculator.calculate_trend_velocity(videos)
        assert velocity < 1.0  # Declining

    def test_trend_velocity_insufficient_data(self):
        """Test trend velocity with insufficient data."""
        calculator = TikTokProxyCalculator()

        videos = [{"createTime": 1700000000, "diggCount": 1000}]
        velocity = calculator.calculate_trend_velocity(videos)

        assert velocity == 1.0  # Default to stable

    def test_determine_trend_growing(self):
        """Test trend determination for growing velocity."""
        calculator = TikTokProxyCalculator()

        assert calculator._determine_trend(1.3) == TrendDirection.GROWING
        assert calculator._determine_trend(1.2) == TrendDirection.GROWING

    def test_determine_trend_declining(self):
        """Test trend determination for declining velocity."""
        calculator = TikTokProxyCalculator()

        assert calculator._determine_trend(0.7) == TrendDirection.DECLINING
        assert calculator._determine_trend(0.8) == TrendDirection.DECLINING

    def test_determine_trend_stable(self):
        """Test trend determination for stable velocity."""
        calculator = TikTokProxyCalculator()

        assert calculator._determine_trend(1.0) == TrendDirection.STABLE
        assert calculator._determine_trend(1.1) == TrendDirection.STABLE
        assert calculator._determine_trend(0.9) == TrendDirection.STABLE


class TestInstagramProxyCalculator:
    """Tests for InstagramProxyCalculator."""

    def test_calculate_basic(self, mock_instagram_raw_data):
        """Test basic proxy score calculation."""
        calculator = InstagramProxyCalculator()
        metrics = calculator.calculate(mock_instagram_raw_data)

        assert metrics.proxy_score is not None
        assert metrics.proxy_score > 0
        assert metrics.confidence == Confidence.PROXY
        assert metrics.source == "apify_instagram"
        assert metrics.post_count == 50000
        assert metrics.daily_posts == 100

    def test_calculate_empty_data(self):
        """Test calculation with empty data."""
        calculator = InstagramProxyCalculator()
        metrics = calculator.calculate({"stats": {}, "posts": []})

        assert metrics.proxy_score == 0
        assert metrics.confidence == Confidence.PROXY

    def test_daily_posts_calculation(self):
        """Test daily posts calculation from timestamps."""
        calculator = InstagramProxyCalculator()

        # 100 posts over 10 days = 10 posts/day
        posts = [
            {"timestamp": 1700000000 + i * 8640}  # ~10 posts per day
            for i in range(100)
        ]

        daily_posts = calculator.calculate_daily_posts(posts)
        assert 9 <= daily_posts <= 11  # Allow some variance

    def test_daily_posts_insufficient_data(self):
        """Test daily posts with insufficient data."""
        calculator = InstagramProxyCalculator()

        posts = [{"timestamp": 1700000000}]
        daily_posts = calculator.calculate_daily_posts(posts)

        assert daily_posts == 0

    def test_saturation_penalty(self):
        """Test that oversaturated hashtags get penalized."""
        calculator = InstagramProxyCalculator()

        # Normal hashtag
        normal_data = {
            "stats": {"post_count": 1000000, "daily_posts": 100, "avg_likes": 500, "avg_comments": 50},
            "posts": [],
        }
        normal_metrics = calculator.calculate(normal_data)

        # Oversaturated hashtag (same daily activity but huge total count)
        saturated_data = {
            "stats": {"post_count": 50000000, "daily_posts": 100, "avg_likes": 500, "avg_comments": 50},
            "posts": [],
        }
        saturated_metrics = calculator.calculate(saturated_data)

        # Saturated should have lower score due to penalty
        assert saturated_metrics.proxy_score < normal_metrics.proxy_score

    def test_related_hashtags_captured(self, mock_instagram_raw_data):
        """Test that related hashtags are captured."""
        calculator = InstagramProxyCalculator()
        metrics = calculator.calculate(mock_instagram_raw_data)

        assert metrics.related_hashtags is not None
        assert "beauty" in metrics.related_hashtags
        assert "skincare" in metrics.related_hashtags
