"""Unit tests for data models."""

import pytest
from datetime import datetime

from src.models.keyword import (
    AmazonMetrics,
    Competition,
    Confidence,
    GoogleMetrics,
    InstagramMetrics,
    MonthlySearchData,
    Platform,
    PlatformMetrics,
    PlatformScore,
    TikTokMetrics,
    TrendDirection,
    UnifiedKeywordData,
    YouTubeMetrics,
)


class TestPlatformMetrics:
    """Tests for PlatformMetrics base class."""

    def test_effective_volume_with_search_volume(self):
        """Test effective volume returns search_volume when present."""
        metrics = PlatformMetrics(
            search_volume=1000,
            proxy_score=500,
            source="test",
        )
        assert metrics.effective_volume == 1000

    def test_effective_volume_with_proxy_score(self):
        """Test effective volume returns proxy_score when no search_volume."""
        metrics = PlatformMetrics(
            proxy_score=500,
            source="test",
        )
        assert metrics.effective_volume == 500

    def test_effective_volume_with_no_data(self):
        """Test effective volume returns 0 when no data."""
        metrics = PlatformMetrics(source="test")
        assert metrics.effective_volume == 0


class TestGoogleMetrics:
    """Tests for GoogleMetrics."""

    def test_google_metrics_creation(self, sample_google_metrics):
        """Test GoogleMetrics creation with all fields."""
        assert sample_google_metrics.search_volume == 10000
        assert sample_google_metrics.trend == TrendDirection.GROWING
        assert sample_google_metrics.competition == Competition.MEDIUM
        assert sample_google_metrics.cpc == 1.50
        assert sample_google_metrics.source == "dataforseo_google"
        assert len(sample_google_metrics.monthly_searches) == 12

    def test_google_metrics_default_confidence(self):
        """Test GoogleMetrics default confidence is HIGH."""
        metrics = GoogleMetrics(source="dataforseo_google")
        assert metrics.confidence == Confidence.HIGH


class TestAmazonMetrics:
    """Tests for AmazonMetrics."""

    def test_effective_volume_prefers_exact(self, sample_amazon_metrics):
        """Test that effective_volume prefers exact_search_volume."""
        assert sample_amazon_metrics.effective_volume == 8000  # exact, not broad

    def test_effective_volume_falls_back_to_broad(self):
        """Test that effective_volume falls back to broad_search_volume."""
        metrics = AmazonMetrics(
            broad_search_volume=15000,
            source="junglescout",
        )
        assert metrics.effective_volume == 15000


class TestTikTokMetrics:
    """Tests for TikTokMetrics."""

    def test_avg_engagement_calculation(self, sample_tiktok_metrics):
        """Test average engagement calculation."""
        expected = 5000.0 + 200.0 + 150.0  # likes + comments + shares
        assert sample_tiktok_metrics.avg_engagement == expected

    def test_default_confidence_is_proxy(self):
        """Test that TikTok metrics default to PROXY confidence."""
        metrics = TikTokMetrics(source="apify_tiktok")
        assert metrics.confidence == Confidence.PROXY


class TestInstagramMetrics:
    """Tests for InstagramMetrics."""

    def test_default_confidence_is_proxy(self):
        """Test that Instagram metrics default to PROXY confidence."""
        metrics = InstagramMetrics(source="apify_instagram")
        assert metrics.confidence == Confidence.PROXY


class TestUnifiedKeywordData:
    """Tests for UnifiedKeywordData."""

    def test_platforms_dict(self, sample_unified_keyword_data):
        """Test platforms_dict property."""
        platforms = sample_unified_keyword_data.platforms_dict

        assert Platform.GOOGLE in platforms
        assert Platform.YOUTUBE in platforms
        assert Platform.AMAZON in platforms
        assert Platform.TIKTOK in platforms
        assert Platform.INSTAGRAM in platforms
        assert platforms[Platform.GOOGLE] is not None

    def test_available_platforms(self, sample_unified_keyword_data):
        """Test available_platforms property."""
        available = sample_unified_keyword_data.available_platforms

        assert len(available) == 5
        assert Platform.GOOGLE in available
        assert Platform.TIKTOK in available

    def test_available_platforms_partial(self, sample_google_metrics):
        """Test available_platforms with partial data."""
        data = UnifiedKeywordData(
            keyword="test",
            google=sample_google_metrics,
        )

        available = data.available_platforms
        assert len(available) == 1
        assert Platform.GOOGLE in available

    def test_get_platform_metrics(self, sample_unified_keyword_data, sample_google_metrics):
        """Test get_platform_metrics method."""
        google = sample_unified_keyword_data.get_platform_metrics(Platform.GOOGLE)
        assert google is not None
        assert google.search_volume == sample_google_metrics.search_volume

    def test_to_rag_document(self, sample_unified_keyword_data):
        """Test RAG document export."""
        sample_unified_keyword_data.unified_demand_score = 65
        sample_unified_keyword_data.cross_platform_trend = TrendDirection.GROWING
        sample_unified_keyword_data.best_platform = Platform.GOOGLE

        doc = sample_unified_keyword_data.to_rag_document()

        assert doc["keyword"] == "test keyword"
        assert doc["unified_demand_score"] == 65
        assert doc["cross_platform_trend"] == "growing"
        assert doc["best_platform"] == "google"
        assert "google" in doc["platforms"]
        assert "collected_at" in doc

    def test_to_rag_document_partial_data(self, sample_google_metrics):
        """Test RAG document export with partial data."""
        data = UnifiedKeywordData(
            keyword="partial test",
            google=sample_google_metrics,
            unified_demand_score=40,
        )

        doc = data.to_rag_document()

        assert doc["keyword"] == "partial test"
        assert "google" in doc["platforms"]
        assert "youtube" not in doc["platforms"]


class TestPlatformScore:
    """Tests for PlatformScore."""

    def test_platform_score_creation(self):
        """Test PlatformScore creation."""
        score = PlatformScore(
            platform=Platform.GOOGLE,
            raw_volume=10000,
            normalized_score=57.5,
            weight=0.3,
            weighted_score=17.25,
        )

        assert score.platform == Platform.GOOGLE
        assert score.raw_volume == 10000
        assert score.normalized_score == 57.5
        assert score.weight == 0.3
        assert score.weighted_score == 17.25

    def test_normalized_score_bounds(self):
        """Test that normalized_score is bounded 0-100."""
        # Should fail validation for out-of-bounds
        with pytest.raises(ValueError):
            PlatformScore(
                platform=Platform.GOOGLE,
                raw_volume=10000,
                normalized_score=150,  # Invalid
                weight=0.3,
                weighted_score=45,
            )

    def test_weight_bounds(self):
        """Test that weight is bounded 0-1."""
        with pytest.raises(ValueError):
            PlatformScore(
                platform=Platform.GOOGLE,
                raw_volume=10000,
                normalized_score=50,
                weight=1.5,  # Invalid
                weighted_score=75,
            )


class TestEnums:
    """Tests for enum values."""

    def test_trend_direction_values(self):
        """Test TrendDirection enum values."""
        assert TrendDirection.GROWING.value == "growing"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.DECLINING.value == "declining"

    def test_competition_values(self):
        """Test Competition enum values."""
        assert Competition.LOW.value == "low"
        assert Competition.MEDIUM.value == "medium"
        assert Competition.HIGH.value == "high"

    def test_confidence_values(self):
        """Test Confidence enum values."""
        assert Confidence.HIGH.value == "high"
        assert Confidence.MEDIUM.value == "medium"
        assert Confidence.PROXY.value == "proxy"

    def test_platform_values(self):
        """Test Platform enum values."""
        assert Platform.GOOGLE.value == "google"
        assert Platform.YOUTUBE.value == "youtube"
        assert Platform.AMAZON.value == "amazon"
        assert Platform.TIKTOK.value == "tiktok"
        assert Platform.INSTAGRAM.value == "instagram"
