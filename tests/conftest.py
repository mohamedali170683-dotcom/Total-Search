"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.config import Settings
from src.models.keyword import (
    AmazonMetrics,
    Competition,
    Confidence,
    GoogleMetrics,
    InstagramMetrics,
    MonthlySearchData,
    Platform,
    TikTokMetrics,
    TrendDirection,
    UnifiedKeywordData,
    YouTubeMetrics,
)


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        dataforseo_login="test_login",
        dataforseo_password="test_password",
        apify_api_token="test_token",
        junglescout_api_key="test_key",
        junglescout_api_key_name="test_key_name",
        database_url="sqlite:///:memory:",
        redis_enabled=False,
    )


@pytest.fixture
def sample_google_metrics() -> GoogleMetrics:
    """Create sample Google metrics."""
    return GoogleMetrics(
        search_volume=10000,
        trend=TrendDirection.GROWING,
        trend_velocity=1.25,
        competition=Competition.MEDIUM,
        cpc=1.50,
        confidence=Confidence.HIGH,
        source="dataforseo_google",
        monthly_searches=[
            MonthlySearchData(year=2026, month=i, search_volume=9000 + i * 100)
            for i in range(1, 13)
        ],
        keyword_difficulty=45.0,
    )


@pytest.fixture
def sample_youtube_metrics() -> YouTubeMetrics:
    """Create sample YouTube metrics."""
    return YouTubeMetrics(
        search_volume=5000,
        trend=TrendDirection.STABLE,
        trend_velocity=1.02,
        confidence=Confidence.HIGH,
        source="dataforseo_youtube",
    )


@pytest.fixture
def sample_amazon_metrics() -> AmazonMetrics:
    """Create sample Amazon metrics."""
    return AmazonMetrics(
        search_volume=8000,
        exact_search_volume=8000,
        broad_search_volume=15000,
        trend=TrendDirection.GROWING,
        trend_velocity=1.15,
        competition=Competition.HIGH,
        confidence=Confidence.HIGH,
        source="junglescout",
        organic_product_count=500,
        sponsored_product_count=25,
    )


@pytest.fixture
def sample_tiktok_metrics() -> TikTokMetrics:
    """Create sample TikTok metrics."""
    return TikTokMetrics(
        proxy_score=25000,
        trend=TrendDirection.GROWING,
        trend_velocity=1.45,
        confidence=Confidence.PROXY,
        source="apify_tiktok",
        hashtag_views=50000000,
        video_count=10000,
        avg_likes=5000.0,
        avg_comments=200.0,
        avg_shares=150.0,
    )


@pytest.fixture
def sample_instagram_metrics() -> InstagramMetrics:
    """Create sample Instagram metrics."""
    return InstagramMetrics(
        proxy_score=15000,
        trend=TrendDirection.STABLE,
        trend_velocity=1.05,
        confidence=Confidence.PROXY,
        source="apify_instagram",
        post_count=5000000,
        daily_posts=500,
        avg_engagement=300.0,
        avg_likes=250.0,
        avg_comments=50.0,
        related_hashtags=["related1", "related2"],
    )


@pytest.fixture
def sample_unified_keyword_data(
    sample_google_metrics,
    sample_youtube_metrics,
    sample_amazon_metrics,
    sample_tiktok_metrics,
    sample_instagram_metrics,
) -> UnifiedKeywordData:
    """Create sample unified keyword data."""
    return UnifiedKeywordData(
        keyword="test keyword",
        google=sample_google_metrics,
        youtube=sample_youtube_metrics,
        amazon=sample_amazon_metrics,
        tiktok=sample_tiktok_metrics,
        instagram=sample_instagram_metrics,
        tags=["test", "sample"],
    )


@pytest.fixture
def mock_dataforseo_response() -> dict:
    """Create mock DataForSEO API response."""
    return {
        "tasks": [
            {
                "status_code": 20000,
                "status_message": "Ok.",
                "result": [
                    {
                        "keyword": "test keyword",
                        "search_volume": 10000,
                        "competition": "MEDIUM",
                        "cpc": 1.50,
                        "keyword_difficulty": 45,
                        "monthly_searches": [
                            {"year": 2026, "month": i, "search_volume": 9000 + i * 100}
                            for i in range(1, 13)
                        ],
                    }
                ],
            }
        ]
    }


@pytest.fixture
def mock_tiktok_raw_data() -> dict:
    """Create mock TikTok scraper raw data."""
    return {
        "hashtag": "testkeyword",
        "stats": {
            "video_count": 100,
            "total_views": 5000000,
            "total_likes": 500000,
            "total_comments": 20000,
            "total_shares": 15000,
            "avg_likes": 5000,
            "avg_comments": 200,
            "avg_shares": 150,
        },
        "videos": [
            {
                "id": f"video_{i}",
                "createTime": 1700000000 + i * 86400,
                "diggCount": 5000 + (i * 100 if i < 50 else i * 50),
                "commentCount": 200 + i * 2,
                "shareCount": 150 + i,
                "playCount": 50000 + i * 1000,
            }
            for i in range(100)
        ],
    }


@pytest.fixture
def mock_instagram_raw_data() -> dict:
    """Create mock Instagram scraper raw data."""
    return {
        "hashtag": "testkeyword",
        "stats": {
            "post_count": 50000,
            "total_likes": 2500000,
            "total_comments": 250000,
            "avg_likes": 500,
            "avg_comments": 50,
            "daily_posts": 100,
            "related_hashtags": ["beauty", "skincare"],
        },
        "posts": [
            {
                "id": f"post_{i}",
                "timestamp": 1700000000 + i * 3600,
                "likesCount": 500 + i * 5,
                "commentsCount": 50 + i,
                "hashtags": ["testkeyword", "beauty"],
            }
            for i in range(100)
        ],
    }
