"""Tests for the keyword pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.pipeline.keyword_pipeline import KeywordPipeline, PipelineOptions
from src.models.keyword import (
    Platform,
    GoogleMetrics,
    YouTubeMetrics,
    TikTokMetrics,
    InstagramMetrics,
    TrendDirection,
    Competition,
    Confidence,
)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.dataforseo_login = MagicMock()
    settings.dataforseo_login.get_secret_value.return_value = "test_login"
    settings.dataforseo_password = MagicMock()
    settings.dataforseo_password.get_secret_value.return_value = "test_password"
    settings.apify_token = MagicMock()
    settings.apify_token.get_secret_value.return_value = "test_token"
    settings.junglescout_api_key = None
    settings.junglescout_api_key_name = None
    settings.default_location_code = 2840
    settings.batch_size = 100
    return settings


@pytest.fixture
def sample_google_metrics():
    """Sample Google metrics."""
    return GoogleMetrics(
        search_volume=10000,
        trend=TrendDirection.GROWING,
        competition=Competition.MEDIUM,
        cpc=1.50,
        confidence=Confidence.HIGH,
        source="dataforseo",
    )


@pytest.fixture
def sample_youtube_metrics():
    """Sample YouTube metrics."""
    return YouTubeMetrics(
        search_volume=5000,
        trend=TrendDirection.STABLE,
        confidence=Confidence.HIGH,
        source="dataforseo",
    )


@pytest.fixture
def sample_tiktok_metrics():
    """Sample TikTok metrics."""
    return TikTokMetrics(
        proxy_score=8000,
        hashtag_views=50000000,
        video_count=100000,
        avg_likes=5000,
        avg_comments=200,
        avg_shares=100,
        trend=TrendDirection.GROWING,
        confidence=Confidence.PROXY,
        source="apify",
    )


@pytest.fixture
def sample_instagram_metrics():
    """Sample Instagram metrics."""
    return InstagramMetrics(
        proxy_score=6000,
        post_count=500000,
        daily_posts=500,
        avg_engagement=1000,
        related_hashtags=["beauty", "skincare"],
        trend=TrendDirection.STABLE,
        confidence=Confidence.PROXY,
        source="apify",
    )


class TestPipelineOptions:
    """Tests for PipelineOptions."""

    def test_default_options(self):
        """Test default pipeline options."""
        options = PipelineOptions()

        assert options.platforms == [Platform.GOOGLE, Platform.YOUTUBE]
        assert options.weight_preset == "balanced"
        assert options.batch_size == 50
        assert options.parallel_fetch is True
        assert options.save_checkpoints is True

    def test_custom_platforms(self):
        """Test custom platform selection."""
        options = PipelineOptions(
            platforms=[Platform.GOOGLE, Platform.AMAZON, Platform.TIKTOK]
        )

        assert len(options.platforms) == 3
        assert Platform.AMAZON in options.platforms

    def test_custom_weights(self):
        """Test custom weight configuration."""
        custom_weights = {
            "google": 0.5,
            "youtube": 0.3,
            "amazon": 0.2,
        }
        options = PipelineOptions(custom_weights=custom_weights)

        assert options.custom_weights == custom_weights


class TestKeywordPipeline:
    """Tests for KeywordPipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, mock_settings):
        """Test pipeline initializes correctly."""
        with patch("src.pipeline.keyword_pipeline.DataForSEOClient") as mock_dataforseo, \
             patch("src.pipeline.keyword_pipeline.ApifyClient") as mock_apify:

            async with KeywordPipeline(settings=mock_settings) as pipeline:
                assert pipeline is not None
                assert pipeline.dataforseo_client is not None
                assert pipeline.apify_client is not None

    @pytest.mark.asyncio
    async def test_fetch_google_data(
        self,
        mock_settings,
        sample_google_metrics,
    ):
        """Test fetching Google data."""
        with patch("src.pipeline.keyword_pipeline.DataForSEOClient") as MockDataForSEO, \
             patch("src.pipeline.keyword_pipeline.ApifyClient"):

            mock_client = AsyncMock()
            mock_client.get_google_search_volume.return_value = {
                "skincare": sample_google_metrics
            }
            MockDataForSEO.return_value = mock_client

            async with KeywordPipeline(settings=mock_settings) as pipeline:
                result = await pipeline._fetch_google_data(["skincare"])

                assert "skincare" in result
                assert result["skincare"].search_volume == 10000

    @pytest.mark.asyncio
    async def test_process_batch_google_only(
        self,
        mock_settings,
        sample_google_metrics,
    ):
        """Test processing a batch with Google only."""
        with patch("src.pipeline.keyword_pipeline.DataForSEOClient") as MockDataForSEO, \
             patch("src.pipeline.keyword_pipeline.ApifyClient"):

            mock_client = AsyncMock()
            mock_client.get_google_search_volume.return_value = {
                "skincare": sample_google_metrics
            }
            MockDataForSEO.return_value = mock_client

            options = PipelineOptions(platforms=[Platform.GOOGLE])

            async with KeywordPipeline(settings=mock_settings) as pipeline:
                results = await pipeline.process_batch(["skincare"], options)

                assert len(results) == 1
                assert results[0].keyword == "skincare"
                assert results[0].unified_demand_score > 0

    @pytest.mark.asyncio
    async def test_process_batch_handles_missing_data(
        self,
        mock_settings,
    ):
        """Test processing handles missing platform data gracefully."""
        with patch("src.pipeline.keyword_pipeline.DataForSEOClient") as MockDataForSEO, \
             patch("src.pipeline.keyword_pipeline.ApifyClient"):

            mock_client = AsyncMock()
            mock_client.get_google_search_volume.return_value = {}  # No data
            MockDataForSEO.return_value = mock_client

            options = PipelineOptions(platforms=[Platform.GOOGLE])

            async with KeywordPipeline(settings=mock_settings) as pipeline:
                results = await pipeline.process_batch(["unknown_keyword"], options)

                # Should still return result, just with zero score
                assert len(results) == 1
                assert results[0].keyword == "unknown_keyword"

    @pytest.mark.asyncio
    async def test_checkpoint_save_load(self, mock_settings, tmp_path):
        """Test checkpoint saving and loading."""
        with patch("src.pipeline.keyword_pipeline.DataForSEOClient"), \
             patch("src.pipeline.keyword_pipeline.ApifyClient"):

            async with KeywordPipeline(settings=mock_settings) as pipeline:
                pipeline.checkpoint_dir = tmp_path

                # Save checkpoint
                processed = ["keyword1", "keyword2", "keyword3"]
                pipeline._save_checkpoint(processed, "test_run")

                # Load checkpoint
                loaded = pipeline._load_checkpoint("test_run")

                assert loaded == set(processed)


class TestPipelineIntegration:
    """Integration-style tests for the pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_run(
        self,
        mock_settings,
        sample_google_metrics,
        sample_youtube_metrics,
    ):
        """Test a full pipeline run with multiple platforms."""
        with patch("src.pipeline.keyword_pipeline.DataForSEOClient") as MockDataForSEO, \
             patch("src.pipeline.keyword_pipeline.ApifyClient"):

            mock_client = AsyncMock()
            mock_client.get_google_search_volume.return_value = {
                "skincare": sample_google_metrics,
                "beauty": sample_google_metrics,
            }
            mock_client.get_youtube_search_volume.return_value = {
                "skincare": sample_youtube_metrics,
                "beauty": sample_youtube_metrics,
            }
            MockDataForSEO.return_value = mock_client

            options = PipelineOptions(
                platforms=[Platform.GOOGLE, Platform.YOUTUBE],
                batch_size=10,
                save_checkpoints=False,
            )

            async with KeywordPipeline(settings=mock_settings) as pipeline:
                results = await pipeline.run(["skincare", "beauty"], options)

                assert len(results) == 2
                for result in results:
                    assert result.unified_demand_score > 0
                    assert Platform.GOOGLE in result.platforms
                    assert Platform.YOUTUBE in result.platforms

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, mock_settings):
        """Test pipeline handles errors gracefully."""
        with patch("src.pipeline.keyword_pipeline.DataForSEOClient") as MockDataForSEO, \
             patch("src.pipeline.keyword_pipeline.ApifyClient"):

            mock_client = AsyncMock()
            mock_client.get_google_search_volume.side_effect = Exception("API Error")
            MockDataForSEO.return_value = mock_client

            options = PipelineOptions(
                platforms=[Platform.GOOGLE],
                continue_on_error=True,
            )

            async with KeywordPipeline(settings=mock_settings) as pipeline:
                # Should not raise, just return empty/partial results
                results = await pipeline.run(["skincare"], options)
                assert isinstance(results, list)
