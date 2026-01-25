"""Main keyword research pipeline orchestrator."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.calculators.proxy_scores import InstagramProxyCalculator, TikTokProxyCalculator
from src.calculators.unified_score import UnifiedScoreCalculator, WeightPresets
from src.clients.apify import ApifyClient
from src.clients.base import batch_items
from src.clients.dataforseo import DataForSEOClient
from src.clients.google_trends import GoogleTrendsClient
from src.clients.junglescout import JungleScoutClient
from src.clients.pinterest import PinterestClient
from src.config import Settings, get_settings
from src.models.keyword import Platform, UnifiedKeywordData

logger = logging.getLogger(__name__)


class PipelineOptions(BaseModel):
    """Configuration options for pipeline execution."""

    # Platform selection
    platforms: list[Platform] = Field(
        default_factory=lambda: list(Platform),
        description="Platforms to fetch data from",
    )

    # Scoring options
    weight_preset: str = Field(
        default="balanced",
        description="Weight preset for unified score calculation",
    )
    custom_weights: dict[Platform, float] | None = Field(
        default=None,
        description="Custom platform weights (overrides preset)",
    )

    # Processing options
    batch_size: int = Field(default=50, description="Keywords per batch")
    parallel_platform_requests: bool = Field(
        default=True,
        description="Fetch from platforms in parallel",
    )
    save_checkpoints: bool = Field(
        default=True,
        description="Save progress checkpoints for resumability",
    )
    checkpoint_dir: Path = Field(
        default=Path("data/checkpoints"),
        description="Directory for checkpoint files",
    )

    # Social media scraping options - reduced for Vercel serverless speed
    tiktok_results_per_hashtag: int = Field(
        default=20,
        description="Number of TikTok videos to scrape per hashtag",
    )
    instagram_results_per_hashtag: int = Field(
        default=20,
        description="Number of Instagram posts to scrape per hashtag",
    )

    # Error handling
    continue_on_error: bool = Field(
        default=True,
        description="Continue processing on individual keyword errors",
    )
    max_errors: int = Field(
        default=10,
        description="Maximum errors before stopping pipeline",
    )


class KeywordPipeline:
    """
    Orchestrates the full data collection and processing pipeline.

    Flow:
    1. Load keywords from input source (CSV, database, API)
    2. Batch keywords for efficient API calls
    3. Fetch data from all platforms (parallel where possible)
    4. Calculate proxy scores for TikTok/Instagram
    5. Calculate unified scores
    6. Save to database
    7. Export for RAG system

    Features:
    - Resumable (tracks progress, can restart from failure point)
    - Rate limit aware (respects API limits)
    - Progress logging
    - Error handling with partial results saving
    """

    def __init__(
        self,
        settings: Settings | None = None,
        dataforseo_client: DataForSEOClient | None = None,
        apify_client: ApifyClient | None = None,
        junglescout_client: JungleScoutClient | None = None,
        google_trends_client: GoogleTrendsClient | None = None,
        pinterest_client: PinterestClient | None = None,
    ):
        self.settings = settings or get_settings()

        # Initialize clients (can be injected for testing)
        self.dataforseo = dataforseo_client or DataForSEOClient(settings=self.settings)
        self.apify = apify_client or ApifyClient(settings=self.settings)
        self.junglescout = junglescout_client or JungleScoutClient(settings=self.settings)
        self.google_trends = google_trends_client or GoogleTrendsClient(settings=self.settings)
        self.pinterest = pinterest_client or PinterestClient(settings=self.settings)

        # Initialize calculators
        self.tiktok_calculator = TikTokProxyCalculator()
        self.instagram_calculator = InstagramProxyCalculator()
        self.unified_calculator = UnifiedScoreCalculator()

        # Pipeline state
        self._processed_keywords: set[str] = set()
        self._error_count = 0
        self._checkpoint_file: Path | None = None

    async def run(
        self,
        keywords: list[str],
        options: PipelineOptions | None = None,
    ) -> list[UnifiedKeywordData]:
        """
        Run the full pipeline for a list of keywords.

        Args:
            keywords: Keywords to process
            options: Pipeline configuration options

        Returns:
            List of UnifiedKeywordData objects
        """
        options = options or PipelineOptions()
        self._error_count = 0

        # Filter out already processed keywords (for resumability)
        pending_keywords = [k for k in keywords if k not in self._processed_keywords]
        logger.info(f"Processing {len(pending_keywords)} keywords ({len(keywords) - len(pending_keywords)} already processed)")

        # Set up checkpoint file
        if options.save_checkpoints:
            options.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._checkpoint_file = options.checkpoint_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Configure unified calculator weights
        if options.custom_weights:
            self.unified_calculator = UnifiedScoreCalculator(weights=options.custom_weights)
        else:
            preset_weights = WeightPresets.get_preset(options.weight_preset)
            self.unified_calculator = UnifiedScoreCalculator(weights=preset_weights)

        results: list[UnifiedKeywordData] = []

        # Process in batches
        batches = batch_items(pending_keywords, options.batch_size)
        total_batches = len(batches)

        for i, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {i}/{total_batches} ({len(batch)} keywords)")

            try:
                batch_results = await self.process_batch(batch, options)
                results.extend(batch_results)

                # Mark keywords as processed
                self._processed_keywords.update(batch)

                # Save checkpoint
                if options.save_checkpoints:
                    self._save_checkpoint(results)

            except Exception as e:
                logger.error(f"Batch {i} failed: {e}")
                self._error_count += 1

                if not options.continue_on_error or self._error_count >= options.max_errors:
                    logger.error("Pipeline stopped due to errors")
                    break

        logger.info(f"Pipeline complete. Processed {len(results)} keywords with {self._error_count} errors")
        return results

    async def process_batch(
        self,
        keywords: list[str],
        options: PipelineOptions,
    ) -> list[UnifiedKeywordData]:
        """
        Process a batch of keywords.

        Args:
            keywords: Keywords to process
            options: Pipeline options

        Returns:
            List of UnifiedKeywordData objects
        """
        # Initialize result containers
        keyword_data: dict[str, UnifiedKeywordData] = {
            kw: UnifiedKeywordData(keyword=kw) for kw in keywords
        }

        # Fetch data from platforms
        if options.parallel_platform_requests:
            await self._fetch_all_platforms_parallel(keywords, keyword_data, options)
        else:
            await self._fetch_all_platforms_sequential(keywords, keyword_data, options)

        # Calculate unified scores
        results = []
        for kw, data in keyword_data.items():
            try:
                self.unified_calculator.calculate(data)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to calculate unified score for '{kw}': {e}")
                data.processing_errors = data.processing_errors or []
                data.processing_errors.append(f"Unified score calculation: {e}")
                results.append(data)

        return results

    async def _fetch_all_platforms_parallel(
        self,
        keywords: list[str],
        keyword_data: dict[str, UnifiedKeywordData],
        options: PipelineOptions,
    ) -> None:
        """Fetch data from all platforms in parallel."""
        tasks = []

        if Platform.GOOGLE in options.platforms:
            tasks.append(self._fetch_google_data(keywords, keyword_data))

        if Platform.YOUTUBE in options.platforms:
            tasks.append(self._fetch_youtube_data(keywords, keyword_data))

        if Platform.AMAZON in options.platforms:
            tasks.append(self._fetch_amazon_data(keywords, keyword_data))

        if Platform.TIKTOK in options.platforms:
            tasks.append(self._fetch_tiktok_data(keywords, keyword_data, options))

        if Platform.INSTAGRAM in options.platforms:
            tasks.append(self._fetch_instagram_data(keywords, keyword_data, options))

        if Platform.PINTEREST in options.platforms:
            tasks.append(self._fetch_pinterest_data(keywords, keyword_data))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_all_platforms_sequential(
        self,
        keywords: list[str],
        keyword_data: dict[str, UnifiedKeywordData],
        options: PipelineOptions,
    ) -> None:
        """Fetch data from all platforms sequentially."""
        if Platform.GOOGLE in options.platforms:
            await self._fetch_google_data(keywords, keyword_data)

        if Platform.YOUTUBE in options.platforms:
            await self._fetch_youtube_data(keywords, keyword_data)

        if Platform.AMAZON in options.platforms:
            await self._fetch_amazon_data(keywords, keyword_data)

        if Platform.TIKTOK in options.platforms:
            await self._fetch_tiktok_data(keywords, keyword_data, options)

        if Platform.INSTAGRAM in options.platforms:
            await self._fetch_instagram_data(keywords, keyword_data, options)

        if Platform.PINTEREST in options.platforms:
            await self._fetch_pinterest_data(keywords, keyword_data)

    async def _fetch_google_data(
        self,
        keywords: list[str],
        keyword_data: dict[str, UnifiedKeywordData],
    ) -> None:
        """Fetch Google search volume data."""
        try:
            logger.debug(f"Fetching Google data for {len(keywords)} keywords")
            metrics = await self.dataforseo.get_google_search_volume(keywords)

            for kw, metric in zip(keywords, metrics):
                keyword_data[kw].google = metric

        except Exception as e:
            logger.error(f"Failed to fetch Google data: {e}")
            for kw in keywords:
                keyword_data[kw].processing_errors = keyword_data[kw].processing_errors or []
                keyword_data[kw].processing_errors.append(f"Google: {e}")

    async def _fetch_youtube_data(
        self,
        keywords: list[str],
        keyword_data: dict[str, UnifiedKeywordData],
    ) -> None:
        """Fetch YouTube search volume data using Google Trends."""
        try:
            logger.debug(f"Fetching YouTube data via Google Trends for {len(keywords)} keywords")
            metrics = await self.google_trends.get_youtube_search_volume(keywords)

            for kw, metric in zip(keywords, metrics):
                keyword_data[kw].youtube = metric

        except Exception as e:
            logger.error(f"Failed to fetch YouTube data: {e}")
            for kw in keywords:
                keyword_data[kw].processing_errors = keyword_data[kw].processing_errors or []
                keyword_data[kw].processing_errors.append(f"YouTube: {e}")

    async def _fetch_amazon_data(
        self,
        keywords: list[str],
        keyword_data: dict[str, UnifiedKeywordData],
    ) -> None:
        """Fetch Amazon search volume data."""
        try:
            logger.debug(f"Fetching Amazon data for {len(keywords)} keywords")
            metrics = await self.junglescout.get_amazon_search_volume(keywords)

            for kw, metric in zip(keywords, metrics):
                keyword_data[kw].amazon = metric

        except Exception as e:
            logger.error(f"Failed to fetch Amazon data: {e}")
            for kw in keywords:
                keyword_data[kw].processing_errors = keyword_data[kw].processing_errors or []
                keyword_data[kw].processing_errors.append(f"Amazon: {e}")

    async def _fetch_tiktok_data(
        self,
        keywords: list[str],
        keyword_data: dict[str, UnifiedKeywordData],
        options: PipelineOptions,
    ) -> None:
        """Fetch and calculate TikTok proxy scores."""
        try:
            logger.debug(f"Fetching TikTok data for {len(keywords)} keywords")
            # Convert keywords to hashtags (remove spaces, etc.)
            hashtags = [kw.replace(" ", "").lower() for kw in keywords]

            raw_data = await self.apify.run_tiktok_hashtag_scraper(
                hashtags,
                results_per_hashtag=options.tiktok_results_per_hashtag,
            )

            for kw, hashtag in zip(keywords, hashtags):
                if data := raw_data.get(hashtag):
                    keyword_data[kw].tiktok = self.tiktok_calculator.calculate(data)

        except Exception as e:
            logger.error(f"Failed to fetch TikTok data: {e}")
            for kw in keywords:
                keyword_data[kw].processing_errors = keyword_data[kw].processing_errors or []
                keyword_data[kw].processing_errors.append(f"TikTok: {e}")

    async def _fetch_instagram_data(
        self,
        keywords: list[str],
        keyword_data: dict[str, UnifiedKeywordData],
        options: PipelineOptions,
    ) -> None:
        """Fetch and calculate Instagram proxy scores."""
        try:
            logger.debug(f"Fetching Instagram data for {len(keywords)} keywords")
            # Convert keywords to hashtags
            hashtags = [kw.replace(" ", "").lower() for kw in keywords]

            raw_data = await self.apify.run_instagram_hashtag_scraper(
                hashtags,
                results_per_hashtag=options.instagram_results_per_hashtag,
            )

            for kw, hashtag in zip(keywords, hashtags):
                if data := raw_data.get(hashtag):
                    keyword_data[kw].instagram = self.instagram_calculator.calculate(data)

        except Exception as e:
            logger.error(f"Failed to fetch Instagram data: {e}")
            for kw in keywords:
                keyword_data[kw].processing_errors = keyword_data[kw].processing_errors or []
                keyword_data[kw].processing_errors.append(f"Instagram: {e}")

    async def _fetch_pinterest_data(
        self,
        keywords: list[str],
        keyword_data: dict[str, UnifiedKeywordData],
    ) -> None:
        """Fetch Pinterest search/interest data."""
        try:
            logger.debug(f"Fetching Pinterest data for {len(keywords)} keywords")
            metrics = await self.pinterest.get_pinterest_search_volume(keywords)

            for kw, metric in zip(keywords, metrics):
                keyword_data[kw].pinterest = metric

        except Exception as e:
            logger.error(f"Failed to fetch Pinterest data: {e}")
            for kw in keywords:
                keyword_data[kw].processing_errors = keyword_data[kw].processing_errors or []
                keyword_data[kw].processing_errors.append(f"Pinterest: {e}")

    def _save_checkpoint(self, results: list[UnifiedKeywordData]) -> None:
        """Save current progress to checkpoint file."""
        if not self._checkpoint_file:
            return

        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "processed_count": len(results),
            "processed_keywords": list(self._processed_keywords),
            "results": [r.model_dump(mode="json") for r in results],
        }

        with open(self._checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        logger.debug(f"Checkpoint saved: {self._checkpoint_file}")

    def load_checkpoint(self, checkpoint_file: Path) -> list[UnifiedKeywordData]:
        """
        Load progress from a checkpoint file.

        Args:
            checkpoint_file: Path to checkpoint file

        Returns:
            List of previously processed UnifiedKeywordData
        """
        with open(checkpoint_file) as f:
            data = json.load(f)

        self._processed_keywords = set(data.get("processed_keywords", []))
        results = [UnifiedKeywordData.model_validate(r) for r in data.get("results", [])]

        logger.info(f"Loaded checkpoint with {len(results)} processed keywords")
        return results

    async def close(self) -> None:
        """Close all client connections."""
        await asyncio.gather(
            self.dataforseo.close(),
            self.apify.close(),
            self.junglescout.close(),
            self.google_trends.close(),
            self.pinterest.close(),
            return_exceptions=True,
        )

    async def __aenter__(self) -> "KeywordPipeline":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
