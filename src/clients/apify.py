"""Apify client for TikTok and Instagram hashtag scraping."""

import asyncio
import logging
from typing import Any

from apify_client import ApifyClientAsync

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ApifyError(Exception):
    """Apify-specific error."""

    pass


class ApifyClient:
    """
    Client for Apify actor runs for social media scraping.

    Actors used:
    - TikTok: "clockworks/tiktok-scraper" for hashtag data
    - Instagram: "apify/instagram-hashtag-scraper" for hashtag stats

    Features:
    - Async actor execution
    - Result polling with timeout
    - Dataset retrieval
    - Error handling for failed runs
    """

    # Actor IDs
    TIKTOK_ACTOR = "clockworks/tiktok-scraper"
    INSTAGRAM_ACTOR = "apify/instagram-hashtag-scraper"

    # Timeouts - reduced for Vercel serverless (60s max)
    DEFAULT_TIMEOUT_SECS = 45  # 45 seconds to stay within Vercel limits
    POLL_INTERVAL_SECS = 3

    def __init__(
        self,
        api_token: str | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.api_token = api_token or self.settings.apify_api_token.get_secret_value()
        self._client: ApifyClientAsync | None = None

        if not self.api_token:
            logger.warning("Apify API token not configured")

    @property
    def client(self) -> ApifyClientAsync:
        """Get or create the Apify client."""
        if self._client is None:
            self._client = ApifyClientAsync(token=self.api_token)
        return self._client

    async def run_tiktok_hashtag_scraper(
        self,
        hashtags: list[str],
        results_per_hashtag: int = 20,  # Reduced for faster results
        timeout_secs: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Run TikTok hashtag scraper for multiple hashtags in parallel.

        Args:
            hashtags: List of hashtags to scrape (without #)
            results_per_hashtag: Number of videos to fetch per hashtag
            timeout_secs: Timeout for the actor run

        Returns:
            Dictionary mapping hashtag to raw scraped data
        """
        timeout_secs = timeout_secs or self.DEFAULT_TIMEOUT_SECS
        results: dict[str, dict[str, Any]] = {}

        # Run all hashtags in parallel for speed
        async def fetch_hashtag(hashtag: str) -> tuple[str, dict]:
            try:
                data = await self._run_single_tiktok_hashtag(
                    hashtag, results_per_hashtag, timeout_secs
                )
                return hashtag, data
            except Exception as e:
                logger.error(f"Failed to scrape TikTok hashtag '{hashtag}': {e}")
                return hashtag, {"error": str(e), "videos": []}

        tasks = [fetch_hashtag(h) for h in hashtags]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, tuple):
                hashtag, data = item
                results[hashtag] = data
            # Exceptions are handled inside fetch_hashtag

        return results

    async def _run_single_tiktok_hashtag(
        self,
        hashtag: str,
        results_limit: int,
        timeout_secs: int,
    ) -> dict[str, Any]:
        """Run TikTok scraper for a single hashtag."""
        # Clean hashtag (remove # if present)
        clean_hashtag = hashtag.lstrip("#")

        actor_input = {
            "hashtags": [clean_hashtag],
            "resultsPerPage": results_limit,
            "shouldDownloadVideos": False,
            "shouldDownloadCovers": False,
            "shouldDownloadSubtitles": False,
            "shouldDownloadSlideshowImages": False,
        }

        logger.info(f"Starting TikTok scraper for #{clean_hashtag}")

        try:
            run = await self.client.actor(self.TIKTOK_ACTOR).call(
                run_input=actor_input,
                timeout_secs=timeout_secs,
            )

            if run.get("status") != "SUCCEEDED":
                raise ApifyError(f"Actor run failed with status: {run.get('status')}")

            # Get dataset items
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                return {"hashtag": clean_hashtag, "videos": [], "stats": {}}

            items = await self._get_dataset_items(dataset_id)

            # Extract hashtag stats from videos
            stats = self._extract_tiktok_hashtag_stats(items, clean_hashtag)

            return {
                "hashtag": clean_hashtag,
                "videos": items,
                "stats": stats,
            }

        except asyncio.TimeoutError:
            raise ApifyError(f"TikTok scraper timed out for #{clean_hashtag}")

    def _extract_tiktok_hashtag_stats(
        self,
        videos: list[dict],
        hashtag: str,
    ) -> dict[str, Any]:
        """Extract aggregated stats from TikTok videos."""
        if not videos:
            return {}

        total_views = 0
        total_likes = 0
        total_comments = 0
        total_shares = 0

        for video in videos:
            total_views += video.get("playCount", 0) or video.get("videoMeta", {}).get(
                "playCount", 0
            )
            total_likes += video.get("diggCount", 0) or video.get("videoMeta", {}).get(
                "diggCount", 0
            )
            total_comments += video.get("commentCount", 0) or video.get("videoMeta", {}).get(
                "commentCount", 0
            )
            total_shares += video.get("shareCount", 0) or video.get("videoMeta", {}).get(
                "shareCount", 0
            )

        video_count = len(videos)

        return {
            "hashtag": hashtag,
            "video_count": video_count,
            "total_views": total_views,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "total_shares": total_shares,
            "avg_likes": total_likes / video_count if video_count else 0,
            "avg_comments": total_comments / video_count if video_count else 0,
            "avg_shares": total_shares / video_count if video_count else 0,
            "avg_views": total_views / video_count if video_count else 0,
        }

    async def run_instagram_hashtag_scraper(
        self,
        hashtags: list[str],
        results_per_hashtag: int = 20,  # Reduced for faster results
        timeout_secs: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Run Instagram hashtag scraper for multiple hashtags in parallel.

        Args:
            hashtags: List of hashtags to scrape (without #)
            results_per_hashtag: Number of posts to fetch per hashtag
            timeout_secs: Timeout for the actor run

        Returns:
            Dictionary mapping hashtag to raw scraped data
        """
        timeout_secs = timeout_secs or self.DEFAULT_TIMEOUT_SECS
        results: dict[str, dict[str, Any]] = {}

        # Run all hashtags in parallel for speed
        async def fetch_hashtag(hashtag: str) -> tuple[str, dict]:
            try:
                data = await self._run_single_instagram_hashtag(
                    hashtag, results_per_hashtag, timeout_secs
                )
                return hashtag, data
            except Exception as e:
                logger.error(f"Failed to scrape Instagram hashtag '{hashtag}': {e}")
                return hashtag, {"error": str(e), "posts": []}

        tasks = [fetch_hashtag(h) for h in hashtags]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, tuple):
                hashtag, data = item
                results[hashtag] = data
            # Exceptions are handled inside fetch_hashtag

        return results

    async def _run_single_instagram_hashtag(
        self,
        hashtag: str,
        results_limit: int,
        timeout_secs: int,
    ) -> dict[str, Any]:
        """Run Instagram scraper for a single hashtag."""
        clean_hashtag = hashtag.lstrip("#")

        actor_input = {
            "hashtags": [clean_hashtag],
            "resultsLimit": results_limit,
            "addParentData": True,
        }

        logger.info(f"Starting Instagram scraper for #{clean_hashtag}")

        try:
            run = await self.client.actor(self.INSTAGRAM_ACTOR).call(
                run_input=actor_input,
                timeout_secs=timeout_secs,
            )

            if run.get("status") != "SUCCEEDED":
                raise ApifyError(f"Actor run failed with status: {run.get('status')}")

            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                return {"hashtag": clean_hashtag, "posts": [], "stats": {}}

            items = await self._get_dataset_items(dataset_id)
            stats = self._extract_instagram_hashtag_stats(items, clean_hashtag)

            return {
                "hashtag": clean_hashtag,
                "posts": items,
                "stats": stats,
            }

        except asyncio.TimeoutError:
            raise ApifyError(f"Instagram scraper timed out for #{clean_hashtag}")

    def _extract_instagram_hashtag_stats(
        self,
        posts: list[dict],
        hashtag: str,
    ) -> dict[str, Any]:
        """Extract aggregated stats from Instagram posts."""
        if not posts:
            return {}

        total_likes = 0
        total_comments = 0
        related_hashtags: set[str] = set()
        timestamps: list[int] = []

        for post in posts:
            total_likes += post.get("likesCount", 0)
            total_comments += post.get("commentsCount", 0)

            # Collect related hashtags
            if caption_hashtags := post.get("hashtags"):
                for h in caption_hashtags:
                    if h.lower() != hashtag.lower():
                        related_hashtags.add(h)

            # Collect timestamps for daily posts calculation
            if timestamp := post.get("timestamp"):
                # Handle both string ISO dates and integer timestamps
                if isinstance(timestamp, str):
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        timestamps.append(int(dt.timestamp()))
                    except (ValueError, TypeError):
                        pass
                elif isinstance(timestamp, (int, float)):
                    timestamps.append(int(timestamp))

        post_count = len(posts)

        # Calculate daily posts from timestamps
        daily_posts = self._calculate_daily_posts(timestamps)

        return {
            "hashtag": hashtag,
            "post_count": post_count,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "avg_likes": total_likes / post_count if post_count else 0,
            "avg_comments": total_comments / post_count if post_count else 0,
            "daily_posts": daily_posts,
            "related_hashtags": list(related_hashtags)[:20],  # Limit to top 20
        }

    def _calculate_daily_posts(self, timestamps: list[int]) -> int:
        """Calculate average daily posts from timestamps."""
        if len(timestamps) < 2:
            return 0

        sorted_ts = sorted(timestamps)
        oldest = sorted_ts[0]
        newest = sorted_ts[-1]

        # Time span in days
        span_days = (newest - oldest) / 86400  # seconds per day

        if span_days < 1:
            return len(timestamps)  # All posts in one day

        return int(len(timestamps) / span_days)

    async def _get_dataset_items(
        self,
        dataset_id: str,
        limit: int = 1000,
    ) -> list[dict]:
        """Get items from an Apify dataset."""
        try:
            dataset = self.client.dataset(dataset_id)
            items_page = await dataset.list_items(limit=limit)
            return items_page.items
        except Exception as e:
            logger.error(f"Failed to get dataset items: {e}")
            return []

    async def get_actor_run_results(self, run_id: str) -> list[dict]:
        """
        Get results from a specific actor run.

        Args:
            run_id: The actor run ID

        Returns:
            List of result items
        """
        try:
            run = await self.client.run(run_id).get()
            if not run:
                return []

            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                return []

            return await self._get_dataset_items(dataset_id)

        except Exception as e:
            logger.error(f"Failed to get actor run results: {e}")
            return []

    async def close(self) -> None:
        """Close the client."""
        # ApifyClientAsync doesn't need explicit closing
        pass

    async def __aenter__(self) -> "ApifyClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
