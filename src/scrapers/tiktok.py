"""TikTok hashtag scraper using Apify actors."""

import logging
from typing import Any

from src.clients.apify import ApifyClient
from src.calculators.proxy_scores import TikTokProxyCalculator
from src.models.keyword import TikTokMetrics

logger = logging.getLogger(__name__)


class TikTokScraper:
    """
    TikTok hashtag data scraper via Apify.

    Uses Apify actors to scrape TikTok hashtag data:
    - clockworks/tiktok-scraper (primary)
    - novi/tiktok-hashtag-api (fallback)

    Collects:
    - Total hashtag views
    - Video count
    - Sample videos with engagement metrics
    - Trend data from recent vs older posts
    """

    def __init__(self, apify_client: ApifyClient):
        self.client = apify_client
        self.calculator = TikTokProxyCalculator()

    async def scrape_hashtag(
        self,
        hashtag: str,
        results_count: int = 50,
    ) -> TikTokMetrics | None:
        """
        Scrape TikTok data for a single hashtag.

        Args:
            hashtag: Hashtag to scrape (without # prefix)
            results_count: Number of videos to fetch for analysis

        Returns:
            TikTokMetrics with proxy score and engagement data
        """
        # Clean hashtag (remove # if present)
        hashtag = hashtag.lstrip("#").lower().strip()

        if not hashtag:
            logger.warning("Empty hashtag provided")
            return None

        try:
            # Use Apify client to run scraper
            raw_data = await self.client.run_tiktok_hashtag_scraper(
                hashtags=[hashtag],
                results_per_hashtag=results_count,
            )

            if not raw_data or hashtag not in raw_data:
                logger.warning(f"No data returned for hashtag: {hashtag}")
                return None

            hashtag_data = raw_data[hashtag]

            # Calculate metrics using proxy calculator
            metrics = self.calculator.calculate(hashtag_data)

            logger.info(
                f"TikTok #{hashtag}: views={metrics.hashtag_views:,}, "
                f"videos={metrics.video_count}, proxy={metrics.proxy_score:,}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error scraping TikTok hashtag '{hashtag}': {e}")
            return None

    async def scrape_hashtags(
        self,
        hashtags: list[str],
        results_count: int = 50,
    ) -> dict[str, TikTokMetrics | None]:
        """
        Scrape TikTok data for multiple hashtags.

        Args:
            hashtags: List of hashtags to scrape
            results_count: Number of videos per hashtag

        Returns:
            Dictionary mapping hashtag to TikTokMetrics
        """
        # Clean hashtags
        clean_hashtags = [h.lstrip("#").lower().strip() for h in hashtags if h]
        clean_hashtags = [h for h in clean_hashtags if h]

        if not clean_hashtags:
            return {}

        try:
            # Batch scrape via Apify
            raw_data = await self.client.run_tiktok_hashtag_scraper(
                hashtags=clean_hashtags,
                results_per_hashtag=results_count,
            )

            results = {}
            for hashtag in clean_hashtags:
                if hashtag in raw_data:
                    metrics = self.calculator.calculate(raw_data[hashtag])
                    results[hashtag] = metrics
                else:
                    results[hashtag] = None

            return results

        except Exception as e:
            logger.error(f"Error batch scraping TikTok hashtags: {e}")
            return {h: None for h in clean_hashtags}

    def keyword_to_hashtag(self, keyword: str) -> str:
        """
        Convert a keyword to TikTok hashtag format.

        Examples:
            "vitamin c serum" -> "vitamincserum"
            "anti-aging" -> "antiaging"
        """
        # Remove special characters and spaces
        hashtag = keyword.lower()
        hashtag = "".join(c for c in hashtag if c.isalnum())
        return hashtag

    async def scrape_keywords(
        self,
        keywords: list[str],
        results_count: int = 50,
    ) -> dict[str, TikTokMetrics | None]:
        """
        Scrape TikTok data for keywords (converts to hashtag format).

        Args:
            keywords: List of keywords
            results_count: Number of videos per hashtag

        Returns:
            Dictionary mapping original keyword to TikTokMetrics
        """
        # Map keywords to hashtags
        keyword_hashtag_map = {
            kw: self.keyword_to_hashtag(kw)
            for kw in keywords
        }

        # Get unique hashtags
        unique_hashtags = list(set(keyword_hashtag_map.values()))

        # Scrape
        hashtag_results = await self.scrape_hashtags(unique_hashtags, results_count)

        # Map back to original keywords
        return {
            kw: hashtag_results.get(hashtag)
            for kw, hashtag in keyword_hashtag_map.items()
        }

    @staticmethod
    def extract_stats_from_videos(videos: list[dict]) -> dict[str, Any]:
        """
        Extract aggregate statistics from video list.

        Useful for manual analysis or debugging.
        """
        if not videos:
            return {}

        total_likes = sum(v.get("likes", 0) or 0 for v in videos)
        total_comments = sum(v.get("comments", 0) or 0 for v in videos)
        total_shares = sum(v.get("shares", 0) or 0 for v in videos)
        total_views = sum(v.get("views", 0) or 0 for v in videos)

        count = len(videos)

        return {
            "video_count": count,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "total_shares": total_shares,
            "total_views": total_views,
            "avg_likes": total_likes / count if count > 0 else 0,
            "avg_comments": total_comments / count if count > 0 else 0,
            "avg_shares": total_shares / count if count > 0 else 0,
            "avg_views": total_views / count if count > 0 else 0,
            "avg_engagement": (total_likes + total_comments + total_shares) / count if count > 0 else 0,
        }
