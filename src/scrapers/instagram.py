"""Instagram hashtag scraper using Apify actors."""

import logging
from typing import Any

from src.clients.apify import ApifyClient
from src.calculators.proxy_scores import InstagramProxyCalculator
from src.models.keyword import InstagramMetrics

logger = logging.getLogger(__name__)


class InstagramScraper:
    """
    Instagram hashtag data scraper via Apify.

    Uses Apify actors to scrape Instagram hashtag data:
    - apify/instagram-hashtag-scraper (primary)
    - apify/instagram-hashtag-stats (for basic stats)

    Collects:
    - Total post count for hashtag
    - Sample posts with engagement metrics
    - Daily posting frequency
    - Related hashtags
    """

    def __init__(self, apify_client: ApifyClient):
        self.client = apify_client
        self.calculator = InstagramProxyCalculator()

    async def scrape_hashtag(
        self,
        hashtag: str,
        results_count: int = 50,
    ) -> InstagramMetrics | None:
        """
        Scrape Instagram data for a single hashtag.

        Args:
            hashtag: Hashtag to scrape (without # prefix)
            results_count: Number of posts to fetch for analysis

        Returns:
            InstagramMetrics with proxy score and engagement data
        """
        # Clean hashtag (remove # if present)
        hashtag = hashtag.lstrip("#").lower().strip()

        if not hashtag:
            logger.warning("Empty hashtag provided")
            return None

        try:
            # Use Apify client to run scraper
            raw_data = await self.client.run_instagram_hashtag_scraper(
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
                f"Instagram #{hashtag}: posts={metrics.post_count:,}, "
                f"daily={metrics.daily_posts}, proxy={metrics.proxy_score:,}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error scraping Instagram hashtag '{hashtag}': {e}")
            return None

    async def scrape_hashtags(
        self,
        hashtags: list[str],
        results_count: int = 50,
    ) -> dict[str, InstagramMetrics | None]:
        """
        Scrape Instagram data for multiple hashtags.

        Args:
            hashtags: List of hashtags to scrape
            results_count: Number of posts per hashtag

        Returns:
            Dictionary mapping hashtag to InstagramMetrics
        """
        # Clean hashtags
        clean_hashtags = [h.lstrip("#").lower().strip() for h in hashtags if h]
        clean_hashtags = [h for h in clean_hashtags if h]

        if not clean_hashtags:
            return {}

        try:
            # Batch scrape via Apify
            raw_data = await self.client.run_instagram_hashtag_scraper(
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
            logger.error(f"Error batch scraping Instagram hashtags: {e}")
            return {h: None for h in clean_hashtags}

    def keyword_to_hashtag(self, keyword: str) -> str:
        """
        Convert a keyword to Instagram hashtag format.

        Instagram hashtags can contain underscores but not spaces.

        Examples:
            "vitamin c serum" -> "vitamincserum"
            "anti-aging" -> "antiaging"
            "self_care" -> "self_care" (underscores preserved)
        """
        # Remove special characters except underscores
        hashtag = keyword.lower()
        hashtag = "".join(c for c in hashtag if c.isalnum() or c == "_")
        # Remove consecutive underscores and trim
        while "__" in hashtag:
            hashtag = hashtag.replace("__", "_")
        return hashtag.strip("_")

    async def scrape_keywords(
        self,
        keywords: list[str],
        results_count: int = 50,
    ) -> dict[str, InstagramMetrics | None]:
        """
        Scrape Instagram data for keywords (converts to hashtag format).

        Args:
            keywords: List of keywords
            results_count: Number of posts per hashtag

        Returns:
            Dictionary mapping original keyword to InstagramMetrics
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
    def extract_stats_from_posts(posts: list[dict]) -> dict[str, Any]:
        """
        Extract aggregate statistics from post list.

        Useful for manual analysis or debugging.
        """
        if not posts:
            return {}

        total_likes = sum(p.get("likesCount", 0) or p.get("likes", 0) or 0 for p in posts)
        total_comments = sum(p.get("commentsCount", 0) or p.get("comments", 0) or 0 for p in posts)

        count = len(posts)

        # Extract related hashtags from captions
        related_hashtags = set()
        for post in posts:
            caption = post.get("caption", "") or ""
            # Find hashtags in caption
            words = caption.split()
            for word in words:
                if word.startswith("#") and len(word) > 1:
                    related_hashtags.add(word.lstrip("#").lower())

        return {
            "post_count": count,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "avg_likes": total_likes / count if count > 0 else 0,
            "avg_comments": total_comments / count if count > 0 else 0,
            "avg_engagement": (total_likes + total_comments) / count if count > 0 else 0,
            "related_hashtags": list(related_hashtags)[:20],  # Top 20
        }

    @staticmethod
    def calculate_daily_posts(posts: list[dict]) -> int:
        """
        Calculate average daily posts from timestamp data.

        Uses the date range of posts to estimate daily frequency.
        """
        if not posts or len(posts) < 2:
            return 0

        from datetime import datetime

        timestamps = []
        for post in posts:
            ts = post.get("timestamp") or post.get("taken_at_timestamp")
            if ts:
                if isinstance(ts, (int, float)):
                    timestamps.append(datetime.fromtimestamp(ts))
                elif isinstance(ts, str):
                    try:
                        timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                    except ValueError:
                        pass

        if len(timestamps) < 2:
            return 0

        timestamps.sort()
        date_range = (timestamps[-1] - timestamps[0]).days

        if date_range <= 0:
            return len(posts)  # All posts on same day

        return len(posts) // date_range

    def get_related_hashtags(self, posts: list[dict], exclude: str = "") -> list[str]:
        """
        Extract related hashtags from post captions.

        Args:
            posts: List of post data
            exclude: Hashtag to exclude (usually the searched one)

        Returns:
            List of related hashtags sorted by frequency
        """
        from collections import Counter

        hashtag_counts: Counter = Counter()
        exclude_lower = exclude.lower().lstrip("#")

        for post in posts:
            caption = post.get("caption", "") or ""
            words = caption.split()
            for word in words:
                if word.startswith("#") and len(word) > 1:
                    tag = word.lstrip("#").lower()
                    if tag != exclude_lower:
                        hashtag_counts[tag] += 1

        # Return top hashtags
        return [tag for tag, _ in hashtag_counts.most_common(20)]
