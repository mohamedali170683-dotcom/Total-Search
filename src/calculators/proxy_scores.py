"""Proxy score calculators for TikTok and Instagram metrics."""

import logging
from typing import Any

from src.models.keyword import (
    Confidence,
    InstagramMetrics,
    TikTokMetrics,
    TrendDirection,
)

logger = logging.getLogger(__name__)


class TikTokProxyCalculator:
    """
    Converts TikTok hashtag metrics into a search-volume-like score.

    Formula focuses on DEMAND signals (views/post counts), NOT engagement:
    - hashtag_views: Total views (all-time) - primary demand signal
    - video_count: Number of videos using hashtag - content creation demand

    The output represents estimated monthly searches/interest based on
    hashtag view counts, calibrated to be comparable to Google search volume.
    """

    # Calibration: Convert hashtag views to monthly search estimate
    # NOTE: We only scrape ~10-20 sample videos, not the full hashtag
    # So we need to extrapolate from the sample to estimate total demand

    # Sample extrapolation factor:
    # Calibrated against Keywordtool.io: naturkosmetik = 9.4K TikTok searches
    # Previous 50x gave 64.4K (6.85x too high)
    # New factor: 50 / 6.85 ≈ 7.3
    SAMPLE_EXTRAPOLATION_FACTOR = 7

    # Conversion factor for views to searches
    VIEWS_TO_SEARCHES_FACTOR = 0.01

    # Video count bonus: More content = more demand
    VIDEO_COUNT_FACTOR = 2.0

    # Normalization bounds
    MIN_PROXY_SCORE = 0
    MAX_PROXY_SCORE = 10_000_000  # Cap at 10M to match Google's scale

    def calculate(self, raw_data: dict[str, Any]) -> TikTokMetrics:
        """
        Calculate TikTok proxy metrics from raw scraped data.

        Args:
            raw_data: Raw data from Apify TikTok scraper

        Returns:
            TikTokMetrics with calculated proxy score based on views/post counts
        """
        stats = raw_data.get("stats", {})
        videos = raw_data.get("videos", [])

        hashtag_views = stats.get("total_views", 0)
        video_count = stats.get("video_count", 0)
        avg_likes = stats.get("avg_likes", 0)
        avg_comments = stats.get("avg_comments", 0)
        avg_shares = stats.get("avg_shares", 0)

        # Calculate trend velocity from video posting frequency (NOT engagement)
        trend_velocity = self._calculate_posting_trend(videos)

        # Determine trend direction
        trend = self._determine_trend(trend_velocity)

        # Calculate proxy score based on DEMAND (views + video count)
        proxy_score = self._calculate_demand_proxy(
            hashtag_views=hashtag_views,
            video_count=video_count,
        )

        return TikTokMetrics(
            proxy_score=proxy_score,
            trend=trend,
            trend_velocity=trend_velocity,
            confidence=Confidence.PROXY,
            source="apify_tiktok",
            hashtag_views=hashtag_views,
            video_count=video_count,
            avg_likes=avg_likes,
            avg_comments=avg_comments,
            avg_shares=avg_shares,
            raw_data=raw_data,
        )

    def _calculate_demand_proxy(
        self,
        hashtag_views: int,
        video_count: int,
    ) -> int:
        """
        Calculate proxy score based on DEMAND metrics only.

        This converts hashtag views and video counts into an estimated
        monthly search volume equivalent.

        NOTE: We only sample ~20 videos, so we extrapolate to estimate
        the full hashtag's demand.
        """
        # Extrapolate from sample to estimate full hashtag metrics
        estimated_total_views = hashtag_views * self.SAMPLE_EXTRAPOLATION_FACTOR
        estimated_total_videos = video_count * self.SAMPLE_EXTRAPOLATION_FACTOR

        # Primary signal: Estimated total views → monthly searches
        # Convert views to search equivalent
        views_as_searches = estimated_total_views * self.VIEWS_TO_SEARCHES_FACTOR

        # Secondary signal: Estimated video count indicates content creator demand
        video_count_signal = estimated_total_videos * self.VIDEO_COUNT_FACTOR

        # Combined demand score
        demand_score = views_as_searches + video_count_signal

        # Apply bounds
        final_score = max(self.MIN_PROXY_SCORE, min(self.MAX_PROXY_SCORE, demand_score))

        return int(final_score)

    def _calculate_posting_trend(self, videos: list[dict]) -> float:
        """
        Calculate trend velocity by comparing posting FREQUENCY over time.

        This measures demand by how many NEW videos are being created,
        not engagement metrics.

        Returns:
            Velocity multiplier (>1 = growing interest, <1 = declining)
        """
        if len(videos) < 4:
            return 1.0  # Not enough data, assume stable

        # Extract timestamps
        timestamps = []
        for video in videos:
            ts = video.get("createTime", 0)
            if not ts:
                # Try ISO format
                iso_ts = video.get("createTimeISO", "")
                if iso_ts:
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
                        ts = int(dt.timestamp())
                    except (ValueError, TypeError):
                        continue
            if ts:
                timestamps.append(ts)

        if len(timestamps) < 4:
            return 1.0

        # Sort timestamps (oldest first)
        timestamps.sort()

        # Split into first half and second half time periods
        mid = len(timestamps) // 2
        first_half = timestamps[:mid]
        second_half = timestamps[mid:]

        # Calculate posting rate for each half
        def posting_rate(ts_list: list[int]) -> float:
            if len(ts_list) < 2:
                return 0
            time_span = ts_list[-1] - ts_list[0]
            if time_span <= 0:
                return len(ts_list)  # All same day
            days = time_span / 86400
            return len(ts_list) / max(days, 1)

        first_rate = posting_rate(first_half)
        second_rate = posting_rate(second_half)

        # Calculate velocity (second half rate / first half rate)
        if first_rate > 0:
            velocity = second_rate / first_rate
        else:
            velocity = 1.5 if second_rate > 0 else 1.0

        # Clamp velocity to reasonable bounds
        return max(0.3, min(3.0, velocity))

    def _determine_trend(self, velocity: float) -> TrendDirection:
        """Determine trend direction from velocity."""
        if velocity > 1.15:
            return TrendDirection.GROWING
        elif velocity < 0.85:
            return TrendDirection.DECLINING
        return TrendDirection.STABLE


class InstagramProxyCalculator:
    """
    Converts Instagram hashtag metrics into a search-volume-like score.

    Formula focuses on DEMAND signals (post counts), NOT engagement:
    - post_count: Total posts using hashtag (all-time)
    - daily_posts: Posts per day (primary demand signal)

    The output represents estimated monthly searches/interest based on
    hashtag post volume, calibrated to be comparable to Google search volume.
    """

    # Calibration: Convert daily posts to monthly search estimate
    # Calibrated against Keywordtool.io Instagram data
    # "naturkosmetik" has ~130.9K search volume with ~956K total posts
    # ~137 searches per 1000 posts = 0.137 ratio
    # 100 daily posts ≈ 15,000 monthly searches
    DAILY_POSTS_TO_SEARCHES = 150  # 1 daily post ≈ 150 monthly searches

    # Post count signal: Total posts indicates historical demand
    # Calibrated: 956K posts ≈ 130.9K searches = ~0.137 ratio
    # Increased from 0.01 to 0.137 for better calibration
    POST_COUNT_FACTOR = 0.137

    # Normalization bounds
    MIN_PROXY_SCORE = 0
    MAX_PROXY_SCORE = 10_000_000

    def calculate(self, raw_data: dict[str, Any]) -> InstagramMetrics:
        """
        Calculate Instagram proxy metrics from raw scraped data.

        Args:
            raw_data: Raw data from Apify Instagram scraper

        Returns:
            InstagramMetrics with calculated proxy score based on post counts
        """
        stats = raw_data.get("stats", {})
        posts = raw_data.get("posts", [])

        post_count = stats.get("post_count", 0)
        daily_posts = stats.get("daily_posts", 0) or self._calculate_daily_posts(posts)
        avg_likes = stats.get("avg_likes", 0)
        avg_comments = stats.get("avg_comments", 0)
        related_hashtags = stats.get("related_hashtags", [])

        # Calculate trend from posting FREQUENCY (not engagement)
        trend_velocity = self._calculate_posting_trend(posts)
        trend = self._determine_trend(trend_velocity)

        # Calculate proxy score based on DEMAND (post counts only)
        proxy_score = self._calculate_demand_proxy(
            post_count=post_count,
            daily_posts=daily_posts,
        )

        return InstagramMetrics(
            proxy_score=proxy_score,
            trend=trend,
            trend_velocity=trend_velocity,
            confidence=Confidence.PROXY,
            source="apify_instagram",
            post_count=post_count,
            daily_posts=daily_posts,
            avg_engagement=avg_likes + avg_comments,
            avg_likes=avg_likes,
            avg_comments=avg_comments,
            related_hashtags=related_hashtags,
            raw_data=raw_data,
        )

    def _calculate_demand_proxy(
        self,
        post_count: int,
        daily_posts: int,
    ) -> int:
        """
        Calculate proxy score based on DEMAND metrics only.

        This converts post counts into an estimated monthly search volume equivalent.
        """
        # Primary signal: Daily posts → monthly searches
        # Logic: Active posting = active interest/demand
        daily_posts_signal = daily_posts * self.DAILY_POSTS_TO_SEARCHES

        # Secondary signal: Total post count indicates historical demand
        post_count_signal = post_count * self.POST_COUNT_FACTOR

        # Combined demand score
        demand_score = daily_posts_signal + post_count_signal

        # Apply bounds
        final_score = max(self.MIN_PROXY_SCORE, min(self.MAX_PROXY_SCORE, demand_score))

        return int(final_score)

    def _calculate_daily_posts(self, posts: list[dict]) -> int:
        """
        Calculate average daily posts from timestamp data.

        Args:
            posts: List of post data with timestamps

        Returns:
            Estimated daily posts
        """
        if len(posts) < 2:
            return 0

        timestamps = []
        for post in posts:
            ts = post.get("timestamp")
            if ts:
                # Handle string timestamps
                if isinstance(ts, str):
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        timestamps.append(int(dt.timestamp()))
                    except (ValueError, TypeError):
                        continue
                elif isinstance(ts, (int, float)):
                    timestamps.append(int(ts))

        if len(timestamps) < 2:
            return 0

        timestamps.sort()
        oldest = timestamps[0]
        newest = timestamps[-1]

        # Time span in days
        span_seconds = newest - oldest
        span_days = span_seconds / 86400  # seconds per day

        if span_days < 1:
            return len(timestamps)

        return int(len(timestamps) / span_days)

    def _calculate_posting_trend(self, posts: list[dict]) -> float:
        """
        Calculate trend velocity by comparing posting FREQUENCY over time.

        This measures demand by how many NEW posts are being created,
        not engagement metrics.

        Returns:
            Velocity multiplier (>1 = growing interest, <1 = declining)
        """
        if len(posts) < 4:
            return 1.0

        # Extract timestamps
        timestamps = []
        for post in posts:
            ts = post.get("timestamp")
            if ts:
                # Handle string timestamps
                if isinstance(ts, str):
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        timestamps.append(int(dt.timestamp()))
                    except (ValueError, TypeError):
                        continue
                elif isinstance(ts, (int, float)):
                    timestamps.append(int(ts))

        if len(timestamps) < 4:
            return 1.0

        # Sort timestamps (oldest first)
        timestamps.sort()

        # Split into first half and second half time periods
        mid = len(timestamps) // 2
        first_half = timestamps[:mid]
        second_half = timestamps[mid:]

        # Calculate posting rate for each half
        def posting_rate(ts_list: list[int]) -> float:
            if len(ts_list) < 2:
                return 0
            time_span = ts_list[-1] - ts_list[0]
            if time_span <= 0:
                return len(ts_list)  # All same day
            days = time_span / 86400
            return len(ts_list) / max(days, 1)

        first_rate = posting_rate(first_half)
        second_rate = posting_rate(second_half)

        # Calculate velocity (second half rate / first half rate)
        if first_rate > 0:
            velocity = second_rate / first_rate
        else:
            velocity = 1.5 if second_rate > 0 else 1.0

        # Clamp velocity to reasonable bounds
        return max(0.3, min(3.0, velocity))

    def _determine_trend(self, velocity: float) -> TrendDirection:
        """Determine trend direction from velocity."""
        if velocity > 1.15:
            return TrendDirection.GROWING
        elif velocity < 0.85:
            return TrendDirection.DECLINING
        return TrendDirection.STABLE
