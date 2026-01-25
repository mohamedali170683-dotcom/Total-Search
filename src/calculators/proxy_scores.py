"""
Engagement score calculators for TikTok and Instagram metrics.

IMPORTANT: These calculators produce ENGAGEMENT metrics, NOT search volume estimates.

What we measure:
- TikTok: Hashtag views, video counts, content engagement
- Instagram: Post counts, posting frequency, community engagement

What we DO NOT measure:
- How many people search for these terms on TikTok/Instagram
- Direct user intent or demand (like Google search volume)

The resulting scores represent "audience engagement" - how much content exists
and how much interaction that content receives. This is valuable data but
fundamentally different from search volume.
"""

import logging
from typing import Any

from src.models.keyword import (
    Confidence,
    InstagramMetrics,
    MetricType,
    TikTokMetrics,
    TrendDirection,
)

logger = logging.getLogger(__name__)


class TikTokEngagementCalculator:
    """
    Calculates TikTok ENGAGEMENT metrics from hashtag data.

    IMPORTANT: This produces ENGAGEMENT scores, NOT search volume estimates.

    What we measure:
    - hashtag_views: Total views on content with this hashtag (algorithm-driven exposure)
    - video_count: Number of videos using hashtag (creator activity/supply)
    - avg_likes, avg_comments, avg_shares: Audience interaction rates

    What this tells you:
    - "X million users have engaged with content about this topic on TikTok"
    - How much content exists (supply side)
    - How audiences interact with this content

    What this does NOT tell you:
    - How many people actively search for this term on TikTok
    - Direct user intent or demand

    The TikTok algorithm pushes content to users; users don't primarily search.
    High engagement ≠ high search demand. They are different metrics.
    """

    # Scaling factors for engagement score calculation
    # These normalize the score to a reasonable range, NOT to match search volume
    VIEWS_WEIGHT = 0.001  # 1M views = 1000 engagement points
    VIDEO_COUNT_WEIGHT = 10  # Each video = 10 engagement points

    # Bounds
    MIN_ENGAGEMENT_SCORE = 0
    MAX_ENGAGEMENT_SCORE = 100_000_000  # Allow large engagement numbers

    def calculate(self, raw_data: dict[str, Any]) -> TikTokMetrics:
        """
        Calculate TikTok ENGAGEMENT metrics from raw scraped data.

        Args:
            raw_data: Raw data from Apify TikTok scraper

        Returns:
            TikTokMetrics with engagement_score (NOT search volume)
        """
        stats = raw_data.get("stats", {})
        videos = raw_data.get("videos", [])

        hashtag_views = stats.get("total_views", 0)
        video_count = stats.get("video_count", 0)
        avg_likes = stats.get("avg_likes", 0)
        avg_comments = stats.get("avg_comments", 0)
        avg_shares = stats.get("avg_shares", 0)

        # Calculate trend velocity from video posting frequency
        trend_velocity = self._calculate_posting_trend(videos)
        trend = self._determine_trend(trend_velocity)

        # Calculate engagement score (NOT search volume)
        engagement_score = self._calculate_engagement_score(
            hashtag_views=hashtag_views,
            video_count=video_count,
            avg_likes=avg_likes,
            avg_comments=avg_comments,
            avg_shares=avg_shares,
        )

        return TikTokMetrics(
            # Use engagement_score, not proxy_score
            engagement_score=engagement_score,
            metric_type=MetricType.ENGAGEMENT,
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
            metric_explanation=(
                f"TikTok audience engagement: {hashtag_views:,} views across "
                f"{video_count:,} videos. This shows content exposure, NOT search demand."
            ),
        )

    def _calculate_engagement_score(
        self,
        hashtag_views: int,
        video_count: int,
        avg_likes: float,
        avg_comments: float,
        avg_shares: float,
    ) -> int:
        """
        Calculate engagement score from TikTok metrics.

        This is an ENGAGEMENT metric showing audience interaction,
        NOT a search volume estimate.

        Components:
        - Views: Primary exposure metric (algorithm-driven)
        - Video count: Content supply
        - Avg engagement: Audience interaction quality
        """
        # Views component (scaled down for reasonable numbers)
        views_component = hashtag_views * self.VIEWS_WEIGHT

        # Video count component
        video_component = video_count * self.VIDEO_COUNT_WEIGHT

        # Engagement quality multiplier (based on avg interactions)
        avg_engagement = avg_likes + avg_comments + (avg_shares * 2)  # Shares weighted higher
        engagement_multiplier = 1.0 + min(avg_engagement / 10000, 1.0)  # Max 2x multiplier

        # Combined engagement score
        engagement_score = (views_component + video_component) * engagement_multiplier

        # Apply bounds
        final_score = max(
            self.MIN_ENGAGEMENT_SCORE,
            min(self.MAX_ENGAGEMENT_SCORE, engagement_score)
        )

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


class InstagramEngagementCalculator:
    """
    Calculates Instagram ENGAGEMENT metrics from hashtag data.

    IMPORTANT: This produces ENGAGEMENT scores, NOT search volume estimates.

    What we measure:
    - post_count: Total posts using hashtag (content supply)
    - daily_posts: Posts per day (creator activity)
    - avg_likes, avg_comments: Audience interaction rates

    What this tells you:
    - "X posts exist with this hashtag, receiving Y average engagement"
    - Community activity around this topic
    - Creator interest in this topic

    What this does NOT tell you:
    - How many people actively search for this term on Instagram
    - Direct user intent or demand

    Instagram's Explore is algorithm-driven; users browse more than search.
    High post counts ≠ high search demand. They are different metrics.
    """

    # Scaling factors for engagement score calculation
    # These normalize the score to a reasonable range
    POST_COUNT_WEIGHT = 0.1  # 10K posts = 1000 engagement points
    DAILY_POSTS_WEIGHT = 100  # 10 daily posts = 1000 engagement points

    # Bounds
    MIN_ENGAGEMENT_SCORE = 0
    MAX_ENGAGEMENT_SCORE = 100_000_000

    def calculate(self, raw_data: dict[str, Any]) -> "InstagramMetrics":
        """
        Calculate Instagram ENGAGEMENT metrics from raw scraped data.

        Args:
            raw_data: Raw data from Apify Instagram scraper

        Returns:
            InstagramMetrics with engagement_score (NOT search volume)
        """
        stats = raw_data.get("stats", {})
        posts = raw_data.get("posts", [])

        post_count = stats.get("post_count", 0)
        daily_posts = stats.get("daily_posts", 0) or self._calculate_daily_posts(posts)
        avg_likes = stats.get("avg_likes", 0)
        avg_comments = stats.get("avg_comments", 0)
        related_hashtags = stats.get("related_hashtags", [])

        # Calculate trend from posting frequency
        trend_velocity = self._calculate_posting_trend(posts)
        trend = self._determine_trend(trend_velocity)

        # Calculate engagement score (NOT search volume)
        engagement_score = self._calculate_engagement_score(
            post_count=post_count,
            daily_posts=daily_posts,
            avg_likes=avg_likes,
            avg_comments=avg_comments,
        )

        return InstagramMetrics(
            # Use engagement_score, not proxy_score
            engagement_score=engagement_score,
            metric_type=MetricType.ENGAGEMENT,
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
            metric_explanation=(
                f"Instagram community engagement: {post_count:,} posts, "
                f"~{daily_posts:,} posts/day. This shows creator activity, NOT search demand."
            ),
        )

    def _calculate_engagement_score(
        self,
        post_count: int,
        daily_posts: int,
        avg_likes: float,
        avg_comments: float,
    ) -> int:
        """
        Calculate engagement score from Instagram metrics.

        This is an ENGAGEMENT metric showing community activity,
        NOT a search volume estimate.

        Components:
        - Post count: Content supply/creator activity
        - Daily posts: Current activity level
        - Avg engagement: Audience interaction quality
        """
        # Post count component (historical content supply)
        post_component = post_count * self.POST_COUNT_WEIGHT

        # Daily posts component (current activity)
        daily_component = daily_posts * self.DAILY_POSTS_WEIGHT

        # Engagement quality multiplier
        avg_engagement = avg_likes + avg_comments
        engagement_multiplier = 1.0 + min(avg_engagement / 5000, 1.0)  # Max 2x multiplier

        # Combined engagement score
        engagement_score = (post_component + daily_component) * engagement_multiplier

        # Apply bounds
        final_score = max(
            self.MIN_ENGAGEMENT_SCORE,
            min(self.MAX_ENGAGEMENT_SCORE, engagement_score)
        )

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

    def _determine_trend_instagram(self, velocity: float) -> TrendDirection:
        """Determine trend direction from velocity (Instagram-specific)."""
        if velocity > 1.15:
            return TrendDirection.GROWING
        elif velocity < 0.85:
            return TrendDirection.DECLINING
        return TrendDirection.STABLE

    # Alias for backward compatibility
    _determine_trend = _determine_trend_instagram


# Backward-compatible aliases (deprecated - use new class names)
TikTokProxyCalculator = TikTokEngagementCalculator
InstagramProxyCalculator = InstagramEngagementCalculator
