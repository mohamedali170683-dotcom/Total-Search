"""
Audience interest calculators for TikTok and Instagram metrics.

IMPORTANT: These calculators produce AUDIENCE INTEREST metrics, NOT search volume estimates.

What we measure:
- TikTok: Average likes per video — how many unique users interact with content
- Instagram: Average likes per post — how many unique users interact with content

Why avg_likes:
- 1 like ≈ 1 unique person interested in this topic
- Average per post/video is sample-size-independent (stable whether we scrape 20 or 50 posts)
- We take max(avg_likes, avg_comments) to avoid double-counting (same person can like AND comment)
- Likes are almost always the higher number (10-50x more than comments)

What this does NOT measure:
- How many people search for these terms on TikTok/Instagram
- Direct user intent or demand (like Google search volume)
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
    Calculates TikTok audience interest from hashtag data.

    Metric: avg_likes per video (the higher of avg_likes vs avg_comments).

    Why avg_likes:
    - 1 like ≈ 1 unique person interested in this topic
    - Sample-size-independent (stable whether we scrape 20 or 50 videos)
    - We use max(avg_likes, avg_comments) to pick the stronger signal
      without double-counting (same person can like AND comment)

    What this tells you:
    - "On average, X people interact with each video about this topic"
    - Higher = more audience interest per piece of content

    What this does NOT tell you:
    - How many people search for this term on TikTok
    - Total reach (that depends on video count, which is supply-side)
    """

    def calculate(self, raw_data: dict[str, Any]) -> TikTokMetrics:
        """
        Calculate TikTok audience interest from raw scraped data.

        The engagement_score = max(avg_likes, avg_comments) per video.
        This represents the average number of unique users interacting
        with content about this topic.
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

        # Primary metric: max(avg_likes, avg_comments) per video
        # This avoids double-counting (same person can like AND comment)
        engagement_score = int(max(avg_likes, avg_comments))

        return TikTokMetrics(
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
                f"TikTok avg. interactions per video: {int(max(avg_likes, avg_comments)):,} "
                f"(from {video_count:,} sampled videos). "
                f"Each interaction ≈ 1 interested person."
            ),
        )

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
    Calculates Instagram audience interest from hashtag data.

    Metric: avg_likes per post (the higher of avg_likes vs avg_comments).

    Why avg_likes:
    - 1 like ≈ 1 unique person interested in this topic
    - Sample-size-independent (stable whether we scrape 20 or 50 posts)
    - We use max(avg_likes, avg_comments) to pick the stronger signal
      without double-counting (same person can like AND comment)

    What this tells you:
    - "On average, X people interact with each post about this topic"
    - Higher = more audience interest per piece of content

    What this does NOT tell you:
    - How many people search for this term on Instagram
    - Total reach (that depends on post count, which is supply-side)
    """

    def calculate(self, raw_data: dict[str, Any]) -> "InstagramMetrics":
        """
        Calculate Instagram audience interest from raw scraped data.

        The engagement_score = max(avg_likes, avg_comments) per post.
        This represents the average number of unique users interacting
        with content about this topic.
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

        # Primary metric: max(avg_likes, avg_comments) per post
        # This avoids double-counting (same person can like AND comment)
        engagement_score = int(max(avg_likes, avg_comments))

        return InstagramMetrics(
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
                f"Instagram avg. interactions per post: {int(max(avg_likes, avg_comments)):,} "
                f"(from {post_count:,} sampled posts). "
                f"Each interaction ≈ 1 interested person."
            ),
        )

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
