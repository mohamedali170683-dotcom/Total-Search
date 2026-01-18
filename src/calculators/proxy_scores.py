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

    Formula components:
    - hashtag_views: Total views (all-time) - normalized to monthly estimate
    - video_count: Number of videos using hashtag
    - avg_engagement: Average likes + comments + shares per video
    - trend_velocity: Recent vs older posts engagement ratio

    The output is calibrated to be roughly comparable to Google search volume
    (i.e., a score of 50,000 indicates similar "demand" as 50,000 Google searches)
    """

    # Calibration constants (tuned based on cross-platform analysis)
    VIEWS_WEIGHT = 0.0001  # Views to demand conversion
    VIDEO_COUNT_WEIGHT = 0.01  # Video count contribution
    ENGAGEMENT_WEIGHT = 0.5  # Engagement multiplier
    TREND_WEIGHT = 10000  # Base weight for trend adjustment
    ESTIMATED_HASHTAG_AGE_MONTHS = 24  # Average assumption for normalization

    # Normalization bounds
    MIN_PROXY_SCORE = 0
    MAX_PROXY_SCORE = 10_000_000  # Cap at 10M to match Google's scale

    def calculate(self, raw_data: dict[str, Any]) -> TikTokMetrics:
        """
        Calculate TikTok proxy metrics from raw scraped data.

        Args:
            raw_data: Raw data from Apify TikTok scraper

        Returns:
            TikTokMetrics with calculated proxy score
        """
        stats = raw_data.get("stats", {})
        videos = raw_data.get("videos", [])

        hashtag_views = stats.get("total_views", 0)
        video_count = stats.get("video_count", 0)
        avg_likes = stats.get("avg_likes", 0)
        avg_comments = stats.get("avg_comments", 0)
        avg_shares = stats.get("avg_shares", 0)

        # Calculate trend velocity from videos
        trend_velocity = self.calculate_trend_velocity(videos)

        # Determine trend direction
        trend = self._determine_trend(trend_velocity)

        # Calculate proxy score
        proxy_score = self._calculate_proxy_score(
            hashtag_views=hashtag_views,
            video_count=video_count,
            avg_engagement=avg_likes + avg_comments + avg_shares,
            trend_velocity=trend_velocity,
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

    def _calculate_proxy_score(
        self,
        hashtag_views: int,
        video_count: int,
        avg_engagement: float,
        trend_velocity: float,
    ) -> int:
        """Calculate the proxy score from components."""
        # Base score from views (normalized to monthly)
        monthly_views = hashtag_views / self.ESTIMATED_HASHTAG_AGE_MONTHS
        views_component = monthly_views * self.VIEWS_WEIGHT

        # Video count component (indicates content creation demand)
        video_component = video_count * self.VIDEO_COUNT_WEIGHT

        # Engagement multiplier (capped at 3x)
        engagement_multiplier = min(3.0, 1 + (avg_engagement / 10000) * self.ENGAGEMENT_WEIGHT)

        # Base score
        base_score = (views_component + video_component) * engagement_multiplier

        # Apply trend adjustment
        trend_adjusted = base_score * trend_velocity

        # Apply bounds
        final_score = max(self.MIN_PROXY_SCORE, min(self.MAX_PROXY_SCORE, trend_adjusted))

        return int(final_score)

    def calculate_trend_velocity(self, videos: list[dict]) -> float:
        """
        Calculate trend velocity by comparing recent vs older video engagement.

        Returns:
            Velocity multiplier (>1 = growing, <1 = declining)
        """
        if len(videos) < 6:
            return 1.0  # Not enough data, assume stable

        # Sort by creation time (newest first)
        sorted_videos = sorted(
            videos,
            key=lambda v: v.get("createTime", 0) or v.get("createTimeISO", ""),
            reverse=True,
        )

        # Split into recent and older halves
        mid = len(sorted_videos) // 2
        recent = sorted_videos[:mid]
        older = sorted_videos[mid:]

        def get_engagement(video: dict) -> int:
            likes = video.get("diggCount", 0) or video.get("stats", {}).get("diggCount", 0) or 0
            comments = video.get("commentCount", 0) or video.get("stats", {}).get("commentCount", 0) or 0
            shares = video.get("shareCount", 0) or video.get("stats", {}).get("shareCount", 0) or 0
            return likes + comments + shares

        recent_engagement = sum(get_engagement(v) for v in recent) / len(recent) if recent else 0
        older_engagement = sum(get_engagement(v) for v in older) / len(older) if older else 0

        if older_engagement > 0:
            velocity = recent_engagement / older_engagement
        else:
            velocity = 1.5 if recent_engagement > 0 else 1.0

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

    Formula components:
    - post_count: Total posts (all-time)
    - daily_posts: Posts per day (strongest demand signal)
    - avg_engagement: Average likes + comments per post
    - saturation_factor: Penalty for oversaturated hashtags
    """

    # Calibration constants
    DAILY_POSTS_WEIGHT = 1.5  # Daily posts is the primary signal
    ENGAGEMENT_MULTIPLIER_CAP = 2.0  # Max engagement boost
    SATURATION_THRESHOLD = 10_000_000  # Posts above this are oversaturated
    SATURATION_PENALTY = 0.5  # Penalty for oversaturated hashtags

    # Normalization bounds
    MIN_PROXY_SCORE = 0
    MAX_PROXY_SCORE = 10_000_000

    def calculate(self, raw_data: dict[str, Any]) -> InstagramMetrics:
        """
        Calculate Instagram proxy metrics from raw scraped data.

        Args:
            raw_data: Raw data from Apify Instagram scraper

        Returns:
            InstagramMetrics with calculated proxy score
        """
        stats = raw_data.get("stats", {})
        posts = raw_data.get("posts", [])

        post_count = stats.get("post_count", 0)
        daily_posts = stats.get("daily_posts", 0) or self.calculate_daily_posts(posts)
        avg_likes = stats.get("avg_likes", 0)
        avg_comments = stats.get("avg_comments", 0)
        related_hashtags = stats.get("related_hashtags", [])

        # Calculate trend from posts
        trend_velocity = self._calculate_trend_velocity(posts)
        trend = self._determine_trend(trend_velocity)

        # Calculate proxy score
        proxy_score = self._calculate_proxy_score(
            post_count=post_count,
            daily_posts=daily_posts,
            avg_engagement=avg_likes + avg_comments,
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

    def _calculate_proxy_score(
        self,
        post_count: int,
        daily_posts: int,
        avg_engagement: float,
    ) -> int:
        """Calculate the proxy score from components."""
        # Daily posts is the primary signal (monthly estimate)
        monthly_posts = daily_posts * 30
        daily_component = monthly_posts * self.DAILY_POSTS_WEIGHT * 100

        # Engagement multiplier (capped)
        engagement_multiplier = min(
            self.ENGAGEMENT_MULTIPLIER_CAP,
            1 + (avg_engagement / 5000),
        )

        # Base score
        base_score = daily_component * engagement_multiplier

        # Apply saturation penalty for oversaturated hashtags
        if post_count > self.SATURATION_THRESHOLD:
            saturation_ratio = self.SATURATION_THRESHOLD / post_count
            base_score *= max(self.SATURATION_PENALTY, saturation_ratio)

        # Apply bounds
        final_score = max(self.MIN_PROXY_SCORE, min(self.MAX_PROXY_SCORE, base_score))

        return int(final_score)

    def calculate_daily_posts(self, posts: list[dict]) -> int:
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
            if ts := post.get("timestamp"):
                timestamps.append(ts)

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

    def _calculate_trend_velocity(self, posts: list[dict]) -> float:
        """Calculate trend velocity from post engagement over time."""
        if len(posts) < 6:
            return 1.0

        # Sort by timestamp (newest first)
        sorted_posts = sorted(
            posts,
            key=lambda p: p.get("timestamp", 0),
            reverse=True,
        )

        mid = len(sorted_posts) // 2
        recent = sorted_posts[:mid]
        older = sorted_posts[mid:]

        def get_engagement(post: dict) -> int:
            return (post.get("likesCount", 0) or 0) + (post.get("commentsCount", 0) or 0)

        recent_engagement = sum(get_engagement(p) for p in recent) / len(recent) if recent else 0
        older_engagement = sum(get_engagement(p) for p in older) / len(older) if older else 0

        if older_engagement > 0:
            velocity = recent_engagement / older_engagement
        else:
            velocity = 1.5 if recent_engagement > 0 else 1.0

        return max(0.3, min(3.0, velocity))

    def _determine_trend(self, velocity: float) -> TrendDirection:
        """Determine trend direction from velocity."""
        if velocity > 1.15:
            return TrendDirection.GROWING
        elif velocity < 0.85:
            return TrendDirection.DECLINING
        return TrendDirection.STABLE
