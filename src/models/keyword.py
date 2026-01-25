"""Pydantic models for keyword metrics across platforms."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TrendDirection(str, Enum):
    """Trend direction for keyword metrics."""

    GROWING = "growing"
    STABLE = "stable"
    DECLINING = "declining"


class Competition(str, Enum):
    """Competition level for keywords."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Confidence(str, Enum):
    """Confidence level of the metrics."""

    HIGH = "high"  # Direct API data (DataForSEO) - verified search volume
    CALIBRATED = "calibrated"  # Proxy with regression model calibration
    MEDIUM = "medium"  # Official API but not direct search volume
    PROXY = "proxy"  # Engagement metrics, NOT search volume


class MetricType(str, Enum):
    """Type of metric being reported."""

    SEARCH_VOLUME = "search_volume"  # Actual search volume (Google, YouTube, Amazon)
    ENGAGEMENT = "engagement"  # Audience engagement metrics (TikTok, Instagram)
    INTEREST_INDEX = "interest_index"  # Relative interest score (Pinterest)


class Platform(str, Enum):
    """Supported platforms."""

    GOOGLE = "google"
    YOUTUBE = "youtube"
    AMAZON = "amazon"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    PINTEREST = "pinterest"


class PlatformMetrics(BaseModel):
    """Base model for platform-specific keyword metrics."""

    # Core metrics
    search_volume: int | None = Field(
        default=None, description="Verified monthly search volume (Google, YouTube, Amazon only)"
    )
    engagement_score: int | None = Field(
        default=None, description="Audience engagement metric (TikTok, Instagram) - NOT search volume"
    )
    interest_score: int | None = Field(
        default=None, description="Relative interest index 0-100 (Pinterest)"
    )

    # Legacy field for backward compatibility
    proxy_score: int | None = Field(
        default=None, description="[DEPRECATED] Use engagement_score instead"
    )

    # Metadata
    metric_type: MetricType = Field(
        default=MetricType.SEARCH_VOLUME,
        description="Type of metric: search_volume, engagement, or interest_index"
    )
    trend: TrendDirection | None = Field(default=None, description="Trend direction")
    trend_velocity: float | None = Field(
        default=None, description="Trend velocity multiplier (>1 = growing)"
    )
    competition: Competition | None = Field(default=None, description="Competition level")
    cpc: float | None = Field(default=None, description="Cost per click (where available)")
    confidence: Confidence = Field(default=Confidence.HIGH, description="Data confidence level")
    source: str = Field(description="API source name")
    raw_data: dict[str, Any] | None = Field(
        default=None, description="Original API response data"
    )
    collected_at: datetime = Field(default_factory=datetime.utcnow)

    # Human-readable explanation
    metric_explanation: str | None = Field(
        default=None, description="Human-readable explanation of what this metric represents"
    )

    @property
    def effective_volume(self) -> int:
        """Get the effective volume based on metric type."""
        if self.metric_type == MetricType.SEARCH_VOLUME:
            return self.search_volume or 0
        elif self.metric_type == MetricType.ENGAGEMENT:
            return self.engagement_score or self.proxy_score or 0
        else:  # INTEREST_INDEX
            return self.interest_score or 0

    @property
    def is_verified_search_data(self) -> bool:
        """Returns True if this is verified search volume data."""
        return self.metric_type == MetricType.SEARCH_VOLUME and self.confidence == Confidence.HIGH


class MonthlySearchData(BaseModel):
    """Monthly search volume data point."""

    year: int
    month: int
    search_volume: int


class GoogleMetrics(PlatformMetrics):
    """Google-specific keyword metrics from DataForSEO."""

    source: str = "dataforseo_google"
    monthly_searches: list[MonthlySearchData] | None = Field(
        default=None, description="12-month search volume history"
    )
    keyword_difficulty: float | None = Field(
        default=None, description="Keyword difficulty score (0-100)"
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def set_confidence(cls, v: Any) -> Confidence:
        """Set confidence based on data availability."""
        return v if v else Confidence.HIGH


class YouTubeMetrics(PlatformMetrics):
    """YouTube-specific keyword metrics from DataForSEO."""

    source: str = "dataforseo_youtube"
    monthly_searches: list[MonthlySearchData] | None = Field(
        default=None, description="12-month search volume history"
    )


class AmazonMetrics(PlatformMetrics):
    """Amazon-specific keyword metrics from Jungle Scout."""

    source: str = "junglescout"
    broad_search_volume: int | None = Field(
        default=None, description="Broad match search volume"
    )
    exact_search_volume: int | None = Field(
        default=None, description="Exact match search volume"
    )
    organic_product_count: int | None = Field(
        default=None, description="Number of organic products for keyword"
    )
    sponsored_product_count: int | None = Field(
        default=None, description="Number of sponsored products"
    )

    @property
    def effective_volume(self) -> int:
        """Use exact search volume as primary, fall back to broad."""
        return self.exact_search_volume or self.broad_search_volume or self.search_volume or 0


class TikTokMetrics(PlatformMetrics):
    """
    TikTok-specific engagement metrics from Apify.

    IMPORTANT: These are ENGAGEMENT metrics, NOT search volume.
    TikTok does not expose search volume data publicly.

    What this measures:
    - How much content exists about this topic
    - How much audience engagement that content receives
    - Creator activity around this keyword/hashtag

    What this does NOT measure:
    - How many people search for this term on TikTok
    - Direct demand/intent (algorithm pushes content, not user searches)
    """

    source: str = "apify_tiktok"
    metric_type: MetricType = MetricType.ENGAGEMENT
    confidence: Confidence = Confidence.PROXY
    metric_explanation: str = "Audience engagement metrics - NOT search volume. Shows content views and interactions."

    # Engagement metrics
    hashtag_views: int | None = Field(
        default=None, description="Total views on content with this hashtag (engagement, not searches)"
    )
    video_count: int | None = Field(
        default=None, description="Number of videos using hashtag (creator activity)"
    )
    avg_likes: float | None = Field(default=None, description="Average likes per video")
    avg_comments: float | None = Field(default=None, description="Average comments per video")
    avg_shares: float | None = Field(default=None, description="Average shares per video")

    @property
    def avg_engagement(self) -> float:
        """Calculate total average engagement per video."""
        likes = self.avg_likes or 0
        comments = self.avg_comments or 0
        shares = self.avg_shares or 0
        return likes + comments + shares

    @property
    def total_engagement(self) -> int:
        """Calculate total engagement (views + interactions)."""
        views = self.hashtag_views or 0
        video_count = self.video_count or 0
        avg_eng = self.avg_engagement
        return views + int(video_count * avg_eng)


class InstagramMetrics(PlatformMetrics):
    """
    Instagram-specific engagement metrics from Apify.

    IMPORTANT: These are ENGAGEMENT metrics, NOT search volume.
    Instagram does not expose search volume data publicly.

    What this measures:
    - Content creation activity (how many posts use this hashtag)
    - Community engagement (likes, comments)
    - Topic popularity among creators

    What this does NOT measure:
    - How many people search for this term on Instagram
    - Direct user intent or demand
    """

    source: str = "apify_instagram"
    metric_type: MetricType = MetricType.ENGAGEMENT
    confidence: Confidence = Confidence.PROXY
    metric_explanation: str = "Community engagement metrics - NOT search volume. Shows posting activity and interactions."

    # Engagement metrics
    post_count: int | None = Field(
        default=None, description="Total posts with hashtag (creator activity, not searches)"
    )
    daily_posts: int | None = Field(
        default=None, description="Average posts per day (content velocity)"
    )
    avg_engagement: float | None = Field(
        default=None, description="Average engagement per post (likes + comments)"
    )
    avg_likes: float | None = Field(default=None, description="Average likes per post")
    avg_comments: float | None = Field(default=None, description="Average comments per post")
    related_hashtags: list[str] | None = Field(
        default=None, description="Related/suggested hashtags"
    )

    @property
    def total_engagement(self) -> int:
        """Calculate total daily engagement."""
        daily = self.daily_posts or 0
        avg_eng = self.avg_engagement or 0
        return int(daily * avg_eng)


class PinterestMetrics(PlatformMetrics):
    """
    Pinterest-specific metrics from Pinterest Trends.

    NOTE: Pinterest provides a relative interest index (0-100), not absolute search volume.
    The interest_score shows relative popularity compared to other terms on Pinterest.

    What this measures:
    - Relative topic interest on Pinterest (0-100 scale)
    - Pin creation activity
    - Trend direction

    For more accurate data, Pinterest Ads API can provide search volume ranges
    (requires Pinterest Business account).
    """

    source: str = "pinterest_trends"
    metric_type: MetricType = MetricType.INTEREST_INDEX
    confidence: Confidence = Confidence.PROXY
    metric_explanation: str = "Relative interest index (0-100) - shows popularity compared to other Pinterest topics."

    # Interest metrics (not absolute volume)
    pinterest_interest_score: int | None = Field(
        default=None, description="Pinterest interest index (0-100, relative not absolute)"
    )
    pin_count: int | None = Field(
        default=None, description="Number of pins for this term (content supply)"
    )
    monthly_searches_estimate: int | None = Field(
        default=None, description="Rough estimate based on interest score - use with caution"
    )
    is_trending: bool = Field(
        default=False, description="Whether the term is currently trending on Pinterest"
    )
    related_terms: list[str] | None = Field(
        default=None, description="Related search terms"
    )

    # Override the base interest_score to use pinterest_interest_score
    @property
    def interest_score(self) -> int | None:
        """Get the interest score."""
        return self.pinterest_interest_score


class PlatformScore(BaseModel):
    """Normalized platform score for unified calculations."""

    platform: Platform
    raw_volume: int
    normalized_score: float = Field(ge=0, le=100, description="Score normalized to 0-100")
    weight: float = Field(ge=0, le=1, description="Platform weight in unified score")
    weighted_score: float = Field(description="normalized_score * weight")


class UnifiedKeywordData(BaseModel):
    """Unified keyword data aggregating all platform metrics."""

    keyword: str = Field(description="The keyword being analyzed")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Platform-specific metrics
    google: GoogleMetrics | None = None
    youtube: YouTubeMetrics | None = None
    amazon: AmazonMetrics | None = None
    tiktok: TikTokMetrics | None = None
    instagram: InstagramMetrics | None = None
    pinterest: "PinterestMetrics | None" = None

    # Unified metrics
    unified_demand_score: int = Field(
        default=0, ge=0, le=100, description="Unified demand score (0-100)"
    )
    cross_platform_trend: TrendDirection = Field(
        default=TrendDirection.STABLE, description="Overall trend across platforms"
    )
    best_platform: Platform | None = Field(
        default=None, description="Platform with highest relative demand"
    )
    platform_scores: list[PlatformScore] | None = Field(
        default=None, description="Individual platform scores"
    )

    # Metadata
    tags: list[str] | None = Field(default=None, description="Categorization tags")
    processing_errors: list[str] | None = Field(
        default=None, description="Errors encountered during processing"
    )

    @property
    def platforms_dict(self) -> dict[Platform, PlatformMetrics | None]:
        """Get all platform metrics as a dictionary."""
        return {
            Platform.GOOGLE: self.google,
            Platform.YOUTUBE: self.youtube,
            Platform.AMAZON: self.amazon,
            Platform.TIKTOK: self.tiktok,
            Platform.INSTAGRAM: self.instagram,
            Platform.PINTEREST: self.pinterest,
        }

    @property
    def available_platforms(self) -> list[Platform]:
        """Get list of platforms with data."""
        return [p for p, m in self.platforms_dict.items() if m is not None]

    def get_platform_metrics(self, platform: Platform) -> PlatformMetrics | None:
        """Get metrics for a specific platform."""
        return self.platforms_dict.get(platform)

    def to_rag_document(self) -> dict[str, Any]:
        """Convert to a format suitable for RAG ingestion."""
        doc = {
            "keyword": self.keyword,
            "unified_demand_score": self.unified_demand_score,
            "cross_platform_trend": self.cross_platform_trend.value,
            "best_platform": self.best_platform.value if self.best_platform else None,
            "platforms": {},
            "search_demand": {},  # Verified search volume platforms
            "engagement_metrics": {},  # Engagement/interest platforms
            "collected_at": self.timestamp.isoformat(),
        }

        for platform, metrics in self.platforms_dict.items():
            if metrics:
                platform_data = {
                    "value": metrics.effective_volume,
                    "metric_type": metrics.metric_type.value,
                    "trend": metrics.trend.value if metrics.trend else None,
                    "competition": metrics.competition.value if metrics.competition else None,
                    "confidence": metrics.confidence.value,
                    "is_verified_search": metrics.is_verified_search_data,
                    "explanation": metrics.metric_explanation,
                }
                doc["platforms"][platform.value] = platform_data

                # Categorize by metric type for easier consumption
                if metrics.metric_type == MetricType.SEARCH_VOLUME:
                    doc["search_demand"][platform.value] = metrics.effective_volume
                else:
                    doc["engagement_metrics"][platform.value] = metrics.effective_volume

        # Calculate totals
        doc["total_verified_search_volume"] = sum(doc["search_demand"].values())
        doc["total_engagement_score"] = sum(doc["engagement_metrics"].values())

        return doc
