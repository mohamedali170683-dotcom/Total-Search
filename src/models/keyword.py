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

    HIGH = "high"
    MEDIUM = "medium"
    PROXY = "proxy"


class Platform(str, Enum):
    """Supported platforms."""

    GOOGLE = "google"
    YOUTUBE = "youtube"
    AMAZON = "amazon"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"


class PlatformMetrics(BaseModel):
    """Base model for platform-specific keyword metrics."""

    search_volume: int | None = Field(
        default=None, description="Actual search volume (Google, YouTube, Amazon)"
    )
    proxy_score: int | None = Field(
        default=None, description="Calculated proxy score (TikTok, Instagram)"
    )
    trend: TrendDirection | None = Field(default=None, description="Trend direction")
    trend_velocity: float | None = Field(
        default=None, description="Trend velocity multiplier (>1 = growing)"
    )
    competition: Competition | None = Field(default=None, description="Competition level")
    cpc: float | None = Field(default=None, description="Cost per click (where available)")
    confidence: Confidence = Field(default=Confidence.HIGH, description="Confidence level")
    source: str = Field(description="API source name")
    raw_data: dict[str, Any] | None = Field(
        default=None, description="Original API response data"
    )
    collected_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def effective_volume(self) -> int:
        """Get the effective volume (search_volume or proxy_score)."""
        return self.search_volume or self.proxy_score or 0


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
    """TikTok-specific hashtag metrics from Apify."""

    source: str = "apify_tiktok"
    confidence: Confidence = Confidence.PROXY
    hashtag_views: int | None = Field(
        default=None, description="Total hashtag views (all-time)"
    )
    video_count: int | None = Field(
        default=None, description="Number of videos using hashtag"
    )
    avg_likes: float | None = Field(default=None, description="Average likes per video")
    avg_comments: float | None = Field(default=None, description="Average comments per video")
    avg_shares: float | None = Field(default=None, description="Average shares per video")

    @property
    def avg_engagement(self) -> float:
        """Calculate total average engagement."""
        likes = self.avg_likes or 0
        comments = self.avg_comments or 0
        shares = self.avg_shares or 0
        return likes + comments + shares


class InstagramMetrics(PlatformMetrics):
    """Instagram-specific hashtag metrics from Apify."""

    source: str = "apify_instagram"
    confidence: Confidence = Confidence.PROXY
    post_count: int | None = Field(
        default=None, description="Total posts with hashtag (all-time)"
    )
    daily_posts: int | None = Field(
        default=None, description="Average posts per day"
    )
    avg_engagement: float | None = Field(
        default=None, description="Average engagement per post"
    )
    avg_likes: float | None = Field(default=None, description="Average likes per post")
    avg_comments: float | None = Field(default=None, description="Average comments per post")
    related_hashtags: list[str] | None = Field(
        default=None, description="Related/suggested hashtags"
    )


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
            "collected_at": self.timestamp.isoformat(),
        }

        for platform, metrics in self.platforms_dict.items():
            if metrics:
                doc["platforms"][platform.value] = {
                    "volume": metrics.effective_volume,
                    "trend": metrics.trend.value if metrics.trend else None,
                    "competition": metrics.competition.value if metrics.competition else None,
                    "confidence": metrics.confidence.value,
                }

        return doc
