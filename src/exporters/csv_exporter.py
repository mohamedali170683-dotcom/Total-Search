"""CSV exporter for keyword data."""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models.keyword import UnifiedKeywordData, Platform

logger = logging.getLogger(__name__)


class CSVExporter:
    """
    Export keyword data to CSV format for analysis and spreadsheet use.

    Supports:
    - Flat CSV with all metrics in columns
    - Summary CSV with key metrics only
    - Platform-specific CSV exports
    """

    def __init__(self, output_dir: str | Path = "data/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_full(
        self,
        keywords: list[UnifiedKeywordData],
        filename: str | None = None,
    ) -> Path:
        """
        Export full keyword data to CSV with all platform metrics.

        Columns include:
        - keyword, unified_score, trend, best_platform
        - google_volume, google_trend, google_competition, google_cpc
        - youtube_volume, youtube_trend
        - amazon_volume, amazon_competition
        - tiktok_proxy, tiktok_views, tiktok_videos
        - instagram_proxy, instagram_posts, instagram_daily

        Args:
            keywords: List of unified keyword data
            filename: Output filename (auto-generated if None)

        Returns:
            Path to the exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keywords_full_{timestamp}.csv"

        output_path = self.output_dir / filename

        # Define all columns
        columns = [
            "keyword",
            "unified_demand_score",
            "cross_platform_trend",
            "best_platform",
            "collected_at",
            "tags",
            # Google
            "google_volume",
            "google_trend",
            "google_competition",
            "google_cpc",
            "google_confidence",
            # YouTube
            "youtube_volume",
            "youtube_trend",
            "youtube_competition",
            "youtube_confidence",
            # Amazon
            "amazon_volume",
            "amazon_exact_volume",
            "amazon_broad_volume",
            "amazon_competition",
            "amazon_confidence",
            # TikTok
            "tiktok_proxy_score",
            "tiktok_hashtag_views",
            "tiktok_video_count",
            "tiktok_avg_likes",
            "tiktok_trend",
            "tiktok_confidence",
            # Instagram
            "instagram_proxy_score",
            "instagram_post_count",
            "instagram_daily_posts",
            "instagram_avg_engagement",
            "instagram_trend",
            "instagram_confidence",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for kw in keywords:
                row = self._to_full_row(kw)
                writer.writerow(row)

        logger.info(f"Exported {len(keywords)} keywords to {output_path}")
        return output_path

    def export_summary(
        self,
        keywords: list[UnifiedKeywordData],
        filename: str | None = None,
    ) -> Path:
        """
        Export summary CSV with key metrics only.

        Columns: keyword, score, trend, best_platform, total_volume

        Args:
            keywords: List of unified keyword data
            filename: Output filename

        Returns:
            Path to the exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keywords_summary_{timestamp}.csv"

        output_path = self.output_dir / filename

        columns = [
            "keyword",
            "unified_demand_score",
            "cross_platform_trend",
            "best_platform",
            "total_effective_volume",
            "platforms_with_data",
            "collected_at",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for kw in keywords:
                row = self._to_summary_row(kw)
                writer.writerow(row)

        logger.info(f"Exported {len(keywords)} keywords (summary) to {output_path}")
        return output_path

    def export_by_platform(
        self,
        keywords: list[UnifiedKeywordData],
        platform: Platform,
        filename: str | None = None,
    ) -> Path:
        """
        Export CSV for a specific platform only.

        Args:
            keywords: List of unified keyword data
            platform: Platform to export
            filename: Output filename

        Returns:
            Path to the exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keywords_{platform.value}_{timestamp}.csv"

        output_path = self.output_dir / filename

        # Platform-specific columns
        base_columns = ["keyword", "unified_demand_score"]
        platform_columns = self._get_platform_columns(platform)

        columns = base_columns + platform_columns

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for kw in keywords:
                metrics = kw.platforms.get(platform)
                if metrics:
                    row = self._to_platform_row(kw, platform, metrics)
                    writer.writerow(row)

        logger.info(f"Exported {platform.value} data for {len(keywords)} keywords to {output_path}")
        return output_path

    def _to_full_row(self, kw: UnifiedKeywordData) -> dict[str, Any]:
        """Convert keyword to full CSV row."""
        row = {
            "keyword": kw.keyword,
            "unified_demand_score": kw.unified_demand_score,
            "cross_platform_trend": kw.cross_platform_trend.value if kw.cross_platform_trend else "",
            "best_platform": kw.best_platform.value if kw.best_platform else "",
            "collected_at": kw.collected_at.isoformat() if kw.collected_at else "",
            "tags": ",".join(kw.tags) if kw.tags else "",
        }

        # Google metrics
        google = kw.platforms.get(Platform.GOOGLE)
        if google:
            row.update({
                "google_volume": google.search_volume or "",
                "google_trend": google.trend.value if google.trend else "",
                "google_competition": google.competition.value if google.competition else "",
                "google_cpc": google.cpc or "",
                "google_confidence": google.confidence.value if google.confidence else "",
            })

        # YouTube metrics
        youtube = kw.platforms.get(Platform.YOUTUBE)
        if youtube:
            row.update({
                "youtube_volume": youtube.search_volume or "",
                "youtube_trend": youtube.trend.value if youtube.trend else "",
                "youtube_competition": youtube.competition.value if youtube.competition else "",
                "youtube_confidence": youtube.confidence.value if youtube.confidence else "",
            })

        # Amazon metrics
        amazon = kw.platforms.get(Platform.AMAZON)
        if amazon:
            row.update({
                "amazon_volume": amazon.search_volume or "",
                "amazon_exact_volume": amazon.exact_search_volume or "",
                "amazon_broad_volume": amazon.broad_search_volume or "",
                "amazon_competition": amazon.competition.value if amazon.competition else "",
                "amazon_confidence": amazon.confidence.value if amazon.confidence else "",
            })

        # TikTok metrics
        tiktok = kw.platforms.get(Platform.TIKTOK)
        if tiktok:
            row.update({
                "tiktok_proxy_score": tiktok.proxy_score or "",
                "tiktok_hashtag_views": tiktok.hashtag_views or "",
                "tiktok_video_count": tiktok.video_count or "",
                "tiktok_avg_likes": tiktok.avg_likes or "",
                "tiktok_trend": tiktok.trend.value if tiktok.trend else "",
                "tiktok_confidence": tiktok.confidence.value if tiktok.confidence else "",
            })

        # Instagram metrics
        instagram = kw.platforms.get(Platform.INSTAGRAM)
        if instagram:
            row.update({
                "instagram_proxy_score": instagram.proxy_score or "",
                "instagram_post_count": instagram.post_count or "",
                "instagram_daily_posts": instagram.daily_posts or "",
                "instagram_avg_engagement": instagram.avg_engagement or "",
                "instagram_trend": instagram.trend.value if instagram.trend else "",
                "instagram_confidence": instagram.confidence.value if instagram.confidence else "",
            })

        return row

    def _to_summary_row(self, kw: UnifiedKeywordData) -> dict[str, Any]:
        """Convert keyword to summary CSV row."""
        # Calculate total effective volume across platforms
        total_volume = sum(
            metrics.effective_volume
            for metrics in kw.platforms.values()
            if metrics and metrics.effective_volume
        )

        # Count platforms with data
        platforms_with_data = sum(
            1 for metrics in kw.platforms.values()
            if metrics and metrics.effective_volume and metrics.effective_volume > 0
        )

        return {
            "keyword": kw.keyword,
            "unified_demand_score": kw.unified_demand_score,
            "cross_platform_trend": kw.cross_platform_trend.value if kw.cross_platform_trend else "",
            "best_platform": kw.best_platform.value if kw.best_platform else "",
            "total_effective_volume": total_volume,
            "platforms_with_data": platforms_with_data,
            "collected_at": kw.collected_at.isoformat() if kw.collected_at else "",
        }

    def _get_platform_columns(self, platform: Platform) -> list[str]:
        """Get columns specific to a platform."""
        columns_map = {
            Platform.GOOGLE: ["volume", "trend", "competition", "cpc", "confidence"],
            Platform.YOUTUBE: ["volume", "trend", "competition", "confidence"],
            Platform.AMAZON: ["volume", "exact_volume", "broad_volume", "competition", "confidence"],
            Platform.TIKTOK: ["proxy_score", "hashtag_views", "video_count", "avg_likes", "trend", "confidence"],
            Platform.INSTAGRAM: ["proxy_score", "post_count", "daily_posts", "avg_engagement", "trend", "confidence"],
        }
        return columns_map.get(platform, [])

    def _to_platform_row(self, kw: UnifiedKeywordData, platform: Platform, metrics: Any) -> dict[str, Any]:
        """Convert keyword to platform-specific CSV row."""
        row = {
            "keyword": kw.keyword,
            "unified_demand_score": kw.unified_demand_score,
        }

        if platform == Platform.GOOGLE:
            row.update({
                "volume": metrics.search_volume or "",
                "trend": metrics.trend.value if metrics.trend else "",
                "competition": metrics.competition.value if metrics.competition else "",
                "cpc": metrics.cpc or "",
                "confidence": metrics.confidence.value if metrics.confidence else "",
            })
        elif platform == Platform.YOUTUBE:
            row.update({
                "volume": metrics.search_volume or "",
                "trend": metrics.trend.value if metrics.trend else "",
                "competition": metrics.competition.value if metrics.competition else "",
                "confidence": metrics.confidence.value if metrics.confidence else "",
            })
        elif platform == Platform.AMAZON:
            row.update({
                "volume": metrics.search_volume or "",
                "exact_volume": metrics.exact_search_volume or "",
                "broad_volume": metrics.broad_search_volume or "",
                "competition": metrics.competition.value if metrics.competition else "",
                "confidence": metrics.confidence.value if metrics.confidence else "",
            })
        elif platform == Platform.TIKTOK:
            row.update({
                "proxy_score": metrics.proxy_score or "",
                "hashtag_views": metrics.hashtag_views or "",
                "video_count": metrics.video_count or "",
                "avg_likes": metrics.avg_likes or "",
                "trend": metrics.trend.value if metrics.trend else "",
                "confidence": metrics.confidence.value if metrics.confidence else "",
            })
        elif platform == Platform.INSTAGRAM:
            row.update({
                "proxy_score": metrics.proxy_score or "",
                "post_count": metrics.post_count or "",
                "daily_posts": metrics.daily_posts or "",
                "avg_engagement": metrics.avg_engagement or "",
                "trend": metrics.trend.value if metrics.trend else "",
                "confidence": metrics.confidence.value if metrics.confidence else "",
            })

        return row
