"""Database repository for keyword data operations."""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings, get_settings
from src.db.models import Base, Keyword, KeywordMetric, UnifiedScore
from src.models.keyword import MetricType, Platform, PlatformMetrics, UnifiedKeywordData

logger = logging.getLogger(__name__)


class KeywordRepository:
    """Repository for keyword database operations."""

    def __init__(self, database_url: str | None = None, settings: Settings | None = None):
        settings = settings or get_settings()
        self.database_url = database_url or settings.database_url

        # Fix for Vercel/Heroku: convert postgres:// to postgresql://
        # SQLAlchemy 1.4+ requires postgresql:// scheme
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)

        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._is_postgres = self.database_url.startswith("postgresql")

    def create_tables(self) -> None:
        """Create all database tables if they don't exist."""
        Base.metadata.create_all(self.engine, checkfirst=True)
        logger.info("Database tables created/verified")

    def drop_tables(self) -> None:
        """Drop all database tables."""
        Base.metadata.drop_all(self.engine)
        logger.info("Database tables dropped")

    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()

    def _get_effective_volume(self, metric: KeywordMetric) -> int:
        """
        Get the effective volume value from a metric based on its type.

        This handles the different metric types:
        - search_volume: For Google, YouTube, Amazon (verified search data)
        - engagement_score: For TikTok, Instagram (audience engagement)
        - interest_score: For Pinterest (relative interest 0-100)
        - proxy_score: Deprecated, for backward compatibility
        """
        metric_type = metric.metric_type or "search_volume"

        if metric_type == "search_volume":
            return metric.search_volume or 0
        elif metric_type == "engagement":
            return metric.engagement_score or metric.proxy_score or 0
        elif metric_type == "interest_index":
            return metric.interest_score or 0
        else:
            # Fallback for backward compatibility
            return metric.search_volume or metric.engagement_score or metric.proxy_score or 0

    def save_keyword_data(
        self,
        keyword_data: UnifiedKeywordData,
        tags: list[str] | None = None,
    ) -> int:
        """
        Save unified keyword data to the database.

        Args:
            keyword_data: UnifiedKeywordData to save
            tags: Optional tags for categorization

        Returns:
            Keyword ID
        """
        with self.get_session() as session:
            # Get or create keyword
            keyword = self._get_or_create_keyword(
                session, keyword_data.keyword, tags
            )

            # Save platform metrics
            for platform, metrics in keyword_data.platforms_dict.items():
                if metrics is not None:
                    self._save_platform_metrics(session, keyword.id, platform, metrics)

            # Save unified score
            self._save_unified_score(session, keyword.id, keyword_data)

            session.commit()
            return keyword.id

    def save_batch(
        self,
        keyword_data_list: list[UnifiedKeywordData],
        tags: list[str] | None = None,
    ) -> list[int]:
        """
        Save a batch of keyword data efficiently.

        Args:
            keyword_data_list: List of UnifiedKeywordData to save
            tags: Optional tags for all keywords

        Returns:
            List of keyword IDs
        """
        keyword_ids = []

        with self.get_session() as session:
            for kw_data in keyword_data_list:
                try:
                    keyword = self._get_or_create_keyword(
                        session, kw_data.keyword, tags
                    )

                    for platform, metrics in kw_data.platforms_dict.items():
                        if metrics is not None:
                            self._save_platform_metrics(
                                session, keyword.id, platform, metrics
                            )

                    self._save_unified_score(session, keyword.id, kw_data)
                    keyword_ids.append(keyword.id)

                except Exception as e:
                    logger.error(f"Failed to save keyword '{kw_data.keyword}': {e}")
                    continue

            session.commit()

        return keyword_ids

    def _get_or_create_keyword(
        self,
        session: Session,
        keyword: str,
        tags: list[str] | None = None,
    ) -> Keyword:
        """Get existing keyword or create new one."""
        import json

        stmt = select(Keyword).where(Keyword.keyword == keyword)
        existing = session.execute(stmt).scalar_one_or_none()

        if existing:
            existing.updated_at = datetime.utcnow()
            if tags:
                existing.tags = json.dumps(tags)
            return existing

        tags_json = json.dumps(tags) if tags else None
        new_keyword = Keyword(keyword=keyword, tags=tags_json)
        session.add(new_keyword)
        session.flush()  # Get the ID
        return new_keyword

    def _save_platform_metrics(
        self,
        session: Session,
        keyword_id: int,
        platform: Platform,
        metrics: PlatformMetrics,
    ) -> None:
        """Save platform-specific metrics using upsert."""
        collected_date = datetime.utcnow().strftime("%Y-%m-%d")

        values = {
            "keyword_id": keyword_id,
            "platform": platform.value,
            "collected_date": collected_date,
            # Metric type classification for honest labeling
            "metric_type": metrics.metric_type.value if metrics.metric_type else MetricType.SEARCH_VOLUME.value,
            # Primary metrics by type
            "search_volume": metrics.search_volume,
            "engagement_score": metrics.engagement_score,
            "interest_score": metrics.interest_score,
            # Deprecated - keep for backward compatibility
            "proxy_score": metrics.proxy_score,
            # Metadata
            "trend": metrics.trend.value if metrics.trend else None,
            "trend_velocity": metrics.trend_velocity,
            "competition": metrics.competition.value if metrics.competition else None,
            "cpc": metrics.cpc,
            "confidence": metrics.confidence.value,
            "metric_explanation": metrics.metric_explanation,
            "raw_data": metrics.raw_data,
        }

        # Build the update set for upsert
        update_set = {
            "metric_type": values["metric_type"],
            "search_volume": values["search_volume"],
            "engagement_score": values["engagement_score"],
            "interest_score": values["interest_score"],
            "proxy_score": values["proxy_score"],
            "trend": values["trend"],
            "trend_velocity": values["trend_velocity"],
            "competition": values["competition"],
            "cpc": values["cpc"],
            "confidence": values["confidence"],
            "metric_explanation": values["metric_explanation"],
            "raw_data": values["raw_data"],
            "collected_at": datetime.utcnow(),
        }

        if self._is_postgres:
            # PostgreSQL upsert using index elements (not constraint name)
            stmt = pg_insert(KeywordMetric).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["keyword_id", "platform", "collected_date"],
                set_=update_set,
            )
        else:
            # SQLite upsert
            stmt = sqlite_insert(KeywordMetric).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["keyword_id", "platform", "collected_date"],
                set_=update_set,
            )

        session.execute(stmt)

    def _save_unified_score(
        self,
        session: Session,
        keyword_id: int,
        keyword_data: UnifiedKeywordData,
    ) -> None:
        """Save unified score data using upsert."""
        platform_scores_json = None
        if keyword_data.platform_scores:
            platform_scores_json = [
                {
                    "platform": ps.platform.value,
                    "raw_volume": ps.raw_volume,
                    "normalized_score": ps.normalized_score,
                    "weight": ps.weight,
                    "weighted_score": ps.weighted_score,
                }
                for ps in keyword_data.platform_scores
            ]

        collected_date = datetime.utcnow().strftime("%Y-%m-%d")

        values = {
            "keyword_id": keyword_id,
            "collected_date": collected_date,
            "unified_demand_score": keyword_data.unified_demand_score,
            "cross_platform_trend": keyword_data.cross_platform_trend.value if keyword_data.cross_platform_trend else None,
            "best_platform": keyword_data.best_platform.value if keyword_data.best_platform else None,
            "platform_scores": platform_scores_json,
        }

        # Build the update set for upsert
        update_set = {
            "unified_demand_score": values["unified_demand_score"],
            "cross_platform_trend": values["cross_platform_trend"],
            "best_platform": values["best_platform"],
            "platform_scores": values["platform_scores"],
            "collected_at": datetime.utcnow(),
        }

        if self._is_postgres:
            # PostgreSQL upsert using index elements (not constraint name)
            stmt = pg_insert(UnifiedScore).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["keyword_id", "collected_date"],
                set_=update_set,
            )
        else:
            # SQLite upsert
            stmt = sqlite_insert(UnifiedScore).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["keyword_id", "collected_date"],
                set_=update_set,
            )

        session.execute(stmt)

    def get_keyword(self, keyword: str) -> UnifiedKeywordData | None:
        """
        Get the latest data for a keyword.

        Args:
            keyword: Keyword to retrieve

        Returns:
            UnifiedKeywordData or None if not found
        """
        with self.get_session() as session:
            stmt = select(Keyword).where(Keyword.keyword == keyword)
            kw = session.execute(stmt).scalar_one_or_none()

            if not kw:
                return None

            return self._build_keyword_data(session, kw)

    def get_keywords_by_tag(self, tag: str) -> list[UnifiedKeywordData]:
        """
        Get all keywords with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of UnifiedKeywordData
        """
        with self.get_session() as session:
            # For SQLite, tags are stored as JSON string, use LIKE for search
            stmt = select(Keyword).where(Keyword.tags.like(f'%"{tag}"%'))
            keywords = session.execute(stmt).scalars().all()

            return [self._build_keyword_data(session, kw) for kw in keywords]

    def get_all_keywords(
        self,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[UnifiedKeywordData]:
        """
        Get all keywords with pagination.

        Args:
            limit: Maximum number of keywords to return
            offset: Number of keywords to skip

        Returns:
            List of UnifiedKeywordData
        """
        with self.get_session() as session:
            stmt = select(Keyword).limit(limit).offset(offset)
            keywords = session.execute(stmt).scalars().all()

            return [self._build_keyword_data(session, kw) for kw in keywords]

    def _build_keyword_data(
        self,
        session: Session,
        keyword: Keyword,
    ) -> UnifiedKeywordData:
        """Build UnifiedKeywordData from database records."""
        # Get latest metrics for each platform
        latest_metrics: dict[str, KeywordMetric] = {}
        for metric in keyword.metrics:
            platform = metric.platform
            if platform not in latest_metrics or metric.collected_at > latest_metrics[platform].collected_at:
                latest_metrics[platform] = metric

        # Get latest unified score
        latest_unified = None
        for unified in keyword.unified_scores:
            if latest_unified is None or unified.collected_at > latest_unified.collected_at:
                latest_unified = unified

        # Build the data object
        data = UnifiedKeywordData(
            keyword=keyword.keyword,
            tags=keyword.tags,
            timestamp=keyword.updated_at,
        )

        # Add platform metrics (simplified - just storing raw_data reference)
        # In production, you'd deserialize the raw_data back to proper metric objects

        if latest_unified:
            data.unified_demand_score = latest_unified.unified_demand_score
            if latest_unified.cross_platform_trend:
                from src.models.keyword import TrendDirection
                data.cross_platform_trend = TrendDirection(latest_unified.cross_platform_trend)
            if latest_unified.best_platform:
                data.best_platform = Platform(latest_unified.best_platform)

        return data

    def export_for_rag(
        self,
        output_format: str = "jsonl",
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Export keyword data in a format suitable for RAG ingestion.

        Args:
            output_format: Output format (jsonl, csv)
            since: Only export data updated since this datetime

        Returns:
            List of dictionaries ready for export
        """
        with self.get_session() as session:
            stmt = select(Keyword)
            if since:
                stmt = stmt.where(Keyword.updated_at >= since)

            keywords = session.execute(stmt).scalars().all()
            export_data = []

            for kw in keywords:
                data = self._build_keyword_data(session, kw)
                export_data.append(data.to_rag_document())

            return export_data

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        with self.get_session() as session:
            from sqlalchemy import func

            keyword_count = session.execute(
                select(func.count(Keyword.id))
            ).scalar()

            metric_count = session.execute(
                select(func.count(KeywordMetric.id))
            ).scalar()

            # Platform distribution
            platform_counts = session.execute(
                select(KeywordMetric.platform, func.count(KeywordMetric.id))
                .group_by(KeywordMetric.platform)
            ).all()

            return {
                "total_keywords": keyword_count,
                "total_metrics": metric_count,
                "metrics_by_platform": {p: c for p, c in platform_counts},
            }

    def get_keyword_history(
        self,
        keyword: str,
        days: int = 90,
    ) -> dict[str, Any]:
        """
        Get historical data for a keyword across all platforms.

        Args:
            keyword: Keyword to get history for
            days: Number of days of history to retrieve

        Returns:
            Dictionary with historical data by platform and date
        """
        from datetime import timedelta

        with self.get_session() as session:
            stmt = select(Keyword).where(Keyword.keyword == keyword)
            kw = session.execute(stmt).scalar_one_or_none()

            if not kw:
                return {"keyword": keyword, "history": {}, "error": "Keyword not found"}

            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Get all metrics since cutoff
            metrics_stmt = (
                select(KeywordMetric)
                .where(KeywordMetric.keyword_id == kw.id)
                .where(KeywordMetric.collected_at >= cutoff_date)
                .order_by(KeywordMetric.collected_at)
            )
            metrics = session.execute(metrics_stmt).scalars().all()

            # Organize by platform and date
            history: dict[str, list[dict]] = {}
            for metric in metrics:
                platform = metric.platform
                if platform not in history:
                    history[platform] = []

                # Get the appropriate metric value based on type
                volume = self._get_effective_volume(metric)
                history[platform].append({
                    "date": metric.collected_date,
                    "volume": volume,
                    "metric_type": metric.metric_type or "search_volume",
                    "trend": metric.trend,
                    "trend_velocity": metric.trend_velocity,
                })

            # Get unified score history
            unified_stmt = (
                select(UnifiedScore)
                .where(UnifiedScore.keyword_id == kw.id)
                .where(UnifiedScore.collected_at >= cutoff_date)
                .order_by(UnifiedScore.collected_at)
            )
            unified_scores = session.execute(unified_stmt).scalars().all()

            unified_history = [
                {
                    "date": us.collected_date,
                    "score": us.unified_demand_score,
                    "trend": us.cross_platform_trend,
                    "best_platform": us.best_platform,
                }
                for us in unified_scores
            ]

            return {
                "keyword": keyword,
                "history": history,
                "unified_history": unified_history,
                "days": days,
            }

    def get_trending_keywords(
        self,
        platform: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Get keywords with the highest growth trends.

        Args:
            platform: Optional platform filter
            limit: Maximum results

        Returns:
            List of trending keywords with metrics
        """
        with self.get_session() as session:
            from sqlalchemy import desc

            # Get keywords with growing trends
            stmt = (
                select(KeywordMetric, Keyword.keyword)
                .join(Keyword, KeywordMetric.keyword_id == Keyword.id)
                .where(KeywordMetric.trend == "growing")
            )

            if platform:
                stmt = stmt.where(KeywordMetric.platform == platform)

            stmt = stmt.order_by(desc(KeywordMetric.trend_velocity)).limit(limit)

            results = session.execute(stmt).all()

            return [
                {
                    "keyword": keyword,
                    "platform": metric.platform,
                    "volume": self._get_effective_volume(metric),
                    "metric_type": metric.metric_type or "search_volume",
                    "trend_velocity": metric.trend_velocity,
                    "collected_at": metric.collected_at.isoformat() if metric.collected_at else None,
                }
                for metric, keyword in results
            ]

    def get_demand_over_time(
        self,
        keywords: list[str],
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get aggregated demand over time for a set of keywords.

        Useful for tracking brand or category demand trends.

        Args:
            keywords: Keywords to aggregate
            days: Number of days of history

        Returns:
            Time series of total demand by platform
        """
        from collections import defaultdict
        from datetime import timedelta

        with self.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Get keyword IDs
            keyword_ids = []
            for kw in keywords:
                stmt = select(Keyword.id).where(Keyword.keyword == kw)
                kw_id = session.execute(stmt).scalar_one_or_none()
                if kw_id:
                    keyword_ids.append(kw_id)

            if not keyword_ids:
                return {"keywords": keywords, "time_series": {}, "error": "No keywords found"}

            # Get metrics for all keywords
            from sqlalchemy import or_
            metrics_stmt = (
                select(KeywordMetric)
                .where(KeywordMetric.keyword_id.in_(keyword_ids))
                .where(KeywordMetric.collected_at >= cutoff_date)
                .order_by(KeywordMetric.collected_date)
            )
            metrics = session.execute(metrics_stmt).scalars().all()

            # Aggregate by date and platform
            time_series: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

            for metric in metrics:
                date = metric.collected_date
                platform = metric.platform
                volume = self._get_effective_volume(metric)
                time_series[date][platform] += volume

            # Convert to sorted list
            sorted_series = []
            for date in sorted(time_series.keys()):
                entry = {"date": date, "platforms": dict(time_series[date])}
                entry["total"] = sum(time_series[date].values())
                sorted_series.append(entry)

            return {
                "keywords": keywords,
                "time_series": sorted_series,
                "days": days,
            }
