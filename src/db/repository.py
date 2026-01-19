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
from src.models.keyword import Platform, PlatformMetrics, UnifiedKeywordData

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
            "search_volume": metrics.search_volume,
            "proxy_score": metrics.proxy_score,
            "trend": metrics.trend.value if metrics.trend else None,
            "trend_velocity": metrics.trend_velocity,
            "competition": metrics.competition.value if metrics.competition else None,
            "cpc": metrics.cpc,
            "confidence": metrics.confidence.value,
            "raw_data": metrics.raw_data,
        }

        # Build the update set for upsert
        update_set = {
            "search_volume": values["search_volume"],
            "proxy_score": values["proxy_score"],
            "trend": values["trend"],
            "trend_velocity": values["trend_velocity"],
            "competition": values["competition"],
            "cpc": values["cpc"],
            "confidence": values["confidence"],
            "raw_data": values["raw_data"],
            "collected_at": datetime.utcnow(),
        }

        if self._is_postgres:
            # PostgreSQL upsert using constraint name
            stmt = pg_insert(KeywordMetric).values(**values)
            stmt = stmt.on_conflict_do_update(
                constraint="idx_keyword_metrics_unique",
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
            # PostgreSQL upsert using constraint name
            stmt = pg_insert(UnifiedScore).values(**values)
            stmt = stmt.on_conflict_do_update(
                constraint="idx_unified_scores_unique",
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
