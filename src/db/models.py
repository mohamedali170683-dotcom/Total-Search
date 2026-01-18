"""SQLAlchemy database models."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Keyword(Base):
    """Keywords table - stores unique keywords."""

    __tablename__ = "keywords"

    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword = Column(String(500), nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    tags = Column(Text, nullable=True)  # JSON-encoded list for SQLite compatibility

    # Relationships
    metrics = relationship("KeywordMetric", back_populates="keyword_rel", cascade="all, delete-orphan")
    unified_scores = relationship("UnifiedScore", back_populates="keyword_rel", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Keyword(id={self.id}, keyword='{self.keyword}')>"

    def get_tags(self) -> list[str]:
        """Get tags as a list."""
        import json
        if self.tags:
            return json.loads(self.tags)
        return []

    def set_tags(self, tags: list[str]) -> None:
        """Set tags from a list."""
        import json
        self.tags = json.dumps(tags) if tags else None


class KeywordMetric(Base):
    """Platform metrics table - stores historical data per platform."""

    __tablename__ = "keyword_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword_id = Column(Integer, ForeignKey("keywords.id", ondelete="CASCADE"), nullable=False)
    platform = Column(String(50), nullable=False)  # google, youtube, amazon, tiktok, instagram
    collected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    collected_date = Column(String(10), nullable=False)  # YYYY-MM-DD for unique constraint

    # Common metrics
    search_volume = Column(Integer, nullable=True)
    proxy_score = Column(Integer, nullable=True)
    trend = Column(String(20), nullable=True)  # growing, stable, declining
    trend_velocity = Column(Float, nullable=True)
    competition = Column(String(20), nullable=True)  # low, medium, high
    cpc = Column(Float, nullable=True)
    confidence = Column(String(20), nullable=True)  # high, medium, proxy

    # Platform-specific data (stored as JSON)
    raw_data = Column(JSON, nullable=True)

    # Relationships
    keyword_rel = relationship("Keyword", back_populates="metrics")

    __table_args__ = (
        Index("idx_keyword_metrics_keyword_platform", "keyword_id", "platform"),
        Index("idx_keyword_metrics_collected_at", "collected_at"),
        Index("idx_keyword_metrics_unique", "keyword_id", "platform", "collected_date", unique=True),
    )

    def __repr__(self) -> str:
        return f"<KeywordMetric(id={self.id}, keyword_id={self.keyword_id}, platform='{self.platform}')>"


class UnifiedScore(Base):
    """Unified scores table - stores calculated cross-platform scores."""

    __tablename__ = "unified_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword_id = Column(Integer, ForeignKey("keywords.id", ondelete="CASCADE"), nullable=False)
    collected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    collected_date = Column(String(10), nullable=False)  # YYYY-MM-DD for unique constraint

    # Unified metrics
    unified_demand_score = Column(Integer, nullable=False)
    cross_platform_trend = Column(String(20), nullable=True)  # growing, stable, declining
    best_platform = Column(String(50), nullable=True)

    # Individual platform normalized scores
    platform_scores = Column(JSON, nullable=True)

    # Relationships
    keyword_rel = relationship("Keyword", back_populates="unified_scores")

    __table_args__ = (
        Index("idx_unified_scores_keyword", "keyword_id"),
        Index("idx_unified_scores_unique", "keyword_id", "collected_date", unique=True),
    )

    def __repr__(self) -> str:
        return f"<UnifiedScore(id={self.id}, keyword_id={self.keyword_id}, score={self.unified_demand_score})>"
