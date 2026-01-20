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


class Brand(Base):
    """Brands table - stores brand tracking information."""

    __tablename__ = "brands"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # JSON-encoded list of keyword variants to track
    variants = Column(Text, nullable=True)

    # Cached aggregated metrics (updated on refresh)
    cached_metrics = Column(JSON, nullable=True)
    last_refreshed = Column(DateTime, nullable=True)

    # Relationships
    competitors = relationship("Competitor", back_populates="brand_rel", cascade="all, delete-orphan")
    alerts = relationship("BrandAlert", back_populates="brand_rel", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_brands_name", "name"),
    )

    def __repr__(self) -> str:
        return f"<Brand(id={self.id}, name='{self.name}')>"

    def get_variants(self) -> list[str]:
        """Get variants as a list."""
        import json
        if self.variants:
            return json.loads(self.variants)
        return []

    def set_variants(self, variants: list[str]) -> None:
        """Set variants from a list."""
        import json
        self.variants = json.dumps(variants) if variants else None


class Competitor(Base):
    """Competitors table - tracks competitor brands."""

    __tablename__ = "competitors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    brand_id = Column(Integer, ForeignKey("brands.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Keywords to track for this competitor
    keywords = Column(Text, nullable=True)  # JSON-encoded list

    # Cached metrics
    cached_metrics = Column(JSON, nullable=True)

    # Relationships
    brand_rel = relationship("Brand", back_populates="competitors")

    __table_args__ = (
        Index("idx_competitors_brand", "brand_id"),
    )

    def __repr__(self) -> str:
        return f"<Competitor(id={self.id}, name='{self.name}')>"

    def get_keywords(self) -> list[str]:
        """Get keywords as a list."""
        import json
        if self.keywords:
            return json.loads(self.keywords)
        return []

    def set_keywords(self, keywords: list[str]) -> None:
        """Set keywords from a list."""
        import json
        self.keywords = json.dumps(keywords) if keywords else None


class BrandAlert(Base):
    """Brand alerts table - stores notifications and insights."""

    __tablename__ = "brand_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    brand_id = Column(Integer, ForeignKey("brands.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    alert_type = Column(String(50), nullable=False)  # opportunity, warning, info
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    platform = Column(String(50), nullable=True)  # Which platform triggered this
    dismissed = Column(Integer, default=0, nullable=False)  # 0=active, 1=dismissed

    # Action data (JSON)
    actions = Column(JSON, nullable=True)

    # Relationships
    brand_rel = relationship("Brand", back_populates="alerts")

    __table_args__ = (
        Index("idx_brand_alerts_brand", "brand_id"),
        Index("idx_brand_alerts_active", "brand_id", "dismissed"),
    )

    def __repr__(self) -> str:
        return f"<BrandAlert(id={self.id}, type='{self.alert_type}', title='{self.title[:30]}...')>"
