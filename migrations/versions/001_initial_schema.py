"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-01-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create keywords table
    op.create_table(
        'keywords',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('keyword', sa.String(length=500), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('tags', sa.ARRAY(sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('keyword')
    )
    op.create_index('ix_keywords_keyword', 'keywords', ['keyword'], unique=True)

    # Create keyword_metrics table
    op.create_table(
        'keyword_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('keyword_id', sa.Integer(), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('collected_at', sa.DateTime(), nullable=False),
        sa.Column('search_volume', sa.Integer(), nullable=True),
        sa.Column('proxy_score', sa.Integer(), nullable=True),
        sa.Column('trend', sa.String(length=20), nullable=True),
        sa.Column('trend_velocity', sa.Float(), nullable=True),
        sa.Column('competition', sa.String(length=20), nullable=True),
        sa.Column('cpc', sa.Float(), nullable=True),
        sa.Column('confidence', sa.String(length=20), nullable=True),
        sa.Column('raw_data', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['keyword_id'], ['keywords.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_keyword_metrics_keyword_platform', 'keyword_metrics', ['keyword_id', 'platform'])
    op.create_index('idx_keyword_metrics_collected_at', 'keyword_metrics', ['collected_at'])

    # Create unified_scores table
    op.create_table(
        'unified_scores',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('keyword_id', sa.Integer(), nullable=False),
        sa.Column('collected_at', sa.DateTime(), nullable=False),
        sa.Column('unified_demand_score', sa.Integer(), nullable=False),
        sa.Column('cross_platform_trend', sa.String(length=20), nullable=True),
        sa.Column('best_platform', sa.String(length=50), nullable=True),
        sa.Column('platform_scores', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['keyword_id'], ['keywords.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_unified_scores_keyword', 'unified_scores', ['keyword_id'])


def downgrade() -> None:
    op.drop_table('unified_scores')
    op.drop_table('keyword_metrics')
    op.drop_table('keywords')
