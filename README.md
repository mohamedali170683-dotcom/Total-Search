# Keyword Research Tool - Multi-Platform Data Aggregator

A Python-based keyword research tool that aggregates search volume and demand metrics from 5 platforms (Google, YouTube, Amazon, TikTok, Instagram) into a unified format suitable for RAG systems.

## Features

- **Multi-Platform Data Collection**: Fetches keyword metrics from Google, YouTube, Amazon, TikTok, and Instagram
- **Proxy Score Calculation**: Converts social media metrics (TikTok/Instagram) into search-volume-like scores
- **Unified Scoring**: Creates a single demand score across all platforms with configurable weights
- **Trend Analysis**: Detects growing, stable, or declining trends across platforms
- **RAG-Ready Export**: Outputs data in JSONL/CSV formats optimized for RAG ingestion
- **Resumable Pipeline**: Supports checkpoints for long-running jobs
- **Docker Support**: Ready for containerized deployment

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL (or SQLite for development)
- API credentials for:
  - DataForSEO (Google & YouTube data)
  - Apify (TikTok & Instagram scraping)
  - Jungle Scout (Amazon data)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd keyword-research-tool

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment file and configure
cp .env.example .env
# Edit .env with your API credentials
```

### Configuration

Edit `.env` with your API credentials:

```env
# DataForSEO
DATAFORSEO_LOGIN=your_login
DATAFORSEO_PASSWORD=your_password

# Apify
APIFY_API_TOKEN=your_token

# Jungle Scout
JUNGLESCOUT_API_KEY=your_key
JUNGLESCOUT_API_KEY_NAME=your_key_name

# Database
DATABASE_URL=sqlite:///data/keywords.db
```

### Basic Usage

```bash
# Run pipeline with keywords from a file
python scripts/run_pipeline.py --input data/input/keywords.csv

# Run with specific keywords
python scripts/run_pipeline.py --keywords "skincare,makeup,beauty"

# Run with specific platforms only
python scripts/run_pipeline.py --keywords "organic food" --platforms google,amazon

# Use different weight preset
python scripts/run_pipeline.py --input keywords.txt --weight-preset ecommerce

# Export for RAG system
python scripts/export_for_rag.py --format jsonl --output data/output/keywords.jsonl
```

## Architecture

```
src/
├── models/           # Pydantic data models
├── clients/          # API clients (DataForSEO, Apify, Jungle Scout)
├── calculators/      # Proxy score and unified score calculators
├── pipeline/         # Main orchestration pipeline
├── db/               # Database models and repository
└── utils/            # Utility functions

scripts/
├── run_pipeline.py   # Main CLI for running the pipeline
└── export_for_rag.py # Export data for RAG ingestion
```

## Data Models

### UnifiedKeywordData

The main output model containing:

```python
{
    "keyword": "organic skincare",
    "unified_demand_score": 72,        # 0-100 normalized score
    "cross_platform_trend": "growing",  # growing/stable/declining
    "best_platform": "amazon",          # Platform with highest demand
    "platforms": {
        "google": {"volume": 12000, "trend": "stable", "competition": "medium"},
        "youtube": {"volume": 5000, "trend": "growing"},
        "amazon": {"volume": 18000, "trend": "growing", "competition": "high"},
        "tiktok": {"volume": 25000, "trend": "growing", "confidence": "proxy"},
        "instagram": {"volume": 15000, "trend": "stable", "confidence": "proxy"}
    }
}
```

## Weight Presets

Configure how platforms contribute to the unified score:

| Preset | Google | YouTube | Amazon | TikTok | Instagram |
|--------|--------|---------|--------|--------|-----------|
| balanced | 0.25 | 0.20 | 0.20 | 0.20 | 0.15 |
| ecommerce | 0.20 | 0.10 | 0.45 | 0.15 | 0.10 |
| content | 0.15 | 0.25 | 0.10 | 0.30 | 0.20 |
| seo | 0.50 | 0.20 | 0.10 | 0.10 | 0.10 |
| video | 0.15 | 0.45 | 0.10 | 0.25 | 0.05 |

## Docker Deployment

```bash
# Start services (database + redis)
docker-compose up -d db redis

# Run pipeline
docker-compose --profile pipeline up pipeline

# Run export
docker-compose --profile export up export

# Development mode
docker-compose --profile dev up dev
```

## API Reference

### DataForSEO Client

```python
from src.clients import DataForSEOClient

async with DataForSEOClient() as client:
    metrics = await client.get_google_search_volume(["keyword1", "keyword2"])
    youtube = await client.get_youtube_search_volume(["keyword1"])
    suggestions = await client.get_keywords_for_site("example.com")
```

### Apify Client

```python
from src.clients import ApifyClient

async with ApifyClient() as client:
    tiktok_data = await client.run_tiktok_hashtag_scraper(
        ["skincare", "beauty"],
        results_per_hashtag=50
    )
    instagram_data = await client.run_instagram_hashtag_scraper(
        ["skincare", "beauty"],
        results_per_hashtag=50
    )
```

### Jungle Scout Client

```python
from src.clients import JungleScoutClient

async with JungleScoutClient() as client:
    metrics = await client.get_amazon_search_volume(["organic food"])
    keywords = await client.get_keywords_by_asin("B0123456789")
```

### Pipeline

```python
from src.pipeline import KeywordPipeline, PipelineOptions
from src.models.keyword import Platform

options = PipelineOptions(
    platforms=[Platform.GOOGLE, Platform.AMAZON],
    weight_preset="ecommerce",
    batch_size=50,
)

async with KeywordPipeline() as pipeline:
    results = await pipeline.run(["keyword1", "keyword2"], options)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_proxy_scores.py -v
```

## Database Migrations

```bash
# Generate a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## License

MIT
