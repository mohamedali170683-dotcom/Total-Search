"""FastAPI application for keyword research tool."""

import asyncio
import io
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for Vercel serverless functions
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Debug: Print path info
print(f"Project root: {project_root}")
print(f"sys.path: {sys.path[:3]}")
print(f"Files in project root: {list(project_root.iterdir())[:10]}")

logger = logging.getLogger(__name__)

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

try:
    from src.calculators.unified_score import UnifiedScoreCalculator, WeightPresets
    from src.clients.apify import ApifyClient
    from src.clients.dataforseo import DataForSEOClient
    from src.config import get_settings
    from src.db.repository import KeywordRepository
    from src.models.keyword import Platform, UnifiedKeywordData
    from src.pipeline.keyword_pipeline import KeywordPipeline, PipelineOptions
    IMPORTS_OK = True
    IMPORT_ERROR = None
except Exception as e:
    import traceback
    IMPORTS_OK = False
    IMPORT_ERROR = traceback.format_exc()
    print(f"Import error: {IMPORT_ERROR}")

# Initialize FastAPI app
app = FastAPI(
    title="Total Search - Cross-Platform Search Intelligence",
    description="Multi-platform keyword research aggregating data from Google, YouTube, Amazon, TikTok, and Instagram",
    version="1.0.0",
)

# CORS middleware for API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    error_details = traceback.format_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url.path),
            "traceback": error_details,
        }
    )

# Templates and static files
templates_dir = project_root / "templates"
print(f"Templates dir: {templates_dir}, exists: {templates_dir.exists()}")
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    # Fallback: try api/templates
    api_templates_dir = Path(__file__).parent / "templates"
    print(f"Trying API templates dir: {api_templates_dir}, exists: {api_templates_dir.exists()}")
    templates = Jinja2Templates(directory=str(templates_dir))  # Will fail gracefully

# Try to mount static files (may not exist in serverless)
try:
    static_dir = project_root / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
except Exception:
    pass

# Initialize repository (only if imports succeeded)
repo = None
repo_error = None
if IMPORTS_OK:
    try:
        repo = KeywordRepository()
        repo.create_tables()
    except Exception as e:
        import traceback
        repo_error = traceback.format_exc()
        print(f"Repository init error: {repo_error}")

# Background task storage
research_tasks: dict[str, dict] = {}


# Request/Response Models
class KeywordResearchRequest(BaseModel):
    """Request model for keyword research."""

    keywords: list[str] = Field(..., min_length=1, max_length=100, description="Keywords to research")
    platforms: list[str] = Field(
        default=["google", "youtube"],
        description="Platforms to query (google, youtube, tiktok, instagram, amazon)"
    )
    weight_preset: str = Field(default="balanced", description="Weight preset for scoring")


class KeywordResearchResponse(BaseModel):
    """Response model for keyword research."""

    task_id: str
    status: str
    message: str


class KeywordData(BaseModel):
    """Keyword data response model."""

    keyword: str
    unified_demand_score: int
    cross_platform_trend: str | None
    best_platform: str | None
    platforms: dict[str, Any]
    collected_at: str


# API Endpoints
@app.get("/")
async def root_redirect():
    """Redirect root to demand distribution dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/demand", status_code=307)


@app.get("/home")
async def home_page(request: Request):
    """Render the main dashboard."""
    # Get real stats and keywords from database if available
    if repo is not None:
        try:
            stats = repo.get_statistics()
            keywords = repo.get_all_keywords(limit=20)
        except Exception as e:
            print(f"Dashboard error: {e}")
            stats = {"total_keywords": 0, "total_metrics": 0, "metrics_by_platform": {}}
            keywords = []
    else:
        stats = {"total_keywords": 0, "total_metrics": 0, "metrics_by_platform": {}}
        keywords = []

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "stats": stats, "keywords": keywords}
    )




@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    if not IMPORTS_OK:
        return {
            "status": "error",
            "imports_ok": False,
            "error": IMPORT_ERROR,
            "project_root": str(project_root),
            "sys_path": sys.path[:5],
        }

    # Check templates
    templates_info = {}
    try:
        dashboard_path = templates_dir / "dashboard.html"
        templates_info = {
            "templates_dir": str(templates_dir),
            "templates_exist": templates_dir.exists(),
            "dashboard_exists": dashboard_path.exists(),
            "templates_contents": list(templates_dir.iterdir()) if templates_dir.exists() else [],
        }
    except Exception as e:
        templates_info = {"error": str(e)}

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "repo_initialized": repo is not None,
        "repo_error": repo_error,
        **templates_info,
    }


@app.get("/test-template")
async def test_template(request: Request):
    """Test minimal Jinja template."""
    try:
        return templates.TemplateResponse(
            "test.html",
            {"request": request, "stats": {"test": "value"}}
        )
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()}
        )


@app.get("/test-simple-dashboard")
async def test_simple_dashboard(request: Request):
    """Test simple dashboard template."""
    try:
        stats = {"total_keywords": 0, "total_metrics": 0, "metrics_by_platform": {}}
        keywords = []
        return templates.TemplateResponse(
            "dashboard_simple.html",
            {"request": request, "stats": stats, "keywords": keywords}
        )
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()}
        )


@app.get("/debug")
async def debug_page():
    """Simple debug page without Jinja templates."""
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head><title>Debug - Total Search</title></head>
    <body style="font-family: sans-serif; padding: 20px;">
        <h1>Total Search - Debug Page</h1>
        <p>This page renders without Jinja templates.</p>
        <h2>System Info:</h2>
        <ul>
            <li>Imports OK: {IMPORTS_OK}</li>
            <li>Repo initialized: {repo is not None}</li>
            <li>Templates dir: {templates_dir}</li>
            <li>Templates exist: {templates_dir.exists()}</li>
        </ul>
        <p><a href="/api/health">Health Check (JSON)</a></p>
    </body>
    </html>
    """)


@app.get("/api/stats")
async def get_stats():
    """Get database statistics."""
    if repo is None:
        return {"total_keywords": 0, "total_metrics": 0, "metrics_by_platform": {}, "error": "Repository not initialized"}
    try:
        stats = repo.get_statistics()
        return stats
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug/platform/{platform}")
async def debug_platform(
    platform: str,
    keyword: str = Query(default="lavera", description="Keyword to test")
):
    """
    Debug endpoint to test individual platform APIs and see raw responses.

    Use this to diagnose why platforms return no data.
    """
    settings = get_settings()
    result = {"platform": platform, "keyword": keyword, "raw_response": None, "error": None}

    try:
        if platform == "google":
            from src.clients.dataforseo import DataForSEOClient
            client = DataForSEOClient(settings=settings)
            try:
                metrics = await client.get_google_search_volume([keyword])
                await client.close()
                result["raw_response"] = {
                    "metrics_count": len(metrics),
                    "first_metric": metrics[0].model_dump() if metrics else None,
                }
            except Exception as e:
                result["error"] = f"Google API error: {str(e)}"

        elif platform == "youtube":
            from src.clients.dataforseo import DataForSEOClient
            client = DataForSEOClient(settings=settings)
            try:
                metrics = await client.get_youtube_search_volume([keyword])
                await client.close()
                result["raw_response"] = {
                    "metrics_count": len(metrics),
                    "first_metric": metrics[0].model_dump() if metrics else None,
                }
            except Exception as e:
                result["error"] = f"YouTube API error: {str(e)}"

        elif platform == "amazon":
            from src.clients.junglescout import JungleScoutClient
            client = JungleScoutClient(settings=settings)
            result["api_configured"] = bool(settings.junglescout_api_key)
            if not settings.junglescout_api_key:
                result["error"] = "Jungle Scout API key not configured"
            else:
                try:
                    metrics = await client.get_amazon_search_volume([keyword])
                    await client.close()
                    result["raw_response"] = {
                        "metrics_count": len(metrics),
                        "first_metric": metrics[0].model_dump() if metrics else None,
                    }
                except Exception as e:
                    result["error"] = f"Amazon API error: {str(e)}"

        elif platform == "tiktok":
            from src.clients.apify import ApifyClient
            client = ApifyClient(settings=settings)
            result["api_configured"] = bool(settings.apify_api_token.get_secret_value())
            try:
                hashtag = keyword.replace(" ", "").lower()
                data = await client.run_tiktok_hashtag_scraper([hashtag], results_per_hashtag=5, timeout_secs=30)
                await client.close()
                result["raw_response"] = {
                    "hashtags_returned": list(data.keys()),
                    "first_result": data.get(hashtag, {}),
                }
            except Exception as e:
                result["error"] = f"TikTok API error: {str(e)}"

        elif platform == "instagram":
            from src.clients.apify import ApifyClient
            client = ApifyClient(settings=settings)
            result["api_configured"] = bool(settings.apify_api_token.get_secret_value())
            try:
                hashtag = keyword.replace(" ", "").lower()
                data = await client.run_instagram_hashtag_scraper([hashtag], results_per_hashtag=5, timeout_secs=30)
                await client.close()
                result["raw_response"] = {
                    "hashtags_returned": list(data.keys()),
                    "first_result": data.get(hashtag, {}),
                }
            except Exception as e:
                result["error"] = f"Instagram API error: {str(e)}"
        else:
            result["error"] = f"Unknown platform: {platform}. Use: google, youtube, amazon, tiktok, instagram"

    except Exception as e:
        import traceback
        result["error"] = f"Unexpected error: {str(e)}"
        result["traceback"] = traceback.format_exc()

    return result


@app.post("/api/research", response_model=KeywordResearchResponse)
async def start_research(
    request: KeywordResearchRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a keyword research task.

    This endpoint starts an async background task and returns immediately.
    Use /api/research/{task_id} to check the status.
    """
    task_id = f"task_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{len(research_tasks)}"

    # Parse platforms
    platforms = []
    for p in request.platforms:
        try:
            platforms.append(Platform(p.lower()))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid platform: {p}")

    # Store task info
    research_tasks[task_id] = {
        "status": "pending",
        "keywords": request.keywords,
        "platforms": [p.value for p in platforms],
        "started_at": datetime.utcnow().isoformat(),
        "results": None,
        "error": None,
    }

    # Start background task
    background_tasks.add_task(
        run_research_task,
        task_id,
        request.keywords,
        platforms,
        request.weight_preset,
    )

    return KeywordResearchResponse(
        task_id=task_id,
        status="pending",
        message=f"Research started for {len(request.keywords)} keywords"
    )


async def run_research_task(
    task_id: str,
    keywords: list[str],
    platforms: list[Platform],
    weight_preset: str,
):
    """Background task to run keyword research."""
    try:
        research_tasks[task_id]["status"] = "running"

        settings = get_settings()
        options = PipelineOptions(
            platforms=platforms,
            weight_preset=weight_preset,
            batch_size=50,
            save_checkpoints=False,
        )

        async with KeywordPipeline(settings=settings) as pipeline:
            results = await pipeline.run(keywords, options)

        # Save to database
        repo.save_batch(results)

        # Store results
        research_tasks[task_id]["status"] = "completed"
        research_tasks[task_id]["completed_at"] = datetime.utcnow().isoformat()
        research_tasks[task_id]["results"] = [
            {
                "keyword": r.keyword,
                "unified_demand_score": r.unified_demand_score,
                "cross_platform_trend": r.cross_platform_trend.value if r.cross_platform_trend else None,
                "best_platform": r.best_platform.value if r.best_platform else None,
            }
            for r in results
        ]

    except Exception as e:
        research_tasks[task_id]["status"] = "failed"
        research_tasks[task_id]["error"] = str(e)


@app.get("/api/research/quick")
async def quick_research(
    keywords: str = Query(..., description="Comma-separated keywords"),
    platforms: str = Query(default="google,youtube", description="Comma-separated platforms"),
):
    """
    Quick synchronous research for a few keywords.

    Use this for small requests (1-5 keywords). For larger requests, use POST /api/research.
    """
    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]

    if len(keyword_list) > 10:
        raise HTTPException(
            status_code=400,
            detail="Too many keywords. Use POST /api/research for more than 10 keywords."
        )

    platform_list = []
    for p in platforms.split(","):
        p = p.strip().lower()
        if p:
            try:
                platform_list.append(Platform(p))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid platform: {p}")

    if not platform_list:
        platform_list = [Platform.GOOGLE, Platform.YOUTUBE]

    settings = get_settings()
    options = PipelineOptions(
        platforms=platform_list,
        weight_preset="balanced",
        batch_size=10,
        save_checkpoints=False,
    )

    try:
        async with KeywordPipeline(settings=settings) as pipeline:
            results = await pipeline.run(keyword_list, options)

        # Save to database
        repo.save_batch(results)

        return {
            "count": len(results),
            "keywords": [r.to_rag_document() for r in results]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/research/{task_id}")
async def get_research_status(task_id: str):
    """Get the status of a research task."""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return research_tasks[task_id]


@app.get("/api/keywords")
async def list_keywords(
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List all researched keywords."""
    keywords = repo.get_all_keywords(limit=limit, offset=offset)

    return {
        "count": len(keywords),
        "offset": offset,
        "keywords": [
            {
                "keyword": kw.keyword,
                "unified_demand_score": kw.unified_demand_score,
                "cross_platform_trend": kw.cross_platform_trend.value if kw.cross_platform_trend else None,
                "best_platform": kw.best_platform.value if kw.best_platform else None,
                "tags": kw.tags,
            }
            for kw in keywords
        ]
    }


@app.get("/api/keywords/{keyword}")
async def get_keyword(keyword: str):
    """Get detailed data for a specific keyword."""
    kw_data = repo.get_keyword(keyword)

    if not kw_data:
        raise HTTPException(status_code=404, detail="Keyword not found")

    return kw_data.to_rag_document()


@app.get("/api/export")
async def export_data(
    format: str = Query(default="json", description="Export format (json or csv)"),
    limit: int = Query(default=1000, le=10000),
):
    """Export keyword data for RAG or analysis."""
    keywords = repo.get_all_keywords(limit=limit)

    data = [kw.to_rag_document() for kw in keywords]

    if format == "csv":
        # Convert to CSV format
        if not data:
            return {"csv": ""}

        headers = ["keyword", "unified_demand_score", "cross_platform_trend", "best_platform"]
        rows = [headers]

        for item in data:
            rows.append([
                item.get("keyword", ""),
                str(item.get("unified_demand_score", "")),
                item.get("cross_platform_trend", ""),
                item.get("best_platform", ""),
            ])

        csv_content = "\n".join([",".join(row) for row in rows])
        return {"csv": csv_content, "count": len(data)}

    return {"data": data, "count": len(data)}


# =============================================================================
# Brand Intelligence API Endpoints
# =============================================================================

@app.get("/brand")
async def brand_dashboard_page(request: Request):
    """Render the brand intelligence dashboard."""
    return templates.TemplateResponse(
        "brand_dashboard.html",
        {"request": request}
    )


@app.get("/api/brands")
async def list_brands():
    """List all tracked brands."""
    if repo is None:
        return {"brands": [], "error": "Repository not initialized"}

    try:
        from src.db.models import Brand
        with repo.get_session() as session:
            from sqlalchemy import select
            stmt = select(Brand).order_by(Brand.updated_at.desc())
            brands = session.execute(stmt).scalars().all()

            return {
                "brands": [
                    {
                        "id": b.id,
                        "name": b.name,
                        "variants": b.get_variants(),
                        "created_at": b.created_at.isoformat(),
                        "last_refreshed": b.last_refreshed.isoformat() if b.last_refreshed else None,
                    }
                    for b in brands
                ]
            }
    except Exception as e:
        return {"brands": [], "error": str(e)}


class CreateBrandRequest(BaseModel):
    """Request model for creating a brand."""
    name: str = Field(..., min_length=1, max_length=200)
    variants: list[str] = Field(default=[])


@app.post("/api/brands")
async def create_brand(request: CreateBrandRequest):
    """Create a new brand to track."""
    if repo is None:
        raise HTTPException(status_code=500, detail="Repository not initialized")

    try:
        from src.db.models import Brand
        import json

        with repo.get_session() as session:
            brand = Brand(
                name=request.name,
                variants=json.dumps(request.variants) if request.variants else None,
            )
            session.add(brand)
            session.commit()
            session.refresh(brand)

            return {
                "brand": {
                    "id": brand.id,
                    "name": brand.name,
                    "variants": brand.get_variants(),
                    "created_at": brand.created_at.isoformat(),
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/brands/{brand_id}/metrics")
async def get_brand_metrics(brand_id: int):
    """Get aggregated metrics for a brand across all platforms."""
    if repo is None:
        raise HTTPException(status_code=500, detail="Repository not initialized")

    try:
        from src.db.models import Brand, Keyword, KeywordMetric, BrandAlert, Competitor
        from sqlalchemy import select, func
        import json

        with repo.get_session() as session:
            # Get the brand
            brand = session.get(Brand, brand_id)
            if not brand:
                raise HTTPException(status_code=404, detail="Brand not found")

            variants = brand.get_variants()

            # Aggregate metrics across all variants and platforms
            platform_metrics = {}
            platforms = ['google', 'youtube', 'amazon', 'tiktok', 'instagram']

            for platform in platforms:
                total_volume = 0
                avg_trend = 0
                count = 0
                has_data = False
                confidence = "none"

                for variant in variants:
                    # Find keyword matching variant
                    stmt = select(Keyword).where(Keyword.keyword == variant)
                    kw = session.execute(stmt).scalar_one_or_none()

                    if kw:
                        # Get latest metrics for this platform
                        metric_stmt = (
                            select(KeywordMetric)
                            .where(KeywordMetric.keyword_id == kw.id)
                            .where(KeywordMetric.platform == platform)
                            .order_by(KeywordMetric.collected_at.desc())
                            .limit(1)
                        )
                        metric = session.execute(metric_stmt).scalar_one_or_none()

                        if metric:
                            # Determine metric type for honest labeling
                            metric_type = getattr(metric, 'metric_type', None) or 'search_volume'

                            # Get the appropriate metric value based on type
                            if metric_type == 'engagement':
                                volume = getattr(metric, 'engagement_score', None) or metric.proxy_score or 0
                            elif metric_type == 'interest_index':
                                volume = getattr(metric, 'interest_score', None) or 0
                            else:
                                volume = metric.search_volume or 0

                            # Mark as "has data" if we have any valid metric
                            if volume > 0:
                                has_data = True

                            total_volume += volume
                            if metric.trend_velocity:
                                avg_trend += metric.trend_velocity
                                count += 1
                            # Track confidence level
                            if metric.confidence == "high":
                                confidence = "high"
                            elif metric.confidence == "medium" and confidence != "high":
                                confidence = "medium"
                            elif confidence == "none":
                                confidence = metric.confidence or "proxy"

                # Determine metric label based on platform
                metric_label = "search_volume"
                if platform in ['tiktok', 'instagram']:
                    metric_label = "engagement"
                elif platform == 'pinterest':
                    metric_label = "interest_index"

                platform_metrics[platform] = {
                    "volume": total_volume,
                    "metricType": metric_label,
                    "metricLabel": _get_metric_label(metric_label),
                    "trend": round(avg_trend / count * 100, 1) if count > 0 else 0,
                    "hasData": has_data,
                    "confidence": confidence,
                }

            # Get active alerts
            alerts_stmt = (
                select(BrandAlert)
                .where(BrandAlert.brand_id == brand_id)
                .where(BrandAlert.dismissed == 0)
                .order_by(BrandAlert.created_at.desc())
                .limit(10)
            )
            alerts = session.execute(alerts_stmt).scalars().all()

            # Get competitors
            competitors_stmt = select(Competitor).where(Competitor.brand_id == brand_id)
            competitors = session.execute(competitors_stmt).scalars().all()

            competitor_data = []
            for comp in competitors:
                comp_metrics = {}
                comp_keywords = comp.get_keywords()

                for platform in platforms:
                    total_vol = 0
                    for kw_text in comp_keywords:
                        kw = session.execute(
                            select(Keyword).where(Keyword.keyword == kw_text)
                        ).scalar_one_or_none()

                        if kw:
                            metric = session.execute(
                                select(KeywordMetric)
                                .where(KeywordMetric.keyword_id == kw.id)
                                .where(KeywordMetric.platform == platform)
                                .order_by(KeywordMetric.collected_at.desc())
                                .limit(1)
                            ).scalar_one_or_none()

                            if metric:
                                # Get appropriate metric value based on type
                                metric_type = getattr(metric, 'metric_type', None) or 'search_volume'
                                if metric_type == 'engagement':
                                    vol = getattr(metric, 'engagement_score', None) or metric.proxy_score or 0
                                elif metric_type == 'interest_index':
                                    vol = getattr(metric, 'interest_score', None) or 0
                                else:
                                    vol = metric.search_volume or 0
                                total_vol += vol

                    comp_metrics[platform] = {"volume": total_vol}

                competitor_data.append({
                    "id": comp.id,
                    "name": comp.name,
                    "metrics": comp_metrics,
                })

            # Generate content opportunities based on related keyword patterns
            opportunities = generate_content_opportunities(brand.name, variants, platform_metrics)

            # Calculate weekly change (mock for now, would need historical data)
            total_volume = sum(m["volume"] for m in platform_metrics.values())
            weekly_change = calculate_weekly_change(session, brand_id, variants)

            return {
                "metrics": platform_metrics,
                "weeklyChange": weekly_change,
                "alerts": [
                    {
                        "id": a.id,
                        "type": a.alert_type,
                        "title": a.title,
                        "description": a.description,
                        "platform": a.platform,
                        "actions": a.actions,
                    }
                    for a in alerts
                ],
                "competitors": competitor_data,
                "opportunities": opportunities,
            }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


def _get_metric_label(metric_type: str) -> str:
    """
    Get human-readable label for metric type.

    This provides honest labeling of what each metric actually measures:
    - search_volume: Verified search data (Google, YouTube, Amazon)
    - engagement: Audience engagement (TikTok, Instagram) - NOT search volume
    - interest_index: Relative interest (Pinterest) - NOT search volume
    """
    labels = {
        "search_volume": "Monthly Searches",
        "engagement": "Audience Engagement",
        "interest_index": "Interest Index",
    }
    return labels.get(metric_type, "Volume")


def _get_metric_explanation(platform: str) -> str:
    """Get explanation of what the metric measures for a platform."""
    explanations = {
        "google": "Verified monthly search volume from Google Ads data",
        "youtube": "Verified monthly search volume from YouTube search data",
        "amazon": "Verified monthly search volume from Amazon search data",
        "tiktok": "Audience engagement (views, likes, shares) - NOT search volume. TikTok is algorithm-driven.",
        "instagram": "Community engagement (posts, likes, comments) - NOT search volume. Instagram is browse-first.",
        "pinterest": "Relative interest index (0-100) - shows topic popularity compared to other Pinterest topics.",
    }
    return explanations.get(platform, "Estimated volume metric")


def generate_content_opportunities(brand_name: str, variants: list[str], metrics: dict) -> list[dict]:
    """Generate content opportunity suggestions based on brand data."""
    opportunities = []

    # Find best performing platform
    best_platform = max(metrics.keys(), key=lambda p: metrics[p]["volume"]) if metrics else "google"

    # Generate keyword-based opportunities
    base_opportunities = [
        {"suffix": "review", "platform": "youtube", "difficulty": "medium"},
        {"suffix": "vs", "platform": "google", "difficulty": "medium"},
        {"suffix": "tutorial", "platform": "youtube", "difficulty": "easy"},
        {"suffix": "alternatives", "platform": "google", "difficulty": "hard"},
        {"suffix": "how to use", "platform": "tiktok", "difficulty": "easy"},
    ]

    for opp in base_opportunities[:4]:
        keyword = f"{brand_name} {opp['suffix']}"
        opportunities.append({
            "keyword": keyword,
            "platform": opp["platform"],
            "volume": int(metrics.get(opp["platform"], {}).get("volume", 0) * 0.1),  # Estimate
            "difficulty": opp["difficulty"],
            "reason": f"High search intent for {opp['suffix']} content on {opp['platform'].title()}",
        })

    return opportunities


def calculate_weekly_change(session, brand_id: int, variants: list[str]) -> float:
    """Calculate week-over-week change in search volume."""
    # This would need historical data to calculate properly
    # For now, return a mock value based on trend velocities
    from src.db.models import Keyword, KeywordMetric
    from sqlalchemy import select

    total_trend = 0
    count = 0

    for variant in variants:
        kw = session.execute(
            select(Keyword).where(Keyword.keyword == variant)
        ).scalar_one_or_none()

        if kw:
            metrics = session.execute(
                select(KeywordMetric)
                .where(KeywordMetric.keyword_id == kw.id)
                .order_by(KeywordMetric.collected_at.desc())
                .limit(5)
            ).scalars().all()

            for m in metrics:
                if m.trend_velocity:
                    total_trend += m.trend_velocity
                    count += 1

    return round(total_trend / count * 100, 1) if count > 0 else 0


class AddVariantsRequest(BaseModel):
    """Request model for adding keyword variants."""
    variants: list[str] = Field(..., min_length=1)


@app.post("/api/brands/{brand_id}/variants")
async def add_brand_variants(brand_id: int, request: AddVariantsRequest):
    """Add keyword variants to a brand."""
    if repo is None:
        raise HTTPException(status_code=500, detail="Repository not initialized")

    try:
        from src.db.models import Brand
        import json

        with repo.get_session() as session:
            brand = session.get(Brand, brand_id)
            if not brand:
                raise HTTPException(status_code=404, detail="Brand not found")

            current_variants = brand.get_variants()
            new_variants = [v for v in request.variants if v not in current_variants]
            current_variants.extend(new_variants)
            brand.set_variants(current_variants)

            session.commit()

            return {"success": True, "added": new_variants, "total": len(current_variants)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AddCompetitorRequest(BaseModel):
    """Request model for adding a competitor."""
    name: str = Field(..., min_length=1, max_length=200)
    keywords: list[str] = Field(default=[])


@app.post("/api/brands/{brand_id}/competitors")
async def add_competitor(brand_id: int, request: AddCompetitorRequest):
    """Add a competitor to track."""
    if repo is None:
        raise HTTPException(status_code=500, detail="Repository not initialized")

    try:
        from src.db.models import Brand, Competitor
        import json

        with repo.get_session() as session:
            brand = session.get(Brand, brand_id)
            if not brand:
                raise HTTPException(status_code=404, detail="Brand not found")

            competitor = Competitor(
                brand_id=brand_id,
                name=request.name,
                keywords=json.dumps(request.keywords) if request.keywords else None,
            )
            session.add(competitor)
            session.commit()
            session.refresh(competitor)

            return {
                "competitor": {
                    "id": competitor.id,
                    "name": competitor.name,
                    "keywords": competitor.get_keywords(),
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/brands/{brand_id}/refresh")
async def refresh_brand_data(
    brand_id: int,
    platforms: str = Query(default="google,youtube,amazon,tiktok,instagram", description="Platforms to research (comma-separated)")
):
    """Refresh data for a brand by re-researching all variants across all platforms."""
    if repo is None:
        raise HTTPException(status_code=500, detail="Repository not initialized")

    try:
        from src.db.models import Brand, Competitor
        from sqlalchemy import select

        with repo.get_session() as session:
            brand = session.get(Brand, brand_id)
            if not brand:
                raise HTTPException(status_code=404, detail="Brand not found")

            variants = brand.get_variants()

            # Also get competitor keywords
            competitors = session.execute(
                select(Competitor).where(Competitor.brand_id == brand_id)
            ).scalars().all()

            all_keywords = list(variants)
            for comp in competitors:
                all_keywords.extend(comp.get_keywords())

            # Remove duplicates - keep reasonable limit for timeout (max 10 keywords)
            all_keywords = list(set(all_keywords))[:10]

        if not all_keywords:
            return {"success": True, "keywords_researched": 0, "message": "No keywords to research"}

        # Parse platforms (default to ALL 5 platforms - this is the core value!)
        platform_list = []
        for p in platforms.split(","):
            p = p.strip().lower()
            if p:
                try:
                    platform_list.append(Platform(p))
                except ValueError:
                    pass

        # Default to all platforms if none specified
        if not platform_list:
            platform_list = [Platform.GOOGLE, Platform.YOUTUBE, Platform.AMAZON, Platform.TIKTOK, Platform.INSTAGRAM]

        # Run synchronously (Vercel serverless doesn't support background tasks well)
        settings = get_settings()

        options = PipelineOptions(
            platforms=platform_list,
            weight_preset="balanced",
            batch_size=10,
            save_checkpoints=False,
        )

        try:
            async with KeywordPipeline(settings=settings) as pipeline:
                results = await pipeline.run(all_keywords, options)

            # Save results
            if results:
                repo.save_batch(results)

            # Update brand last_refreshed timestamp
            with repo.get_session() as session:
                brand = session.get(Brand, brand_id)
                if brand:
                    brand.last_refreshed = datetime.utcnow()
                    session.commit()

            return {
                "success": True,
                "keywords_researched": len(results) if results else 0,
                "platforms_checked": [p.value for p in platform_list],
                "results": [
                    {
                        "keyword": r.keyword,
                        "unified_demand_score": r.unified_demand_score,
                        "best_platform": r.best_platform.value if r.best_platform else None,
                    }
                    for r in (results or [])
                ]
            }

        except Exception as e:
            import traceback
            raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}\n{traceback.format_exc()}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def run_brand_refresh_task_unused(brand_id: int, keywords: list[str]):
    """Background task to refresh brand keyword data."""
    try:
        from src.db.models import Brand

        settings = get_settings()
        platforms = [Platform.GOOGLE, Platform.YOUTUBE, Platform.AMAZON, Platform.TIKTOK, Platform.INSTAGRAM]

        options = PipelineOptions(
            platforms=platforms,
            weight_preset="balanced",
            batch_size=20,
            save_checkpoints=False,
        )

        async with KeywordPipeline(settings=settings) as pipeline:
            results = await pipeline.run(keywords, options)

        # Save results
        repo.save_batch(results)

        # Update brand last_refreshed timestamp
        with repo.get_session() as session:
            brand = session.get(Brand, brand_id)
            if brand:
                brand.last_refreshed = datetime.utcnow()
                session.commit()

        # Generate alerts based on results
        await generate_brand_alerts(brand_id, results)

    except Exception as e:
        print(f"Brand refresh error: {e}")


async def generate_brand_alerts(brand_id: int, results: list):
    """Generate alerts based on research results."""
    try:
        from src.db.models import BrandAlert
        import json

        with repo.get_session() as session:
            for result in results:
                # Check for significant trend changes
                if result.cross_platform_trend and result.cross_platform_trend.value == "growing":
                    # Find the platform with highest growth
                    if result.platform_scores:
                        for ps in result.platform_scores:
                            if ps.weighted_score > 50:
                                alert = BrandAlert(
                                    brand_id=brand_id,
                                    alert_type="opportunity",
                                    title=f"'{result.keyword}' trending on {ps.platform.value.title()}",
                                    description=f"Search volume is growing. Consider creating content.",
                                    platform=ps.platform.value,
                                    actions=json.dumps([
                                        {"label": "Create Content", "type": "create_content"},
                                        {"label": "Dismiss", "type": "dismiss"},
                                    ]),
                                )
                                session.add(alert)

            session.commit()

    except Exception as e:
        print(f"Alert generation error: {e}")


# =============================================================================
# Search Distribution API - Core Feature for Cross-Platform Analysis
# =============================================================================

@app.get("/demand")
async def demand_distribution_page(request: Request):
    """Render the demand distribution dashboard - the main analysis view."""
    return templates.TemplateResponse(
        "demand_distribution.html",
        {"request": request}
    )


@app.get("/compare")
async def compare_page(request: Request):
    """Render the competitor comparison dashboard."""
    return templates.TemplateResponse(
        "compare.html",
        {"request": request}
    )


class DemandAnalysisRequest(BaseModel):
    """Request for demand distribution analysis."""
    keywords: list[str] = Field(..., min_length=1, max_length=50, description="Keywords to analyze")
    include_competitors: bool = Field(default=False, description="Include competitor comparison")
    competitor_keywords: list[str] | None = Field(default=None, description="Competitor keywords to compare")


@app.get("/api/demand/analyze")
async def analyze_demand_distribution(
    keywords: str = Query(..., description="Comma-separated keywords to analyze"),
    refresh: bool = Query(default=False, description="Force refresh data from APIs"),
    country: str = Query(default="us", description="Country code (e.g., us, de, uk, fr)"),
    language: str = Query(default="en", description="Language code (e.g., en, de, fr, es)"),
    weight_preset: str = Query(default="general", description="Audience profile for platform weights"),
):
    """
    Analyze demand distribution across all platforms for given keywords.

    Returns:
    - Total demand volume across all platforms
    - Per-platform breakdown with volumes and percentages
    - Platform-specific insights and recommendations
    - Trend analysis
    """
    if repo is None:
        raise HTTPException(status_code=500, detail="Repository not initialized")

    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]

    if len(keyword_list) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 keywords per request")

    try:
        from src.db.models import Keyword, KeywordMetric
        from sqlalchemy import select

        # Check if we have recent data (< 24 hours) or need to refresh
        results = {}
        keywords_to_fetch = []

        with repo.get_session() as session:
            for kw in keyword_list:
                # Check for existing data
                stmt = select(Keyword).where(Keyword.keyword == kw)
                existing = session.execute(stmt).scalar_one_or_none()

                if existing and not refresh:
                    # Get metrics for all platforms
                    metrics_stmt = (
                        select(KeywordMetric)
                        .where(KeywordMetric.keyword_id == existing.id)
                        .order_by(KeywordMetric.collected_at.desc())
                    )
                    metrics = session.execute(metrics_stmt).scalars().all()

                    # Check if data is recent (within 24 hours)
                    from datetime import datetime, timedelta
                    recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                    recent_metrics = [m for m in metrics if m.collected_at > recent_cutoff]

                    if recent_metrics:
                        results[kw] = _aggregate_keyword_metrics(recent_metrics)
                    else:
                        keywords_to_fetch.append(kw)
                else:
                    keywords_to_fetch.append(kw)

        # Fetch data for keywords we don't have
        if keywords_to_fetch:
            fetched_data = await _fetch_demand_data(keywords_to_fetch, country=country, language=language)
            results.update(fetched_data)

        # Validate weight preset
        if weight_preset not in WEIGHT_PRESETS:
            weight_preset = "general"

        # Collect demo platforms across all keywords
        all_demo_platforms = set()
        for kw_data in results.values():
            all_demo_platforms.update(kw_data.get("demo_platforms", []))

        # Calculate demand distribution
        distribution = _calculate_demand_distribution(results, weight_preset=weight_preset)

        # Mark demo platforms in the platform list
        for plat in distribution.get("platforms", []):
            if plat["platform"] in all_demo_platforms:
                plat["is_demo"] = True

        return {
            "keywords": keyword_list,
            "demand_index": distribution["demand_index"],
            "weight_preset": weight_preset,
            "weight_preset_label": WEIGHT_PRESET_LABELS.get(weight_preset, "General"),
            "total_demand": distribution["total_volume"],  # backward compat
            "search_volume_total": distribution["search_volume_total"],
            "social_engagement_total": distribution["social_engagement_total"],
            "pinterest_interest": distribution["pinterest_interest"],
            "platforms": distribution["platforms"],
            "distribution_summary": distribution["summary"],
            "insights": distribution["insights"],
            "recommendations": distribution["recommendations"],
            "keyword_details": results,
            "demo_platforms": list(all_demo_platforms),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}\n{traceback.format_exc()}")


@app.get("/api/demand/compare")
async def compare_demand(
    brand_keywords: str = Query(..., description="Your brand keywords (comma-separated)"),
    competitor_keywords: str = Query(..., description="Competitor keywords (comma-separated)"),
):
    """
    Compare demand distribution between your brand and competitors.

    Shows share of voice across platforms.
    """
    brand_kws = [k.strip() for k in brand_keywords.split(",") if k.strip()]
    competitor_kws = [k.strip() for k in competitor_keywords.split(",") if k.strip()]

    if len(brand_kws) + len(competitor_kws) > 30:
        raise HTTPException(status_code=400, detail="Maximum 30 total keywords")

    try:
        # Fetch data for both sets
        brand_data = await _fetch_demand_data(brand_kws)
        competitor_data = await _fetch_demand_data(competitor_kws)

        brand_distribution = _calculate_demand_distribution(brand_data)
        competitor_distribution = _calculate_demand_distribution(competitor_data)

        # Calculate share of voice
        share_of_voice = _calculate_share_of_voice(brand_distribution, competitor_distribution)

        return {
            "brand": {
                "keywords": brand_kws,
                "total_demand": brand_distribution["total_volume"],
                "platforms": brand_distribution["platforms"],
            },
            "competitor": {
                "keywords": competitor_kws,
                "total_demand": competitor_distribution["total_volume"],
                "platforms": competitor_distribution["platforms"],
            },
            "share_of_voice": share_of_voice,
            "insights": _generate_competitive_insights(brand_distribution, competitor_distribution),
        }

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


async def _fetch_demand_data(keywords: list[str], country: str = "us", language: str = "en") -> dict:
    """Fetch demand data from all platforms for keywords.

    Args:
        keywords: List of keywords to research
        country: Country code (e.g., us, de, uk, fr, es, it, nl, au, ca, br, mx, jp)
        language: Language code (e.g., en, de, fr, es, it, nl, pt, ja)
    """
    settings = get_settings()

    # Map country codes to DataForSEO location codes
    country_location_map = {
        "us": 2840,    # United States
        "de": 2276,    # Germany
        "uk": 2826,    # United Kingdom
        "gb": 2826,    # United Kingdom (alternate)
        "fr": 2250,    # France
        "es": 2724,    # Spain
        "it": 2380,    # Italy
        "nl": 2528,    # Netherlands
        "au": 2036,    # Australia
        "ca": 2124,    # Canada
        "br": 2076,    # Brazil
        "mx": 2484,    # Mexico
        "jp": 2392,    # Japan
    }

    # Use all 6 platforms
    all_platforms = [
        Platform.GOOGLE, Platform.YOUTUBE, Platform.AMAZON,
        Platform.TIKTOK, Platform.INSTAGRAM, Platform.PINTEREST
    ]

    # Get location code for the country
    location_code = country_location_map.get(country.lower(), 2840)  # Default to US

    options = PipelineOptions(
        platforms=all_platforms,
        weight_preset="balanced",
        batch_size=10,
        save_checkpoints=False,
        tiktok_results_per_hashtag=10,
        instagram_results_per_hashtag=10,
        location_code=location_code,
        language_code=language.lower(),
    )

    results = {}

    try:
        async with KeywordPipeline(settings=settings) as pipeline:
            pipeline_results = await pipeline.run(keywords, options)

        # Save to database
        if repo and pipeline_results:
            repo.save_batch(pipeline_results)

        # Convert to dict format
        for result in pipeline_results:
            results[result.keyword] = {
                "unified_score": result.unified_demand_score,
                "trend": result.cross_platform_trend.value if result.cross_platform_trend else "stable",
                "best_platform": result.best_platform.value if result.best_platform else None,
                "platforms": {}
            }

            # Add per-platform data with honest labeling
            for platform, metrics in result.platforms_dict.items():
                if metrics:
                    # Determine metric type for honest labeling
                    metric_type = metrics.metric_type.value if metrics.metric_type else "search_volume"
                    results[result.keyword]["platforms"][platform.value] = {
                        "volume": metrics.effective_volume,
                        "metricType": metric_type,
                        "metricLabel": _get_metric_label(metric_type),
                        "explanation": metrics.metric_explanation or _get_metric_explanation(platform.value),
                        "trend": metrics.trend.value if metrics.trend else "stable",
                        "trend_velocity": metrics.trend_velocity,
                        "confidence": metrics.confidence.value,
                        "isVerifiedSearchData": metrics.is_verified_search_data,
                    }
                else:
                    results[result.keyword]["platforms"][platform.value] = {
                        "volume": 0,
                        "metricType": "unknown",
                        "metricLabel": "No Data",
                        "explanation": "No data available for this platform",
                        "trend": None,
                        "trend_velocity": None,
                        "confidence": "none",
                        "isVerifiedSearchData": False,
                    }

        # Inject demo data for TikTok, Instagram, Pinterest when real data is 0
        from src.demo_trends import generate_demo_volume

        demo_platforms_set = {"tiktok", "instagram", "pinterest"}
        for kw, kw_data in results.items():
            google_volume = kw_data.get("platforms", {}).get("google", {}).get("volume", 0)
            for plat in demo_platforms_set:
                plat_data = kw_data.get("platforms", {}).get(plat, {})
                if not plat_data.get("volume"):
                    demo = generate_demo_volume(kw, plat, google_volume=google_volume)
                    kw_data["platforms"][plat] = {
                        "volume": demo["volume"],
                        "trend": demo["trend"],
                        "trend_velocity": demo["trend_velocity"],
                        "confidence": "demo",
                    }
                    kw_data.setdefault("demo_platforms", []).append(plat)

    except Exception as e:
        import logging
        logging.error(f"Error fetching demand data: {e}")
        # Return empty results for failed keywords
        for kw in keywords:
            if kw not in results:
                results[kw] = {
                    "unified_score": 0,
                    "trend": "unknown",
                    "best_platform": None,
                    "platforms": {},
                    "error": str(e),
                }

    return results


def _aggregate_keyword_metrics(metrics: list) -> dict:
    """Aggregate metrics from database into demand data format."""
    result = {
        "unified_score": 0,
        "trend": "stable",
        "best_platform": None,
        "platforms": {}
    }

    platform_volumes = {}

    for metric in metrics:
        platform = metric.platform
        volume = metric.search_volume or metric.proxy_score or 0

        # Keep most recent per platform
        if platform not in result["platforms"] or volume > result["platforms"][platform].get("volume", 0):
            result["platforms"][platform] = {
                "volume": volume,
                "trend": metric.trend,
                "trend_velocity": metric.trend_velocity,
                "confidence": metric.confidence or "proxy",
            }
            platform_volumes[platform] = volume

    # Determine best platform
    if platform_volumes:
        result["best_platform"] = max(platform_volumes, key=platform_volumes.get)

    return result


def _normalize_to_100(volume: int) -> float:
    """Normalize a raw volume to 0-100 using log scaling (matches unified_score.py)."""
    MIN_VOL = 10
    MAX_VOL = 10_000_000
    if volume < MIN_VOL:
        return 0.0
    log_vol = math.log(volume, 10)
    log_max = math.log(MAX_VOL, 10)
    log_min = math.log(MIN_VOL, 10)
    return max(0.0, min(100.0, ((log_vol - log_min) / (log_max - log_min)) * 100))


WEIGHT_PRESETS = {
    "general": {
        "google": 0.30, "amazon": 0.22, "youtube": 0.15,
        "instagram": 0.14, "tiktok": 0.11, "pinterest": 0.08,
    },
    "ecommerce": {
        "amazon": 0.32, "google": 0.25, "pinterest": 0.14,
        "instagram": 0.12, "youtube": 0.10, "tiktok": 0.07,
    },
    "beauty": {
        "tiktok": 0.25, "instagram": 0.22, "google": 0.18,
        "amazon": 0.15, "pinterest": 0.12, "youtube": 0.08,
    },
    "gen_z": {
        "tiktok": 0.26, "instagram": 0.22, "youtube": 0.18,
        "google": 0.18, "pinterest": 0.08, "amazon": 0.08,
    },
    "b2b_tech": {
        "google": 0.40, "youtube": 0.22, "amazon": 0.18,
        "instagram": 0.08, "tiktok": 0.07, "pinterest": 0.05,
    },
    "video_content": {
        "youtube": 0.30, "tiktok": 0.25, "instagram": 0.18,
        "google": 0.15, "pinterest": 0.07, "amazon": 0.05,
    },
}

WEIGHT_PRESET_LABELS = {
    "general": "General",
    "ecommerce": "E-commerce",
    "beauty": "Beauty & Lifestyle",
    "gen_z": "Gen Z Audience",
    "b2b_tech": "B2B / Technology",
    "video_content": "Video & Content",
}


def _calculate_demand_distribution(keyword_data: dict, weight_preset: str = "general") -> dict:
    """Calculate unified demand distribution across all platforms."""
    SEARCH_PLATFORMS = {"google", "youtube", "amazon"}
    SOCIAL_PLATFORMS = {"tiktok", "instagram"}

    METRIC_TYPE_MAP = {
        "google": "search_volume", "youtube": "search_volume", "amazon": "search_volume",
        "tiktok": "engagement", "instagram": "engagement",
        "pinterest": "interest_index",
    }
    METRIC_LABEL_MAP = {
        "search_volume": "Monthly Searches",
        "engagement": "Avg. Interactions / Post",
        "interest_index": "Interest Index (0-100)",
    }

    platform_totals = {
        "google": 0, "youtube": 0, "amazon": 0,
        "tiktok": 0, "instagram": 0, "pinterest": 0,
    }
    platform_trends = {p: [] for p in platform_totals}
    platform_keyword_values: dict[str, list[int]] = {p: [] for p in SOCIAL_PLATFORMS}

    # Aggregate volumes across all keywords
    for kw, data in keyword_data.items():
        platforms = data.get("platforms", {})
        for platform, metrics in platforms.items():
            if platform in platform_totals:
                volume = metrics.get("volume", 0) if isinstance(metrics, dict) else 0
                if platform in SEARCH_PLATFORMS:
                    platform_totals[platform] += volume
                elif platform in SOCIAL_PLATFORMS:
                    platform_keyword_values[platform].append(volume)
                else:
                    platform_totals[platform] = max(platform_totals[platform], volume)
                trend = metrics.get("trend") if isinstance(metrics, dict) else None
                if trend:
                    platform_trends[platform].append(trend)

    for p in SOCIAL_PLATFORMS:
        values = platform_keyword_values[p]
        platform_totals[p] = int(sum(values) / len(values)) if values else 0

    # Compute totals by metric type (kept for backward compat / detail display)
    search_volume_total = sum(platform_totals[p] for p in SEARCH_PLATFORMS)
    social_engagement_total = sum(platform_totals[p] for p in SOCIAL_PLATFORMS)
    pinterest_interest = platform_totals.get("pinterest", 0)
    total_volume = sum(platform_totals.values())

    # --- Normalize all platforms to 0-100 for unified comparison ---
    normalized_scores = {}
    for p, vol in platform_totals.items():
        if p == "pinterest":
            normalized_scores[p] = float(vol)  # already 0-100
        else:
            normalized_scores[p] = round(_normalize_to_100(vol), 1)

    # Compute Search Index (weighted average of normalized scores)
    weights = WEIGHT_PRESETS.get(weight_preset, WEIGHT_PRESETS["general"])
    weighted_sum = 0.0
    total_weight = 0.0
    for p, score in normalized_scores.items():
        if score > 0:
            w = weights.get(p, 0)
            weighted_sum += score * w
            total_weight += w
    demand_index = int(weighted_sum / total_weight) if total_weight > 0 else 0

    # Total normalized (for pie chart percentages)
    total_normalized = sum(normalized_scores.values())

    # Build platform breakdown
    platforms = []
    for platform, volume in sorted(platform_totals.items(), key=lambda x: -normalized_scores.get(x[0], 0)):
        metric_type = METRIC_TYPE_MAP[platform]
        norm_score = normalized_scores[platform]
        norm_pct = round((norm_score / total_normalized * 100), 1) if total_normalized > 0 else 0

        # Aggregate trend
        trends = platform_trends[platform]
        if trends:
            growing = trends.count("growing")
            declining = trends.count("declining")
            trend = "growing" if growing > declining else ("declining" if declining > growing else "stable")
        else:
            trend = "stable"

        platforms.append({
            "platform": platform,
            "volume": volume,
            "normalized_score": norm_score,
            "percentage": norm_pct,
            "metric_type": metric_type,
            "metric_label": METRIC_LABEL_MAP[metric_type],
            "trend": trend,
            "display_name": _get_platform_display_name(platform),
            "icon": _get_platform_icon(platform),
            "color": _get_platform_color(platform),
        })

    # Find top platform by normalized score
    top_platform_obj = platforms[0] if platforms else None

    summary = {
        "demand_index": demand_index,
        "search_volume_total": search_volume_total,
        "social_engagement_total": social_engagement_total,
        "pinterest_interest": pinterest_interest,
        "total_volume": total_volume,
        "top_platform": top_platform_obj["platform"] if top_platform_obj else None,
        "top_platform_name": top_platform_obj["display_name"] if top_platform_obj else None,
        "top_platform_score": top_platform_obj["normalized_score"] if top_platform_obj else 0,
        "platform_count": len([p for p in platforms if p["volume"] > 0]),
    }

    insights = _generate_demand_insights(platforms, summary)
    recommendations = _generate_demand_recommendations(platforms, summary)

    return {
        "demand_index": demand_index,
        "total_volume": total_volume,
        "search_volume_total": search_volume_total,
        "social_engagement_total": social_engagement_total,
        "pinterest_interest": pinterest_interest,
        "platforms": platforms,
        "summary": summary,
        "insights": insights,
        "recommendations": recommendations,
    }


def _get_platform_display_name(platform: str) -> str:
    """Get display name for platform."""
    names = {
        "google": "Google Search",
        "youtube": "YouTube",
        "amazon": "Amazon",
        "tiktok": "TikTok",
        "instagram": "Instagram",
        "pinterest": "Pinterest",
    }
    return names.get(platform, platform.title())


def _get_platform_icon(platform: str) -> str:
    """Get Font Awesome icon class for platform."""
    icons = {
        "google": "fab fa-google",
        "youtube": "fab fa-youtube",
        "amazon": "fab fa-amazon",
        "tiktok": "fab fa-tiktok",
        "instagram": "fab fa-instagram",
        "pinterest": "fab fa-pinterest",
    }
    return icons.get(platform, "fas fa-search")


def _get_platform_color(platform: str) -> str:
    """Get brand color for platform."""
    colors = {
        "google": "#4285F4",
        "youtube": "#FF0000",
        "amazon": "#FF9900",
        "tiktok": "#000000",
        "instagram": "#E1306C",
        "pinterest": "#E60023",
    }
    return colors.get(platform, "#6B7280")


def _generate_demand_insights(platforms: list, summary: dict) -> list[dict]:
    """Generate insights about demand distribution across all platforms."""
    insights = []

    demand_index = summary.get("demand_index", 0)
    top_name = summary.get("top_platform_name", "")
    top_score = summary.get("top_platform_score", 0)

    # Insight: Overall demand strength
    if demand_index >= 60:
        insights.append({
            "type": "opportunity",
            "title": f"Strong cross-platform demand (Index: {demand_index}/100)",
            "description": f"Demand signals are strong across multiple platforms. {top_name} leads with a normalized score of {top_score}/100.",
            "priority": "high",
        })
    elif demand_index >= 30:
        insights.append({
            "type": "opportunity",
            "title": f"Moderate demand detected (Index: {demand_index}/100)",
            "description": f"There is meaningful demand across platforms. {top_name} shows the strongest signal at {top_score}/100.",
            "priority": "medium",
        })

    # Insight: Platform-specific growth
    for p in platforms:
        if p["volume"] > 0 and p["trend"] == "growing":
            insights.append({
                "type": "trend",
                "title": f"{p['display_name']} demand is growing",
                "description": f"Demand on {p['display_name']} is trending upward (score: {p['normalized_score']}/100). Consider increasing investment on this platform.",
                "priority": "medium",
                "platform": p["platform"],
            })

    # Insight: Amazon e-commerce signal
    amazon_data = next((p for p in platforms if p["platform"] == "amazon"), None)
    if amazon_data and amazon_data["normalized_score"] > 20:
        insights.append({
            "type": "ecommerce",
            "title": f"Amazon demand score: {amazon_data['normalized_score']}/100",
            "description": "Strong purchase intent. Ensure your Amazon presence and advertising are optimized.",
            "priority": "high",
            "platform": "amazon",
        })

    # Insight: Social platform demand
    tiktok_data = next((p for p in platforms if p["platform"] == "tiktok"), None)
    instagram_data = next((p for p in platforms if p["platform"] == "instagram"), None)
    if tiktok_data and tiktok_data["volume"] > 0:
        insights.append({
            "type": "social",
            "title": f"TikTok: ~{tiktok_data['volume']:,} interactions per video",
            "description": f"On average, {tiktok_data['volume']:,} users interact with each TikTok video about this topic (score: {tiktok_data['normalized_score']}/100).",
            "priority": "medium",
            "platform": "tiktok",
        })
    if instagram_data and instagram_data["volume"] > 0:
        insights.append({
            "type": "social",
            "title": f"Instagram: ~{instagram_data['volume']:,} interactions per post",
            "description": f"On average, {instagram_data['volume']:,} users interact with each Instagram post about this topic (score: {instagram_data['normalized_score']}/100).",
            "priority": "medium",
            "platform": "instagram",
        })

    # Insight: Pinterest interest
    pinterest_data = next((p for p in platforms if p["platform"] == "pinterest"), None)
    if pinterest_data and pinterest_data["volume"] > 30:
        insights.append({
            "type": "visual",
            "title": f"Pinterest interest: {pinterest_data['volume']}/100",
            "description": f"Pinterest users show interest in this topic (score: {pinterest_data['normalized_score']}/100). Create visual content and shopping pins.",
            "priority": "medium",
            "platform": "pinterest",
        })

    return insights[:5]


def _generate_demand_recommendations(platforms: list, summary: dict) -> list[dict]:
    """Generate actionable recommendations based on demand distribution."""
    recommendations = []

    platform_data = {p["platform"]: p for p in platforms}

    # Recommendation based on top platform
    if summary["top_platform"]:
        top = platform_data[summary["top_platform"]]
        recommendations.append({
            "platform": top["platform"],
            "action": f"Prioritize {top['display_name']}",
            "description": f"With a demand score of {top['normalized_score']}/100, this is your primary channel for this keyword set.",
            "tactics": _get_platform_tactics(top["platform"]),
        })

    # Recommendations for underutilized high-volume platforms
    for p in platforms[1:4]:  # Next 3 platforms after top
        if p["volume"] > 0 and p["normalized_score"] > 15:
            recommendations.append({
                "platform": p["platform"],
                "action": f"Expand to {p['display_name']}",
                "description": f"Demand score of {p['normalized_score']}/100 represents a significant opportunity.",
                "tactics": _get_platform_tactics(p["platform"]),
            })

    return recommendations[:4]  # Limit to top 4 recommendations


def _get_platform_tactics(platform: str) -> list[str]:
    """Get tactical recommendations for each platform."""
    tactics = {
        "google": ["Google Ads Search campaigns", "SEO optimization", "Google Shopping (if e-commerce)"],
        "youtube": ["Video ad campaigns", "Creator partnerships", "SEO for video content"],
        "amazon": ["Sponsored Products", "Amazon DSP", "Listing optimization"],
        "tiktok": ["Spark Ads", "Hashtag challenges", "Creator marketplace"],
        "instagram": ["Reels ads", "Shopping tags", "Influencer partnerships"],
        "pinterest": ["Shopping pins", "Idea pins", "Pinterest Ads"],
    }
    return tactics.get(platform, ["Platform-specific advertising", "Content optimization"])


def _calculate_share_of_voice(brand: dict, competitor: dict) -> dict:
    """Calculate share of voice between brand and competitor."""
    share_of_voice = {"overall": {}, "by_platform": {}}

    # Overall share
    brand_total = brand["total_volume"]
    competitor_total = competitor["total_volume"]
    total = brand_total + competitor_total

    if total > 0:
        share_of_voice["overall"] = {
            "brand": round(brand_total / total * 100, 1),
            "competitor": round(competitor_total / total * 100, 1),
        }

    # Per-platform share
    brand_platforms = {p["platform"]: p["volume"] for p in brand["platforms"]}
    competitor_platforms = {p["platform"]: p["volume"] for p in competitor["platforms"]}

    all_platforms = set(brand_platforms.keys()) | set(competitor_platforms.keys())

    for platform in all_platforms:
        brand_vol = brand_platforms.get(platform, 0)
        comp_vol = competitor_platforms.get(platform, 0)
        platform_total = brand_vol + comp_vol

        if platform_total > 0:
            share_of_voice["by_platform"][platform] = {
                "brand": round(brand_vol / platform_total * 100, 1),
                "competitor": round(comp_vol / platform_total * 100, 1),
                "brand_volume": brand_vol,
                "competitor_volume": comp_vol,
            }

    return share_of_voice


def _generate_competitive_insights(brand: dict, competitor: dict) -> list[dict]:
    """Generate insights from competitive comparison."""
    insights = []

    brand_platforms = {p["platform"]: p for p in brand["platforms"]}
    competitor_platforms = {p["platform"]: p for p in competitor["platforms"]}

    # Find platforms where competitor is stronger
    for platform in brand_platforms:
        brand_vol = brand_platforms[platform]["volume"]
        comp_vol = competitor_platforms.get(platform, {}).get("volume", 0)

        if comp_vol > brand_vol * 1.5:  # Competitor is 50%+ stronger
            insights.append({
                "type": "gap",
                "title": f"Competitor dominates on {_get_platform_display_name(platform)}",
                "description": f"They have {comp_vol:,} vs your {brand_vol:,}. Consider increasing presence.",
                "platform": platform,
            })
        elif brand_vol > comp_vol * 1.5:  # We're stronger
            insights.append({
                "type": "strength",
                "title": f"You lead on {_get_platform_display_name(platform)}",
                "description": f"You have {brand_vol:,} vs their {comp_vol:,}. Maintain this advantage.",
                "platform": platform,
            })

    return insights


# =============================================================================
# Google Trends Intelligence API
# =============================================================================

@app.get("/api/trends/intelligence")
async def get_trends_intelligence(
    keywords: str = Query(..., description="Comma-separated keywords (max 5)"),
    country: str = Query(default="", description="Country code (e.g., US, DE, UK) or empty for worldwide"),
    timeframe: str = Query(default="today 12-m", description="Timeframe (today 12-m, today 3-m, today 1-m)"),
    include_correlation: bool = Query(default=True, description="Include cross-platform correlation analysis"),
):
    """
    Get comprehensive Google Trends intelligence for keywords.

    Returns:
    - 12-month interest trend chart data (Google Web + YouTube + Amazon + TikTok + Instagram)
    - Geographic hotspots (top 10 countries)
    - Rising queries (emerging search terms)
    - Top related queries
    - Seasonality analysis
    - Cross-platform trend correlation (if include_correlation=True)

    Note: TikTok and Instagram trends are simulated demo data showing
    what the full product would look like with a KeywordTool.io license.
    Amazon uses real DataForSEO data when available, with demo fallback.
    """
    try:
        from src.clients.google_trends import GoogleTrendsClient
        from src.demo_trends import generate_demo_trend

        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()][:5]

        if not keyword_list:
            raise HTTPException(status_code=400, detail="At least one keyword is required")

        settings = get_settings()

        # Map common country codes
        geo_map = {
            "us": "US", "de": "DE", "uk": "GB", "gb": "GB",
            "fr": "FR", "es": "ES", "it": "IT", "nl": "NL",
            "au": "AU", "ca": "CA", "br": "BR", "mx": "MX", "jp": "JP",
        }
        geo = geo_map.get(country.lower(), country.upper()) if country else ""

        async with GoogleTrendsClient(settings=settings) as client:
            data = await client.get_trends_intelligence(
                keywords=keyword_list,
                timeframe=timeframe,
                geo=geo,
            )

            # Fetch multi-platform trends (Google Web + YouTube)
            multi_platform = {}
            correlation_analysis = {}
            if include_correlation:
                try:
                    multi_platform = await client.get_multi_platform_trends(
                        keywords=keyword_list,
                        timeframe=timeframe,
                        geo=geo,
                    )
                except Exception as e:
                    logger.warning(f"Multi-platform trends failed: {e}")
                    multi_platform = {"platforms": {}}

                # Get Google Web series as base for demo data generation
                google_web_series = (
                    multi_platform.get("platforms", {})
                    .get("google_web", {})
                    .get("interest_over_time", [])
                )
                primary_kw = keyword_list[0]

                # --- Amazon trends (real DataForSEO data with demo fallback) ---
                try:
                    from src.clients.dataforseo import DataForSEOClient

                    location_map = {
                        "US": 2840, "DE": 2276, "GB": 2826, "FR": 2250,
                        "ES": 2724, "IT": 2380, "NL": 2528, "AU": 2036,
                        "CA": 2124, "BR": 2076, "MX": 2484, "JP": 2392,
                    }
                    location_code = location_map.get(geo, 2840)

                    async with DataForSEOClient(settings=settings) as dfs_client:
                        amazon_metrics = await dfs_client.get_amazon_search_volume(
                            keywords=[primary_kw],
                            location_code=location_code,
                        )

                    if amazon_metrics and amazon_metrics[0].search_volume:
                        amazon_series = generate_demo_trend(
                            google_web_series, primary_kw, "amazon"
                        )
                        if amazon_series:
                            multi_platform.setdefault("platforms", {})["amazon"] = {
                                "interest_over_time": amazon_series,
                                "demo": True,
                                "real_volume": amazon_metrics[0].search_volume,
                            }
                    else:
                        amazon_series = generate_demo_trend(
                            google_web_series, primary_kw, "amazon"
                        )
                        if amazon_series:
                            multi_platform.setdefault("platforms", {})["amazon"] = {
                                "interest_over_time": amazon_series,
                                "demo": True,
                            }
                except Exception as e:
                    logger.warning(f"Amazon trends failed: {e}")
                    amazon_series = generate_demo_trend(
                        google_web_series, primary_kw, "amazon"
                    )
                    if amazon_series:
                        multi_platform.setdefault("platforms", {})["amazon"] = {
                            "interest_over_time": amazon_series,
                            "demo": True,
                        }

                # --- TikTok trends (demo data) ---
                tiktok_series = generate_demo_trend(
                    google_web_series, primary_kw, "tiktok"
                )
                if tiktok_series:
                    multi_platform.setdefault("platforms", {})["tiktok"] = {
                        "interest_over_time": tiktok_series,
                        "demo": True,
                    }

                # --- Instagram trends (demo data) ---
                instagram_series = generate_demo_trend(
                    google_web_series, primary_kw, "instagram"
                )
                if instagram_series:
                    multi_platform.setdefault("platforms", {})["instagram"] = {
                        "interest_over_time": instagram_series,
                        "demo": True,
                    }

                # Compute correlation across all platforms
                if multi_platform.get("platforms"):
                    correlation_analysis = _compute_trend_correlation(
                        multi_platform.get("platforms", {}),
                        keyword_list[0],
                    )

        # Enrich with insights
        insights = _generate_trends_insights(data)

        # Track which platforms are demo vs real
        demo_platforms = [
            p for p, d in multi_platform.get("platforms", {}).items()
            if d.get("demo")
        ]

        response = {
            "keywords": keyword_list,
            "country": geo or "Worldwide",
            "timeframe": timeframe,
            "interest_over_time": data.get("interest_over_time", []),
            "interest_by_region": data.get("interest_by_region", []),
            "related_queries": data.get("related_queries", {}),
            "rising_queries": data.get("rising_queries", []),
            "seasonality": data.get("seasonality"),
            "insights": insights,
        }

        if include_correlation:
            response["multi_platform_trends"] = multi_platform.get("platforms", {})
            response["correlation_analysis"] = correlation_analysis
            response["demo_platforms"] = demo_platforms

        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trends intelligence: {str(e)}\n{traceback.format_exc()}"
        )


# ---------------------------------------------------------------------------
# Behavioral Search Intelligence
# ---------------------------------------------------------------------------

@app.get("/api/behavioral/analysis")
async def analyze_behavioral_intelligence(
    keyword: str = Query(..., description="Category keyword to analyze (e.g., naturkosmetik)"),
    brands: str = Query(default="", description="Comma-separated brand names for share of search"),
    country: str = Query(default="de", description="Country code (e.g., de, us, uk)"),
    language: str = Query(default="de", description="Language code (e.g., de, en)"),
):
    """
    Behavioral search intelligence: funnel mapping, share of search, modifier analysis.

    Analyzes search data to extract actionable behavioral insights:
    1. Funnel Stage Mapping — where is category demand? (awareness/consideration/purchase)
    2. Share of Search — brand share of total category search (predicts market share)
    3. Behavioral Modifiers — macro consumer behavior patterns (price vs quality, etc.)
    """
    import traceback

    try:
        from src.clients.dataforseo import DataForSEOClient
        from src.config import get_settings
        from src.behavioral_analysis import (
            get_all_behavioral_keywords,
            analyze_funnel,
            analyze_share_of_search,
            analyze_modifier_pairs,
        )

        settings = get_settings()
        brand_list = [b.strip() for b in brands.split(",") if b.strip()]

        # Get all keywords needed in a single list
        all_keywords = get_all_behavioral_keywords(keyword, brand_list, language)

        # Country → location code mapping
        country_location_map = {
            "us": 2840, "de": 2276, "uk": 2826, "gb": 2826,
            "fr": 2250, "es": 2724, "it": 2380, "nl": 2528,
            "au": 2036, "ca": 2124,
        }
        location_code = country_location_map.get(country.lower(), 2840)

        # Language → DataForSEO language code
        lang_map = {"de": "de", "en": "en", "fr": "fr", "es": "es", "it": "it", "nl": "nl"}
        lang_code = lang_map.get(language.lower(), "en")

        # Fetch all volumes in one DataForSEO call
        keyword_volumes: dict[str, int] = {}
        async with DataForSEOClient(settings=settings) as client:
            metrics = await client.get_google_search_volume(
                all_keywords,
                location_code=location_code,
                language_code=lang_code,
            )
            # Metrics are returned in the same order as input keywords
            for kw, m in zip(all_keywords, metrics):
                keyword_volumes[kw] = m.search_volume or 0

        # Run all three analyses
        funnel = analyze_funnel(keyword, keyword_volumes, language)

        share_of_search = None
        if brand_list:
            brand_vols = {b: keyword_volumes.get(b, 0) for b in brand_list}
            cat_generic_vol = keyword_volumes.get(keyword, 0)
            share_of_search = analyze_share_of_search(keyword, brand_vols, cat_generic_vol)

        modifiers = analyze_modifier_pairs(keyword_volumes, language)

        return {
            "status": "ok",
            "keyword": keyword,
            "country": country,
            "language": language,
            "keywords_analyzed": len(keyword_volumes),
            "funnel_analysis": funnel.to_dict(),
            "share_of_search": share_of_search.to_dict() if share_of_search else None,
            "behavioral_modifiers": modifiers.to_dict(),
        }

    except Exception as e:
        logger.error(f"Behavioral analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Behavioral analysis failed: {str(e)}\n{traceback.format_exc()}"
        )


@app.get("/api/search-landscape")
async def get_search_landscape(
    keyword: str = Query(..., description="Keyword to analyze SERP landscape for"),
    brand: str = Query(default="", description="Brand name to check presence in results"),
    country: str = Query(default="de", description="Country code (e.g., de, us, uk)"),
    language: str = Query(default="de", description="Language code"),
):
    """
    Search Landscape: Who owns the search results for this keyword?

    Fetches the live Google SERP and analyzes:
    - Top ranking domains and their types (brand, marketplace, content, retailer)
    - Whether the specified brand appears or is absent
    - People Also Ask questions (reveals searcher intent)
    - SERP features present (featured snippets, videos, knowledge graph)
    - Domain category breakdown (what type of sites dominate)
    - Actionable insights based on the competitive landscape
    """
    try:
        from src.clients.dataforseo import DataForSEOClient
        from src.config import get_settings

        settings = get_settings()

        country_location_map = {
            "us": 2840, "de": 2276, "uk": 2826, "gb": 2826,
            "fr": 2250, "es": 2724, "it": 2380, "nl": 2528,
            "au": 2036, "ca": 2124, "br": 2076, "mx": 2484, "jp": 2392,
        }
        location_code = country_location_map.get(country.lower(), 2840)
        lang_map = {"de": "de", "en": "en", "fr": "fr", "es": "es", "it": "it", "nl": "nl", "pt": "pt", "ja": "ja"}
        lang_code = lang_map.get(language.lower(), "en")

        async with DataForSEOClient(settings=settings) as client:
            serp_data = await client.get_serp_results(
                keyword=keyword,
                location_code=location_code,
                language_code=lang_code,
                depth=20,
            )

        organic = serp_data.get("organic_results", [])
        serp_features = serp_data.get("serp_features", [])
        people_also_ask = serp_data.get("people_also_ask", [])

        # --- Classify domains ---
        MARKETPLACES = {"amazon", "ebay", "etsy", "zalando", "otto", "kaufland", "idealo", "aliexpress", "walmart", "target"}
        SOCIAL_PLATFORMS = {"youtube", "tiktok", "instagram", "pinterest", "facebook", "twitter", "reddit", "linkedin"}
        CONTENT_SIGNALS = {"blog", "magazin", "magazine", "wiki", "ratgeber", "guide", "advice", "tipps", "test", "vergleich", "review"}
        RETAILERS = {"dm", "rossmann", "douglas", "mueller", "thalia", "mediamarkt", "saturn", "notino", "flaconi", "parfumdreams", "sephora", "boots"}

        domain_results = []
        brand_lower = brand.strip().lower()
        brand_found_at = None

        for r in organic:
            domain = r.get("domain", "").lower()
            domain_root = domain.replace("www.", "").split(".")[0] if domain else ""
            title = r.get("title", "").lower()
            url = r.get("url", "").lower()
            position = r.get("position", 0)

            # Classify domain type
            if domain_root in MARKETPLACES or any(mp in domain for mp in MARKETPLACES):
                domain_type = "marketplace"
            elif domain_root in SOCIAL_PLATFORMS or any(sp in domain for sp in SOCIAL_PLATFORMS):
                domain_type = "social"
            elif domain_root in RETAILERS or any(rt in domain for rt in RETAILERS):
                domain_type = "retailer"
            elif any(sig in url for sig in CONTENT_SIGNALS) or any(sig in title for sig in CONTENT_SIGNALS):
                domain_type = "content"
            else:
                domain_type = "brand"

            # Check if brand appears
            is_brand = False
            if brand_lower and (brand_lower in domain or brand_lower in title or brand_lower in url):
                is_brand = True
                if brand_found_at is None:
                    brand_found_at = position

            domain_results.append({
                "position": position,
                "domain": r.get("domain", ""),
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "type": domain_type,
                "is_brand": is_brand,
            })

        # --- Domain type breakdown ---
        type_counts = {}
        for dr in domain_results:
            t = dr["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        total_results = len(domain_results)
        type_breakdown = []
        type_labels = {
            "brand": "Brand Sites",
            "marketplace": "Marketplaces",
            "content": "Content / Reviews",
            "retailer": "Retailers",
            "social": "Social Platforms",
        }
        for dtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            type_breakdown.append({
                "type": dtype,
                "label": type_labels.get(dtype, dtype.title()),
                "count": count,
                "percentage": round(count / total_results * 100, 1) if total_results else 0,
            })

        # --- Unique domains (deduplicated top domains) ---
        seen_domains = set()
        unique_domains = []
        for dr in domain_results:
            root = dr["domain"].replace("www.", "")
            if root not in seen_domains:
                seen_domains.add(root)
                unique_domains.append({
                    "domain": root,
                    "best_position": dr["position"],
                    "type": dr["type"],
                    "is_brand": dr["is_brand"],
                })

        # --- Generate insights ---
        insights = []

        if brand_lower:
            if brand_found_at is not None:
                if brand_found_at <= 3:
                    insights.append({
                        "type": "positive",
                        "title": f"{brand.title()} ranks #{brand_found_at}",
                        "description": f"Strong position — your brand is in the top 3 for \"{keyword}\"."
                    })
                elif brand_found_at <= 10:
                    insights.append({
                        "type": "warning",
                        "title": f"{brand.title()} ranks #{brand_found_at}",
                        "description": f"Page 1 but not top 3. Competitors above you are capturing more clicks."
                    })
                else:
                    insights.append({
                        "type": "negative",
                        "title": f"{brand.title()} ranks #{brand_found_at}",
                        "description": f"Below the fold — most searchers will never see your result."
                    })
            else:
                insights.append({
                    "type": "negative",
                    "title": f"{brand.title()} not found in top 20",
                    "description": f"Your brand doesn't appear in search results for \"{keyword}\". This is a major visibility gap."
                })

        # Marketplace dominance
        mp_count = type_counts.get("marketplace", 0)
        if mp_count >= 3:
            mp_domains = [d["domain"] for d in unique_domains if d["type"] == "marketplace"][:3]
            insights.append({
                "type": "warning",
                "title": f"Marketplace-dominated SERP",
                "description": f"{mp_count} of {total_results} results are marketplaces ({', '.join(mp_domains)}). Organic brand visibility is limited — consider marketplace optimization."
            })

        # Content opportunity
        content_count = type_counts.get("content", 0)
        if content_count >= 4:
            insights.append({
                "type": "opportunity",
                "title": "Content-driven SERP",
                "description": f"Google favors informational content ({content_count} results are reviews/guides). Creating authoritative content could earn ranking."
            })
        elif content_count <= 1 and total_results >= 10:
            insights.append({
                "type": "opportunity",
                "title": "Content gap detected",
                "description": "Very few content/review sites rank. An informational content strategy could fill this gap."
            })

        # People Also Ask intent
        if people_also_ask:
            question_types = []
            for paa in people_also_ask[:8]:
                q = paa.get("question", "").lower()
                if any(w in q for w in ["what", "was ist", "wie", "how"]):
                    question_types.append("educational")
                elif any(w in q for w in ["best", "beste", "top", "review", "test"]):
                    question_types.append("comparison")
                elif any(w in q for w in ["buy", "kaufen", "price", "preis", "cost", "kosten"]):
                    question_types.append("transactional")
                else:
                    question_types.append("general")

            dominant_intent = max(set(question_types), key=question_types.count) if question_types else "general"
            intent_labels = {
                "educational": "Educational (people want to understand)",
                "comparison": "Comparative (people want to evaluate options)",
                "transactional": "Transactional (people want to buy)",
                "general": "Mixed intent",
            }
            insights.append({
                "type": "info",
                "title": f"Search intent: {intent_labels.get(dominant_intent, 'Mixed')}",
                "description": f"\"People Also Ask\" questions reveal {dominant_intent} intent. Align your content strategy accordingly."
            })

        # SERP features
        feature_types = [f["type"] for f in serp_features]
        if "featured_snippet" in feature_types:
            fs = next(f for f in serp_features if f["type"] == "featured_snippet")
            insights.append({
                "type": "info",
                "title": f"Featured Snippet owned by {fs.get('domain', 'unknown')}",
                "description": "A featured snippet captures ~30% of clicks. Structuring content to answer the query directly could win this position."
            })

        if "video" in feature_types:
            insights.append({
                "type": "opportunity",
                "title": "Video results in SERP",
                "description": "Google shows video results for this query. YouTube content optimized for this keyword could appear here."
            })

        return {
            "status": "ok",
            "keyword": keyword,
            "brand": brand if brand else None,
            "country": country,
            "results_count": total_results,
            "brand_position": brand_found_at,
            "organic_results": domain_results[:10],
            "unique_domains": unique_domains[:15],
            "type_breakdown": type_breakdown,
            "serp_features": serp_features,
            "people_also_ask": people_also_ask[:8],
            "insights": insights,
        }

    except Exception as e:
        logger.error(f"Search landscape failed: {e}")
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Search landscape failed: {str(e)}\n{traceback.format_exc()}"
        )


@app.get("/api/trends/daily")
async def get_daily_trending_searches(
    country: str = Query(default="united_states", description="Country (e.g., united_states, germany, japan)"),
):
    """
    Get today's trending searches for a country.

    Shows what's currently trending on Google in the selected country.
    Useful for identifying newsjacking opportunities and timely content.
    """
    try:
        from src.clients.google_trends import GoogleTrendsClient

        settings = get_settings()

        async with GoogleTrendsClient(settings=settings) as client:
            trending = await client.get_trending_searches(country=country)

        return {
            "country": country,
            "trending_searches": trending,
            "count": len(trending),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trending searches: {str(e)}"
        )



def _compute_trend_correlation(
    platforms_data: dict,
    primary_keyword: str,
) -> dict:
    """
    Compute cross-platform trend correlation analysis.

    Uses Spearman rank correlation (robust to non-linear relationships and
    outliers) with cross-correlation lag detection, plus Granger causality
    testing to determine if one platform's trends statistically predict
    another's.

    Returns pairs with correlation metrics, lag info, Granger causality
    p-values, and human-readable insights.
    """
    try:
        import numpy as np
        from scipy.stats import spearmanr
    except ImportError:
        return {"error": "numpy/scipy not installed"}

    # Granger causality is optional — statsmodels is heavier
    granger_available = False
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        granger_available = True
    except ImportError:
        pass

    results: dict = {"pairs": [], "insights": [], "method": "spearman"}
    if granger_available:
        results["method"] = "spearman+granger"

    # Extract time series values per platform
    series_by_platform: dict[str, list[float]] = {}

    for platform_name, platform_data in platforms_data.items():
        iot = platform_data.get("interest_over_time", [])
        if not iot:
            continue

        if platform_name in ("tiktok", "amazon", "instagram"):
            values = [float(entry.get(platform_name, 0)) for entry in iot]
        else:
            values = [float(entry.get(primary_keyword, 0)) for entry in iot]

        if values and any(v > 0 for v in values):
            series_by_platform[platform_name] = values

    if len(series_by_platform) < 2:
        return results

    def _safe_granger(cause: np.ndarray, effect: np.ndarray, max_lag: int) -> dict:
        """Run Granger causality test, return best lag and p-value."""
        if not granger_available or len(cause) < max_lag + 3:
            return {"tested": False}
        try:
            data = np.column_stack([effect, cause])
            test_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            best_p = 1.0
            best_lag = 1
            for lag_val, res in test_result.items():
                p_val = res[0]["ssr_ftest"][1]
                if p_val < best_p:
                    best_p = p_val
                    best_lag = lag_val
            return {"tested": True, "p_value": round(best_p, 4), "lag": best_lag}
        except Exception:
            return {"tested": False}

    platform_names = list(series_by_platform.keys())
    for i in range(len(platform_names)):
        for j in range(i + 1, len(platform_names)):
            p1, p2 = platform_names[i], platform_names[j]
            s1, s2 = series_by_platform[p1], series_by_platform[p2]

            min_len = min(len(s1), len(s2))
            if min_len < 4:
                continue
            a1 = np.array(s1[-min_len:], dtype=float)
            a2 = np.array(s2[-min_len:], dtype=float)

            if np.std(a1) == 0 or np.std(a2) == 0:
                continue

            # Spearman rank correlation (base, no lag)
            base_rho, base_p = spearmanr(a1, a2)
            base_rho = float(base_rho)
            base_p = float(base_p)

            # Cross-correlate with lags 1-4 using Spearman
            best_lag = 0
            best_rho = abs(base_rho)
            best_rho_signed = base_rho
            best_p_val = base_p

            max_lag = min(5, min_len - 2)
            for lag in range(1, max_lag):
                # p2 leads p1 by 'lag' periods
                rho, p = spearmanr(a1[lag:], a2[:-lag])
                if abs(rho) > best_rho:
                    best_rho = abs(rho)
                    best_rho_signed = float(rho)
                    best_p_val = float(p)
                    best_lag = lag

                # p1 leads p2 by 'lag' periods
                rho, p = spearmanr(a1[:-lag], a2[lag:])
                if abs(rho) > best_rho:
                    best_rho = abs(rho)
                    best_rho_signed = float(rho)
                    best_p_val = float(p)
                    best_lag = -lag

            # Granger causality: does p2 Granger-cause p1? And vice versa?
            granger_max = min(4, min_len // 3)
            gc_p2_causes_p1 = _safe_granger(a2, a1, granger_max)
            gc_p1_causes_p2 = _safe_granger(a1, a2, granger_max)

            pair_result = {
                "platform_a": p1,
                "platform_b": p2,
                "spearman_rho": round(best_rho_signed, 3),
                "p_value": round(best_p_val, 4),
                "lag_periods": best_lag,
                "relationship": "synchronized",
                "significant": best_p_val < 0.05,
            }

            # Include Granger results if available
            if gc_p2_causes_p1.get("tested"):
                pair_result["granger_b_causes_a"] = {
                    "p_value": gc_p2_causes_p1["p_value"],
                    "lag": gc_p2_causes_p1["lag"],
                    "significant": gc_p2_causes_p1["p_value"] < 0.05,
                }
            if gc_p1_causes_p2.get("tested"):
                pair_result["granger_a_causes_b"] = {
                    "p_value": gc_p1_causes_p2["p_value"],
                    "lag": gc_p1_causes_p2["lag"],
                    "significant": gc_p1_causes_p2["p_value"] < 0.05,
                }

            # Also keep "correlation" key for backward compatibility
            pair_result["correlation"] = pair_result["spearman_rho"]

            # Generate insight text
            p1_label = p1.replace("_", " ").title()
            p2_label = p2.replace("_", " ").title()

            # Priority 1: Granger causality (strongest evidence)
            gc_insight_added = False
            if gc_p2_causes_p1.get("tested") and gc_p2_causes_p1["p_value"] < 0.05:
                pair_result["relationship"] = f"{p2} predicts {p1} (Granger, lag {gc_p2_causes_p1['lag']})"
                results["insights"].append(
                    f"{p2_label} statistically predicts {p1_label} trends "
                    f"(Granger p={gc_p2_causes_p1['p_value']:.3f}, lag={gc_p2_causes_p1['lag']}). "
                    f"Discovery on {p2_label} leads to search on {p1_label}."
                )
                gc_insight_added = True
            elif gc_p1_causes_p2.get("tested") and gc_p1_causes_p2["p_value"] < 0.05:
                pair_result["relationship"] = f"{p1} predicts {p2} (Granger, lag {gc_p1_causes_p2['lag']})"
                results["insights"].append(
                    f"{p1_label} statistically predicts {p2_label} trends "
                    f"(Granger p={gc_p1_causes_p2['p_value']:.3f}, lag={gc_p1_causes_p2['lag']}). "
                    f"Search intent on {p1_label} precedes {p2_label} engagement."
                )
                gc_insight_added = True

            # Priority 2: Spearman lag relationship
            if not gc_insight_added:
                if best_rho > 0.4 and best_lag > 0 and best_p_val < 0.05:
                    pair_result["relationship"] = f"{p2} leads {p1} by ~{best_lag} week(s)"
                    results["insights"].append(
                        f"{p2_label} trends lead {p1_label} by ~{best_lag} week(s) "
                        f"(ρ={best_rho_signed:.2f}, p={best_p_val:.3f})."
                    )
                elif best_rho > 0.4 and best_lag < 0 and best_p_val < 0.05:
                    pair_result["relationship"] = f"{p1} leads {p2} by ~{abs(best_lag)} week(s)"
                    results["insights"].append(
                        f"{p1_label} trends lead {p2_label} by ~{abs(best_lag)} week(s) "
                        f"(ρ={best_rho_signed:.2f}, p={best_p_val:.3f})."
                    )
                elif best_rho > 0.6 and best_p_val < 0.05:
                    pair_result["relationship"] = "strongly synchronized"
                    results["insights"].append(
                        f"{p1_label} and {p2_label} are strongly correlated "
                        f"(ρ={best_rho_signed:.2f}, p={best_p_val:.3f}). "
                        f"Interest moves together across platforms."
                    )
                elif best_rho < 0.2 or best_p_val >= 0.05:
                    pair_result["relationship"] = "independent"
                    results["insights"].append(
                        f"{p1_label} and {p2_label} appear independent "
                        f"(ρ={best_rho_signed:.2f}, p={best_p_val:.3f})."
                    )

            results["pairs"].append(pair_result)

    return results


def _generate_trends_insights(data: dict) -> list[dict]:
    """Generate actionable insights from Google Trends data."""
    insights = []

    # Seasonality insight
    seasonality = data.get("seasonality")
    if seasonality:
        if seasonality.get("seasonality_strength") in ["high", "medium"]:
            insights.append({
                "type": "seasonality",
                "title": f"Peak interest in {seasonality.get('peak_month', 'N/A')}",
                "description": f"Search interest peaks in {seasonality.get('peak_month')} "
                              f"(index: {seasonality.get('peak_interest', 0)}) and is lowest in "
                              f"{seasonality.get('low_month')} (index: {seasonality.get('low_interest', 0)}). "
                              f"Plan campaigns around peak periods.",
                "priority": "high",
                "icon": "fa-calendar-alt",
            })

    # Rising queries insight
    rising = data.get("rising_queries", [])
    breakout_queries = [q for q in rising if q.get("value") == "Breakout"]
    if breakout_queries:
        query_list = ", ".join([q.get("query", "") for q in breakout_queries[:3]])
        insights.append({
            "type": "breakout",
            "title": f"{len(breakout_queries)} breakout queries detected",
            "description": f"These emerging searches have >5000% growth: {query_list}. "
                          f"Consider creating content around these topics.",
            "priority": "high",
            "icon": "fa-rocket",
        })
    elif rising:
        top_rising = rising[0] if rising else {}
        insights.append({
            "type": "rising",
            "title": "Rising search terms identified",
            "description": f"Top rising query: '{top_rising.get('query', 'N/A')}' "
                          f"with {top_rising.get('value', 'N/A')}% growth. "
                          f"These represent emerging opportunities.",
            "priority": "medium",
            "icon": "fa-arrow-trend-up",
        })

    # Geographic insight
    regions = data.get("interest_by_region", [])
    if regions and len(regions) >= 2:
        top_region = regions[0].get("region", "N/A")
        first_kw = data.get("keywords", [""])[0]
        top_interest = regions[0].get(first_kw, 0) if first_kw else 0

        insights.append({
            "type": "geographic",
            "title": f"Highest interest in {top_region}",
            "description": f"{top_region} shows the strongest interest (index: {top_interest}). "
                          f"Consider localizing content and ads for this market.",
            "priority": "medium",
            "icon": "fa-globe",
        })

    # Trend direction insight
    time_series = data.get("interest_over_time", [])
    if len(time_series) >= 6:
        first_kw = data.get("keywords", [""])[0]
        if first_kw:
            recent = [t.get(first_kw, 0) for t in time_series[-3:]]
            older = [t.get(first_kw, 0) for t in time_series[-6:-3]]

            recent_avg = sum(recent) / len(recent) if recent else 0
            older_avg = sum(older) / len(older) if older else 0

            if older_avg > 0:
                change = ((recent_avg - older_avg) / older_avg) * 100
                if change > 15:
                    insights.append({
                        "type": "trend_up",
                        "title": f"Interest growing +{change:.0f}%",
                        "description": "Search interest has increased in recent months. "
                                      "This indicates growing demand - good time to invest.",
                        "priority": "high",
                        "icon": "fa-chart-line",
                    })
                elif change < -15:
                    insights.append({
                        "type": "trend_down",
                        "title": f"Interest declining {change:.0f}%",
                        "description": "Search interest has decreased recently. "
                                      "Consider pivoting strategy or targeting different keywords.",
                        "priority": "medium",
                        "icon": "fa-chart-line-down",
                    })

    return insights[:5]


# =============================================================================
# Historical Data & Trending API Endpoints
# =============================================================================

@app.get("/api/demand/history")
async def get_keyword_history(
    keyword: str = Query(..., description="Keyword to get history for"),
    days: int = Query(default=90, ge=1, le=365, description="Days of history"),
):
    """
    Get historical demand data for a single keyword across all platforms.

    Returns time series data showing how demand has changed over time.
    """
    if repo is None:
        raise HTTPException(status_code=500, detail="Repository not initialized")

    try:
        history = repo.get_keyword_history(keyword, days=days)

        if history.get("error"):
            raise HTTPException(status_code=404, detail=history["error"])

        # Enrich with platform display names and colors
        enriched_history = {}
        for platform, data_points in history.get("history", {}).items():
            enriched_history[platform] = {
                "display_name": _get_platform_display_name(platform),
                "color": _get_platform_color(platform),
                "icon": _get_platform_icon(platform),
                "data": data_points,
            }

        return {
            "keyword": keyword,
            "days": days,
            "platforms": enriched_history,
            "unified_history": history.get("unified_history", []),
            "summary": _generate_history_summary(history),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.get("/api/demand/trending")
async def get_trending_keywords(
    platform: str | None = Query(default=None, description="Filter by platform"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of results"),
):
    """
    Get keywords with the highest growth trends.

    Useful for identifying emerging opportunities.
    """
    if repo is None:
        raise HTTPException(status_code=500, detail="Repository not initialized")

    try:
        trending = repo.get_trending_keywords(platform=platform, limit=limit)

        # Group by keyword for cleaner response
        keyword_trends = {}
        for item in trending:
            kw = item["keyword"]
            if kw not in keyword_trends:
                keyword_trends[kw] = {
                    "keyword": kw,
                    "platforms": [],
                    "max_velocity": 0,
                }

            keyword_trends[kw]["platforms"].append({
                "platform": item["platform"],
                "display_name": _get_platform_display_name(item["platform"]),
                "volume": item["volume"],
                "trend_velocity": item["trend_velocity"],
            })

            if item["trend_velocity"] and item["trend_velocity"] > keyword_trends[kw]["max_velocity"]:
                keyword_trends[kw]["max_velocity"] = item["trend_velocity"]

        # Sort by max velocity
        sorted_keywords = sorted(
            keyword_trends.values(),
            key=lambda x: x["max_velocity"] or 0,
            reverse=True
        )

        return {
            "trending": sorted_keywords,
            "filter": {"platform": platform},
            "count": len(sorted_keywords),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trending: {str(e)}")


@app.get("/api/demand/timeseries")
async def get_demand_timeseries(
    keywords: str = Query(..., description="Comma-separated keywords"),
    days: int = Query(default=30, ge=1, le=90, description="Days of history"),
):
    """
    Get aggregated demand time series for multiple keywords.

    Useful for tracking brand/category demand over time.
    """
    if repo is None:
        raise HTTPException(status_code=500, detail="Repository not initialized")

    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]

    if len(keyword_list) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 keywords")

    try:
        timeseries = repo.get_demand_over_time(keyword_list, days=days)

        if timeseries.get("error"):
            raise HTTPException(status_code=404, detail=timeseries["error"])

        # Calculate growth metrics
        time_series_data = timeseries.get("time_series", [])
        growth_analysis = _analyze_timeseries_growth(time_series_data)

        # Enrich time series with platform colors
        enriched_series = []
        for entry in time_series_data:
            enriched_entry = {
                "date": entry["date"],
                "total": entry["total"],
                "platforms": {}
            }
            for platform, volume in entry.get("platforms", {}).items():
                enriched_entry["platforms"][platform] = {
                    "volume": volume,
                    "color": _get_platform_color(platform),
                }
            enriched_series.append(enriched_entry)

        return {
            "keywords": keyword_list,
            "days": days,
            "time_series": enriched_series,
            "growth": growth_analysis,
            "platform_totals": _calculate_platform_totals(time_series_data),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get timeseries: {str(e)}")


def _generate_history_summary(history: dict) -> dict:
    """Generate summary statistics from historical data."""
    summary = {
        "platforms_tracked": len(history.get("history", {})),
        "data_points": 0,
        "first_date": None,
        "last_date": None,
        "overall_trend": "stable",
    }

    all_dates = []
    trend_scores = []

    for platform, data_points in history.get("history", {}).items():
        summary["data_points"] += len(data_points)
        for dp in data_points:
            if dp.get("date"):
                all_dates.append(dp["date"])
            if dp.get("trend") == "growing":
                trend_scores.append(1)
            elif dp.get("trend") == "declining":
                trend_scores.append(-1)
            else:
                trend_scores.append(0)

    if all_dates:
        summary["first_date"] = min(all_dates)
        summary["last_date"] = max(all_dates)

    if trend_scores:
        avg_trend = sum(trend_scores) / len(trend_scores)
        if avg_trend > 0.3:
            summary["overall_trend"] = "growing"
        elif avg_trend < -0.3:
            summary["overall_trend"] = "declining"

    return summary


def _analyze_timeseries_growth(time_series: list) -> dict:
    """Analyze growth patterns in time series data."""
    if len(time_series) < 2:
        return {"status": "insufficient_data", "change_percent": 0}

    # Compare first half vs second half
    midpoint = len(time_series) // 2
    first_half = time_series[:midpoint]
    second_half = time_series[midpoint:]

    first_total = sum(entry.get("total", 0) for entry in first_half)
    second_total = sum(entry.get("total", 0) for entry in second_half)

    if first_total > 0:
        change_percent = ((second_total - first_total) / first_total) * 100
    else:
        change_percent = 100 if second_total > 0 else 0

    status = "stable"
    if change_percent > 10:
        status = "growing"
    elif change_percent < -10:
        status = "declining"

    return {
        "status": status,
        "change_percent": round(change_percent, 1),
        "first_period_total": first_total,
        "second_period_total": second_total,
    }


def _calculate_platform_totals(time_series: list) -> dict:
    """Calculate total volume by platform from time series."""
    totals = {}

    for entry in time_series:
        for platform, volume in entry.get("platforms", {}).items():
            if platform not in totals:
                totals[platform] = {
                    "volume": 0,
                    "display_name": _get_platform_display_name(platform),
                    "color": _get_platform_color(platform),
                }
            totals[platform]["volume"] += volume

    return totals


# =============================================================================
# Export API - PDF and PowerPoint Reports
# =============================================================================

@app.get("/api/export/pdf")
async def export_demand_pdf(
    keywords: str = Query(..., description="Comma-separated keywords"),
    title: str = Query(default="Search Distribution Analysis", description="Report title"),
    refresh: bool = Query(default=False, description="Force refresh data"),
):
    """
    Export demand analysis as a PDF report.

    Generates a professional PDF with:
    - Executive summary
    - Platform breakdown table
    - Distribution pie chart
    - Key insights
    - Recommendations
    """
    from fastapi.responses import StreamingResponse

    # First get the analysis data
    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]

    if len(keyword_list) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 keywords")

    try:
        # Get analysis data (reuse existing logic)
        from src.db.models import Keyword, KeywordMetric
        from sqlalchemy import select
        from datetime import timedelta

        results = {}
        keywords_to_fetch = []

        if repo:
            with repo.get_session() as session:
                for kw in keyword_list:
                    stmt = select(Keyword).where(Keyword.keyword == kw)
                    existing = session.execute(stmt).scalar_one_or_none()

                    if existing and not refresh:
                        metrics_stmt = (
                            select(KeywordMetric)
                            .where(KeywordMetric.keyword_id == existing.id)
                            .order_by(KeywordMetric.collected_at.desc())
                        )
                        metrics = session.execute(metrics_stmt).scalars().all()
                        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                        recent_metrics = [m for m in metrics if m.collected_at > recent_cutoff]

                        if recent_metrics:
                            results[kw] = _aggregate_keyword_metrics(recent_metrics)
                        else:
                            keywords_to_fetch.append(kw)
                    else:
                        keywords_to_fetch.append(kw)

        if keywords_to_fetch:
            fetched_data = await _fetch_demand_data(keywords_to_fetch)
            results.update(fetched_data)

        distribution = _calculate_demand_distribution(results)

        analysis_data = {
            "keywords": keyword_list,
            "total_demand": distribution["total_volume"],
            "platforms": distribution["platforms"],
            "distribution_summary": distribution["summary"],
            "insights": distribution["insights"],
            "recommendations": distribution["recommendations"],
        }

        # Generate PDF
        try:
            from src.services.export import generate_demand_report_pdf
            pdf_bytes = generate_demand_report_pdf(analysis_data, title)
        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"PDF export requires reportlab library. Error: {str(e)}"
            )

        # Return as downloadable file
        filename = f"demand_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")


@app.get("/api/export/pptx")
async def export_demand_pptx(
    keywords: str = Query(..., description="Comma-separated keywords"),
    title: str = Query(default="Search Distribution Analysis", description="Report title"),
    refresh: bool = Query(default=False, description="Force refresh data"),
):
    """
    Export demand analysis as a PowerPoint presentation.

    Generates a professional PPTX with:
    - Title slide
    - Executive summary with key metrics
    - Platform breakdown table
    - Key insights
    - Recommendations
    """
    from fastapi.responses import StreamingResponse

    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]

    if len(keyword_list) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 keywords")

    try:
        # Get analysis data (same as PDF)
        from src.db.models import Keyword, KeywordMetric
        from sqlalchemy import select
        from datetime import timedelta

        results = {}
        keywords_to_fetch = []

        if repo:
            with repo.get_session() as session:
                for kw in keyword_list:
                    stmt = select(Keyword).where(Keyword.keyword == kw)
                    existing = session.execute(stmt).scalar_one_or_none()

                    if existing and not refresh:
                        metrics_stmt = (
                            select(KeywordMetric)
                            .where(KeywordMetric.keyword_id == existing.id)
                            .order_by(KeywordMetric.collected_at.desc())
                        )
                        metrics = session.execute(metrics_stmt).scalars().all()
                        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                        recent_metrics = [m for m in metrics if m.collected_at > recent_cutoff]

                        if recent_metrics:
                            results[kw] = _aggregate_keyword_metrics(recent_metrics)
                        else:
                            keywords_to_fetch.append(kw)
                    else:
                        keywords_to_fetch.append(kw)

        if keywords_to_fetch:
            fetched_data = await _fetch_demand_data(keywords_to_fetch)
            results.update(fetched_data)

        distribution = _calculate_demand_distribution(results)

        analysis_data = {
            "keywords": keyword_list,
            "total_demand": distribution["total_volume"],
            "platforms": distribution["platforms"],
            "distribution_summary": distribution["summary"],
            "insights": distribution["insights"],
            "recommendations": distribution["recommendations"],
        }

        # Generate PPTX
        try:
            from src.services.export import generate_demand_report_pptx
            pptx_bytes = generate_demand_report_pptx(analysis_data, title)
        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"PowerPoint export requires python-pptx library. Error: {str(e)}"
            )

        filename = f"demand_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pptx"

        return StreamingResponse(
            io.BytesIO(pptx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pptx_bytes)),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PowerPoint export failed: {str(e)}")


@app.get("/api/export/comparison/pdf")
async def export_comparison_pdf(
    brand_keywords: str = Query(..., description="Your brand keywords"),
    competitor_keywords: str = Query(..., description="Competitor keywords"),
    brand_name: str = Query(default="Your Brand", description="Brand name for report"),
    competitor_name: str = Query(default="Competitor", description="Competitor name for report"),
):
    """
    Export competitor comparison as a PDF report.

    Includes share of voice analysis across platforms.
    """
    from fastapi.responses import StreamingResponse

    brand_kws = [k.strip() for k in brand_keywords.split(",") if k.strip()]
    competitor_kws = [k.strip() for k in competitor_keywords.split(",") if k.strip()]

    if len(brand_kws) + len(competitor_kws) > 30:
        raise HTTPException(status_code=400, detail="Maximum 30 total keywords")

    try:
        brand_data = await _fetch_demand_data(brand_kws)
        competitor_data = await _fetch_demand_data(competitor_kws)

        brand_distribution = _calculate_demand_distribution(brand_data)
        competitor_distribution = _calculate_demand_distribution(competitor_data)

        share_of_voice = _calculate_share_of_voice(brand_distribution, competitor_distribution)

        # Build comparison analysis data
        analysis_data = {
            "keywords": brand_kws + competitor_kws,
            "total_demand": brand_distribution["total_volume"] + competitor_distribution["total_volume"],
            "platforms": brand_distribution["platforms"],
            "distribution_summary": {
                **brand_distribution["summary"],
                "comparison_type": "share_of_voice",
                "brand_share": share_of_voice["overall"].get("brand", 0),
                "competitor_share": share_of_voice["overall"].get("competitor", 0),
            },
            "insights": _generate_competitive_insights(brand_distribution, competitor_distribution),
            "recommendations": brand_distribution.get("recommendations", []),
        }

        try:
            from src.services.export import generate_demand_report_pdf
            pdf_bytes = generate_demand_report_pdf(
                analysis_data,
                title=f"{brand_name} vs {competitor_name} - Competitive Analysis"
            )
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"PDF export requires reportlab. Error: {str(e)}")

        filename = f"comparison_{brand_name}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison export failed: {str(e)}")


# =============================================================================
# Calibration API Endpoints
# =============================================================================

@app.get("/api/calibration/status")
async def get_calibration_status():
    """
    Get the status of calibration models for each platform.

    Shows how many calibration points are collected and whether models are fitted.
    """
    try:
        from src.calibration.regression_models import (
            get_tiktok_calibrator,
            get_instagram_calibrator,
            get_youtube_calibrator,
            get_pinterest_calibrator,
        )

        return {
            "tiktok": get_tiktok_calibrator().get_model_stats(),
            "instagram": get_instagram_calibrator().get_model_stats(),
            "youtube": get_youtube_calibrator().get_model_stats(),
            "pinterest": get_pinterest_calibrator().get_model_stats(),
        }
    except ImportError:
        return {"error": "Calibration module not available"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/calibration/keywords")
async def get_calibration_keywords():
    """Get the recommended calibration keywords to collect data for."""
    try:
        from src.calibration.calibration_keywords import (
            get_all_calibration_keywords,
            get_calibration_keywords_by_volume,
        )

        return {
            "keywords": get_all_calibration_keywords(),
            "by_volume": get_calibration_keywords_by_volume(),
            "total_count": len(get_all_calibration_keywords()),
        }
    except ImportError:
        return {"error": "Calibration module not available"}


class CalibrationDataRequest(BaseModel):
    """Request model for submitting calibration data from Keywordtool.io."""
    keyword: str = Field(..., min_length=1, max_length=200)
    tiktok_volume: int = Field(default=0, ge=0)
    instagram_volume: int = Field(default=0, ge=0)
    youtube_volume: int = Field(default=0, ge=0)
    pinterest_volume: int = Field(default=0, ge=0)


@app.post("/api/calibration/add")
async def add_calibration_data(request: CalibrationDataRequest):
    """
    Add calibration data from Keywordtool.io for a keyword.

    This collects our raw metrics for the keyword and pairs it with
    the ground truth from Keywordtool.io.
    """
    try:
        from src.calibration.regression_models import (
            CalibrationPoint,
            get_tiktok_calibrator,
            get_instagram_calibrator,
            get_youtube_calibrator,
            get_pinterest_calibrator,
        )

        # Fetch our raw metrics for this keyword
        settings = get_settings()
        keyword = request.keyword

        # Collect metrics from our APIs
        raw_metrics = {}

        # TikTok
        if request.tiktok_volume > 0:
            try:
                from src.clients.apify import ApifyClient
                async with ApifyClient(settings=settings) as client:
                    hashtag = keyword.replace(" ", "").lower()
                    data = await client.run_tiktok_hashtag_scraper([hashtag], results_per_hashtag=20, timeout_secs=45)
                    raw_metrics["tiktok"] = data.get(hashtag, {})
            except Exception as e:
                raw_metrics["tiktok_error"] = str(e)

        # Instagram
        if request.instagram_volume > 0:
            try:
                from src.clients.apify import ApifyClient
                async with ApifyClient(settings=settings) as client:
                    hashtag = keyword.replace(" ", "").lower()
                    data = await client.run_instagram_hashtag_scraper([hashtag], results_per_hashtag=20, timeout_secs=45)
                    raw_metrics["instagram"] = data.get(hashtag, {})
            except Exception as e:
                raw_metrics["instagram_error"] = str(e)

        # YouTube (Google Trends)
        if request.youtube_volume > 0:
            try:
                from src.clients.google_trends import GoogleTrendsClient
                async with GoogleTrendsClient(settings=settings) as client:
                    metrics = await client.get_youtube_search_volume([keyword], geo="DE")
                    if metrics:
                        raw_metrics["youtube"] = metrics[0].raw_data or {}
            except Exception as e:
                raw_metrics["youtube_error"] = str(e)

        # Pinterest
        if request.pinterest_volume > 0:
            try:
                from src.clients.pinterest import PinterestClient
                async with PinterestClient(settings=settings) as client:
                    metrics = await client.get_search_volume(keyword, country="DE")
                    raw_metrics["pinterest"] = metrics.raw_data or {}
            except Exception as e:
                raw_metrics["pinterest_error"] = str(e)

        # Add calibration points
        added = []

        # TikTok
        if request.tiktok_volume > 0 and "tiktok" in raw_metrics:
            stats = raw_metrics["tiktok"].get("stats", {})
            point = CalibrationPoint(
                keyword=keyword,
                keywordtool_volume=request.tiktok_volume,
                tiktok_views=stats.get("total_views", 0),
                tiktok_video_count=stats.get("video_count", 0),
                tiktok_avg_likes=stats.get("avg_likes", 0),
                tiktok_avg_shares=stats.get("avg_shares", 0),
            )
            get_tiktok_calibrator().add_calibration_point(point)
            added.append("tiktok")

        # Instagram
        if request.instagram_volume > 0 and "instagram" in raw_metrics:
            stats = raw_metrics["instagram"].get("stats", {})
            point = CalibrationPoint(
                keyword=keyword,
                keywordtool_volume=request.instagram_volume,
                instagram_post_count=stats.get("post_count", 0),
                instagram_daily_posts=stats.get("daily_posts", 0),
                instagram_avg_likes=stats.get("avg_likes", 0),
                instagram_avg_comments=stats.get("avg_comments", 0),
            )
            get_instagram_calibrator().add_calibration_point(point)
            added.append("instagram")

        # YouTube
        if request.youtube_volume > 0 and "youtube" in raw_metrics:
            point = CalibrationPoint(
                keyword=keyword,
                keywordtool_volume=request.youtube_volume,
                youtube_trends_index=raw_metrics["youtube"].get("trends_index", 0),
            )
            get_youtube_calibrator().add_calibration_point(point)
            added.append("youtube")

        # Pinterest
        if request.pinterest_volume > 0 and "pinterest" in raw_metrics:
            point = CalibrationPoint(
                keyword=keyword,
                keywordtool_volume=request.pinterest_volume,
                pinterest_interest_score=raw_metrics["pinterest"].get("interest_score", 0),
            )
            get_pinterest_calibrator().add_calibration_point(point)
            added.append("pinterest")

        return {
            "success": True,
            "keyword": keyword,
            "platforms_added": added,
            "raw_metrics_collected": {k: v for k, v in raw_metrics.items() if not k.endswith("_error")},
            "errors": {k: v for k, v in raw_metrics.items() if k.endswith("_error")},
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="Calibration module not available")
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Failed to add calibration data: {str(e)}\n{traceback.format_exc()}")


@app.post("/api/calibration/fit")
async def fit_calibration_models():
    """
    Fit regression models using collected calibration data.

    Call this after adding enough calibration points (10+ per platform).
    """
    try:
        from src.calibration.regression_models import (
            get_tiktok_calibrator,
            get_instagram_calibrator,
            get_youtube_calibrator,
            get_pinterest_calibrator,
        )

        results = {
            "tiktok": get_tiktok_calibrator().fit_model(),
            "instagram": get_instagram_calibrator().fit_model(),
            "youtube": get_youtube_calibrator().fit_model(),
            "pinterest": get_pinterest_calibrator().fit_model(),
        }

        return {
            "success": True,
            "models": results,
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="Calibration module not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fit models: {str(e)}")


# Vercel serverless handler
app_handler = app
