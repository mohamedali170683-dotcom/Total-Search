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
project_root = Path(__file__).parent.resolve()
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
                            # Only mark as "has data" if we have real search volume (not just proxy)
                            if metric.search_volume and metric.search_volume > 0:
                                has_data = True
                            elif metric.proxy_score and metric.proxy_score > 0:
                                # For social platforms (TikTok, Instagram), proxy score is valid data
                                if platform in ['tiktok', 'instagram']:
                                    has_data = True

                            volume = metric.search_volume or metric.proxy_score or 0
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

                platform_metrics[platform] = {
                    "volume": total_volume,
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
                                total_vol += metric.search_volume or metric.proxy_score or 0

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

        # Calculate demand distribution
        distribution = _calculate_demand_distribution(results, weight_preset=weight_preset)

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

            # Add per-platform data
            for platform, metrics in result.platforms_dict.items():
                if metrics:
                    results[result.keyword]["platforms"][platform.value] = {
                        "volume": metrics.effective_volume,
                        "trend": metrics.trend.value if metrics.trend else "stable",
                        "trend_velocity": metrics.trend_velocity,
                        "confidence": metrics.confidence.value,
                    }
                else:
                    results[result.keyword]["platforms"][platform.value] = {
                        "volume": 0,
                        "trend": None,
                        "trend_velocity": None,
                        "confidence": "none",
                    }

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
    - 12-month interest trend chart data (Google Web + YouTube + TikTok)
    - Geographic hotspots (top 10 countries)
    - Rising queries (emerging search terms)
    - Top related queries
    - Seasonality analysis
    - Cross-platform trend correlation (if include_correlation=True)
    """
    try:
        from src.clients.google_trends import GoogleTrendsClient
        from src.clients.tickertrends import TickerTrendsClient

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

                # Fetch TikTok trends if configured
                tiktok_data = {}
                try:
                    tt_client = TickerTrendsClient(settings=settings)
                    if tt_client.is_configured and keyword_list:
                        tiktok_data = await tt_client.get_hashtag_trends(
                            hashtag=keyword_list[0].replace(" ", ""),
                        )
                        if tiktok_data.get("status") == "ok":
                            multi_platform.setdefault("platforms", {})["tiktok"] = {
                                "interest_over_time": tiktok_data.get("interest_over_time", []),
                            }
                except Exception as e:
                    logger.warning(f"TikTok trends failed: {e}")

                # Compute correlation
                if multi_platform.get("platforms"):
                    correlation_analysis = _compute_trend_correlation(
                        multi_platform.get("platforms", {}),
                        keyword_list[0],
                    )

        # Enrich with insights
        insights = _generate_trends_insights(data)

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

        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trends intelligence: {str(e)}\n{traceback.format_exc()}"
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

    Extracts time series for the primary keyword from each platform,
    computes Pearson correlation, and detects lead/lag relationships.
    """
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy not installed"}

    results: dict = {"pairs": [], "insights": []}

    # Extract time series values per platform
    series_by_platform: dict[str, list[float]] = {}

    for platform_name, platform_data in platforms_data.items():
        iot = platform_data.get("interest_over_time", [])
        if not iot:
            continue

        if platform_name == "tiktok":
            # TikTok uses "tiktok" key instead of keyword name
            values = [float(entry.get("tiktok", 0)) for entry in iot]
        else:
            values = [float(entry.get(primary_keyword, 0)) for entry in iot]

        if values and any(v > 0 for v in values):
            series_by_platform[platform_name] = values

    if len(series_by_platform) < 2:
        return results

    # Compare each pair
    platform_names = list(series_by_platform.keys())
    for i in range(len(platform_names)):
        for j in range(i + 1, len(platform_names)):
            p1, p2 = platform_names[i], platform_names[j]
            s1, s2 = series_by_platform[p1], series_by_platform[p2]

            # Align lengths (trim to shorter)
            min_len = min(len(s1), len(s2))
            if min_len < 4:
                continue
            a1 = np.array(s1[-min_len:], dtype=float)
            a2 = np.array(s2[-min_len:], dtype=float)

            # Skip if either series is constant
            if np.std(a1) == 0 or np.std(a2) == 0:
                continue

            # Base correlation (no lag)
            base_corr = float(np.corrcoef(a1, a2)[0, 1])

            # Cross-correlate with lags 1-4 to find lead/lag
            best_lag = 0
            best_corr = abs(base_corr)
            best_corr_signed = base_corr

            for lag in range(1, min(5, min_len - 2)):
                # p2 leads p1 by 'lag' periods (shift p2 back)
                corr_val = float(np.corrcoef(a1[lag:], a2[:-lag])[0, 1])
                if abs(corr_val) > best_corr:
                    best_corr = abs(corr_val)
                    best_corr_signed = corr_val
                    best_lag = lag

                # p1 leads p2 by 'lag' periods (shift p1 back)
                corr_val = float(np.corrcoef(a1[:-lag], a2[lag:])[0, 1])
                if abs(corr_val) > best_corr:
                    best_corr = abs(corr_val)
                    best_corr_signed = corr_val
                    best_lag = -lag

            # Generate insight
            pair_result = {
                "platform_a": p1,
                "platform_b": p2,
                "correlation": round(best_corr_signed, 3),
                "lag_periods": best_lag,
                "relationship": "synchronized",
            }

            if best_corr > 0.4 and best_lag > 0:
                pair_result["relationship"] = f"{p2} leads {p1} by ~{best_lag} week(s)"
                results["insights"].append(
                    f"{p2.replace('_', ' ').title()} trends lead "
                    f"{p1.replace('_', ' ').title()} by ~{best_lag} week(s) "
                    f"(r={best_corr_signed:.2f}). Social signals may predict search demand."
                )
            elif best_corr > 0.4 and best_lag < 0:
                pair_result["relationship"] = f"{p1} leads {p2} by ~{abs(best_lag)} week(s)"
                results["insights"].append(
                    f"{p1.replace('_', ' ').title()} trends lead "
                    f"{p2.replace('_', ' ').title()} by ~{abs(best_lag)} week(s) "
                    f"(r={best_corr_signed:.2f})."
                )
            elif best_corr > 0.6:
                pair_result["relationship"] = "strongly synchronized"
                results["insights"].append(
                    f"{p1.replace('_', ' ').title()} and "
                    f"{p2.replace('_', ' ').title()} trends are strongly correlated "
                    f"(r={best_corr_signed:.2f}). Interest moves together across platforms."
                )
            elif best_corr < 0.2:
                pair_result["relationship"] = "independent"
                results["insights"].append(
                    f"{p1.replace('_', ' ').title()} and "
                    f"{p2.replace('_', ' ').title()} trends appear independent "
                    f"(r={best_corr_signed:.2f}). Platforms serve different audiences."
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


# Vercel serverless handler
app_handler = app
