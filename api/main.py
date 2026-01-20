"""FastAPI application for keyword research tool."""

import asyncio
import json
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
    title="Total Search - Keyword Research Tool",
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
                            volume = metric.search_volume or metric.proxy_score or 0
                            total_volume += volume
                            if metric.trend_velocity:
                                avg_trend += metric.trend_velocity
                                count += 1

                platform_metrics[platform] = {
                    "volume": total_volume,
                    "trend": round(avg_trend / count * 100, 1) if count > 0 else 0,
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
async def refresh_brand_data(brand_id: int):
    """Refresh data for a brand by re-researching all variants (synchronous for Vercel)."""
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

            # Remove duplicates
            all_keywords = list(set(all_keywords))

        if not all_keywords:
            return {"success": True, "keywords_researched": 0, "message": "No keywords to research"}

        # Run synchronously (Vercel serverless doesn't support background tasks well)
        settings = get_settings()
        platforms = [Platform.GOOGLE, Platform.YOUTUBE, Platform.AMAZON, Platform.TIKTOK, Platform.INSTAGRAM]

        options = PipelineOptions(
            platforms=platforms,
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


# Vercel serverless handler
app_handler = app
