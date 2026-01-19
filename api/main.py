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
    stats = {"total_keywords": 0, "total_metrics": 0, "metrics_by_platform": {}}
    keywords = []
    return templates.TemplateResponse(
        "dashboard_simple.html",
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


# Vercel serverless handler
app_handler = app
