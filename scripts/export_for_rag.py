#!/usr/bin/env python3
"""
Export keyword data in formats suitable for RAG ingestion.

Usage:
    python scripts/export_for_rag.py --format jsonl --output data/output/keywords.jsonl
    python scripts/export_for_rag.py --format csv --since 2026-01-01
    python scripts/export_for_rag.py --format parquet --output data/output/keywords.parquet
"""

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from src.db.repository import KeywordRepository

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@click.command()
@click.option(
    "--format", "-f",
    "output_format",
    type=click.Choice(["jsonl", "csv", "json"]),
    default="jsonl",
    help="Output format",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (default: stdout or auto-generated)",
)
@click.option(
    "--since",
    type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]),
    default=None,
    help="Only export data updated since this date",
)
@click.option(
    "--tag",
    type=str,
    default=None,
    help="Filter by tag",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of keywords to export",
)
@click.option(
    "--include-raw/--no-include-raw",
    default=False,
    help="Include raw API response data",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
def cli(
    output_format: str,
    output: Path | None,
    since: datetime | None,
    tag: str | None,
    limit: int | None,
    include_raw: bool,
    log_level: str,
) -> None:
    """Export keyword data for RAG system ingestion."""
    setup_logging(log_level)

    # Initialize repository
    repo = KeywordRepository()

    # Show database stats
    stats = repo.get_statistics()
    console.print("\n[bold]Database Statistics[/bold]")
    console.print(f"Total keywords: {stats['total_keywords']}")
    console.print(f"Total metrics: {stats['total_metrics']}")
    if stats['metrics_by_platform']:
        console.print("Metrics by platform:")
        for platform, count in stats['metrics_by_platform'].items():
            console.print(f"  - {platform}: {count}")
    console.print()

    # Fetch data
    console.print("[bold]Fetching data...[/bold]")

    if tag:
        keyword_data = repo.get_keywords_by_tag(tag)
    else:
        keyword_data = repo.get_all_keywords(limit=limit or 100000)

    # Filter by date if specified
    if since:
        keyword_data = [
            kw for kw in keyword_data
            if kw.timestamp and kw.timestamp >= since
        ]

    # Apply limit
    if limit:
        keyword_data = keyword_data[:limit]

    console.print(f"[green]Found {len(keyword_data)} keywords to export[/green]")

    if not keyword_data:
        console.print("[yellow]No data to export[/yellow]")
        return

    # Convert to export format
    export_data = [kw.to_rag_document() for kw in keyword_data]

    # Generate output path if not specified
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(f"data/output/keywords_{timestamp}.{output_format}")

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Export based on format
    if output_format == "jsonl":
        export_jsonl(export_data, output)
    elif output_format == "csv":
        export_csv(export_data, output)
    elif output_format == "json":
        export_json(export_data, output)

    console.print(f"[green]✓ Exported {len(export_data)} keywords to {output}[/green]")

    # Show sample
    display_sample(export_data[:5])


def export_jsonl(data: list[dict], output_path: Path) -> None:
    """Export data as JSONL (JSON Lines)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, default=str, ensure_ascii=False) + "\n")


def export_csv(data: list[dict], output_path: Path) -> None:
    """Export data as CSV."""
    if not data:
        return

    # Flatten the data for CSV
    flattened = []
    for item in data:
        flat = {
            "keyword": item.get("keyword"),
            "unified_demand_score": item.get("unified_demand_score"),
            "cross_platform_trend": item.get("cross_platform_trend"),
            "best_platform": item.get("best_platform"),
            "collected_at": item.get("collected_at"),
        }

        # Add platform-specific columns
        platforms = item.get("platforms", {})
        for platform, metrics in platforms.items():
            if metrics:
                flat[f"{platform}_volume"] = metrics.get("volume")
                flat[f"{platform}_trend"] = metrics.get("trend")
                flat[f"{platform}_competition"] = metrics.get("competition")

        flattened.append(flat)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if flattened:
            writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
            writer.writeheader()
            writer.writerows(flattened)


def export_json(data: list[dict], output_path: Path) -> None:
    """Export data as a single JSON array."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str, ensure_ascii=False, indent=2)


def display_sample(data: list[dict]) -> None:
    """Display a sample of exported data."""
    if not data:
        return

    console.print("\n[bold]Sample Data[/bold]")
    table = Table()
    table.add_column("Keyword", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Trend", style="yellow")
    table.add_column("Best Platform", style="magenta")

    for item in data:
        table.add_row(
            str(item.get("keyword", ""))[:40],
            str(item.get("unified_demand_score", 0)),
            str(item.get("cross_platform_trend", "-")),
            str(item.get("best_platform", "-")),
        )

    console.print(table)


if __name__ == "__main__":
    cli()
