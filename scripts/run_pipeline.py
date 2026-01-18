#!/usr/bin/env python3
"""
CLI to run the keyword research pipeline.

Usage:
    python scripts/run_pipeline.py --input data/keywords.csv --output data/output/
    python scripts/run_pipeline.py --keywords "skincare,makeup,beauty" --platforms google,amazon
    python scripts/run_pipeline.py --resume data/checkpoints/checkpoint_20260118_120000.json
"""

import asyncio
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config import get_settings
from src.db.repository import KeywordRepository
from src.models.keyword import Platform
from src.pipeline.keyword_pipeline import KeywordPipeline, PipelineOptions

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_keywords_from_csv(file_path: Path) -> list[str]:
    """Load keywords from a CSV file."""
    keywords = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # Skip empty rows
                keywords.append(row[0].strip())
    return keywords


def load_keywords_from_text(file_path: Path) -> list[str]:
    """Load keywords from a plain text file (one per line)."""
    with open(file_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_platforms(platforms_str: str) -> list[Platform]:
    """Parse comma-separated platform names."""
    platforms = []
    for p in platforms_str.split(","):
        p = p.strip().lower()
        try:
            platforms.append(Platform(p))
        except ValueError:
            console.print(f"[yellow]Warning: Unknown platform '{p}', skipping[/yellow]")
    return platforms


@click.command()
@click.option(
    "--input", "-i",
    "input_file",
    type=click.Path(exists=True, path_type=Path),
    help="Input file (CSV or TXT) containing keywords",
)
@click.option(
    "--keywords", "-k",
    type=str,
    help="Comma-separated list of keywords",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("data/output"),
    help="Output directory for results",
)
@click.option(
    "--platforms", "-p",
    type=str,
    default="google,youtube,amazon,tiktok,instagram",
    help="Comma-separated list of platforms to query",
)
@click.option(
    "--weight-preset",
    type=click.Choice(["balanced", "ecommerce", "content", "seo", "video"]),
    default="balanced",
    help="Weight preset for unified score calculation",
)
@click.option(
    "--batch-size",
    type=int,
    default=50,
    help="Number of keywords per batch",
)
@click.option(
    "--resume",
    type=click.Path(exists=True, path_type=Path),
    help="Resume from checkpoint file",
)
@click.option(
    "--save-to-db/--no-save-to-db",
    default=True,
    help="Save results to database",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
def cli(
    input_file: Path | None,
    keywords: str | None,
    output: Path,
    platforms: str,
    weight_preset: str,
    batch_size: int,
    resume: Path | None,
    save_to_db: bool,
    log_level: str,
) -> None:
    """Run the keyword research pipeline."""
    setup_logging(log_level)

    # Validate input
    if not input_file and not keywords and not resume:
        console.print("[red]Error: Must provide --input, --keywords, or --resume[/red]")
        raise SystemExit(1)

    # Load keywords
    keyword_list: list[str] = []

    if keywords:
        keyword_list = [k.strip() for k in keywords.split(",")]

    if input_file:
        if input_file.suffix == ".csv":
            keyword_list.extend(load_keywords_from_csv(input_file))
        else:
            keyword_list.extend(load_keywords_from_text(input_file))

    # Remove duplicates while preserving order
    keyword_list = list(dict.fromkeys(keyword_list))

    # Parse platforms
    platform_list = parse_platforms(platforms)

    console.print(f"\n[bold]Keyword Research Pipeline[/bold]")
    console.print(f"Keywords: {len(keyword_list)}")
    console.print(f"Platforms: {', '.join(p.value for p in platform_list)}")
    console.print(f"Weight Preset: {weight_preset}")
    console.print()

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Configure pipeline options
    options = PipelineOptions(
        platforms=platform_list,
        weight_preset=weight_preset,
        batch_size=batch_size,
        save_checkpoints=True,
        checkpoint_dir=output / "checkpoints",
    )

    # Run pipeline
    asyncio.run(run_pipeline(
        keyword_list,
        options,
        output,
        resume,
        save_to_db,
    ))


async def run_pipeline(
    keywords: list[str],
    options: PipelineOptions,
    output_dir: Path,
    resume_file: Path | None,
    save_to_db: bool,
) -> None:
    """Execute the pipeline asynchronously."""
    settings = get_settings()

    async with KeywordPipeline(settings=settings) as pipeline:
        # Load checkpoint if resuming
        existing_results = []
        if resume_file:
            existing_results = pipeline.load_checkpoint(resume_file)
            console.print(f"[green]Resumed from checkpoint with {len(existing_results)} keywords[/green]")

        # Run pipeline with progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing keywords...", total=None)

            results = await pipeline.run(keywords, options)

        # Combine with existing results
        all_results = existing_results + results

        console.print(f"\n[green]✓ Processed {len(results)} keywords[/green]")

        # Display summary table
        display_results_summary(all_results[:10])  # Show top 10

        # Save to database
        if save_to_db:
            console.print("\n[bold]Saving to database...[/bold]")
            repo = KeywordRepository()
            repo.create_tables()  # Ensure tables exist
            repo.save_batch(all_results)
            console.print("[green]✓ Saved to database[/green]")

        # Export results
        output_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        export_results(all_results, output_file)
        console.print(f"[green]✓ Exported to {output_file}[/green]")


def display_results_summary(results: list) -> None:
    """Display a summary table of results."""
    table = Table(title="Top Results")
    table.add_column("Keyword", style="cyan")
    table.add_column("Unified Score", justify="right", style="green")
    table.add_column("Trend", style="yellow")
    table.add_column("Best Platform", style="magenta")
    table.add_column("Platforms", justify="right")

    for r in results:
        platforms_count = len(r.available_platforms)
        table.add_row(
            r.keyword[:30],
            str(r.unified_demand_score),
            r.cross_platform_trend.value if r.cross_platform_trend else "-",
            r.best_platform.value if r.best_platform else "-",
            str(platforms_count),
        )

    console.print(table)


def export_results(results: list, output_file: Path) -> None:
    """Export results to JSONL file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_rag_document(), default=str) + "\n")


if __name__ == "__main__":
    cli()
