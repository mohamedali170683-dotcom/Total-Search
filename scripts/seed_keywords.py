#!/usr/bin/env python3
"""
Seed initial keyword list into the database.

Usage:
    python scripts/seed_keywords.py --file data/keywords.csv
    python scripts/seed_keywords.py --keywords "skincare,makeup,beauty"
    python scripts/seed_keywords.py --category beauty --count 100
"""

import asyncio
import csv
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.db.repository import KeywordRepository

console = Console()
logger = logging.getLogger(__name__)

# Predefined keyword lists by category
CATEGORY_KEYWORDS = {
    "beauty": [
        "skincare routine", "vitamin c serum", "retinol", "hyaluronic acid",
        "niacinamide", "salicylic acid", "moisturizer", "sunscreen spf",
        "face wash", "cleanser", "toner", "serum", "eye cream", "face mask",
        "anti aging", "acne treatment", "dark spots", "wrinkles",
        "dry skin", "oily skin", "sensitive skin", "combination skin",
        "korean skincare", "natural skincare", "organic beauty",
    ],
    "fitness": [
        "home workout", "weight loss", "muscle building", "protein powder",
        "creatine", "pre workout", "yoga", "pilates", "hiit workout",
        "strength training", "cardio", "gym equipment", "resistance bands",
        "dumbbells", "kettlebell", "running shoes", "fitness tracker",
        "meal prep", "calorie deficit", "intermittent fasting",
    ],
    "tech": [
        "smartphone", "laptop", "wireless earbuds", "smart watch",
        "tablet", "gaming pc", "monitor", "keyboard", "mouse",
        "webcam", "microphone", "ring light", "usb hub", "power bank",
        "portable charger", "bluetooth speaker", "noise cancelling",
        "mechanical keyboard", "gaming headset", "streaming setup",
    ],
    "fashion": [
        "summer dress", "winter coat", "sneakers", "handbag",
        "sunglasses", "jewelry", "watch", "belt", "scarf",
        "jeans", "t shirt", "hoodie", "blazer", "boots",
        "sandals", "activewear", "loungewear", "sustainable fashion",
        "vintage clothing", "designer bags",
    ],
    "home": [
        "air purifier", "robot vacuum", "coffee maker", "blender",
        "instant pot", "air fryer", "mattress", "pillows", "bedding",
        "curtains", "rug", "lamp", "desk chair", "standing desk",
        "bookshelf", "storage bins", "organization", "home decor",
        "plants", "candles",
    ],
}


def load_keywords_from_csv(filepath: Path) -> list[tuple[str, list[str]]]:
    """Load keywords from CSV file.

    Expected format:
    keyword,tags
    "vitamin c serum","skincare,beauty"
    """
    keywords = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keyword = row.get("keyword", "").strip()
            tags_str = row.get("tags", "")
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            if keyword:
                keywords.append((keyword, tags))
    return keywords


def parse_inline_keywords(keywords_str: str) -> list[tuple[str, list[str]]]:
    """Parse comma-separated keywords string."""
    keywords = []
    for kw in keywords_str.split(","):
        kw = kw.strip()
        if kw:
            keywords.append((kw, []))
    return keywords


def get_category_keywords(category: str, count: int) -> list[tuple[str, list[str]]]:
    """Get keywords from predefined category."""
    category_lower = category.lower()
    if category_lower not in CATEGORY_KEYWORDS:
        available = ", ".join(CATEGORY_KEYWORDS.keys())
        raise ValueError(f"Unknown category '{category}'. Available: {available}")

    keywords = CATEGORY_KEYWORDS[category_lower][:count]
    return [(kw, [category_lower]) for kw in keywords]


@click.command()
@click.option(
    "--file", "-f",
    type=click.Path(exists=True, path_type=Path),
    help="CSV file with keywords (columns: keyword, tags)"
)
@click.option(
    "--keywords", "-k",
    type=str,
    help="Comma-separated list of keywords"
)
@click.option(
    "--category", "-c",
    type=click.Choice(list(CATEGORY_KEYWORDS.keys()), case_sensitive=False),
    help="Predefined category to seed"
)
@click.option(
    "--count", "-n",
    type=int,
    default=25,
    help="Number of keywords from category (default: 25)"
)
@click.option(
    "--tags", "-t",
    type=str,
    default="",
    help="Additional tags to add to all keywords"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be added without saving"
)
def main(
    file: Path | None,
    keywords: str | None,
    category: str | None,
    count: int,
    tags: str,
    dry_run: bool,
):
    """Seed keywords into the database."""
    console.print("[bold blue]Keyword Seeder[/bold blue]")
    console.print("=" * 50)

    # Collect keywords from all sources
    all_keywords: list[tuple[str, list[str]]] = []

    if file:
        console.print(f"\n[cyan]Loading from file:[/cyan] {file}")
        all_keywords.extend(load_keywords_from_csv(file))

    if keywords:
        console.print(f"\n[cyan]Parsing inline keywords[/cyan]")
        all_keywords.extend(parse_inline_keywords(keywords))

    if category:
        console.print(f"\n[cyan]Loading category:[/cyan] {category} (up to {count})")
        all_keywords.extend(get_category_keywords(category, count))

    if not all_keywords:
        console.print("[red]No keywords provided![/red]")
        console.print("Use --file, --keywords, or --category to specify keywords")
        raise SystemExit(1)

    # Add extra tags if provided
    extra_tags = [t.strip() for t in tags.split(",") if t.strip()]
    if extra_tags:
        all_keywords = [
            (kw, list(set(kw_tags + extra_tags)))
            for kw, kw_tags in all_keywords
        ]

    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_keywords = []
    for kw, kw_tags in all_keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append((kw, kw_tags))

    # Display table
    table = Table(title=f"Keywords to Seed ({len(unique_keywords)} total)")
    table.add_column("Keyword", style="cyan")
    table.add_column("Tags", style="green")

    for kw, kw_tags in unique_keywords[:20]:  # Show first 20
        table.add_row(kw, ", ".join(kw_tags) if kw_tags else "-")

    if len(unique_keywords) > 20:
        table.add_row(f"... and {len(unique_keywords) - 20} more", "")

    console.print(table)

    if dry_run:
        console.print("\n[yellow]DRY RUN - no changes made[/yellow]")
        return

    # Save to database
    console.print("\n[cyan]Saving to database...[/cyan]")

    settings = get_settings()
    repo = KeywordRepository(settings=settings)
    repo.create_tables()

    added = 0
    skipped = 0

    for kw, kw_tags in unique_keywords:
        try:
            # Check if keyword exists
            existing = repo.get_keyword(kw)
            if existing:
                skipped += 1
            else:
                # Add keyword (simplified - just save the keyword entry)
                repo._get_or_create_keyword(kw, kw_tags)
                added += 1
        except Exception as e:
            console.print(f"[red]Error adding '{kw}': {e}[/red]")

    console.print(f"\n[green]Done![/green]")
    console.print(f"  Added: {added}")
    console.print(f"  Skipped (existing): {skipped}")

    # Show stats
    stats = repo.get_statistics()
    console.print(f"\n[bold]Database Stats:[/bold]")
    console.print(f"  Total keywords: {stats.get('total_keywords', 0)}")
    console.print(f"  Total metrics: {stats.get('total_metrics', 0)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
