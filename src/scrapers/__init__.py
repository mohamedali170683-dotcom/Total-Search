"""Social media scrapers using Apify actors."""

from src.scrapers.tiktok import TikTokScraper
from src.scrapers.instagram import InstagramScraper

__all__ = ["TikTokScraper", "InstagramScraper"]
