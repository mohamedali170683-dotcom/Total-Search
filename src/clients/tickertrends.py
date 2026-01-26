"""TickerTrends API client for TikTok hashtag view time-series data.

Provides monthly time-series data for TikTok hashtag views going back 3 years.
Used for cross-platform trend correlation with Google Trends data.

Authentication: API key as query parameter (?key=...).
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from src.clients.base import BaseAPIClient, RateLimiter
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class TickerTrendsClient(BaseAPIClient):
    """Client for TickerTrends API — TikTok hashtag view time series."""

    BASE_URL = "https://api.tickertrends.io"
    CALLS_PER_MINUTE = 30

    def __init__(self, settings: Settings | None = None):
        settings = settings or get_settings()
        self._api_key = (
            settings.tickertrends_api_key.get_secret_value()
            if settings.tickertrends_api_key
            else ""
        )
        self._endpoint_path = settings.tickertrends_endpoint or "/search-volume"

        super().__init__(
            base_url=self.BASE_URL,
            settings=settings,
            rate_limiter=RateLimiter(self.CALLS_PER_MINUTE),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _get_default_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "User-Agent": "TotalSearch/1.0",
        }

    async def get_hashtag_trends(
        self,
        hashtag: str,
        months: int = 12,
    ) -> dict[str, Any]:
        """
        Get TikTok hashtag view time series.

        Args:
            hashtag: TikTok hashtag (without #).
            months: Number of months to retrieve (default 12).

        Returns:
            Dict with status, hashtag, and interest_over_time (0-100 normalized).
        """
        if not self.is_configured:
            return {
                "status": "not_configured",
                "hashtag": hashtag,
                "interest_over_time": [],
            }

        try:
            params = {
                "term": hashtag,
                "key": self._api_key,
            }

            data = await self.get(self._endpoint_path, params=params)
            time_series = self._normalize_time_series(data, months)

            return {
                "status": "ok",
                "hashtag": hashtag,
                "interest_over_time": time_series,
            }

        except Exception as e:
            logger.warning(f"TickerTrends API error for '{hashtag}': {e}")
            return {
                "status": "error",
                "hashtag": hashtag,
                "interest_over_time": [],
                "error": str(e),
            }

    def _normalize_time_series(
        self,
        raw_data: dict[str, Any],
        months: int,
    ) -> list[dict[str, Any]]:
        """
        Normalize TickerTrends response to 0-100 scale matching Google Trends.

        Handles multiple possible response structures from the API.
        """
        # Try to extract the time series entries from various response shapes
        entries = []
        if isinstance(raw_data, dict):
            data_field = raw_data.get("data", raw_data.get("results", []))
            if isinstance(data_field, list) and data_field:
                # Could be [{term, values: [{timestamp, value}]}] or [{timestamp, value}]
                first = data_field[0]
                if isinstance(first, dict) and "values" in first:
                    entries = first["values"]
                else:
                    entries = data_field
        elif isinstance(raw_data, list):
            entries = raw_data

        if not entries:
            return []

        # Extract date + raw value pairs
        values = []
        for entry in entries:
            ts = entry.get("timestamp") or entry.get("date") or entry.get("month", "")
            val = entry.get("value") or entry.get("views") or entry.get("volume", 0)
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0
            values.append({"date": str(ts), "raw_value": val})

        # Keep only the most recent N months
        if len(values) > months:
            values = values[-months:]

        # Normalize to 0-100 (same scale as Google Trends)
        max_val = max((v["raw_value"] for v in values), default=1) or 1

        normalized = []
        for v in values:
            normalized.append({
                "date": v["date"],
                "tiktok": round((v["raw_value"] / max_val) * 100),
            })

        return normalized
