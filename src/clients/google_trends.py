"""Google Trends client for YouTube search volume estimation."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import httpx
from pytrends.request import TrendReq

from src.config import Settings, get_settings
from src.models.keyword import (
    Confidence,
    MonthlySearchData,
    TrendDirection,
    YouTubeMetrics,
)

logger = logging.getLogger(__name__)


class GoogleTrendsClient:
    """
    Client for Google Trends API to get YouTube search interest.

    This uses the pytrends library to access Google Trends data
    and specifically targets YouTube search trends.

    Note: Google Trends provides relative interest (0-100 scale),
    not absolute search volume. We convert this to estimated volume.
    """

    # Base multiplier to convert Google Trends index to estimated searches
    # Calibrated against Keywordtool.io: "naturkosmetik" has ~70K YouTube searches
    # If trends index is ~50, then 50 * 1400 = 70,000
    YOUTUBE_TRENDS_MULTIPLIER = 1400

    # Timeout for requests
    REQUEST_TIMEOUT = 30

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._pytrends: TrendReq | None = None

    @property
    def pytrends(self) -> TrendReq:
        """Get or create the pytrends client."""
        if self._pytrends is None:
            self._pytrends = TrendReq(
                hl="en-US",
                tz=360,
                timeout=(10, 25),
                retries=2,
                backoff_factor=0.1,
            )
        return self._pytrends

    async def get_youtube_search_volume(
        self,
        keywords: list[str],
        timeframe: str = "today 12-m",
        geo: str = "US",
    ) -> list[YouTubeMetrics]:
        """
        Get YouTube search volume estimates from Google Trends.

        Args:
            keywords: List of keywords to analyze
            timeframe: Google Trends timeframe (default: last 12 months)
            geo: Geographic location code (default: US)

        Returns:
            List of YouTubeMetrics objects with estimated search volume
        """
        results: list[YouTubeMetrics] = []

        # Process keywords in batches of 5 (Google Trends limit)
        batch_size = 5
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i : i + batch_size]
            batch_results = await self._fetch_youtube_trends_batch(batch, timeframe, geo)
            results.extend(batch_results)

            # Small delay to avoid rate limiting
            if i + batch_size < len(keywords):
                await asyncio.sleep(1)

        return results

    async def _fetch_youtube_trends_batch(
        self,
        keywords: list[str],
        timeframe: str,
        geo: str = "US",
    ) -> list[YouTubeMetrics]:
        """Fetch YouTube trends for a batch of keywords."""
        metrics_list: list[YouTubeMetrics] = []

        try:
            # Run in thread pool since pytrends is synchronous
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                self._sync_fetch_youtube_trends,
                keywords,
                timeframe,
                geo,
            )

            for keyword in keywords:
                if keyword in data:
                    keyword_data = data[keyword]
                    metrics_list.append(
                        YouTubeMetrics(
                            search_volume=keyword_data.get("estimated_volume"),
                            trend=keyword_data.get("trend"),
                            trend_velocity=keyword_data.get("trend_velocity"),
                            monthly_searches=keyword_data.get("monthly_searches"),
                            confidence=Confidence.PROXY,
                            source="google_trends_youtube",
                            raw_data=keyword_data,
                        )
                    )
                else:
                    metrics_list.append(
                        YouTubeMetrics(
                            source="google_trends_youtube",
                            confidence=Confidence.PROXY,
                            raw_data={"error": "No data returned"},
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to fetch YouTube trends: {e}")
            for keyword in keywords:
                metrics_list.append(
                    YouTubeMetrics(
                        source="google_trends_youtube",
                        confidence=Confidence.PROXY,
                        raw_data={"error": str(e)},
                    )
                )

        return metrics_list

    def _sync_fetch_youtube_trends(
        self,
        keywords: list[str],
        timeframe: str,
        geo: str = "US",
    ) -> dict[str, dict[str, Any]]:
        """Synchronous method to fetch YouTube trends data."""
        result: dict[str, dict[str, Any]] = {}

        try:
            logger.info(f"Fetching YouTube trends for {keywords} in geo={geo}")

            # Build payload for YouTube search
            # gprop='youtube' targets YouTube specifically
            self.pytrends.build_payload(
                keywords,
                cat=0,
                timeframe=timeframe,
                geo=geo,
                gprop="youtube",
            )

            # Get interest over time
            interest_df = self.pytrends.interest_over_time()

            if interest_df.empty:
                logger.warning(f"No Google Trends YouTube data for keywords: {keywords} (geo={geo})")
                # Try without geo restriction as fallback
                logger.info(f"Retrying without geo restriction...")
                self.pytrends.build_payload(
                    keywords,
                    cat=0,
                    timeframe=timeframe,
                    gprop="youtube",
                )
                interest_df = self.pytrends.interest_over_time()

                if interest_df.empty:
                    logger.warning(f"Still no data after fallback for: {keywords}")
                    return result

            for keyword in keywords:
                if keyword not in interest_df.columns:
                    continue

                keyword_series = interest_df[keyword]

                # Get latest value (most recent interest)
                latest_value = int(keyword_series.iloc[-1]) if len(keyword_series) > 0 else 0

                # Calculate estimated volume
                estimated_volume = latest_value * self.YOUTUBE_TRENDS_MULTIPLIER

                # Build monthly search data
                monthly_searches = []
                for date, value in keyword_series.items():
                    # Convert pandas Timestamp to datetime
                    dt = date.to_pydatetime() if hasattr(date, "to_pydatetime") else date
                    monthly_searches.append(
                        MonthlySearchData(
                            year=dt.year,
                            month=dt.month,
                            search_volume=int(value) * self.YOUTUBE_TRENDS_MULTIPLIER,
                        )
                    )

                # Calculate trend
                trend, velocity = self._calculate_trend(monthly_searches)

                result[keyword] = {
                    "trends_index": latest_value,
                    "estimated_volume": estimated_volume,
                    "trend": trend,
                    "trend_velocity": velocity,
                    "monthly_searches": monthly_searches,
                }

        except Exception as e:
            logger.error(f"Error in sync fetch: {e}")
            raise

        return result

    def _calculate_trend(
        self,
        monthly_searches: list[MonthlySearchData],
    ) -> tuple[TrendDirection | None, float | None]:
        """Calculate trend direction and velocity from monthly data."""
        if not monthly_searches or len(monthly_searches) < 3:
            return None, None

        # Sort by date (oldest first)
        sorted_data = sorted(monthly_searches, key=lambda x: (x.year, x.month))

        # Compare recent 3 data points vs previous 3
        if len(sorted_data) >= 6:
            recent = sum(m.search_volume for m in sorted_data[-3:]) / 3
            previous = sum(m.search_volume for m in sorted_data[-6:-3]) / 3

            if previous > 0:
                velocity = recent / previous
            else:
                velocity = 1.0 if recent == 0 else 2.0

            if velocity > 1.15:
                trend = TrendDirection.GROWING
            elif velocity < 0.85:
                trend = TrendDirection.DECLINING
            else:
                trend = TrendDirection.STABLE

            return trend, round(velocity, 2)

        return TrendDirection.STABLE, 1.0

    async def close(self) -> None:
        """Close the client."""
        pass

    async def __aenter__(self) -> "GoogleTrendsClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
