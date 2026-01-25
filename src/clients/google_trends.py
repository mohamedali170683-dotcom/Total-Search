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
            # Note: retries parameter removed due to urllib3 2.x compatibility issue
            # pytrends uses deprecated 'method_whitelist' which is not supported
            self._pytrends = TrendReq(
                hl="en-US",
                tz=360,
                timeout=(10, 25),
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
                if keyword in data and data[keyword].get("estimated_volume"):
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
                    # Fallback: estimate YouTube volume based on keyword type
                    # YouTube typically has 3-8x Google search volume for video-friendly topics
                    estimated = self._estimate_youtube_volume(keyword)
                    metrics_list.append(
                        YouTubeMetrics(
                            search_volume=estimated,
                            trend=TrendDirection.STABLE,
                            trend_velocity=1.0,
                            confidence=Confidence.PROXY,
                            source="youtube_estimate",
                            raw_data={"estimated": True, "keyword": keyword},
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to fetch YouTube trends: {e}")
            for keyword in keywords:
                # Fallback estimation
                estimated = self._estimate_youtube_volume(keyword)
                metrics_list.append(
                    YouTubeMetrics(
                        search_volume=estimated,
                        trend=TrendDirection.STABLE,
                        trend_velocity=1.0,
                        confidence=Confidence.PROXY,
                        source="youtube_estimate",
                        raw_data={"error": str(e), "estimated": True},
                    )
                )

        return metrics_list

    def _estimate_youtube_volume(self, keyword: str) -> int:
        """
        Estimate YouTube search volume based on keyword characteristics.

        YouTube-heavy categories get higher multipliers.
        Base estimate calibrated against Keywordtool.io data.
        """
        keyword_lower = keyword.lower()

        # YouTube-heavy categories with multipliers
        # These topics have high YouTube search relative to Google
        youtube_categories = {
            "tutorial": 8.0, "how to": 8.0, "review": 7.0, "unboxing": 7.0,
            "recipe": 6.0, "makeup": 6.0, "hairstyle": 6.0, "workout": 6.0,
            "gaming": 7.0, "music": 8.0, "vlog": 7.0, "diy": 5.0,
            "beauty": 5.0, "skincare": 5.0, "fashion": 4.0, "travel": 4.0,
            "food": 5.0, "fitness": 5.0, "yoga": 6.0, "meditation": 5.0,
            "cosmetic": 4.0, "natural": 3.0, "organic": 3.0,
        }

        # Base multiplier (YouTube typically 3-5x Google for general topics)
        multiplier = 4.0

        # Check for YouTube-heavy categories
        for category, cat_multiplier in youtube_categories.items():
            if category in keyword_lower:
                multiplier = max(multiplier, cat_multiplier)
                break

        # Base volume estimate (will be multiplied by Google volume ratio if available)
        # For standalone estimate, use conservative base of 10K
        base_volume = 10000

        estimated = int(base_volume * multiplier / 4.0)  # Normalize around 10K base

        logger.info(f"YouTube estimate for '{keyword}': {estimated} (multiplier: {multiplier})")
        return estimated

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

    async def get_trends_intelligence(
        self,
        keywords: list[str],
        timeframe: str = "today 12-m",
        geo: str = "",
    ) -> dict[str, Any]:
        """
        Get comprehensive Google Trends intelligence for keywords.

        Returns:
        - Interest over time (12-month trend chart data)
        - Interest by region (geographic hotspots)
        - Related queries (rising and top queries)
        - Seasonality analysis

        Args:
            keywords: Keywords to analyze (max 5)
            timeframe: Google Trends timeframe
            geo: Geographic location (empty = worldwide)

        Returns:
            Dictionary with trends intelligence data
        """
        # Limit to 5 keywords (Google Trends limit)
        keywords = keywords[:5]

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                self._sync_get_trends_intelligence,
                keywords,
                timeframe,
                geo,
            )
            return data
        except Exception as e:
            logger.error(f"Failed to get trends intelligence: {e}")
            return {
                "error": str(e),
                "keywords": keywords,
                "interest_over_time": [],
                "interest_by_region": [],
                "related_queries": {},
                "seasonality": None,
            }

    def _sync_get_trends_intelligence(
        self,
        keywords: list[str],
        timeframe: str,
        geo: str,
    ) -> dict[str, Any]:
        """Synchronous method to fetch comprehensive trends data."""
        result: dict[str, Any] = {
            "keywords": keywords,
            "timeframe": timeframe,
            "geo": geo or "Worldwide",
            "interest_over_time": [],
            "interest_by_region": [],
            "related_queries": {},
            "rising_queries": [],
            "seasonality": None,
        }

        try:
            # Build payload for web search (not YouTube)
            self.pytrends.build_payload(
                keywords,
                cat=0,
                timeframe=timeframe,
                geo=geo if geo else "",
            )

            # 1. Interest Over Time
            try:
                interest_df = self.pytrends.interest_over_time()
                if not interest_df.empty:
                    time_series = []
                    for date, row in interest_df.iterrows():
                        dt = date.to_pydatetime() if hasattr(date, "to_pydatetime") else date
                        entry = {"date": dt.strftime("%Y-%m-%d")}
                        for kw in keywords:
                            if kw in row:
                                entry[kw] = int(row[kw])
                        time_series.append(entry)
                    result["interest_over_time"] = time_series

                    # Calculate seasonality
                    result["seasonality"] = self._analyze_seasonality(interest_df, keywords)
            except Exception as e:
                logger.warning(f"Failed to get interest over time: {e}")

            # 2. Interest By Region
            try:
                region_df = self.pytrends.interest_by_region(
                    resolution="COUNTRY",
                    inc_low_vol=True,
                    inc_geo_code=True,
                )
                if not region_df.empty:
                    regions = []
                    # Get top 10 regions by first keyword
                    primary_kw = keywords[0] if keywords else None
                    if primary_kw and primary_kw in region_df.columns:
                        sorted_df = region_df.sort_values(by=primary_kw, ascending=False).head(10)
                        for idx, row in sorted_df.iterrows():
                            region_entry = {"region": idx}
                            for kw in keywords:
                                if kw in row:
                                    region_entry[kw] = int(row[kw])
                            regions.append(region_entry)
                    result["interest_by_region"] = regions
            except Exception as e:
                logger.warning(f"Failed to get interest by region: {e}")

            # 3. Related Queries
            try:
                related = self.pytrends.related_queries()
                if related:
                    for kw in keywords:
                        if kw in related and related[kw]:
                            kw_related = {}

                            # Top queries
                            top_df = related[kw].get("top")
                            if top_df is not None and not top_df.empty:
                                kw_related["top"] = top_df.head(10).to_dict("records")

                            # Rising queries (these are the gold!)
                            rising_df = related[kw].get("rising")
                            if rising_df is not None and not rising_df.empty:
                                rising_list = rising_df.head(10).to_dict("records")
                                kw_related["rising"] = rising_list

                                # Add to overall rising queries list
                                for q in rising_list:
                                    q["source_keyword"] = kw
                                    result["rising_queries"].append(q)

                            result["related_queries"][kw] = kw_related
            except Exception as e:
                logger.warning(f"Failed to get related queries: {e}")

            # Sort rising queries by value (growth rate)
            result["rising_queries"] = sorted(
                result["rising_queries"],
                key=lambda x: self._parse_rising_value(x.get("value", 0)),
                reverse=True,
            )[:15]  # Top 15 rising queries

        except Exception as e:
            logger.error(f"Error fetching trends intelligence: {e}")
            result["error"] = str(e)

        return result

    def _parse_rising_value(self, value: Any) -> int:
        """Parse rising query value (can be 'Breakout' or a number)."""
        if value == "Breakout":
            return 10000  # Very high priority
        try:
            return int(str(value).replace("%", "").replace("+", "").replace(",", ""))
        except (ValueError, TypeError):
            return 0

    def _analyze_seasonality(
        self,
        interest_df: Any,
        keywords: list[str],
    ) -> dict[str, Any] | None:
        """Analyze seasonality patterns in the interest data."""
        if interest_df.empty or not keywords:
            return None

        try:
            primary_kw = keywords[0]
            if primary_kw not in interest_df.columns:
                return None

            series = interest_df[primary_kw]

            # Group by month to find seasonal patterns
            monthly_avg: dict[int, list[int]] = {}
            for date, value in series.items():
                month = date.month
                if month not in monthly_avg:
                    monthly_avg[month] = []
                monthly_avg[month].append(int(value))

            # Calculate average interest per month
            month_averages = {
                month: sum(values) / len(values)
                for month, values in monthly_avg.items()
            }

            if not month_averages:
                return None

            # Find peak and low months
            peak_month = max(month_averages, key=month_averages.get)
            low_month = min(month_averages, key=month_averages.get)

            month_names = [
                "", "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]

            # Calculate seasonality strength (variance)
            avg_interest = sum(month_averages.values()) / len(month_averages)
            variance = sum((v - avg_interest) ** 2 for v in month_averages.values()) / len(month_averages)
            seasonality_strength = "high" if variance > 200 else "medium" if variance > 50 else "low"

            return {
                "peak_month": month_names[peak_month],
                "peak_month_num": peak_month,
                "peak_interest": round(month_averages[peak_month], 1),
                "low_month": month_names[low_month],
                "low_month_num": low_month,
                "low_interest": round(month_averages[low_month], 1),
                "seasonality_strength": seasonality_strength,
                "monthly_averages": {
                    month_names[m]: round(v, 1)
                    for m, v in sorted(month_averages.items())
                },
            }

        except Exception as e:
            logger.warning(f"Failed to analyze seasonality: {e}")
            return None

    async def get_trending_searches(
        self,
        country: str = "united_states",
    ) -> list[dict[str, Any]]:
        """
        Get currently trending searches for a country.

        Args:
            country: Country name (e.g., 'united_states', 'germany', 'japan')

        Returns:
            List of trending search terms
        """
        try:
            loop = asyncio.get_event_loop()
            trending = await loop.run_in_executor(
                None,
                self._sync_get_trending_searches,
                country,
            )
            return trending
        except Exception as e:
            logger.error(f"Failed to get trending searches: {e}")
            return []

    def _sync_get_trending_searches(self, country: str) -> list[dict[str, Any]]:
        """Synchronous method to get trending searches."""
        try:
            trending_df = self.pytrends.trending_searches(pn=country)
            if trending_df is not None and not trending_df.empty:
                return [{"query": str(q), "rank": i + 1} for i, q in enumerate(trending_df[0].tolist()[:20])]
        except Exception as e:
            logger.warning(f"Failed to fetch trending searches: {e}")
        return []

    async def close(self) -> None:
        """Close the client."""
        pass

    async def __aenter__(self) -> "GoogleTrendsClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
