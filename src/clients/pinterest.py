"""Pinterest API client for search volume and trend data."""

import logging
import re
from typing import Any

import httpx

from src.clients.base import BaseAPIClient, RateLimiter
from src.config import Settings, get_settings
from src.models.keyword import (
    Confidence,
    PinterestMetrics,
    TrendDirection,
)

logger = logging.getLogger(__name__)


class PinterestClient(BaseAPIClient):
    """
    Client for Pinterest search data.

    Pinterest doesn't have a public keyword volume API, so we use multiple approaches:
    1. Pinterest Trends page scraping (public data)
    2. Pinterest Ads API (if credentials available)
    3. Proxy estimation based on pin counts and engagement

    This provides directional data for demand measurement across platforms.
    """

    # Pinterest Trends base URL (public, no auth needed)
    TRENDS_URL = "https://trends.pinterest.com"
    ADS_API_URL = "https://api.pinterest.com/v5"

    # Rate limiting (conservative for scraping)
    CALLS_PER_MINUTE = 30

    # Calibration multiplier for converting interest score (0-100) to search volume
    # Based on validation: "naturkosmetik" has ~2,300 searches on Pinterest
    # Interest score 20 * 115 = 2,300
    # This is calibrated against Keywordtool.io Pinterest data
    INTEREST_TO_VOLUME_MULTIPLIER = 115

    def __init__(
        self,
        access_token: str | None = None,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        super().__init__(
            base_url=self.TRENDS_URL,
            settings=settings,
            rate_limiter=RateLimiter(self.CALLS_PER_MINUTE),
        )

        self.access_token = access_token or getattr(settings, 'pinterest_access_token', None)
        self._ads_client: httpx.AsyncClient | None = None

    def _get_default_headers(self) -> dict[str, str]:
        """Get default headers for requests."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json, text/html",
            "Accept-Language": "en-US,en;q=0.9",
        }
        return headers

    async def get_pinterest_search_volume(
        self,
        keywords: list[str],
        country: str = "US",
    ) -> list[PinterestMetrics]:
        """
        Get Pinterest search/interest data for keywords.

        Args:
            keywords: List of keywords to analyze
            country: Country code (default: US)

        Returns:
            List of PinterestMetrics objects
        """
        results = []

        for keyword in keywords:
            try:
                metrics = await self._fetch_keyword_data(keyword, country)
                results.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to fetch Pinterest data for '{keyword}': {e}")
                results.append(
                    PinterestMetrics(
                        source="pinterest_trends",
                        confidence=Confidence.PROXY,
                        raw_data={"error": str(e)},
                    )
                )

        return results

    async def _fetch_keyword_data(
        self,
        keyword: str,
        country: str,
    ) -> PinterestMetrics:
        """Fetch data for a single keyword using Pinterest Trends."""
        # Clean keyword for URL
        clean_keyword = keyword.lower().replace(" ", "-")

        try:
            # Try to get trends data from Pinterest Trends page
            trends_data = await self._fetch_trends_data(clean_keyword, country)

            if trends_data:
                return self._parse_trends_response(keyword, trends_data)

            # Fallback: estimate from search API (public endpoint)
            search_data = await self._fetch_search_estimates(keyword)
            return self._calculate_proxy_metrics(keyword, search_data)

        except Exception as e:
            logger.debug(f"Pinterest fetch error for '{keyword}': {e}")
            # Return proxy estimate based on keyword characteristics
            return self._estimate_from_keyword(keyword)

    async def _fetch_trends_data(
        self,
        keyword: str,
        country: str,
    ) -> dict | None:
        """Fetch data from Pinterest Trends API endpoint."""
        try:
            # Pinterest Trends has a JSON API endpoint
            params = {
                "keyword": keyword,
                "region": country.lower(),
            }

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{self.TRENDS_URL}/api/trends",
                    params=params,
                    headers=self._get_default_headers(),
                )

                if response.status_code == 200:
                    return response.json()

        except Exception as e:
            logger.debug(f"Pinterest Trends API unavailable: {e}")

        return None

    async def _fetch_search_estimates(
        self,
        keyword: str,
    ) -> dict:
        """Fetch search volume estimates from Pinterest search suggestions."""
        try:
            # Pinterest search suggestions endpoint (public)
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://www.pinterest.com/resource/BaseSearchResource/get/",
                    params={
                        "source_url": f"/search/pins/?q={keyword}",
                        "data": '{"options":{"query":"' + keyword + '"}}'
                    },
                    headers=self._get_default_headers(),
                )

                if response.status_code == 200:
                    data = response.json()
                    return self._extract_search_metrics(data)

        except Exception as e:
            logger.debug(f"Pinterest search API error: {e}")

        return {}

    def _extract_search_metrics(self, data: dict) -> dict:
        """Extract metrics from Pinterest search response."""
        metrics = {
            "result_count": 0,
            "has_shopping": False,
            "related_searches": [],
        }

        try:
            resource_response = data.get("resource_response", {})
            results = resource_response.get("data", {}).get("results", [])

            metrics["result_count"] = len(results)

            # Check for shopping indicators
            for result in results[:20]:
                if result.get("is_product") or result.get("rich_summary"):
                    metrics["has_shopping"] = True
                    break

            # Get related searches if available
            related = resource_response.get("data", {}).get("guides", [])
            metrics["related_searches"] = [r.get("term", "") for r in related[:10]]

        except Exception as e:
            logger.debug(f"Error parsing Pinterest response: {e}")

        return metrics

    def _parse_trends_response(
        self,
        keyword: str,
        data: dict,
    ) -> PinterestMetrics:
        """Parse Pinterest Trends API response into metrics."""
        try:
            # Extract trend data
            trend_data = data.get("data", {})

            # Get normalized interest score (0-100)
            interest_score = trend_data.get("normalized_interest", 50)

            # Get historical data points
            time_series = trend_data.get("time_series", [])

            # Calculate proxy search volume using calibrated multiplier
            # Pinterest interest score 0-100 maps to search volume
            # Calibrated: score 20 ≈ 2,300 searches (validated against Keywordtool.io)
            proxy_volume = int(interest_score * self.INTEREST_TO_VOLUME_MULTIPLIER)

            # Determine trend
            trend, trend_velocity = self._calculate_trend_from_series(time_series)

            # Get related terms
            related = trend_data.get("related_terms", [])

            return PinterestMetrics(
                proxy_score=proxy_volume,
                trend=trend,
                trend_velocity=trend_velocity,
                confidence=Confidence.PROXY,
                source="pinterest_trends",
                interest_score=interest_score,
                pin_count=trend_data.get("pin_count"),
                monthly_searches_estimate=proxy_volume,
                is_trending=interest_score > 70,
                related_terms=related[:10] if related else None,
                raw_data=data,
            )

        except Exception as e:
            logger.warning(f"Error parsing Pinterest trends: {e}")
            return self._estimate_from_keyword(keyword)

    def _calculate_trend_from_series(
        self,
        time_series: list[dict],
    ) -> tuple[TrendDirection | None, float | None]:
        """Calculate trend from time series data."""
        if not time_series or len(time_series) < 4:
            return TrendDirection.STABLE, 1.0

        try:
            # Get recent vs older values
            values = [point.get("value", 0) for point in time_series]

            mid = len(values) // 2
            recent_avg = sum(values[mid:]) / len(values[mid:])
            older_avg = sum(values[:mid]) / len(values[:mid])

            if older_avg > 0:
                velocity = recent_avg / older_avg
            else:
                velocity = 1.5 if recent_avg > 0 else 1.0

            if velocity > 1.15:
                trend = TrendDirection.GROWING
            elif velocity < 0.85:
                trend = TrendDirection.DECLINING
            else:
                trend = TrendDirection.STABLE

            return trend, round(velocity, 2)

        except Exception:
            return TrendDirection.STABLE, 1.0

    def _calculate_proxy_metrics(
        self,
        keyword: str,
        search_data: dict,
    ) -> PinterestMetrics:
        """Calculate proxy metrics from search data."""
        result_count = search_data.get("result_count", 0)
        has_shopping = search_data.get("has_shopping", False)
        related = search_data.get("related_searches", [])

        # Estimate interest based on result density
        # More results = more content = more interest
        if result_count > 100:
            interest_score = 80
        elif result_count > 50:
            interest_score = 60
        elif result_count > 20:
            interest_score = 40
        else:
            interest_score = 20

        # Shopping keywords tend to have higher commercial intent
        if has_shopping:
            interest_score = min(100, interest_score + 15)

        # Convert to proxy volume using calibrated multiplier
        proxy_volume = int(interest_score * self.INTEREST_TO_VOLUME_MULTIPLIER)

        return PinterestMetrics(
            proxy_score=proxy_volume,
            trend=TrendDirection.STABLE,
            trend_velocity=1.0,
            confidence=Confidence.PROXY,
            source="pinterest_search",
            interest_score=interest_score,
            monthly_searches_estimate=proxy_volume,
            is_trending=interest_score > 70,
            related_terms=related if related else None,
            raw_data=search_data,
        )

    def _estimate_from_keyword(self, keyword: str) -> PinterestMetrics:
        """Estimate metrics based on keyword characteristics."""
        # Pinterest-heavy categories
        pinterest_categories = {
            "recipe": 80, "diy": 75, "decor": 75, "fashion": 70,
            "wedding": 85, "home": 70, "craft": 70, "outfit": 65,
            "hairstyle": 70, "makeup": 65, "nail": 60, "garden": 60,
            "food": 65, "dessert": 70, "style": 60, "design": 55,
            "tattoo": 65, "fitness": 50, "workout": 45, "travel": 55,
        }

        keyword_lower = keyword.lower()
        interest_score = 30  # Base score

        for category, score in pinterest_categories.items():
            if category in keyword_lower:
                interest_score = max(interest_score, score)
                break

        proxy_volume = int(interest_score * self.INTEREST_TO_VOLUME_MULTIPLIER)

        return PinterestMetrics(
            proxy_score=proxy_volume,
            trend=TrendDirection.STABLE,
            trend_velocity=1.0,
            confidence=Confidence.PROXY,
            source="pinterest_estimate",
            interest_score=interest_score,
            monthly_searches_estimate=proxy_volume,
            is_trending=False,
            raw_data={"estimated": True, "keyword": keyword},
        )

    async def get_trending_keywords(
        self,
        category: str | None = None,
        country: str = "US",
        limit: int = 20,
    ) -> list[dict]:
        """
        Get currently trending keywords on Pinterest.

        Args:
            category: Optional category filter
            country: Country code
            limit: Max results to return

        Returns:
            List of trending keyword data
        """
        try:
            params = {"region": country.lower()}
            if category:
                params["category"] = category

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{self.TRENDS_URL}/api/trending",
                    params=params,
                    headers=self._get_default_headers(),
                )

                if response.status_code == 200:
                    data = response.json()
                    trends = data.get("data", {}).get("trends", [])
                    return trends[:limit]

        except Exception as e:
            logger.warning(f"Failed to fetch Pinterest trends: {e}")

        return []
