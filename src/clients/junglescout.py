"""Jungle Scout API client for Amazon search volume data."""

import logging
from typing import Any

from src.clients.base import BaseAPIClient, RateLimiter, batch_items
from src.config import Settings, get_settings
from src.models.keyword import (
    AmazonMetrics,
    Competition,
    Confidence,
    TrendDirection,
)

logger = logging.getLogger(__name__)


# Marketplace configurations
MARKETPLACE_CONFIG = {
    "us": {"country_code": "us", "marketplace_id": "ATVPDKIKX0DER"},
    "uk": {"country_code": "uk", "marketplace_id": "A1F83G8C2ARO7P"},
    "de": {"country_code": "de", "marketplace_id": "A1PA6795UKMFR9"},
    "fr": {"country_code": "fr", "marketplace_id": "A13V1IB3VIYZZH"},
    "ca": {"country_code": "ca", "marketplace_id": "A2EUQ1WTGCTBG2"},
    "it": {"country_code": "it", "marketplace_id": "APJ6JRA9NG5V4"},
    "es": {"country_code": "es", "marketplace_id": "A1RKKUPIHCS9HS"},
    "mx": {"country_code": "mx", "marketplace_id": "A1AM78C64UM0Y8"},
    "jp": {"country_code": "jp", "marketplace_id": "A1VC38T7YXB528"},
}


class JungleScoutClient(BaseAPIClient):
    """
    Client for Jungle Scout API interactions.

    Endpoints:
    - Keywords by Keyword: GET /keywords/keywords_by_keyword_query
    - Keywords by ASIN: GET /keywords/keywords_by_asin_query
    - Historical Search Volume: GET /keywords/historical_search_volume

    Features:
    - Batch keyword lookups (up to 10 keywords per request)
    - Marketplace selection (US, UK, DE, etc.)
    - Rate limiting (varies by tier)
    """

    BASE_URL = "https://developer.junglescout.com/api"
    MAX_KEYWORDS_PER_REQUEST = 10
    CALLS_PER_MINUTE = 60  # Conservative default, adjust based on tier

    def __init__(
        self,
        api_key: str | None = None,
        api_key_name: str | None = None,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        super().__init__(
            base_url=self.BASE_URL,
            settings=settings,
            rate_limiter=RateLimiter(self.CALLS_PER_MINUTE),
        )

        self.api_key = api_key or settings.junglescout_api_key.get_secret_value()
        self.api_key_name = api_key_name or settings.junglescout_api_key_name

        if not self.api_key or not self.api_key_name:
            logger.warning("Jungle Scout credentials not configured")

    def _get_default_headers(self) -> dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"{self.api_key_name}:{self.api_key}",
            "Content-Type": "application/vnd.api+json",
            "Accept": "application/vnd.junglescout.v1+json",
        }

    async def get_amazon_search_volume(
        self,
        keywords: list[str],
        marketplace: str | None = None,
    ) -> list[AmazonMetrics]:
        """
        Get Amazon search volume for keywords.

        Args:
            keywords: List of keywords to analyze
            marketplace: Amazon marketplace (us, uk, de, etc.)

        Returns:
            List of AmazonMetrics objects
        """
        marketplace = marketplace or self.settings.default_marketplace
        marketplace_config = MARKETPLACE_CONFIG.get(marketplace.lower(), MARKETPLACE_CONFIG["us"])

        all_metrics: list[AmazonMetrics] = []

        # Process in batches of MAX_KEYWORDS_PER_REQUEST
        for batch in batch_items(keywords, self.MAX_KEYWORDS_PER_REQUEST):
            batch_metrics = await self._fetch_keywords_batch(batch, marketplace_config)
            all_metrics.extend(batch_metrics)

        return all_metrics

    async def _fetch_keywords_batch(
        self,
        keywords: list[str],
        marketplace_config: dict,
    ) -> list[AmazonMetrics]:
        """Fetch Amazon metrics for a batch of keywords."""
        # Build the query parameters
        params = {
            "marketplace": marketplace_config["country_code"],
            "search_terms": ",".join(keywords),
        }

        try:
            response = await self.get(
                "/keywords/keywords_by_keyword_query",
                params=params,
            )
            return self._parse_keywords_response(response, keywords)

        except Exception as e:
            logger.error(f"Failed to fetch Amazon search volume: {e}")
            return [
                AmazonMetrics(
                    source="junglescout",
                    confidence=Confidence.PROXY,
                    raw_data={"error": str(e)},
                )
                for _ in keywords
            ]

    def _parse_keywords_response(
        self,
        response: dict[str, Any],
        keywords: list[str],
    ) -> list[AmazonMetrics]:
        """Parse Jungle Scout keywords response."""
        metrics_list: list[AmazonMetrics] = []
        keyword_to_data: dict[str, dict] = {}

        # Extract data from response
        data_items = response.get("data", [])
        for item in data_items:
            attributes = item.get("attributes", {})
            keyword = attributes.get("name", "").lower()
            keyword_to_data[keyword] = attributes

        # Build metrics for each requested keyword
        for keyword in keywords:
            data = keyword_to_data.get(keyword.lower())

            if not data:
                metrics_list.append(
                    AmazonMetrics(
                        source="junglescout",
                        confidence=Confidence.PROXY,
                    )
                )
                continue

            # Extract search volumes
            exact_volume = data.get("exact_match_search_volume")
            broad_volume = data.get("broad_match_search_volume")

            # Use exact volume as primary
            search_volume = exact_volume or broad_volume

            # Determine trend from monthly data if available
            trend, trend_velocity = self._calculate_trend(data.get("monthly_trend", []))

            # Map dominant category
            competition = self._estimate_competition(data)

            metrics_list.append(
                AmazonMetrics(
                    search_volume=search_volume,
                    exact_search_volume=exact_volume,
                    broad_search_volume=broad_volume,
                    trend=trend,
                    trend_velocity=trend_velocity,
                    competition=competition,
                    confidence=Confidence.HIGH if search_volume else Confidence.PROXY,
                    source="junglescout",
                    organic_product_count=data.get("organic_product_count"),
                    sponsored_product_count=data.get("sponsored_product_count"),
                    raw_data=data,
                )
            )

        return metrics_list

    async def get_keywords_by_asin(
        self,
        asin: str,
        marketplace: str | None = None,
        limit: int = 50,
    ) -> list[str]:
        """
        Reverse ASIN lookup for keyword ideas.

        Args:
            asin: Amazon ASIN to analyze
            marketplace: Amazon marketplace
            limit: Maximum number of keywords to return

        Returns:
            List of keywords
        """
        marketplace = marketplace or self.settings.default_marketplace
        marketplace_config = MARKETPLACE_CONFIG.get(marketplace.lower(), MARKETPLACE_CONFIG["us"])

        params = {
            "marketplace": marketplace_config["country_code"],
            "asin": asin,
            "page[size]": min(limit, 100),
        }

        try:
            response = await self.get(
                "/keywords/keywords_by_asin_query",
                params=params,
            )

            keywords = []
            for item in response.get("data", []):
                if keyword := item.get("attributes", {}).get("name"):
                    keywords.append(keyword)

            return keywords[:limit]

        except Exception as e:
            logger.error(f"Failed to get keywords for ASIN {asin}: {e}")
            return []

    async def get_historical_search_volume(
        self,
        keyword: str,
        marketplace: str | None = None,
    ) -> dict[str, Any]:
        """
        Get historical search volume data for a keyword.

        Args:
            keyword: Keyword to analyze
            marketplace: Amazon marketplace

        Returns:
            Historical search volume data
        """
        marketplace = marketplace or self.settings.default_marketplace
        marketplace_config = MARKETPLACE_CONFIG.get(marketplace.lower(), MARKETPLACE_CONFIG["us"])

        params = {
            "marketplace": marketplace_config["country_code"],
            "search_term": keyword,
        }

        try:
            response = await self.get(
                "/keywords/historical_search_volume",
                params=params,
            )
            return response

        except Exception as e:
            logger.error(f"Failed to get historical data for '{keyword}': {e}")
            return {}

    def _calculate_trend(
        self,
        monthly_data: list[dict] | None,
    ) -> tuple[TrendDirection | None, float | None]:
        """Calculate trend from monthly search volume data."""
        if not monthly_data or len(monthly_data) < 3:
            return None, None

        # Sort by date
        sorted_data = sorted(monthly_data, key=lambda x: x.get("month", ""))

        # Compare recent 3 months vs previous 3 months
        if len(sorted_data) >= 6:
            recent_volumes = [
                d.get("exact_match_search_volume", 0) or 0
                for d in sorted_data[-3:]
            ]
            previous_volumes = [
                d.get("exact_match_search_volume", 0) or 0
                for d in sorted_data[-6:-3]
            ]

            recent_avg = sum(recent_volumes) / 3
            previous_avg = sum(previous_volumes) / 3

            if previous_avg > 0:
                velocity = recent_avg / previous_avg
            else:
                velocity = 1.0 if recent_avg == 0 else 2.0

            if velocity > 1.15:
                trend = TrendDirection.GROWING
            elif velocity < 0.85:
                trend = TrendDirection.DECLINING
            else:
                trend = TrendDirection.STABLE

            return trend, round(velocity, 2)

        return TrendDirection.STABLE, 1.0

    def _estimate_competition(self, data: dict) -> Competition | None:
        """Estimate competition from sponsored/organic product counts."""
        sponsored = data.get("sponsored_product_count", 0) or 0
        organic = data.get("organic_product_count", 0) or 0

        if sponsored == 0 and organic == 0:
            return None

        # Heuristic: high sponsored ratio = high competition
        total = sponsored + organic
        sponsored_ratio = sponsored / total if total > 0 else 0

        if sponsored_ratio > 0.3 or sponsored > 20:
            return Competition.HIGH
        elif sponsored_ratio > 0.15 or sponsored > 10:
            return Competition.MEDIUM
        else:
            return Competition.LOW
