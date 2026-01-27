"""DataForSEO API client for Google, YouTube, and Amazon search volume data."""

import base64
import logging
from typing import Any

from src.clients.base import BaseAPIClient, RateLimiter, batch_items
from src.config import Settings, get_settings
from src.models.keyword import (
    AmazonMetrics,
    Competition,
    Confidence,
    GoogleMetrics,
    MonthlySearchData,
    TrendDirection,
    YouTubeMetrics,
)

logger = logging.getLogger(__name__)


class DataForSEOClient(BaseAPIClient):
    """
    Client for DataForSEO API interactions.

    Endpoints implemented:
    - Google Ads Search Volume: /v3/keywords_data/google_ads/search_volume/live
    - Clickstream Bulk Volume: /v3/keywords_data/clickstream_data/bulk_search_volume/live
    - Google Ads Keywords for Site: /v3/keywords_data/google_ads/keywords_for_site/live

    Features:
    - Batch requests (up to 1000 keywords per request)
    - Rate limiting handling (2000 requests/minute)
    - Error retry logic with exponential backoff
    - Response caching support
    """

    BASE_URL = "https://api.dataforseo.com"
    MAX_KEYWORDS_PER_REQUEST = 1000
    CALLS_PER_MINUTE = 2000

    def __init__(
        self,
        login: str | None = None,
        password: str | None = None,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        super().__init__(
            base_url=self.BASE_URL,
            settings=settings,
            rate_limiter=RateLimiter(self.CALLS_PER_MINUTE),
        )

        self.login = login or settings.dataforseo_login
        self.password = password or settings.dataforseo_password.get_secret_value()

        if not self.login or not self.password:
            logger.warning("DataForSEO credentials not configured")

    def _get_default_headers(self) -> dict[str, str]:
        """Get authorization headers."""
        credentials = f"{self.login}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
        }

    async def get_google_search_volume(
        self,
        keywords: list[str],
        location_code: int | None = None,
        language_code: str | None = None,
    ) -> list[GoogleMetrics]:
        """
        Get Google search volume for keywords.

        Args:
            keywords: List of keywords to analyze
            location_code: Location code (default: 2840 for US)
            language_code: Language code (default: en)

        Returns:
            List of GoogleMetrics objects
        """
        location_code = location_code or self.settings.default_location_code
        language_code = language_code or self.settings.default_language_code

        all_metrics: list[GoogleMetrics] = []

        # Process in batches of MAX_KEYWORDS_PER_REQUEST
        for batch in batch_items(keywords, self.MAX_KEYWORDS_PER_REQUEST):
            batch_metrics = await self._fetch_google_search_volume_batch(
                batch, location_code, language_code
            )
            all_metrics.extend(batch_metrics)

        return all_metrics

    async def _fetch_google_search_volume_batch(
        self,
        keywords: list[str],
        location_code: int,
        language_code: str,
    ) -> list[GoogleMetrics]:
        """Fetch Google search volume for a batch of keywords."""
        payload = [
            {
                "keywords": keywords,
                "location_code": location_code,
                "language_code": language_code,
                "date_from": None,  # Last 12 months by default
                "include_serp_info": False,
            }
        ]

        try:
            response = await self.post(
                "/v3/keywords_data/google_ads/search_volume/live",
                json_data=payload,
            )
            return self._parse_google_search_volume_response(response, keywords)
        except Exception as e:
            logger.error(f"Failed to fetch Google search volume: {e}")
            # Return empty metrics for failed keywords
            return [
                GoogleMetrics(
                    source="dataforseo_google",
                    confidence=Confidence.PROXY,
                    raw_data={"error": str(e)},
                )
                for _ in keywords
            ]

    def _parse_google_search_volume_response(
        self,
        response: dict[str, Any],
        keywords: list[str],
    ) -> list[GoogleMetrics]:
        """Parse DataForSEO Google search volume response."""
        metrics_list: list[GoogleMetrics] = []
        keyword_to_metrics: dict[str, dict] = {}

        # Extract results from response
        tasks = response.get("tasks", [])
        for task in tasks:
            if task.get("status_code") != 20000:
                logger.warning(f"Task failed: {task.get('status_message')}")
                continue

            results = task.get("result", [])
            for result in results:
                keyword = result.get("keyword", "").lower()
                keyword_to_metrics[keyword] = result

        # Build metrics for each requested keyword
        for keyword in keywords:
            result = keyword_to_metrics.get(keyword.lower())

            if not result:
                metrics_list.append(
                    GoogleMetrics(
                        source="dataforseo_google",
                        search_volume=None,
                        confidence=Confidence.PROXY,
                    )
                )
                continue

            # Parse monthly searches
            monthly_searches = None
            if result.get("monthly_searches"):
                monthly_searches = [
                    MonthlySearchData(
                        year=ms.get("year"),
                        month=ms.get("month"),
                        search_volume=ms.get("search_volume", 0),
                    )
                    for ms in result["monthly_searches"]
                ]

            # Determine trend from monthly data
            trend, trend_velocity = self._calculate_trend(monthly_searches)

            # Map competition
            competition = self._map_competition(result.get("competition"))

            metrics_list.append(
                GoogleMetrics(
                    search_volume=result.get("search_volume"),
                    trend=trend,
                    trend_velocity=trend_velocity,
                    competition=competition,
                    cpc=result.get("cpc"),
                    confidence=Confidence.HIGH,
                    source="dataforseo_google",
                    monthly_searches=monthly_searches,
                    keyword_difficulty=result.get("keyword_difficulty"),
                    raw_data=result,
                )
            )

        return metrics_list

    async def get_youtube_search_volume(
        self,
        keywords: list[str],
        location_code: int | None = None,
        language_code: str | None = None,
    ) -> list[YouTubeMetrics]:
        """
        Get YouTube search volume for keywords.

        Uses the clickstream data endpoint which includes YouTube data.

        Args:
            keywords: List of keywords to analyze
            location_code: Location code (default: 2840 for US)
            language_code: Language code (default: en)

        Returns:
            List of YouTubeMetrics objects
        """
        location_code = location_code or self.settings.default_location_code
        language_code = language_code or self.settings.default_language_code

        all_metrics: list[YouTubeMetrics] = []

        for batch in batch_items(keywords, self.MAX_KEYWORDS_PER_REQUEST):
            batch_metrics = await self._fetch_youtube_search_volume_batch(
                batch, location_code, language_code
            )
            all_metrics.extend(batch_metrics)

        return all_metrics

    async def _fetch_youtube_search_volume_batch(
        self,
        keywords: list[str],
        location_code: int,
        language_code: str,
    ) -> list[YouTubeMetrics]:
        """Fetch YouTube search volume for a batch of keywords."""
        # Use clickstream endpoint which provides YouTube data
        payload = [
            {
                "keywords": keywords,
                "location_code": location_code,
                "language_code": language_code,
            }
        ]

        try:
            response = await self.post(
                "/v3/keywords_data/clickstream_data/bulk_search_volume/live",
                json_data=payload,
            )
            return self._parse_youtube_search_volume_response(response, keywords)
        except Exception as e:
            logger.error(f"Failed to fetch YouTube search volume: {e}")
            return [
                YouTubeMetrics(
                    source="dataforseo_youtube",
                    confidence=Confidence.PROXY,
                    raw_data={"error": str(e)},
                )
                for _ in keywords
            ]

    def _parse_youtube_search_volume_response(
        self,
        response: dict[str, Any],
        keywords: list[str],
    ) -> list[YouTubeMetrics]:
        """Parse DataForSEO clickstream response for YouTube data."""
        metrics_list: list[YouTubeMetrics] = []
        keyword_to_metrics: dict[str, dict] = {}

        tasks = response.get("tasks", [])
        for task in tasks:
            if task.get("status_code") != 20000:
                continue

            results = task.get("result", [])
            for result in results:
                keyword = result.get("keyword", "").lower()
                keyword_to_metrics[keyword] = result

        for keyword in keywords:
            result = keyword_to_metrics.get(keyword.lower())

            if not result:
                metrics_list.append(
                    YouTubeMetrics(
                        source="dataforseo_youtube",
                        confidence=Confidence.PROXY,
                    )
                )
                continue

            # Parse monthly data if available
            monthly_searches = None
            if result.get("monthly_searches"):
                monthly_searches = [
                    MonthlySearchData(
                        year=ms.get("year"),
                        month=ms.get("month"),
                        search_volume=ms.get("search_volume", 0),
                    )
                    for ms in result["monthly_searches"]
                ]

            trend, trend_velocity = self._calculate_trend(monthly_searches)

            metrics_list.append(
                YouTubeMetrics(
                    search_volume=result.get("search_volume"),
                    trend=trend,
                    trend_velocity=trend_velocity,
                    confidence=Confidence.HIGH if result.get("search_volume") else Confidence.PROXY,
                    source="dataforseo_youtube",
                    monthly_searches=monthly_searches,
                    raw_data=result,
                )
            )

        return metrics_list

    async def get_keywords_for_site(
        self,
        domain: str,
        location_code: int | None = None,
        language_code: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        """
        Get keyword suggestions based on a domain.

        Args:
            domain: Target domain to analyze
            location_code: Location code
            language_code: Language code
            limit: Maximum number of keywords to return

        Returns:
            List of suggested keywords
        """
        location_code = location_code or self.settings.default_location_code
        language_code = language_code or self.settings.default_language_code

        payload = [
            {
                "target": domain,
                "location_code": location_code,
                "language_code": language_code,
                "include_serp_info": False,
                "limit": limit,
            }
        ]

        try:
            response = await self.post(
                "/v3/keywords_data/google_ads/keywords_for_site/live",
                json_data=payload,
            )

            keywords = []
            tasks = response.get("tasks", [])
            for task in tasks:
                if task.get("status_code") != 20000:
                    continue
                results = task.get("result", [])
                for result in results:
                    if keyword := result.get("keyword"):
                        keywords.append(keyword)

            return keywords[:limit]

        except Exception as e:
            logger.error(f"Failed to get keywords for site {domain}: {e}")
            return []

    async def get_amazon_search_volume(
        self,
        keywords: list[str],
        location_code: int | None = None,
        language_code: str | None = None,
    ) -> list[AmazonMetrics]:
        """
        Get Amazon search volume for keywords.

        Uses the DataForSEO Labs Amazon Bulk Search Volume endpoint.

        Args:
            keywords: List of keywords to analyze
            location_code: Location code (default: 2840 for US)
            language_code: Language code (default: en)

        Returns:
            List of AmazonMetrics objects
        """
        location_code = location_code or self.settings.default_location_code
        language_code = language_code or self.settings.default_language_code

        all_metrics: list[AmazonMetrics] = []

        for batch in batch_items(keywords, self.MAX_KEYWORDS_PER_REQUEST):
            batch_metrics = await self._fetch_amazon_search_volume_batch(
                batch, location_code, language_code
            )
            all_metrics.extend(batch_metrics)

        return all_metrics

    async def _fetch_amazon_search_volume_batch(
        self,
        keywords: list[str],
        location_code: int,
        language_code: str,
    ) -> list[AmazonMetrics]:
        """Fetch Amazon search volume for a batch of keywords."""
        # DataForSEO Labs Amazon Bulk Search Volume endpoint
        payload = [
            {
                "keywords": keywords,
                "location_code": location_code,
                "language_code": language_code,
            }
        ]

        try:
            response = await self.post(
                "/v3/dataforseo_labs/amazon/bulk_search_volume/live",
                json_data=payload,
            )
            return self._parse_amazon_search_volume_response(response, keywords)
        except Exception as e:
            logger.error(f"Failed to fetch Amazon search volume: {e}")
            return [
                AmazonMetrics(
                    source="dataforseo_amazon",
                    confidence=Confidence.PROXY,
                    raw_data={"error": str(e)},
                )
                for _ in keywords
            ]

    def _parse_amazon_search_volume_response(
        self,
        response: dict[str, Any],
        keywords: list[str],
    ) -> list[AmazonMetrics]:
        """Parse DataForSEO Amazon search volume response."""
        metrics_list: list[AmazonMetrics] = []
        keyword_to_metrics: dict[str, dict] = {}

        tasks = response.get("tasks", [])
        for task in tasks:
            if task.get("status_code") != 20000:
                logger.warning(f"Amazon task failed: {task.get('status_message')}")
                continue

            results = task.get("result", [])
            for result in results:
                # Result contains items array with keyword data
                items = result.get("items", [])
                for item in items:
                    keyword = item.get("keyword", "").lower()
                    keyword_to_metrics[keyword] = item

        for keyword in keywords:
            result = keyword_to_metrics.get(keyword.lower())

            if not result:
                metrics_list.append(
                    AmazonMetrics(
                        source="dataforseo_amazon",
                        confidence=Confidence.PROXY,
                    )
                )
                continue

            # Amazon returns search_volume directly
            search_volume = result.get("search_volume")

            metrics_list.append(
                AmazonMetrics(
                    search_volume=search_volume,
                    confidence=Confidence.HIGH if search_volume else Confidence.PROXY,
                    source="dataforseo_amazon",
                    raw_data=result,
                )
            )

        return metrics_list

    async def get_serp_results(
        self,
        keyword: str,
        location_code: int | None = None,
        language_code: str | None = None,
        depth: int = 20,
    ) -> dict[str, Any]:
        """
        Get Google SERP (Search Engine Results Page) for a keyword.

        Returns the top organic results with domain, title, position,
        and SERP features (featured snippets, people also ask, etc.).

        Args:
            keyword: The keyword to check SERP for.
            location_code: Location code (default: 2840 for US).
            language_code: Language code (default: en).
            depth: Number of results to fetch (default: 20).

        Returns:
            Dict with organic_results, serp_features, and people_also_ask.
        """
        location_code = location_code or self.settings.default_location_code
        language_code = language_code or self.settings.default_language_code

        payload = [
            {
                "keyword": keyword,
                "location_code": location_code,
                "language_code": language_code,
                "device": "desktop",
                "os": "windows",
                "depth": depth,
            }
        ]

        try:
            response = await self.post(
                "/v3/serp/google/organic/live/regular",
                json_data=payload,
            )
            return self._parse_serp_response(response, depth=depth)
        except Exception as e:
            logger.error(f"Failed to fetch SERP results for '{keyword}': {e}")
            return {"organic_results": [], "serp_features": [], "people_also_ask": []}

    def _parse_serp_response(self, response: dict[str, Any], depth: int = 20) -> dict[str, Any]:
        """Parse DataForSEO SERP response into structured results."""
        organic_results = []
        serp_features = []
        people_also_ask = []

        tasks = response.get("tasks", [])
        for task in tasks:
            if task.get("status_code") != 20000:
                logger.warning(f"SERP task failed: {task.get('status_message')}")
                continue

            results = task.get("result", [])
            for result in results:
                items = result.get("items", [])
                for item in items:
                    item_type = item.get("type", "")

                    if item_type == "organic":
                        organic_results.append({
                            "position": item.get("rank_group"),
                            "domain": item.get("domain", ""),
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "description": item.get("description", ""),
                            "breadcrumb": item.get("breadcrumb", ""),
                        })

                    elif item_type == "featured_snippet":
                        serp_features.append({
                            "type": "featured_snippet",
                            "domain": item.get("domain", ""),
                            "title": item.get("title", ""),
                            "description": item.get("description", ""),
                        })

                    elif item_type == "people_also_ask":
                        items_paa = item.get("items", [])
                        if isinstance(items_paa, list):
                            for paa in items_paa:
                                people_also_ask.append({
                                    "question": paa.get("title", ""),
                                    "domain": paa.get("domain", ""),
                                    "url": paa.get("url", ""),
                                })
                        elif item.get("title"):
                            people_also_ask.append({
                                "question": item.get("title", ""),
                                "domain": item.get("domain", ""),
                                "url": item.get("url", ""),
                            })

                    elif item_type == "knowledge_graph":
                        serp_features.append({
                            "type": "knowledge_graph",
                            "title": item.get("title", ""),
                            "description": item.get("description", ""),
                        })

                    elif item_type == "local_pack":
                        serp_features.append({
                            "type": "local_pack",
                            "title": item.get("title", ""),
                        })

                    elif item_type == "video":
                        serp_features.append({
                            "type": "video",
                            "domain": item.get("domain", ""),
                            "title": item.get("title", ""),
                        })

        return {
            "organic_results": organic_results[:depth],
            "serp_features": serp_features,
            "people_also_ask": people_also_ask,
        }

    def _calculate_trend(
        self,
        monthly_searches: list[MonthlySearchData] | None,
    ) -> tuple[TrendDirection | None, float | None]:
        """Calculate trend direction and velocity from monthly data."""
        if not monthly_searches or len(monthly_searches) < 3:
            return None, None

        # Sort by date (oldest first)
        sorted_data = sorted(monthly_searches, key=lambda x: (x.year, x.month))

        # Compare recent 3 months vs previous 3 months
        if len(sorted_data) >= 6:
            recent = sum(m.search_volume for m in sorted_data[-3:]) / 3
            previous = sum(m.search_volume for m in sorted_data[-6:-3]) / 3

            if previous > 0:
                velocity = recent / previous
            else:
                velocity = 1.0 if recent == 0 else 2.0  # Assume growing if went from 0

            if velocity > 1.15:
                trend = TrendDirection.GROWING
            elif velocity < 0.85:
                trend = TrendDirection.DECLINING
            else:
                trend = TrendDirection.STABLE

            return trend, round(velocity, 2)

        return TrendDirection.STABLE, 1.0

    def _map_competition(self, competition_value: str | None) -> Competition | None:
        """Map DataForSEO competition string to enum."""
        if not competition_value:
            return None

        mapping = {
            "LOW": Competition.LOW,
            "MEDIUM": Competition.MEDIUM,
            "HIGH": Competition.HIGH,
        }
        return mapping.get(competition_value.upper())
