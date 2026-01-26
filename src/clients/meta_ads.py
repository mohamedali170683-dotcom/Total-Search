"""Meta Marketing API client for audience reach estimation.

Uses Meta's delivery_estimate endpoint to get estimated audience size
(DAU and MAU) for interest-based targeting on Facebook and Instagram.
No ad spend required — read-only audience sizing.

Endpoints used:
    1. GET /search?type=adinterest — maps keywords to Meta interest IDs
    2. GET /act_{id}/delivery_estimate — returns audience size estimates
"""

import asyncio
import json
import logging
from typing import Any

import httpx

from src.clients.base import RateLimiter
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v21.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

AGE_GROUPS = [
    (18, 24),
    (25, 34),
    (35, 44),
    (45, 54),
    (55, 65),
]


class MetaAdsError(Exception):
    """Meta Marketing API error."""

    def __init__(self, message: str, error_code: int | None = None):
        super().__init__(message)
        self.error_code = error_code


class MetaAdsClient:
    """
    Client for Meta Marketing API audience reach estimation.

    Maps keywords to Meta interest targeting IDs, then queries
    delivery_estimate for audience size. Supports demographic
    breakdowns by age group and gender.
    """

    CALLS_PER_MINUTE = 30

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._access_token = self.settings.meta_access_token.get_secret_value()
        self._ad_account_id = self.settings.meta_ad_account_id
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = RateLimiter(self.CALLS_PER_MINUTE)

    @property
    def is_configured(self) -> bool:
        return bool(self._access_token and self._ad_account_id)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0, connect=5.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "MetaAdsClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make a GET request to the Graph API with rate limiting."""
        await self._rate_limiter.acquire()
        params["access_token"] = self._access_token

        response = await self.client.get(url, params=params)
        data = response.json()

        if "error" in data:
            error = data["error"]
            code = error.get("code", 0)
            msg = error.get("message", "Unknown Meta API error")

            if code == 190:
                raise MetaAdsError(
                    f"Meta access token expired or invalid: {msg}", error_code=code
                )
            if code == 4:
                raise MetaAdsError(
                    f"Meta API rate limit exceeded: {msg}", error_code=code
                )
            raise MetaAdsError(f"Meta API error ({code}): {msg}", error_code=code)

        return data

    async def search_interests(self, keyword: str) -> list[dict[str, Any]]:
        """
        Search for Meta interest targeting objects matching a keyword.

        Returns list of interest objects sorted by audience_size (largest first).
        Each object has: id, name, audience_size, path, topic.
        Returns empty list if no match found.
        """
        try:
            data = await self._get(
                f"{GRAPH_API_BASE}/search",
                params={"type": "adinterest", "q": keyword},
            )

            interests = data.get("data", [])
            return sorted(
                [
                    {
                        "id": i["id"],
                        "name": i["name"],
                        "audience_size": i.get("audience_size", 0),
                        "path": i.get("path", []),
                        "topic": i.get("topic", ""),
                    }
                    for i in interests
                ],
                key=lambda x: x["audience_size"],
                reverse=True,
            )

        except MetaAdsError:
            raise
        except Exception as e:
            logger.warning(f"Interest search failed for '{keyword}': {e}")
            return []

    async def get_delivery_estimate(
        self,
        interest_id: str,
        interest_name: str,
        country_code: str = "US",
        age_min: int = 18,
        age_max: int = 65,
        genders: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Get delivery estimate (audience reach) for an interest + geo.

        Args:
            interest_id: Meta interest ID from search_interests
            interest_name: Interest name for targeting_spec
            country_code: ISO 2-letter country code (uppercase)
            age_min: Minimum age (default 18)
            age_max: Maximum age (default 65)
            genders: [1]=male, [2]=female, None=all

        Returns:
            Dict with estimate_dau, estimate_mau_lower_bound,
            estimate_mau_upper_bound, estimate_ready.
        """
        targeting_spec: dict[str, Any] = {
            "geo_locations": {"countries": [country_code]},
            "age_min": age_min,
            "age_max": age_max,
            "flexible_spec": [
                {"interests": [{"id": interest_id, "name": interest_name}]}
            ],
        }

        if genders:
            targeting_spec["genders"] = genders

        data = await self._get(
            f"{GRAPH_API_BASE}/act_{self._ad_account_id}/delivery_estimate",
            params={
                "optimization_goal": "REACH",
                "targeting_spec": json.dumps(targeting_spec),
            },
        )

        estimates = data.get("data", [{}])
        estimate = estimates[0] if estimates else {}

        return {
            "estimate_dau": estimate.get("estimate_dau", 0),
            "estimate_mau_lower_bound": estimate.get("estimate_mau_lower_bound", 0),
            "estimate_mau_upper_bound": estimate.get("estimate_mau_upper_bound", 0),
            "estimate_ready": estimate.get("estimate_ready", False),
        }

    async def get_audience_reach(
        self,
        keywords: list[str],
        country_code: str = "US",
    ) -> dict[str, Any]:
        """
        Map keywords to Meta interests and get audience size estimates.

        This is the main entry point called by the API endpoint.

        Returns structured dict with:
            - matched_keywords: keywords that mapped to Meta interests with reach data
            - unmatched_keywords: keywords with no Meta interest match
            - total_reach: aggregated audience reach (largest single interest)
            - primary_interest_id: ID of the best-matched interest (for demographics)
        """
        matched = []
        unmatched = []
        primary_interest_id = None
        primary_interest_name = None
        max_audience = 0

        # Step 1: Search interests for all keywords in parallel
        search_tasks = [self.search_interests(kw) for kw in keywords]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Build list of (keyword, best_interest) pairs
        keyword_interests: list[tuple[str, dict[str, Any] | None]] = []
        for kw, result in zip(keywords, search_results):
            if isinstance(result, Exception):
                logger.warning(f"Interest search failed for '{kw}': {result}")
                keyword_interests.append((kw, None))
            elif result:
                keyword_interests.append((kw, result[0]))  # Take top match
            else:
                keyword_interests.append((kw, None))

        # Step 2: Get delivery estimates for matched keywords in parallel
        estimate_tasks = []
        estimate_indices = []
        for i, (kw, interest) in enumerate(keyword_interests):
            if interest:
                estimate_tasks.append(
                    self.get_delivery_estimate(
                        interest_id=interest["id"],
                        interest_name=interest["name"],
                        country_code=country_code,
                    )
                )
                estimate_indices.append(i)

        estimate_results = await asyncio.gather(
            *estimate_tasks, return_exceptions=True
        )

        # Step 3: Assemble results
        estimate_map: dict[int, dict[str, Any]] = {}
        for idx, result in zip(estimate_indices, estimate_results):
            if not isinstance(result, Exception):
                estimate_map[idx] = result
            else:
                logger.warning(f"Delivery estimate failed: {result}")

        for i, (kw, interest) in enumerate(keyword_interests):
            if interest is None:
                unmatched.append(kw)
                continue

            estimate = estimate_map.get(i)
            if estimate is None:
                unmatched.append(kw)
                continue

            mau_upper = estimate.get("estimate_mau_upper_bound", 0)
            entry = {
                "keyword": kw,
                "interest_id": interest["id"],
                "interest_name": interest["name"],
                "interest_audience_size": interest["audience_size"],
                "estimate_dau": estimate["estimate_dau"],
                "estimate_mau_lower_bound": estimate["estimate_mau_lower_bound"],
                "estimate_mau_upper_bound": mau_upper,
            }
            matched.append(entry)

            if mau_upper > max_audience:
                max_audience = mau_upper
                primary_interest_id = interest["id"]
                primary_interest_name = interest["name"]

        # Total reach: use the largest single interest (not sum, since audiences overlap)
        total_reach = {
            "estimate_dau": max(
                (m["estimate_dau"] for m in matched), default=0
            ),
            "estimate_mau_lower_bound": max(
                (m["estimate_mau_lower_bound"] for m in matched), default=0
            ),
            "estimate_mau_upper_bound": max(
                (m["estimate_mau_upper_bound"] for m in matched), default=0
            ),
            "platforms": ["facebook", "instagram"],
        }

        return {
            "matched_keywords": matched,
            "unmatched_keywords": unmatched,
            "total_reach": total_reach,
            "primary_interest_id": primary_interest_id,
            "primary_interest_name": primary_interest_name,
        }

    async def get_demographic_breakdown(
        self,
        interest_id: str,
        interest_name: str,
        country_code: str = "US",
    ) -> dict[str, Any]:
        """
        Get audience breakdown by age group and gender.

        Makes parallel API calls for each age/gender segment.

        Returns:
            Dict with age_groups list and gender_split ratios.
        """
        tasks = []
        segments = []

        for age_min, age_max in AGE_GROUPS:
            for gender_id, gender_label in [(1, "male"), (2, "female")]:
                tasks.append(
                    self.get_delivery_estimate(
                        interest_id=interest_id,
                        interest_name=interest_name,
                        country_code=country_code,
                        age_min=age_min,
                        age_max=age_max,
                        genders=[gender_id],
                    )
                )
                segments.append(
                    {"age_min": age_min, "age_max": age_max, "gender": gender_label}
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build age group breakdown
        age_groups: dict[str, dict[str, int]] = {}
        total_male = 0
        total_female = 0

        for segment, result in zip(segments, results):
            if isinstance(result, Exception):
                logger.warning(f"Demographic call failed: {result}")
                continue

            age_range = f"{segment['age_min']}-{segment['age_max']}+"  if segment["age_max"] == 65 else f"{segment['age_min']}-{segment['age_max']}"
            dau = result.get("estimate_dau", 0)

            if age_range not in age_groups:
                age_groups[age_range] = {"male": 0, "female": 0, "total": 0}

            age_groups[age_range][segment["gender"]] = dau
            age_groups[age_range]["total"] += dau

            if segment["gender"] == "male":
                total_male += dau
            else:
                total_female += dau

        total_all = total_male + total_female
        gender_split = {
            "male": round(total_male / total_all, 2) if total_all > 0 else 0.5,
            "female": round(total_female / total_all, 2) if total_all > 0 else 0.5,
        }

        return {
            "age_groups": [
                {"range": age_range, **data}
                for age_range, data in age_groups.items()
            ],
            "gender_split": gender_split,
            "total_dau": total_all,
        }
