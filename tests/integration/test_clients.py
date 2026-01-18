"""Integration tests for API clients with mocked responses."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.clients.dataforseo import DataForSEOClient
from src.clients.junglescout import JungleScoutClient
from src.models.keyword import Competition, Confidence, TrendDirection


class TestDataForSEOClient:
    """Integration tests for DataForSEO client."""

    @pytest.fixture
    def client(self, settings):
        """Create DataForSEO client with test settings."""
        return DataForSEOClient(
            login=settings.dataforseo_login,
            password=settings.dataforseo_password.get_secret_value(),
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_get_google_search_volume(self, client, mock_dataforseo_response):
        """Test Google search volume fetching with mocked response."""
        with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_dataforseo_response

            metrics = await client.get_google_search_volume(["test keyword"])

            assert len(metrics) == 1
            assert metrics[0].search_volume == 10000
            assert metrics[0].competition == Competition.MEDIUM
            assert metrics[0].cpc == 1.50
            assert metrics[0].confidence == Confidence.HIGH

    @pytest.mark.asyncio
    async def test_get_google_search_volume_batch(self, client, mock_dataforseo_response):
        """Test batch Google search volume fetching."""
        with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_dataforseo_response

            keywords = ["keyword1", "keyword2", "keyword3"]
            metrics = await client.get_google_search_volume(keywords)

            # Even if API returns one result, we should get metrics for all keywords
            assert len(metrics) == len(keywords)

    @pytest.mark.asyncio
    async def test_get_google_search_volume_error_handling(self, client):
        """Test error handling in Google search volume fetching."""
        with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("API Error")

            metrics = await client.get_google_search_volume(["test"])

            # Should return metrics with error info instead of raising
            assert len(metrics) == 1
            assert metrics[0].confidence == Confidence.PROXY
            assert "error" in metrics[0].raw_data

    @pytest.mark.asyncio
    async def test_get_keywords_for_site(self, client):
        """Test keyword suggestions for domain."""
        mock_response = {
            "tasks": [
                {
                    "status_code": 20000,
                    "result": [
                        {"keyword": "skincare routine"},
                        {"keyword": "best moisturizer"},
                        {"keyword": "anti aging cream"},
                    ],
                }
            ]
        }

        with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            keywords = await client.get_keywords_for_site("example.com", limit=10)

            assert len(keywords) == 3
            assert "skincare routine" in keywords

    def test_calculate_trend_growing(self, client):
        """Test trend calculation for growing keyword."""
        from src.models.keyword import MonthlySearchData

        # Increasing monthly data
        monthly_data = [
            MonthlySearchData(year=2026, month=i, search_volume=1000 + i * 200)
            for i in range(1, 13)
        ]

        trend, velocity = client._calculate_trend(monthly_data)

        assert trend == TrendDirection.GROWING
        assert velocity > 1.0

    def test_calculate_trend_declining(self, client):
        """Test trend calculation for declining keyword."""
        from src.models.keyword import MonthlySearchData

        # Decreasing monthly data
        monthly_data = [
            MonthlySearchData(year=2026, month=i, search_volume=3000 - i * 200)
            for i in range(1, 13)
        ]

        trend, velocity = client._calculate_trend(monthly_data)

        assert trend == TrendDirection.DECLINING
        assert velocity < 1.0


class TestJungleScoutClient:
    """Integration tests for Jungle Scout client."""

    @pytest.fixture
    def client(self, settings):
        """Create Jungle Scout client with test settings."""
        return JungleScoutClient(
            api_key=settings.junglescout_api_key.get_secret_value(),
            api_key_name=settings.junglescout_api_key_name,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_get_amazon_search_volume(self, client):
        """Test Amazon search volume fetching with mocked response."""
        mock_response = {
            "data": [
                {
                    "attributes": {
                        "name": "test keyword",
                        "exact_match_search_volume": 8000,
                        "broad_match_search_volume": 15000,
                        "organic_product_count": 500,
                        "sponsored_product_count": 25,
                    }
                }
            ]
        }

        with patch.object(client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            metrics = await client.get_amazon_search_volume(["test keyword"])

            assert len(metrics) == 1
            assert metrics[0].exact_search_volume == 8000
            assert metrics[0].broad_search_volume == 15000
            assert metrics[0].organic_product_count == 500

    @pytest.mark.asyncio
    async def test_get_keywords_by_asin(self, client):
        """Test reverse ASIN keyword lookup."""
        mock_response = {
            "data": [
                {"attributes": {"name": "keyword 1"}},
                {"attributes": {"name": "keyword 2"}},
                {"attributes": {"name": "keyword 3"}},
            ]
        }

        with patch.object(client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            keywords = await client.get_keywords_by_asin("B0123456789")

            assert len(keywords) == 3
            assert "keyword 1" in keywords

    def test_estimate_competition_high(self, client):
        """Test competition estimation for high-competition keyword."""
        data = {"sponsored_product_count": 30, "organic_product_count": 500}
        competition = client._estimate_competition(data)
        assert competition == Competition.HIGH

    def test_estimate_competition_low(self, client):
        """Test competition estimation for low-competition keyword."""
        data = {"sponsored_product_count": 5, "organic_product_count": 500}
        competition = client._estimate_competition(data)
        assert competition == Competition.LOW

    def test_estimate_competition_no_data(self, client):
        """Test competition estimation with no data."""
        data = {"sponsored_product_count": 0, "organic_product_count": 0}
        competition = client._estimate_competition(data)
        assert competition is None
