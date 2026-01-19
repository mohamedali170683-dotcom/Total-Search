"""Tests for the exporters."""

import json
import csv
import pytest
from pathlib import Path
from datetime import datetime

from src.exporters.json_exporter import JSONExporter
from src.exporters.csv_exporter import CSVExporter
from src.models.keyword import (
    Platform,
    UnifiedKeywordData,
    GoogleMetrics,
    YouTubeMetrics,
    TrendDirection,
    Competition,
    Confidence,
)


@pytest.fixture
def sample_keyword_data():
    """Create sample keyword data for testing."""
    return UnifiedKeywordData(
        keyword="skincare",
        collected_at=datetime(2026, 1, 19, 10, 0, 0),
        platforms={
            Platform.GOOGLE: GoogleMetrics(
                search_volume=50000,
                trend=TrendDirection.GROWING,
                competition=Competition.MEDIUM,
                cpc=1.50,
                confidence=Confidence.HIGH,
                source="dataforseo",
            ),
            Platform.YOUTUBE: YouTubeMetrics(
                search_volume=25000,
                trend=TrendDirection.STABLE,
                confidence=Confidence.HIGH,
                source="dataforseo",
            ),
        },
        unified_demand_score=72,
        cross_platform_trend=TrendDirection.GROWING,
        best_platform=Platform.GOOGLE,
        tags=["beauty", "health"],
    )


@pytest.fixture
def multiple_keywords(sample_keyword_data):
    """Create multiple keyword data entries."""
    kw1 = sample_keyword_data

    kw2 = UnifiedKeywordData(
        keyword="vitamin c serum",
        collected_at=datetime(2026, 1, 19, 10, 0, 0),
        platforms={
            Platform.GOOGLE: GoogleMetrics(
                search_volume=30000,
                trend=TrendDirection.GROWING,
                competition=Competition.HIGH,
                cpc=2.50,
                confidence=Confidence.HIGH,
                source="dataforseo",
            ),
        },
        unified_demand_score=65,
        cross_platform_trend=TrendDirection.GROWING,
        best_platform=Platform.GOOGLE,
        tags=["beauty"],
    )

    kw3 = UnifiedKeywordData(
        keyword="retinol",
        collected_at=datetime(2026, 1, 19, 10, 0, 0),
        platforms={
            Platform.GOOGLE: GoogleMetrics(
                search_volume=40000,
                trend=TrendDirection.STABLE,
                competition=Competition.MEDIUM,
                cpc=1.75,
                confidence=Confidence.HIGH,
                source="dataforseo",
            ),
        },
        unified_demand_score=68,
        cross_platform_trend=TrendDirection.STABLE,
        best_platform=Platform.GOOGLE,
        tags=["beauty", "anti-aging"],
    )

    return [kw1, kw2, kw3]


class TestJSONExporter:
    """Tests for JSONExporter."""

    def test_export_jsonl(self, tmp_path, multiple_keywords):
        """Test JSONL export."""
        exporter = JSONExporter(output_dir=tmp_path)
        output_path = exporter.export_jsonl(multiple_keywords, "test.jsonl")

        assert output_path.exists()

        # Read and verify
        with open(output_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Each line should be valid JSON
        for line in lines:
            doc = json.loads(line)
            assert "keyword" in doc
            assert "unified_demand_score" in doc

    def test_export_json(self, tmp_path, multiple_keywords):
        """Test JSON array export."""
        exporter = JSONExporter(output_dir=tmp_path)
        output_path = exporter.export_json(multiple_keywords, "test.json")

        assert output_path.exists()

        with open(output_path, "r") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "keywords" in data
        assert len(data["keywords"]) == 3
        assert data["metadata"]["total_keywords"] == 3

    def test_export_json_no_metadata(self, tmp_path, sample_keyword_data):
        """Test JSON export without metadata."""
        exporter = JSONExporter(output_dir=tmp_path)
        output_path = exporter.export_json(
            [sample_keyword_data],
            "test_no_meta.json",
            include_metadata=False,
        )

        with open(output_path, "r") as f:
            data = json.load(f)

        # Should be a list, not dict with metadata
        assert isinstance(data, list)
        assert len(data) == 1

    def test_export_for_embedding(self, tmp_path, sample_keyword_data):
        """Test embedding-optimized export."""
        exporter = JSONExporter(output_dir=tmp_path)
        output_path = exporter.export_for_embedding(
            [sample_keyword_data],
            "test_embedding.jsonl",
        )

        assert output_path.exists()

        with open(output_path, "r") as f:
            doc = json.loads(f.readline())

        assert "id" in doc
        assert "text" in doc
        assert "metadata" in doc
        assert "skincare" in doc["text"].lower()

    def test_auto_filename(self, tmp_path, sample_keyword_data):
        """Test automatic filename generation."""
        exporter = JSONExporter(output_dir=tmp_path)
        output_path = exporter.export_jsonl([sample_keyword_data])

        assert output_path.exists()
        assert output_path.suffix == ".jsonl"
        assert "keywords_" in output_path.name


class TestCSVExporter:
    """Tests for CSVExporter."""

    def test_export_full(self, tmp_path, multiple_keywords):
        """Test full CSV export."""
        exporter = CSVExporter(output_dir=tmp_path)
        output_path = exporter.export_full(multiple_keywords, "test_full.csv")

        assert output_path.exists()

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

        # Check columns exist
        assert "keyword" in rows[0]
        assert "unified_demand_score" in rows[0]
        assert "google_volume" in rows[0]
        assert "google_cpc" in rows[0]

    def test_export_summary(self, tmp_path, multiple_keywords):
        """Test summary CSV export."""
        exporter = CSVExporter(output_dir=tmp_path)
        output_path = exporter.export_summary(multiple_keywords, "test_summary.csv")

        assert output_path.exists()

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

        # Summary should have fewer columns
        assert "keyword" in rows[0]
        assert "unified_demand_score" in rows[0]
        assert "total_effective_volume" in rows[0]
        # Should NOT have platform-specific columns
        assert "google_cpc" not in rows[0]

    def test_export_by_platform(self, tmp_path, multiple_keywords):
        """Test platform-specific CSV export."""
        exporter = CSVExporter(output_dir=tmp_path)
        output_path = exporter.export_by_platform(
            multiple_keywords,
            Platform.GOOGLE,
            "test_google.csv",
        )

        assert output_path.exists()

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # All keywords have Google data
        assert len(rows) == 3

        # Should have Google-specific columns
        assert "volume" in rows[0]
        assert "cpc" in rows[0]

    def test_empty_export(self, tmp_path):
        """Test export with empty data."""
        exporter = CSVExporter(output_dir=tmp_path)
        output_path = exporter.export_full([], "test_empty.csv")

        assert output_path.exists()

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 0

    def test_special_characters_in_keyword(self, tmp_path):
        """Test export handles special characters."""
        kw = UnifiedKeywordData(
            keyword="vitamin c, serum & cream",
            collected_at=datetime.now(),
            platforms={},
            unified_demand_score=50,
            cross_platform_trend=TrendDirection.STABLE,
            best_platform=None,
        )

        exporter = CSVExporter(output_dir=tmp_path)
        output_path = exporter.export_full([kw], "test_special.csv")

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["keyword"] == "vitamin c, serum & cream"

    def test_auto_filename(self, tmp_path, sample_keyword_data):
        """Test automatic filename generation."""
        exporter = CSVExporter(output_dir=tmp_path)
        output_path = exporter.export_summary([sample_keyword_data])

        assert output_path.exists()
        assert output_path.suffix == ".csv"
        assert "keywords_summary_" in output_path.name
