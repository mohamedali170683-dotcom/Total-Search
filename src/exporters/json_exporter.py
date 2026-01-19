"""JSON exporter for keyword data - optimized for RAG ingestion."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models.keyword import UnifiedKeywordData

logger = logging.getLogger(__name__)


class JSONExporter:
    """
    Export keyword data to JSON/JSONL formats for RAG system ingestion.

    Supports:
    - JSONL (one JSON object per line) - recommended for RAG
    - JSON array format
    - Nested JSON with metadata
    """

    def __init__(self, output_dir: str | Path = "data/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_jsonl(
        self,
        keywords: list[UnifiedKeywordData],
        filename: str | None = None,
        include_raw_data: bool = False,
    ) -> Path:
        """
        Export keywords to JSONL format (one JSON object per line).

        This is the recommended format for RAG ingestion as it:
        - Allows streaming/chunked processing
        - Each line is a complete document
        - Easy to append new data

        Args:
            keywords: List of unified keyword data
            filename: Output filename (auto-generated if None)
            include_raw_data: Include raw API responses

        Returns:
            Path to the exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keywords_{timestamp}.jsonl"

        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            for kw in keywords:
                doc = self._to_rag_document(kw, include_raw_data)
                f.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")

        logger.info(f"Exported {len(keywords)} keywords to {output_path}")
        return output_path

    def export_json(
        self,
        keywords: list[UnifiedKeywordData],
        filename: str | None = None,
        include_raw_data: bool = False,
        include_metadata: bool = True,
    ) -> Path:
        """
        Export keywords to JSON array format.

        Args:
            keywords: List of unified keyword data
            filename: Output filename (auto-generated if None)
            include_raw_data: Include raw API responses
            include_metadata: Include export metadata (timestamp, count, etc.)

        Returns:
            Path to the exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keywords_{timestamp}.json"

        output_path = self.output_dir / filename

        documents = [
            self._to_rag_document(kw, include_raw_data)
            for kw in keywords
        ]

        if include_metadata:
            output = {
                "metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "total_keywords": len(keywords),
                    "format_version": "1.0",
                },
                "keywords": documents,
            }
        else:
            output = documents

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exported {len(keywords)} keywords to {output_path}")
        return output_path

    def export_for_embedding(
        self,
        keywords: list[UnifiedKeywordData],
        filename: str | None = None,
    ) -> Path:
        """
        Export keywords in a format optimized for embedding generation.

        Creates text representations suitable for vector embedding:
        - Combines keyword with platform metrics into natural language
        - Includes trend and competition context

        Args:
            keywords: List of unified keyword data
            filename: Output filename

        Returns:
            Path to the exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keywords_embedding_{timestamp}.jsonl"

        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            for kw in keywords:
                doc = {
                    "id": kw.keyword.lower().replace(" ", "_"),
                    "text": self._generate_embedding_text(kw),
                    "metadata": {
                        "keyword": kw.keyword,
                        "score": kw.unified_demand_score,
                        "trend": kw.cross_platform_trend.value if kw.cross_platform_trend else None,
                        "best_platform": kw.best_platform.value if kw.best_platform else None,
                    }
                }
                f.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")

        logger.info(f"Exported {len(keywords)} keywords for embedding to {output_path}")
        return output_path

    def _to_rag_document(
        self,
        kw: UnifiedKeywordData,
        include_raw_data: bool = False,
    ) -> dict[str, Any]:
        """Convert UnifiedKeywordData to RAG-friendly document."""
        doc = kw.to_rag_document()

        if not include_raw_data and "platforms" in doc:
            # Remove raw_data from each platform
            for platform_data in doc.get("platforms", {}).values():
                if isinstance(platform_data, dict):
                    platform_data.pop("raw_data", None)

        return doc

    def _generate_embedding_text(self, kw: UnifiedKeywordData) -> str:
        """Generate natural language text for embedding."""
        parts = [f"Keyword: {kw.keyword}"]

        # Add score context
        score = kw.unified_demand_score
        if score >= 70:
            parts.append(f"High demand keyword with unified score of {score}/100.")
        elif score >= 40:
            parts.append(f"Moderate demand keyword with unified score of {score}/100.")
        else:
            parts.append(f"Lower demand keyword with unified score of {score}/100.")

        # Add trend
        if kw.cross_platform_trend:
            trend = kw.cross_platform_trend.value
            parts.append(f"The trend across platforms is {trend}.")

        # Add best platform
        if kw.best_platform:
            parts.append(f"Best performing platform: {kw.best_platform.value}.")

        # Add platform-specific volumes
        platform_info = []
        for platform, metrics in kw.platforms.items():
            if metrics:
                vol = metrics.effective_volume
                if vol and vol > 0:
                    platform_info.append(f"{platform}: {vol:,} searches")

        if platform_info:
            parts.append(f"Platform volumes: {', '.join(platform_info)}.")

        return " ".join(parts)
