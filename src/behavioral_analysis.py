"""Behavioral search intelligence: funnel mapping, share of search, modifier analysis.

Three analysis engines that extract actionable behavioral insights from search data:

1. Funnel Stage Mapping — classifies search demand into awareness/consideration/purchase
   stages using modifier-based keyword grouping. Different categories show dramatically
   different funnel profiles (validated: natural cosmetics = 46% purchase vs protein
   powder = 79% consideration).

2. Share of Search — calculates each brand's share of total category search demand.
   Based on Les Binet's research showing share of search predicts market share with
   83% accuracy across 30 case studies. Uses sum-of-brands as denominator (not a
   single category term, which is too narrow).

3. Modifier Trend Analysis — tracks macro behavioral modifiers (günstig vs beste,
   test vs erfahrungen) to detect long-term consumer behavior shifts. Works at the
   bare modifier level, not product-specific (validated with real pytrends data).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Funnel modifier definitions by language
# ---------------------------------------------------------------------------

FUNNEL_MODIFIERS: dict[str, dict[str, list[str]]] = {
    "de": {
        "awareness": ["was ist", "vorteile", "wirkung", "erklärt", "bedeutung"],
        "consideration": ["beste", "bester", "bestes", "test", "vergleich",
                          "erfahrungen", "bewertung", "empfehlung", "vs"],
        "purchase": ["kaufen", "bestellen", "online shop", "günstig kaufen",
                      "preis", "angebot", "rabatt"],
    },
    "en": {
        "awareness": ["what is", "benefits", "how does", "explained", "meaning"],
        "consideration": ["best", "review", "comparison", "vs", "top",
                          "recommended", "rating"],
        "purchase": ["buy", "order", "shop", "price", "deal", "discount",
                      "coupon", "cheap"],
    },
}

# Macro behavioral modifier pairs for trend analysis
BEHAVIORAL_MODIFIER_PAIRS: dict[str, list[dict]] = {
    "de": [
        {"a": "günstig", "b": "beste", "hypothesis": "price_vs_quality",
         "label": "Price-driven vs quality-driven search behavior"},
        {"a": "test", "b": "erfahrungen", "hypothesis": "expert_vs_peer",
         "label": "Expert reviews vs peer experience (trust shift)"},
        {"a": "kaufen", "b": "vergleich", "hypothesis": "buy_vs_compare",
         "label": "Direct purchase intent vs comparison shopping"},
    ],
    "en": [
        {"a": "cheap", "b": "best", "hypothesis": "price_vs_quality",
         "label": "Price-driven vs quality-driven search behavior"},
        {"a": "review", "b": "buy", "hypothesis": "research_vs_purchase",
         "label": "Research intent vs purchase readiness"},
        {"a": "how to", "b": "buy", "hypothesis": "learn_vs_purchase",
         "label": "Learning intent vs buying intent"},
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FunnelStageResult:
    stage: str  # awareness, consideration, purchase
    volume: int
    percentage: float
    keywords: list[dict]  # [{keyword, volume}]


@dataclass
class FunnelAnalysis:
    category: str
    language: str
    stages: list[FunnelStageResult]
    dominant_stage: str
    insight: str
    total_volume: int

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "language": self.language,
            "total_volume": self.total_volume,
            "dominant_stage": self.dominant_stage,
            "insight": self.insight,
            "stages": [
                {
                    "stage": s.stage,
                    "volume": s.volume,
                    "percentage": s.percentage,
                    "keywords": s.keywords,
                }
                for s in self.stages
            ],
        }


@dataclass
class ShareOfSearchResult:
    brand: str
    volume: int
    share_percentage: float
    trend: str  # from DataForSEO monthly data if available


@dataclass
class ShareOfSearchAnalysis:
    category: str
    total_category_volume: int
    branded_volume: int
    unbranded_percentage: float
    brands: list[ShareOfSearchResult]
    insight: str

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "total_category_volume": self.total_category_volume,
            "branded_volume": self.branded_volume,
            "unbranded_percentage": round(self.unbranded_percentage, 1),
            "insight": self.insight,
            "brands": [
                {
                    "brand": b.brand,
                    "volume": b.volume,
                    "share_percentage": round(b.share_percentage, 1),
                    "trend": b.trend,
                }
                for b in self.brands
            ],
        }


@dataclass
class ModifierPairResult:
    modifier_a: str
    modifier_b: str
    hypothesis: str
    label: str
    volume_a: int
    volume_b: int
    ratio: float  # b / a (>1 means b dominates)
    insight: str


@dataclass
class BehavioralModifierAnalysis:
    language: str
    pairs: list[ModifierPairResult]

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "pairs": [
                {
                    "modifier_a": p.modifier_a,
                    "modifier_b": p.modifier_b,
                    "hypothesis": p.hypothesis,
                    "label": p.label,
                    "volume_a": p.volume_a,
                    "volume_b": p.volume_b,
                    "ratio": round(p.ratio, 2),
                    "insight": p.insight,
                }
                for p in self.pairs
            ],
        }


# ---------------------------------------------------------------------------
# Feature 3: Funnel Stage Mapping
# ---------------------------------------------------------------------------

def build_funnel_keywords(
    category_keyword: str,
    language: str = "de",
) -> dict[str, list[str]]:
    """Generate funnel-stage keywords by combining category with modifiers.

    Example for "naturkosmetik" (de):
        awareness:     ["was ist naturkosmetik", "naturkosmetik vorteile", ...]
        consideration: ["beste naturkosmetik", "naturkosmetik test", ...]
        purchase:      ["naturkosmetik kaufen", "naturkosmetik bestellen", ...]
    """
    modifiers = FUNNEL_MODIFIERS.get(language, FUNNEL_MODIFIERS["en"])
    result = {}
    for stage, mods in modifiers.items():
        keywords = []
        for mod in mods:
            # Try both orders: "modifier keyword" and "keyword modifier"
            keywords.append(f"{mod} {category_keyword}")
            keywords.append(f"{category_keyword} {mod}")
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique.append(kw)
        result[stage] = unique
    return result


def analyze_funnel(
    category_keyword: str,
    keyword_volumes: dict[str, int],
    language: str = "de",
) -> FunnelAnalysis:
    """Classify keyword volumes into funnel stages and generate insight.

    Args:
        category_keyword: The base category (e.g., "naturkosmetik").
        keyword_volumes: Dict of keyword → monthly search volume.
        language: Language code for modifier selection.

    Returns:
        FunnelAnalysis with stage breakdown and actionable insight.
    """
    funnel_kws = build_funnel_keywords(category_keyword, language)

    stages = []
    total = 0
    for stage_name, kw_list in funnel_kws.items():
        stage_keywords = []
        stage_vol = 0
        for kw in kw_list:
            vol = keyword_volumes.get(kw, 0) or keyword_volumes.get(kw.lower(), 0)
            if vol > 0:
                stage_keywords.append({"keyword": kw, "volume": vol})
                stage_vol += vol
        stages.append(FunnelStageResult(
            stage=stage_name,
            volume=stage_vol,
            percentage=0,  # calculated below
            keywords=sorted(stage_keywords, key=lambda x: x["volume"], reverse=True),
        ))
        total += stage_vol

    # Calculate percentages
    for s in stages:
        s.percentage = round(s.volume / total * 100, 1) if total > 0 else 0

    # Determine dominant stage
    dominant = max(stages, key=lambda s: s.volume)
    dominant_stage = dominant.stage

    # Generate insight
    insight = _generate_funnel_insight(category_keyword, stages, dominant_stage)

    return FunnelAnalysis(
        category=category_keyword,
        language=language,
        stages=stages,
        dominant_stage=dominant_stage,
        insight=insight,
        total_volume=total,
    )


def _generate_funnel_insight(
    category: str,
    stages: list[FunnelStageResult],
    dominant: str,
) -> str:
    """Generate actionable insight text from funnel distribution."""
    stage_map = {s.stage: s for s in stages}
    aw = stage_map.get("awareness")
    co = stage_map.get("consideration")
    pu = stage_map.get("purchase")

    if dominant == "awareness":
        return (
            f'"{category}" is in the education phase — {aw.percentage}% of search demand '
            f"is awareness-focused. Invest in educational content (explainers, guides, "
            f'"what is" articles) to capture early-stage interest before competitors.'
        )
    elif dominant == "consideration":
        return (
            f'"{category}" is comparison-heavy — {co.percentage}% of searches are '
            f"consideration-stage (reviews, comparisons, tests). Create authoritative "
            f"comparison content and ensure strong presence in review platforms."
        )
    else:
        return (
            f'"{category}" is purchase-ready — {pu.percentage}% of search demand '
            f"is transactional. Optimize for conversion: pricing pages, shop SEO, "
            f"product availability, and last-click attribution channels."
        )


# ---------------------------------------------------------------------------
# Feature 2: Share of Search
# ---------------------------------------------------------------------------

def analyze_share_of_search(
    category: str,
    brand_volumes: dict[str, int],
    category_generic_volume: int = 0,
) -> ShareOfSearchAnalysis:
    """Calculate brand share of search within a category.

    Uses sum of all brand volumes + generic category volume as denominator
    (Les Binet method). This avoids the problem where individual brand volumes
    exceed a narrow category term.

    Args:
        category: Category name for labeling.
        brand_volumes: Dict of brand name → monthly search volume.
        category_generic_volume: Volume for the generic category term (optional).

    Returns:
        ShareOfSearchAnalysis with brand shares and insight.
    """
    total_branded = sum(brand_volumes.values())
    total = total_branded + category_generic_volume

    if total == 0:
        return ShareOfSearchAnalysis(
            category=category,
            total_category_volume=0,
            branded_volume=0,
            unbranded_percentage=0,
            brands=[],
            insight="Insufficient search volume data.",
        )

    brands = []
    for brand, vol in sorted(brand_volumes.items(), key=lambda x: x[1], reverse=True):
        brands.append(ShareOfSearchResult(
            brand=brand,
            volume=vol,
            share_percentage=vol / total * 100 if total > 0 else 0,
            trend="",
        ))

    unbranded_pct = category_generic_volume / total * 100 if total > 0 else 0

    # Generate insight
    leader = brands[0] if brands else None
    if leader and len(brands) >= 2:
        runner_up = brands[1]
        gap = leader.share_percentage - runner_up.share_percentage
        insight = (
            f"{leader.brand.title()} leads with {leader.share_percentage:.0f}% share of search "
            f"in {category}, {gap:.0f}pp ahead of {runner_up.brand.title()}. "
        )
        if unbranded_pct > 30:
            insight += (
                f"Note: {unbranded_pct:.0f}% of category searches are unbranded — "
                f"significant opportunity to capture generic demand."
            )
        elif unbranded_pct < 10:
            insight += (
                f"Only {unbranded_pct:.0f}% of searches are unbranded — "
                f"this is a brand-dominated category where awareness is key."
            )
    else:
        insight = f"Share of search data for {category}."

    return ShareOfSearchAnalysis(
        category=category,
        total_category_volume=total,
        branded_volume=total_branded,
        unbranded_percentage=unbranded_pct,
        brands=brands,
        insight=insight,
    )


# ---------------------------------------------------------------------------
# Feature 1: Behavioral Modifier Pairs
# ---------------------------------------------------------------------------

def analyze_modifier_pairs(
    keyword_volumes: dict[str, int],
    language: str = "de",
) -> BehavioralModifierAnalysis:
    """Analyze macro behavioral modifier volumes to detect consumer behavior patterns.

    Compares search volumes for opposing modifier pairs (e.g., "günstig" vs "beste")
    to reveal market-level behavioral orientation.

    Args:
        keyword_volumes: Dict of keyword → monthly search volume.
            Must include the bare modifier terms.
        language: Language code for modifier pair selection.

    Returns:
        BehavioralModifierAnalysis with pair comparisons and insights.
    """
    pairs_config = BEHAVIORAL_MODIFIER_PAIRS.get(language, BEHAVIORAL_MODIFIER_PAIRS["en"])

    pairs = []
    for cfg in pairs_config:
        vol_a = keyword_volumes.get(cfg["a"], 0)
        vol_b = keyword_volumes.get(cfg["b"], 0)

        if vol_a == 0 and vol_b == 0:
            continue

        ratio = vol_b / vol_a if vol_a > 0 else float("inf")
        insight = _generate_modifier_insight(cfg, vol_a, vol_b, ratio)

        pairs.append(ModifierPairResult(
            modifier_a=cfg["a"],
            modifier_b=cfg["b"],
            hypothesis=cfg["hypothesis"],
            label=cfg["label"],
            volume_a=vol_a,
            volume_b=vol_b,
            ratio=ratio,
            insight=insight,
        ))

    return BehavioralModifierAnalysis(language=language, pairs=pairs)


def _generate_modifier_insight(
    cfg: dict,
    vol_a: int,
    vol_b: int,
    ratio: float,
) -> str:
    """Generate insight for a modifier pair comparison."""
    hypothesis = cfg["hypothesis"]

    if hypothesis == "price_vs_quality":
        if ratio > 2:
            return (
                f'Quality-driven behavior dominates: "{cfg["b"]}" is searched {ratio:.1f}x more '
                f'than "{cfg["a"]}". Consumers prioritize quality over price — position '
                f"products on value and excellence, not discounts."
            )
        elif ratio < 0.5:
            return (
                f'Price-driven behavior dominates: "{cfg["a"]}" is searched {1/ratio:.1f}x more '
                f'than "{cfg["b"]}". Market is price-sensitive — consider competitive pricing '
                f"and deal-oriented messaging."
            )
        else:
            return (
                f'Price and quality are balanced: "{cfg["a"]}" ({vol_a:,}) vs "{cfg["b"]}" '
                f"({vol_b:,}). Market has mixed intent — segment messaging by audience."
            )

    elif hypothesis == "expert_vs_peer":
        if ratio > 1.5:
            return (
                f'Peer experience dominates: "{cfg["b"]}" outperforms "{cfg["a"]}" by {ratio:.1f}x. '
                f"Consumers trust peer reviews over expert tests — invest in UGC and "
                f"customer testimonials."
            )
        elif ratio < 0.7:
            return (
                f'Expert authority dominates: "{cfg["a"]}" leads "{cfg["b"]}" by {1/ratio:.1f}x. '
                f"Consumers rely on expert testing — pursue editorial reviews and "
                f"certification partnerships."
            )
        else:
            return (
                f'Both expert and peer reviews matter: "{cfg["a"]}" ({vol_a:,}) vs "{cfg["b"]}" '
                f"({vol_b:,}). Build credibility through both channels."
            )

    elif hypothesis in ("buy_vs_compare", "research_vs_purchase", "learn_vs_purchase"):
        if ratio > 1.5:
            return (
                f'Research intent dominates: "{cfg["b"]}" is {ratio:.1f}x higher than '
                f'"{cfg["a"]}". Market is still in evaluation mode — invest in mid-funnel '
                f"comparison content."
            )
        elif ratio < 0.7:
            return (
                f'Purchase intent dominates: "{cfg["a"]}" leads by {1/ratio:.1f}x. '
                f"Consumers are ready to buy — optimize conversion paths and "
                f"transactional search terms."
            )
        else:
            return (
                f'Balanced research and purchase intent: "{cfg["a"]}" ({vol_a:,}) vs '
                f'"{cfg["b"]}" ({vol_b:,}). Cover both informational and transactional terms.'
            )

    return f'"{cfg["a"]}" ({vol_a:,}) vs "{cfg["b"]}" ({vol_b:,}): ratio {ratio:.1f}x'


# ---------------------------------------------------------------------------
# Combined analysis
# ---------------------------------------------------------------------------

def get_all_behavioral_keywords(
    category_keyword: str,
    brands: list[str] | None = None,
    language: str = "de",
) -> list[str]:
    """Return all keywords needed for behavioral analysis in a single list.

    This allows the caller to fetch all volumes in one DataForSEO API call,
    then pass the results to each analysis function.
    """
    keywords = set()

    # Funnel keywords
    funnel_kws = build_funnel_keywords(category_keyword, language)
    for stage_kws in funnel_kws.values():
        keywords.update(stage_kws)

    # Brand names (for share of search)
    if brands:
        keywords.update(brands)

    # Category generic term
    keywords.add(category_keyword)

    # Behavioral modifier pairs (bare modifiers)
    pairs = BEHAVIORAL_MODIFIER_PAIRS.get(language, BEHAVIORAL_MODIFIER_PAIRS["en"])
    for pair in pairs:
        keywords.add(pair["a"])
        keywords.add(pair["b"])

    return sorted(keywords)
