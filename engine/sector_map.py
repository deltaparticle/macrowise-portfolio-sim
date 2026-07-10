"""Regex-driven sector/theme tagging for the 264 Indian indices.

Each index gets one or more tags across 3 dimensions:
  - sector: IT / Banks / Auto / ... (industry sector)
  - size:   Largecap / Midcap / Smallcap / Broad ...
  - style:  Momentum / Quality / Value / LowVol / Growth / Dividend / MultiFactor / EqualWeight
  - theme:  ESG / Shariah / PSU / Defence / Manufacturing / Infra / Consumption / Digital
Indices flagged `exclude=True` (leveraged, inverse, futures, VIX, USD, dividend-points)
are dropped from the default optimization pool.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR


SECTOR_RULES: list[tuple[str, str]] = [
    ("IT",             r"\b(information technology|focused it|nifty it|bse it|teck|digital|internet|it & telecom)\b"),
    ("Banks",          r"\b(bank|bankex)\b"),
    ("FinancialServices", r"\b(financial services|financials ex|nbfc|insurance|capital markets|housing finance|diversified financials)\b"),
    ("Auto",           r"\b(auto|mobility|ev & new age|automotive)\b"),
    ("Pharma",         r"\b(pharma|healthcare|hospitals)\b"),
    ("FMCG",           r"\b(fmcg|fast moving consumer)\b"),
    ("Consumer",       r"\b(consumer durables|consumer discretionary|consumer services|india consumption|non.cyclical consumer|premium consumption|new age consumption|retail|midsmall india consumption|multicap consumption)\b"),
    ("Energy",         r"\b(energy|oil & gas|oil and gas|power|utilities)\b"),
    ("Metal",          r"\b(metal)\b"),
    ("Realty",         r"\b(realty|reit|housing)\b"),
    ("Infrastructure", r"\b(infrastructure|construction|cement|industrials|capital goods|transportation|logistics|commercial & transport|core housing|infra)\b"),
    ("Telecom",        r"\b(telecommunication|telecom)\b"),
    ("Media",          r"\b(media|waves)\b"),
    ("Defence",        r"\b(defence)\b"),
    ("Manufacturing",  r"\b(manufacturing)\b"),
    ("Chemicals",      r"\b(chemicals)\b"),
    ("Commodities",    r"\b(commodities)\b"),
    ("Services",       r"\b(services sector|services|tourism)\b"),
    ("Rural",          r"\b(rural)\b"),
    ("MNC",            r"\b(mnc)\b"),
]

SIZE_RULES: list[tuple[str, str]] = [
    ("Largecap",     r"\b(largecap|large cap|large.midcap|nifty 50|nifty50|sensex|bse sensex|bse 100|nifty 100|nifty100)\b"),
    ("Midcap",       r"\b(midcap|mid cap|midcap 100|midcap 150|midcap 50)\b"),
    ("Smallcap",     r"\b(smallcap|small cap)\b"),
    ("Microcap",     r"\b(microcap)\b"),
    ("MidSmallcap",  r"\b(midsmall|mid.small)\b"),
    ("Multicap",     r"\b(multicap|multi cap|total market)\b"),
    ("Broad",        r"\b(bse 200|bse 500|bse 1000|nifty 200|nifty 500|allcap|next 50|next 250|next 500|sensex next)\b"),
]

STYLE_RULES: list[tuple[str, str]] = [
    ("Momentum",     r"\b(momentum)\b"),
    ("Quality",      r"\b(quality)\b"),
    ("Value",        r"\b(value|enhanced value)\b"),
    ("LowVolatility", r"\b(low volatility|low.volatility)\b"),
    ("Alpha",        r"\b(alpha)\b"),
    ("Dividend",     r"\b(dividend)\b"),
    ("Growth",       r"\b(growth)\b"),
    ("HighBeta",     r"\b(high beta)\b"),
    ("MultiFactor",  r"\b(multifactor|multi.factor|momentum quality|mqvl)\b"),
    ("EqualWeight",  r"\b(equal weight|equal.cap|equal size)\b"),
    ("Liquid",       r"\b(liquid)\b"),
    ("IPO",          r"\b(ipo|sme emerge)\b"),
    ("Focused",      r"\b(focused)\b"),
]

THEME_RULES: list[tuple[str, str]] = [
    ("ESG",          r"\b(esg|clean environment)\b"),
    ("Shariah",      r"\b(shariah)\b"),
    ("PSU",          r"\b(psu|cpse|pse|public sector|bharat 22|railways psu)\b"),
    ("CorporateGroup", r"\b(corporate group|birla group|tata group|mahindra group|select business groups|5 corporate groups|maatr)\b"),
    ("Digital",      r"\b(digital|internet)\b"),
    ("Housing",      r"\b(housing)\b"),
    ("Sectoral",     r"\b(sector leaders|india 150|india select|top 10|top 15|top 20)\b"),
]

# Exclusions from default optimization pool
EXCLUDE_RULES = [
    r"\binverse\b",
    r"\bleverage\b",
    r"\bfutures\b",
    r"\bvix\b",
    r"\busd\b",
    r"\bdividend points\b",
    r"\b2x\b",
    r"\b1x inverse\b",
]

# Broad-market anchor set — used for CAPM/Black-Litterman "market" proxy
MARKET_PROXY_CANDIDATES = [
    "nifty_500", "nifty500", "bse_500", "nifty_total_market",
    "nifty_50", "bse_sensex", "bse_100", "nifty_100",
]


@dataclass
class IndexTag:
    slug: str
    display: str
    sectors: list[str] = field(default_factory=list)
    sizes: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    exclude: bool = False


def _match_all(text: str, rules: list[tuple[str, str]]) -> list[str]:
    hits = []
    for tag, pattern in rules:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(tag)
    return hits


def tag_index(slug: str, display: str | None = None) -> IndexTag:
    display = display or slug.replace("_", " ")
    text = display.lower()
    exclude = any(re.search(p, text, re.IGNORECASE) for p in EXCLUDE_RULES)
    return IndexTag(
        slug=slug,
        display=display,
        sectors=_match_all(text, SECTOR_RULES),
        sizes=_match_all(text, SIZE_RULES),
        styles=_match_all(text, STYLE_RULES),
        themes=_match_all(text, THEME_RULES),
        exclude=exclude,
    )


def build_map(data_dir: Path = DATA_DIR) -> dict[str, IndexTag]:
    out = {}
    for p in sorted(data_dir.glob("*_yfinance.csv")):
        slug = p.stem.replace("_yfinance", "")
        display = slug.replace("_", " ").title()
        out[slug] = tag_index(slug, display)
    return out


def indices_for_sectors(
    sectors: list[str],
    tag_map: dict[str, IndexTag] | None = None,
    include_excluded: bool = False,
    match_any: bool = True,
) -> list[str]:
    """Return slugs whose sector/theme tags match any (or all) of `sectors`."""
    tag_map = tag_map or build_map()
    wanted = {s.lower() for s in sectors}
    picked = []
    for slug, tag in tag_map.items():
        if tag.exclude and not include_excluded:
            continue
        tags = {t.lower() for t in tag.sectors + tag.themes + tag.sizes + tag.styles}
        if match_any:
            if wanted & tags:
                picked.append(slug)
        else:
            if wanted <= tags:
                picked.append(slug)
    return picked


def market_proxy(tag_map: dict[str, IndexTag] | None = None) -> str | None:
    tag_map = tag_map or build_map()
    for cand in MARKET_PROXY_CANDIDATES:
        if cand in tag_map:
            return cand
    return None


def available_sectors(tag_map: dict[str, IndexTag] | None = None) -> dict[str, list[str]]:
    tag_map = tag_map or build_map()
    out = {"sectors": set(), "sizes": set(), "styles": set(), "themes": set()}
    for t in tag_map.values():
        if t.exclude:
            continue
        out["sectors"].update(t.sectors)
        out["sizes"].update(t.sizes)
        out["styles"].update(t.styles)
        out["themes"].update(t.themes)
    return {k: sorted(v) for k, v in out.items()}


if __name__ == "__main__":
    import json
    tm = build_map()
    print(f"Tagged {len(tm)} indices ({sum(1 for t in tm.values() if t.exclude)} excluded).")
    print(json.dumps(available_sectors(tm), indent=2))
