#!/usr/bin/env python3
"""
apply_country_extraction_fix.py

Post-hoc correction for two classes of producer-country extraction errors
discovered during reviewer W1 investigation:

  Bug 1 — PREFIX match (Niger→Nigeria, Dominica→Dominican Republic, Guinea→Guinea-Bissau):
    The original code used `f", {c_low}"` which is a prefix of longer country
    names (e.g. ", niger" is a prefix of ", nigeria"). Fixed by word-boundary regex.

  Bug 2 — INSTITUTION-NAME substring (Georgia→Georgia Tech, Mexico→New Mexico):
    Country names that appear as the first word of well-known US institution names
    were matched as those countries. Fixed by scrubbing known institution fragments
    before re-testing the match.

Strategy:
  - Re-apply the fixed extract_countries_from_text() to the stored author_affiliations_raw
    column (which holds the same aff_text that the original function received).
  - Keep author_institution_countries and author_email_countries unchanged (they were
    produced by different, unaffected functions).
  - Recompute author_all_inferred_countries as the union of the three corrected columns.
  - Recompute author_countries_final using the same regex-vs-LLM merge logic as the
    original pipeline.

Usage:
    python apply_country_extraction_fix.py \\
        --input  author_affiliation_all_merged.csv \\
        --output author_affiliation_all_merged_corrected.csv \\
        --report country_fix_report.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


# ── Same exclusion patterns as added to author_affiliation_extract.py ─────────
INSTITUTION_EXCLUSION_PATTERNS: Dict[str, List[str]] = {
    "georgia": [
        "georgia institute of technology", "georgia tech", "gatech",
        "georgia state university", "university of georgia",
        "georgia southern", "georgia college", "georgia gwinnett",
        "kennesaw state",
    ],
    "mexico": [
        "new mexico state", "new mexico institute", "university of new mexico",
        "new mexico tech", "new mexico highlands",
    ],
}

# ── Minimal country list mirrored from author_affiliation_extract.py ─────────
# We import it directly if available; otherwise rebuild a stripped-down version.
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from author_affiliation_extract import COUNTRY_LIST, COUNTRY_ALIASES
except ImportError:
    # Fallback: build from pycountry
    try:
        import pycountry
        _c = set()
        for c in pycountry.countries:
            _c.add(c.name)
            if hasattr(c, "common_name"):
                _c.add(c.common_name)
        COUNTRY_LIST = sorted(_c, key=lambda x: len(x), reverse=True)
    except ImportError:
        raise SystemExit("Cannot import COUNTRY_LIST. Ensure author_affiliation_extract.py is in the same directory.")
    COUNTRY_ALIASES = {
        "usa": "United States", "u.s.a.": "United States", "u.s.": "United States",
        "us": "United States", "uk": "United Kingdom", "u.k.": "United Kingdom",
        "republic of korea": "South Korea", "uae": "United Arab Emirates",
        "prc": "China", "p.r. china": "China",
    }


def _wb_pattern(name_lower: str) -> re.Pattern:
    return re.compile(r"(?<![a-z])" + re.escape(name_lower) + r"(?![a-z])")


_COMPILED: List[Tuple[re.Pattern, str]] = []
_ALIAS_COMPILED: List[Tuple[re.Pattern, str]] = []
_EXCL_COMPILED: Dict[str, Tuple[re.Pattern, List[str]]] = {}


def _build() -> None:
    for alias, canon in COUNTRY_ALIASES.items():
        _ALIAS_COMPILED.append((_wb_pattern(alias.lower()), canon))
    for country in COUNTRY_LIST:
        _COMPILED.append((_wb_pattern(country.lower()), country))
    for country, frags in INSTITUTION_EXCLUSION_PATTERNS.items():
        _EXCL_COMPILED[country] = (_wb_pattern(country), frags)


_build()


def fixed_extract_countries(text: str) -> List[str]:
    """
    Re-implementation of extract_countries_from_text with both bugs fixed.
    Operates on the same aff_text string that the original function received.
    """
    if not text or pd.isna(text):
        return []
    low = str(text).lower()
    found: List[str] = []

    for pat, canon in _ALIAS_COMPILED:
        if pat.search(low):
            found.append(canon)

    for pat, country in _COMPILED:
        if pat.search(low):
            key = country.lower()
            if key in _EXCL_COMPILED:
                excl_pat, frags = _EXCL_COMPILED[key]
                scrubbed = low
                for frag in frags:
                    scrubbed = scrubbed.replace(frag, " ")
                if not excl_pat.search(scrubbed):
                    continue
            found.append(country)

    normalized: List[str] = []
    for c in found:
        if c == "United States of America":
            c = "United States"
        elif c == "UK":
            c = "United Kingdom"
        normalized.append(c)

    return sorted(set(normalized))


def _safe_str(val) -> str:
    """Convert a pandas cell value to a clean string, treating NaN/None as empty."""
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def _split(cell) -> List[str]:
    s = _safe_str(cell)
    if not s:
        return []
    return [c.strip() for c in s.split(";") if c.strip() and c.strip().lower() != "nan"]


def _join(items: List[str]) -> str:
    seen: Set[str] = set()
    out = []
    for item in items:
        if item and item.lower() != "nan" and item not in seen:
            seen.add(item)
            out.append(item)
    return "; ".join(out)


def choose_final_countries(regex: str, llm: str) -> Tuple[str, str]:
    """Mirror of merge_author_affiliation_regex_llm.choose_final_countries."""
    r_ok = bool(regex and regex.strip())
    l_ok = bool(llm and llm.strip())
    if r_ok and not l_ok:
        return regex, "regex"
    if not r_ok and l_ok:
        return llm, "llm_fallback"
    if r_ok and l_ok:
        r_list = _split(regex)
        l_list = _split(llm)
        if set(l_list) == set(r_list):
            return regex, "regex"
        if len(l_list) > len(r_list):
            return llm, "llm_fallback"
        return regex, "regex"
    return "", ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {len(df)} rows from {args.input}")

    report_rows = []
    new_aff_countries = []
    new_all_countries = []
    new_final_countries = []
    new_final_sources = []

    for _, row in df.iterrows():
        aff_text = str(row.get("author_affiliations_raw", "") or "")

        # Step 1: Re-extract text-based countries with fixed function
        corrected_aff = fixed_extract_countries(aff_text)

        # Step 2: Keep email and institution inference unchanged (they were correct)
        inst_countries = _split(str(row.get("author_institution_countries", "") or ""))
        email_countries = _split(str(row.get("author_email_countries", "") or ""))

        # Step 3: Recompute all-inferred union
        all_countries = sorted(set(corrected_aff + inst_countries + email_countries))

        # Step 4: Recompute final using same merge logic as original pipeline
        regex_candidate = _join(all_countries)
        llm_candidate = _safe_str(row.get("llm_countries", ""))
        final, source = choose_final_countries(regex_candidate, llm_candidate)

        # Track changes
        original_aff = _split(str(row.get("author_affiliation_countries", "") or ""))
        original_final = _split(str(row.get("author_countries_final", "") or ""))
        removed = sorted(set(original_final) - set(_split(final)))
        added = sorted(set(_split(final)) - set(original_final))

        if removed or added:
            report_rows.append({
                "paper_url": row.get("paper_url", ""),
                "original_aff_countries": _join(original_aff),
                "corrected_aff_countries": _join(corrected_aff),
                "original_final": _join(original_final),
                "corrected_final": final,
                "removed": _join(removed),
                "added": _join(added),
                "affiliations_sample": aff_text[:300],
            })

        new_aff_countries.append(_join(corrected_aff))
        new_all_countries.append(_join(all_countries))
        new_final_countries.append(final)
        new_final_sources.append(source + "|post_hoc_fix" if removed or added else source)

    df["author_affiliation_countries"] = new_aff_countries
    df["author_all_inferred_countries"] = new_all_countries
    df["author_countries_regex_final_candidate"] = new_all_countries
    df["author_countries_final"] = new_final_countries
    df["author_countries_final_source"] = new_final_sources

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote corrected CSV → {args.output}")

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(args.report, index=False)
    print(f"Wrote fix report ({len(report_df)} rows changed) → {args.report}")

    by_removed: Dict[str, int] = {}
    by_added: Dict[str, int] = {}
    for r in report_rows:
        for c in _split(r["removed"]):
            by_removed[c] = by_removed.get(c, 0) + 1
        for c in _split(r["added"]):
            by_added[c] = by_added.get(c, 0) + 1

    print("\nCountries removed (paper-level):")
    for c, n in sorted(by_removed.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")
    if by_added:
        print("\nCountries added (paper-level):")
        for c, n in sorted(by_added.items(), key=lambda x: -x[1]):
            print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
