#!/usr/bin/env python3
"""
Build final normalized AtlasNLP v2 analysis CSVs.

Inputs
------
1) atlasnlp_audit_sampling_frame.csv
   - authoritative 18,035-row order / IDs
   - existing cleaned language fields
   - task and producer metadata

2) content_country_v2_merged.csv
   - 18,031 successfully processed v2.2 records

Outputs
-------
- atlasnlp_core_v2_all_normalized.csv
- atlasnlp_core_v2_eligible.csv
- atlasnlp_core_v2_explicit.csv
- atlasnlp_core_v2_inferred_only.csv
- atlasnlp_core_v2_explicit_plus_inferred.csv
- atlasnlp_core_v2_explicit_country_expanded.csv
- atlasnlp_core_v2_explicit_plus_inferred_country_expanded.csv
- country_normalization_qc.csv
- normalization_report.json
- normalization_report.txt

Design principles
-----------------
- Preserve the original 18,035-row order.
- Reuse the prior cleaned language fields verbatim; do not re-extract languages.
- Normalize new content-country outputs to the same 197-entity geopolitical ontology:
  195 states used by the prior AtlasNLP ontology + Taiwan + Kosovo.
- Map common aliases and subnational/territorial labels to the governing country
  when that was the prior project convention.
- Drop non-country aggregate labels and genuinely ambiguous country strings rather
  than guessing.
- Normalize explicit and inferred lists separately, then remove inferred duplicates
  when the same normalized country is already explicit.
- Keep the four failed v2 records in the 18,035-row master as PROCESSING_FAILURE.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CANONICAL_197 = [
    "Afghanistan","Albania","Algeria","Andorra","Angola","Antigua and Barbuda",
    "Argentina","Armenia","Australia","Austria","Azerbaijan","Bahamas","Bahrain",
    "Bangladesh","Barbados","Belarus","Belgium","Belize","Benin","Bhutan","Bolivia",
    "Bosnia and Herzegovina","Botswana","Brazil","Brunei","Bulgaria","Burkina Faso",
    "Burundi","Cabo Verde","Cambodia","Cameroon","Canada","Central African Republic",
    "Chad","Chile","China","Colombia","Comoros","Congo (Congo-Brazzaville)",
    "Costa Rica","Cote d'Ivoire","Croatia","Cuba","Cyprus","Czechia",
    "Democratic Republic of the Congo","Denmark","Djibouti","Dominica",
    "Dominican Republic","Ecuador","Egypt","El Salvador","Equatorial Guinea",
    "Eritrea","Estonia","Eswatini","Ethiopia","Fiji","Finland","France","Gabon",
    "Gambia","Georgia","Germany","Ghana","Greece","Grenada","Guatemala","Guinea",
    "Guinea-Bissau","Guyana","Haiti","Honduras","Hungary","Iceland","India",
    "Indonesia","Iran","Iraq","Ireland","Israel","Italy","Jamaica","Japan","Jordan",
    "Kazakhstan","Kenya","Kiribati","Kosovo","Kuwait","Kyrgyzstan","Laos","Latvia",
    "Lebanon","Lesotho","Liberia","Libya","Liechtenstein","Lithuania","Luxembourg",
    "Madagascar","Malawi","Malaysia","Maldives","Mali","Malta","Marshall Islands",
    "Mauritania","Mauritius","Mexico","Micronesia","Moldova","Monaco","Mongolia",
    "Montenegro","Morocco","Mozambique","Myanmar","Namibia","Nauru","Nepal",
    "Netherlands","New Zealand","Nicaragua","Niger","Nigeria","North Korea",
    "North Macedonia","Norway","Oman","Pakistan","Palau","Palestine","Panama",
    "Papua New Guinea","Paraguay","Peru","Philippines","Poland","Portugal","Qatar",
    "Romania","Russia","Rwanda","Saint Kitts and Nevis","Saint Lucia",
    "Saint Vincent and the Grenadines","Samoa","San Marino","São Tomé and Príncipe",
    "Saudi Arabia","Senegal","Serbia","Seychelles","Sierra Leone","Singapore",
    "Slovakia","Slovenia","Solomon Islands","Somalia","South Africa","South Korea",
    "South Sudan","Spain","Sri Lanka","Sudan","Suriname","Sweden","Switzerland",
    "Syria","Taiwan","Tajikistan","Tanzania","Thailand","Timor-Leste","Togo","Tonga",
    "Trinidad and Tobago","Tunisia","Turkey","Turkmenistan","Tuvalu","Uganda",
    "Ukraine","United Arab Emirates","United Kingdom","United States","Uruguay",
    "Uzbekistan","Vanuatu","Vatican City","Venezuela","Vietnam","Yemen","Zambia",
    "Zimbabwe",
]
CANONICAL_SET = set(CANONICAL_197)

# Deterministic aliases. These are normalization only; they do not create geography
# where the extractor did not already propose it.
COUNTRY_ALIASES = {
    # common abbreviations / official-name variants
    "USA": "United States",
    "US": "United States",
    "U.S.": "United States",
    "U.S.A.": "United States",
    "United States of America": "United States",
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "Britain": "United Kingdom",
    "Great Britain": "United Kingdom",
    "The Czech Republic": "Czechia",
    "Czech Republic": "Czechia",
    "Russian Federation": "Russia",
    "Republic of Korea": "South Korea",
    "The Republic of Korea": "South Korea",
    "Korea, Republic of": "South Korea",
    "Republic of Moldova": "Moldova",
    "FYR Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    "Bosnia": "Bosnia and Herzegovina",
    "Saint Vincent": "Saint Vincent and the Grenadines",
    "United Emirates": "United Arab Emirates",
    "UAE": "United Arab Emirates",
    "Viet Nam": "Vietnam",
    "Türkiye": "Turkey",
    "Ivory Coast": "Cote d'Ivoire",
    "Côte d’Ivoire": "Cote d'Ivoire",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "State of Palestine": "Palestine",
    "Cape Verde": "Cabo Verde",
    "Swaziland": "Eswatini",
    "East Timor": "Timor-Leste",
    "Burma": "Myanmar",
    "Brunei Darussalam": "Brunei",
    "Lao People's Democratic Republic": "Laos",
    "Iran, Islamic Republic of": "Iran",
    "Bolivia, Plurinational State of": "Bolivia",
    "Venezuela, Bolivarian Republic of": "Venezuela",
    "Tanzania, United Republic of": "Tanzania",
    "Syrian Arab Republic": "Syria",
    "Micronesia, Federated States of": "Micronesia",

    # subnational / territorial labels -> governing country under AtlasNLP ontology
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
    "Wales": "United Kingdom",
    "Northern Ireland": "United Kingdom",
    "Hong Kong": "China",
    "Hong Kong SAR": "China",
    "Hong Kong S.A.R.": "China",
    "Puerto Rico": "United States",
    "Greenland": "Denmark",
    "Faroe Islands": "Denmark",
    "French Guiana": "France",
    "Réunion": "France",
    "Reunion": "France",
}

# Aggregate/non-country values are dropped. "Congo" is intentionally not guessed
# between the Republic of the Congo and the Democratic Republic of the Congo.
DROP_TOKENS = {
    "Worldwide (multiple countries)": "AGGREGATE_NOT_COUNTRY",
    "Multiple countries (50 countries; 31 Indo-European countries used in analysis)": "AGGREGATE_NOT_COUNTRY",
    "Worldwide": "AGGREGATE_NOT_COUNTRY",
    "World": "AGGREGATE_NOT_COUNTRY",
    "Multiple countries": "AGGREGATE_NOT_COUNTRY",
    "Congo": "AMBIGUOUS_COUNTRY_NAME",
}


def parse_listish(value: str) -> List[str]:
    if value is None:
        return []
    s = str(value).strip()
    if not s or s in {"[]", "None", "nan", "Not stated"}:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return [s]


def parse_bool(value: str) -> Optional[bool]:
    s = str(value).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def normalize_token(raw: str) -> Tuple[Optional[str], str, str]:
    token = re.sub(r"\s+", " ", str(raw).strip())

    if token in DROP_TOKENS:
        return None, "DROP", DROP_TOKENS[token]

    normalized = COUNTRY_ALIASES.get(token, token)

    if normalized not in CANONICAL_SET:
        return None, "UNMAPPED", "NOT_IN_197_ENTITY_ONTOLOGY"

    if normalized != token:
        return normalized, "MAP", f"{token} -> {normalized}"

    return normalized, "UNCHANGED", ""


def normalize_country_list(values: Iterable[str]):
    normalized: List[str] = []
    changes: List[str] = []
    dropped: List[str] = []
    unmapped: List[str] = []

    for raw in values:
        value, action, note = normalize_token(raw)
        if action == "MAP":
            changes.append(note)
        elif action == "DROP":
            dropped.append(f"{raw} [{note}]")
        elif action == "UNMAPPED":
            unmapped.append(f"{raw} [{note}]")

        if value and value not in normalized:
            normalized.append(value)

    return normalized, changes, dropped, unmapped


def language_list_from_cleaned_string(value: str) -> List[str]:
    s = str(value or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def load_csv(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, rows: List[Dict[str, str]], fields: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sampling_frame", required=True)
    p.add_argument("--v2_merged", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    sampling_path = Path(args.sampling_frame)
    v2_path = Path(args.v2_merged)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampling_fields, sampling = load_csv(sampling_path)
    v2_fields, v2_rows = load_csv(v2_path)

    # ---------------------------
    # Integrity checks
    # ---------------------------
    if len(sampling) != 18035:
        raise ValueError(f"Expected 18,035 sampling-frame rows, found {len(sampling)}")

    source_ids = [str(r["source_row_index"]) for r in sampling]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate source_row_index values in sampling frame")

    v2_ids = [str(r["record_id"]) for r in v2_rows]
    if len(v2_ids) != len(set(v2_ids)):
        raise ValueError("Duplicate record_id values in v2 merged CSV")

    unexpected = sorted(set(v2_ids) - set(source_ids), key=lambda x: int(x))
    if unexpected:
        raise ValueError(f"Unexpected v2 record_ids: {unexpected[:20]}")

    v2_by_id = {str(r["record_id"]): r for r in v2_rows}
    missing_v2 = sorted(set(source_ids) - set(v2_ids), key=lambda x: int(x))

    # Language integrity: reuse cleaned fields exactly.
    language_count_mismatches = []
    for r in sampling:
        langs = language_list_from_cleaned_string(r.get("canonical_language_string_cleaned", ""))
        raw_count = str(r.get("canonical_language_count_cleaned", "")).strip()
        expected_count = int(float(raw_count)) if raw_count else 0
        if len(langs) != expected_count:
            language_count_mismatches.append(
                (r["source_row_index"], expected_count, len(langs), langs)
            )

    if language_count_mismatches:
        raise ValueError(
            f"Language count mismatch in {len(language_count_mismatches)} rows; "
            f"first={language_count_mismatches[0]}"
        )

    # ---------------------------
    # Build normalized master
    # ---------------------------
    normalized_rows: List[Dict[str, str]] = []
    qc_counter = Counter()
    qc_records = defaultdict(set)

    for base in sampling:
        rid = str(base["source_row_index"])
        out = dict(base)

        # Existing language normalization is authoritative.
        canonical_langs = language_list_from_cleaned_string(
            base.get("canonical_language_string_cleaned", "")
        )
        out["canonical_language_list_cleaned"] = json_list(canonical_langs)
        out["has_canonical_language_cleaned"] = str(bool(canonical_langs))
        out["language_normalization_source_v2"] = (
            "atlasnlp_audit_sampling_frame.canonical_language_string_cleaned"
        )

        v2 = v2_by_id.get(rid)

        if v2 is None:
            out.update({
                "v2_processing_status": "FAILURE",
                "v2_source_shard": "",
                "v2_paper_id": "",
                "v2_dataset_match_status": "",
                "v2_dataset_relationship": "",
                "v2_primary_eligible": "",
                "v2_content_country_status_raw": "PROCESSING_FAILURE",
                "v2_explicit_content_countries_raw": "[]",
                "v2_inferred_content_countries_raw": "[]",
                "content_country_status_normalized": "PROCESSING_FAILURE",
                "explicit_content_countries_normalized": "[]",
                "inferred_content_countries_normalized": "[]",
                "explicit_plus_inferred_countries_normalized": "[]",
                "n_explicit_countries_normalized": "0",
                "n_inferred_countries_normalized": "0",
                "n_any_countries_normalized": "0",
                "has_explicit_country_normalized": "False",
                "has_any_country_normalized": "False",
                "multi_country_explicit_normalized": "False",
                "multi_country_any_normalized": "False",
                "country_normalization_changed": "False",
                "country_normalization_changes": "[]",
                "country_normalization_dropped_tokens": "[]",
                "country_normalization_unmapped_tokens": "[]",
            })
            normalized_rows.append(out)
            continue

        explicit_raw = parse_listish(v2.get("explicit_content_countries", ""))
        inferred_raw = parse_listish(v2.get("inferred_content_countries", ""))

        explicit, ex_changes, ex_dropped, ex_unmapped = normalize_country_list(explicit_raw)
        inferred, in_changes, in_dropped, in_unmapped = normalize_country_list(inferred_raw)

        # Explicit evidence outranks inferred evidence after normalization.
        explicit_set = set(explicit)
        inferred_before = list(inferred)
        inferred = [c for c in inferred if c not in explicit_set]
        inferred_dedup_removed = [c for c in inferred_before if c in explicit_set]

        all_countries = explicit + [c for c in inferred if c not in explicit_set]

        for raw in explicit_raw + inferred_raw:
            normalized, action, note = normalize_token(raw)
            if action != "UNCHANGED":
                qc_counter[(raw, normalized or "", action, note)] += 1
                qc_records[(raw, normalized or "", action, note)].add(rid)

        if inferred_dedup_removed:
            for c in inferred_dedup_removed:
                key = (c, c, "REMOVE_INFERRED_DUPLICATE", "EXPLICIT_EVIDENCE_OUTRANKS_INFERRED")
                qc_counter[key] += 1
                qc_records[key].add(rid)

        primary_eligible = parse_bool(v2.get("primary_eligible", ""))

        if primary_eligible is True:
            if explicit:
                normalized_status = "EXPLICIT"
            elif inferred:
                normalized_status = "INFERRED_ONLY"
            else:
                normalized_status = "UNATTRIBUTED"
        else:
            normalized_status = v2.get("content_country_status", "") or "EXCLUDED_AMBIGUOUS"

        changes = ex_changes + in_changes
        dropped = ex_dropped + in_dropped
        unmapped = ex_unmapped + in_unmapped
        changed = bool(changes or dropped or unmapped or inferred_dedup_removed)

        out.update({
            "v2_processing_status": "SUCCESS",
            "v2_source_shard": v2.get("source_shard", ""),
            "v2_paper_id": v2.get("paper_id", ""),
            "v2_dataset_match_status": v2.get("dataset_match_status", ""),
            "v2_dataset_relationship": v2.get("dataset_relationship", ""),
            "v2_primary_eligible": str(primary_eligible) if primary_eligible is not None else "",
            "v2_content_country_status_raw": v2.get("content_country_status", ""),
            "v2_explicit_content_countries_raw": json_list(explicit_raw),
            "v2_inferred_content_countries_raw": json_list(inferred_raw),
            "content_country_status_normalized": normalized_status,
            "explicit_content_countries_normalized": json_list(explicit),
            "inferred_content_countries_normalized": json_list(inferred),
            "explicit_plus_inferred_countries_normalized": json_list(all_countries),
            "n_explicit_countries_normalized": str(len(explicit)),
            "n_inferred_countries_normalized": str(len(inferred)),
            "n_any_countries_normalized": str(len(all_countries)),
            "has_explicit_country_normalized": str(bool(explicit)),
            "has_any_country_normalized": str(bool(all_countries)),
            "multi_country_explicit_normalized": str(len(explicit) > 1),
            "multi_country_any_normalized": str(len(all_countries) > 1),
            "country_normalization_changed": str(changed),
            "country_normalization_changes": json_list(
                changes + [f"inferred duplicate removed: {c}" for c in inferred_dedup_removed]
            ),
            "country_normalization_dropped_tokens": json_list(dropped),
            "country_normalization_unmapped_tokens": json_list(unmapped),
        })

        normalized_rows.append(out)

    # All normalized tokens must be ontology-valid.
    observed_normalized = set()
    all_unmapped = []
    for r in normalized_rows:
        observed_normalized.update(parse_listish(r["explicit_content_countries_normalized"]))
        observed_normalized.update(parse_listish(r["inferred_content_countries_normalized"]))
        all_unmapped.extend(parse_listish(r["country_normalization_unmapped_tokens"]))

    invalid_after = sorted(observed_normalized - CANONICAL_SET)
    if invalid_after:
        raise ValueError(f"Invalid normalized country tokens remain: {invalid_after}")
    if all_unmapped:
        raise ValueError(f"Unmapped country tokens remain: {sorted(set(all_unmapped))}")

    # ---------------------------
    # Analysis subsets
    # ---------------------------
    eligible = [
        r for r in normalized_rows
        if r["v2_processing_status"] == "SUCCESS"
        and r["v2_primary_eligible"] == "True"
    ]
    explicit = [
        r for r in eligible
        if r["content_country_status_normalized"] == "EXPLICIT"
    ]
    inferred_only = [
        r for r in eligible
        if r["content_country_status_normalized"] == "INFERRED_ONLY"
    ]
    explicit_plus_inferred = [
        r for r in eligible
        if r["content_country_status_normalized"] in {"EXPLICIT", "INFERRED_ONLY"}
    ]

    new_fields = [
        "canonical_language_list_cleaned",
        "has_canonical_language_cleaned",
        "language_normalization_source_v2",
        "v2_processing_status",
        "v2_source_shard",
        "v2_paper_id",
        "v2_dataset_match_status",
        "v2_dataset_relationship",
        "v2_primary_eligible",
        "v2_content_country_status_raw",
        "v2_explicit_content_countries_raw",
        "v2_inferred_content_countries_raw",
        "content_country_status_normalized",
        "explicit_content_countries_normalized",
        "inferred_content_countries_normalized",
        "explicit_plus_inferred_countries_normalized",
        "n_explicit_countries_normalized",
        "n_inferred_countries_normalized",
        "n_any_countries_normalized",
        "has_explicit_country_normalized",
        "has_any_country_normalized",
        "multi_country_explicit_normalized",
        "multi_country_any_normalized",
        "country_normalization_changed",
        "country_normalization_changes",
        "country_normalization_dropped_tokens",
        "country_normalization_unmapped_tokens",
    ]
    master_fields = sampling_fields + [f for f in new_fields if f not in sampling_fields]

    write_csv(out_dir / "atlasnlp_core_v2_all_normalized.csv", normalized_rows, master_fields)
    write_csv(out_dir / "atlasnlp_core_v2_eligible.csv", eligible, master_fields)
    write_csv(out_dir / "atlasnlp_core_v2_explicit.csv", explicit, master_fields)
    write_csv(out_dir / "atlasnlp_core_v2_inferred_only.csv", inferred_only, master_fields)
    write_csv(
        out_dir / "atlasnlp_core_v2_explicit_plus_inferred.csv",
        explicit_plus_inferred,
        master_fields,
    )

    # ---------------------------
    # Expanded association files
    # ---------------------------
    expanded_base_fields = [
        "source_row_index",
        "audit_record_id",
        "paper_url",
        "Dataset name",
        "normalized_task_category",
        "canonical_language_string_cleaned",
        "canonical_language_count_cleaned",
        "canonical_language_list_cleaned",
        "producer_country_list",
        "v2_dataset_relationship",
        "content_country_status_normalized",
    ]

    explicit_expanded = []
    sensitivity_expanded = []

    for r in eligible:
        ex = parse_listish(r["explicit_content_countries_normalized"])
        inf = parse_listish(r["inferred_content_countries_normalized"])

        for country in ex:
            row = {k: r.get(k, "") for k in expanded_base_fields}
            row["content_country"] = country
            row["country_evidence_tier"] = "EXPLICIT"
            explicit_expanded.append(row)
            sensitivity_expanded.append(dict(row))

        for country in inf:
            row = {k: r.get(k, "") for k in expanded_base_fields}
            row["content_country"] = country
            row["country_evidence_tier"] = "INFERRED"
            sensitivity_expanded.append(row)

    expanded_fields = expanded_base_fields + ["content_country", "country_evidence_tier"]
    write_csv(
        out_dir / "atlasnlp_core_v2_explicit_country_expanded.csv",
        explicit_expanded,
        expanded_fields,
    )
    write_csv(
        out_dir / "atlasnlp_core_v2_explicit_plus_inferred_country_expanded.csv",
        sensitivity_expanded,
        expanded_fields,
    )

    # ---------------------------
    # Country-normalization QC table
    # ---------------------------
    qc_rows = []
    for key, count in sorted(qc_counter.items(), key=lambda kv: (-kv[1], kv[0][0])):
        raw, normalized, action, note = key
        ids = sorted(qc_records[key], key=lambda x: int(x))
        qc_rows.append({
            "raw_country_token": raw,
            "normalized_country_token": normalized,
            "action": action,
            "reason": note,
            "occurrences": str(count),
            "affected_record_count": str(len(ids)),
            "affected_record_ids": ";".join(ids),
        })

    qc_fields = [
        "raw_country_token",
        "normalized_country_token",
        "action",
        "reason",
        "occurrences",
        "affected_record_count",
        "affected_record_ids",
    ]
    write_csv(out_dir / "country_normalization_qc.csv", qc_rows, qc_fields)

    # ---------------------------
    # Report
    # ---------------------------
    status_counts = Counter(r["content_country_status_normalized"] for r in normalized_rows)
    relationship_counts = Counter(
        r["v2_dataset_relationship"]
        for r in normalized_rows
        if r["v2_processing_status"] == "SUCCESS"
    )

    language_present = sum(
        1 for r in normalized_rows if r["has_canonical_language_cleaned"] == "True"
    )

    report = {
        "input_sampling_frame_rows": len(sampling),
        "input_v2_success_rows": len(v2_rows),
        "final_master_rows": len(normalized_rows),
        "missing_v2_record_ids_marked_processing_failure": missing_v2,
        "country_ontology_size": len(CANONICAL_197),
        "country_ontology_definition": "195-state AtlasNLP ontology + Taiwan + Kosovo",
        "normalized_country_unique_count_observed": len(observed_normalized),
        "unmapped_country_tokens_remaining": 0,
        "language_normalization": {
            "source": "existing canonical_language_string_cleaned / canonical_language_count_cleaned",
            "records_with_canonical_language": language_present,
            "records_without_canonical_language": len(normalized_rows) - language_present,
            "language_count_mismatches": 0,
        },
        "normalized_status_counts": dict(status_counts),
        "primary_eligible_records": len(eligible),
        "explicit_records": len(explicit),
        "inferred_only_records": len(inferred_only),
        "explicit_plus_inferred_records": len(explicit_plus_inferred),
        "unattributed_eligible_records": sum(
            1 for r in eligible if r["content_country_status_normalized"] == "UNATTRIBUTED"
        ),
        "explicit_country_associations": len(explicit_expanded),
        "explicit_plus_inferred_country_associations": len(sensitivity_expanded),
        "dataset_relationship_counts_successful_records": dict(relationship_counts),
        "country_normalization_qc_rows": len(qc_rows),
        "notes": [
            "One v2 EXPLICIT row containing only an aggregate multi-country label becomes UNATTRIBUTED after normalization.",
            "The ambiguous token 'Congo' is dropped rather than guessed; the affected record retains other valid inferred countries.",
            "Explicit evidence outranks inferred evidence when alias normalization collapses both tiers to the same country.",
            "The four v2 processing failures remain in the 18,035-row master and are never relabeled UNATTRIBUTED.",
        ],
    }

    with open(out_dir / "normalization_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    report_lines = [
        "AtlasNLP v2 final normalization report",
        "======================================",
        f"Original rows: {len(sampling):,}",
        f"v2 successful rows: {len(v2_rows):,}",
        f"Processing failures retained in master: {len(missing_v2):,}",
        "",
        "Normalized analytical funnel:",
        f"  Primary eligible: {len(eligible):,}",
        f"  Explicit: {len(explicit):,}",
        f"  Inferred only: {len(inferred_only):,}",
        f"  Explicit + inferred: {len(explicit_plus_inferred):,}",
        f"  Eligible unattributed: {report['unattributed_eligible_records']:,}",
        "",
        "Country associations:",
        f"  Explicit associations: {len(explicit_expanded):,}",
        f"  Explicit + inferred associations: {len(sensitivity_expanded):,}",
        "",
        "Language normalization:",
        f"  Existing cleaned language fields reused: yes",
        f"  Rows with canonical language: {language_present:,}",
        f"  Rows without canonical language: {len(normalized_rows)-language_present:,}",
        f"  Count mismatches: 0",
        "",
        "Country normalization:",
        f"  Canonical ontology size: {len(CANONICAL_197)}",
        f"  Unique normalized countries observed: {len(observed_normalized)}",
        f"  Unmapped tokens remaining: 0",
        f"  QC transformation/drop rows: {len(qc_rows)}",
        "",
        f"Missing v2 record IDs: {', '.join(missing_v2)}",
    ]
    (out_dir / "normalization_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
