# =============================================================================
# validation_triage.py — Triage and classify raw LLM extraction outputs
#
# Paper section: §3.4 "Validation of Automated Extraction"
# Role in pipeline: Step 5 — after shard results are merged into acl_raw_results.csv,
#   this script checks each record for anomalies, contradictions, and missing fields,
#   then routes records into:
#     - machine_followup (auto-fixable issues)
#     - human_review (ambiguous cases needing manual judgment)
#     - contradictions (logically inconsistent field values)
#
# Run from compiled_ACL/:
#   python validation_triage.py
#
# Input:  validation_triage_autofixed.csv  (output of auto_fix_contradictions.py)
# Output: contra_fixed_validation_triage.csv, contra_fixed_machine_followup.csv,
#         contra_fixedhuman_review.csv, contra_fixed_contradictions.csv
# =============================================================================
import re
import json
import pandas as pd
from pathlib import Path

#INPUT_CSV = "acl_raw_results.csv"
INPUT_CSV = "validation_triage_autofixed.csv"

OUTPUT_TRIAGE = "contra_fixed_validation_triage.csv"
OUTPUT_MACHINE = "contra_fixed_machine_followup.csv"
OUTPUT_HUMAN = "contra_fixedhuman_review.csv"
OUTPUT_CONTRADICTIONS = "contra_fixed_contradictions.csv"
OUTPUT_ANOMALY = "contra_fixed_anomaly_summary.txt"


URL_RE = re.compile(r"https?://\S+", re.I)

LANGUAGE_COUNTRY_MAP = {
    "Bulgarian": "Bulgaria",
    "Finnish": "Finland",
    "Hungarian": "Hungary",
    "Estonian": "Estonia",
    "Icelandic": "Iceland",
    "Latvian": "Latvia",
    "Lithuanian": "Lithuania",
    "Slovenian": "Slovenia",
    "Croatian": "Croatia",
    "Serbian": "Serbia",
    "Slovak": "Slovakia",
    "Maltese": "Malta",
}

KNOWN_LICENSE_HINTS = [
    "mit", "apache", "cc-by", "cc by", "gpl", "bsd", "creative commons", "odc"
]

MODALITY_HINTS_AUDIO = [
    "speech recognition", "automatic speech recognition", "asr",
    "spoken language understanding", "text-to-speech", "tts", "audio"
]

MODALITY_HINTS_IMAGE = [
    "image captioning", "visual question answering", "vqa",
    "vision-language", "image", "ocr"
]

MODALITY_HINTS_VIDEO = [
    "video", "sign language", "video captioning"
]


def split_multi_value_cell(value: str):
    if pd.isna(value):
        return []
    value = str(value).strip()
    if not value or value.lower() == "not stated":
        return []
    parts = re.split(r"\s*,\s*|\s*;\s*", value)
    return [p.strip() for p in parts if p.strip()]


def normalize_country_list(value: str):
    country_map = {
        "UK": "United Kingdom",
        "U.K.": "United Kingdom",
        "US": "United States",
        "U.S.": "United States",
        "USA": "United States",
    }
    items = split_multi_value_cell(value)
    out = []
    seen = set()
    for x in items:
        x = country_map.get(x, x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_language_list(value: str):
    lang_map = {
        "EN": "English",
        "eng": "English",
        "ASL": "American Sign Language",
        "BSL": "British Sign Language",
        "ISL": "Indian Sign Language",
        "GSL": "Greek Sign Language",
        "LSA": "Argentine Sign Language",
    }
    items = split_multi_value_cell(value)
    out = []
    seen = set()
    for x in items:
        x = lang_map.get(x, x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def infer_primary_modality(row):
    text = " ".join([
        str(row.get("Task Category", "")),
        str(row.get("Short description", "")),
        str(row.get("Dataset name", "")),
    ]).lower()

    multimodal = str(row.get("Is the dataset multimodal?", "")).strip()

    has_audio = any(h in text for h in MODALITY_HINTS_AUDIO)
    has_image = any(h in text for h in MODALITY_HINTS_IMAGE)
    has_video = any(h in text for h in MODALITY_HINTS_VIDEO)

    # conservative derived modality
    if has_video and ("sign language" in text or "video" in text):
        # sign-language datasets are often video-based, sometimes with text alignment
        return "Video" if multimodal != "Yes" else "Text + Video"
    if has_audio and has_image:
        return "Audio + Image"
    if has_audio and multimodal == "Yes":
        return "Text + Audio"
    if has_image and multimodal == "Yes":
        return "Text + Image"
    if has_video and multimodal == "Yes":
        return "Text + Video"
    if has_audio:
        return "Audio"
    if has_image:
        return "Image"
    if has_video:
        return "Video"
    return "Text"


def is_url_like(value: str) -> bool:
    if pd.isna(value):
        return False
    value = str(value).strip()
    return bool(URL_RE.search(value))


def looks_like_license(value: str) -> bool:
    if pd.isna(value):
        return False
    value = str(value).lower()
    return any(h in value for h in KNOWN_LICENSE_HINTS)


def assign_flags(row):
    flags = []

    countries = normalize_country_list(row.get("All Countries Covered (you can select multiple countries)", ""))
    langs = normalize_language_list(row.get("Languages covered", ""))

    country_attr = str(row.get("Country Attribution Method", "Not stated")).strip()
    multi_country = str(row.get("Multi Country?", "Not stated")).strip()
    lang_cov = str(row.get("Language coverage type", "Not stated")).strip()
    dataset_link = str(row.get("Dataset Link (URL)", "Not stated")).strip()
    license_val = str(row.get("License", "Not stated")).strip()
    origin_type = str(row.get("Origin Type", "Not stated")).strip()
    dataset_name = str(row.get("Dataset name", "")).strip()
    task_cat = str(row.get("Task Category", "")).strip()
    desc = str(row.get("Short description", "")).strip()
    multimodal = str(row.get("Is the dataset multimodal?", "Not stated")).strip()
    multiple_choice = str(row.get("Is dataset in multiple choice format?", "Not stated")).strip()

    # language-based country inference
    if not countries and langs:
        inferred = [LANGUAGE_COUNTRY_MAP.get(l) for l in langs if l in LANGUAGE_COUNTRY_MAP]

        if inferred:
            flags.append("COUNTRY_INFERABLE_FROM_LANGUAGE")

    # Missingness
    if dataset_link == "Not stated":
        flags.append("LINK_MISSING")
    if license_val == "Not stated":
        flags.append("LICENSE_MISSING")
    if origin_type == "Not stated":
        flags.append("ORIGIN_TYPE_MISSING")
    if country_attr == "Not stated":
        flags.append("COUNTRY_ATTRIBUTION_MISSING")
    if langs and not countries:
        flags.append("COUNTRY_MISSING_BUT_LANGUAGE_PRESENT")
    if not countries and not langs:
        flags.append("COUNTRIES_MISSING")
    if multiple_choice =="Not stated":
        flags.append("MULTIPLE_CHOICE_MISSING")

    # Contradictions
    if countries and country_attr == "Not stated":
        flags.append("COUNTRY_ATTRIBUTION_CONTRADICTION")

    if len(countries) > 1 and multi_country == "No":
        flags.append("MULTI_COUNTRY_CONTRADICTION")
    if len(countries) == 1 and multi_country == "Yes":
        flags.append("MULTI_COUNTRY_CONTRADICTION")

    if len(langs) > 1 and lang_cov == "Monolingual":
        flags.append("LANGUAGE_COUNT_CONTRADICTION")
    if len(langs) == 1 and lang_cov == "Multilingual":
        flags.append("LANGUAGE_COUNT_CONTRADICTION")

    # Bad column-value patterns
    if dataset_link != "Not stated" and not is_url_like(dataset_link):
        flags.append("DATASET_LINK_NOT_URL")

    if license_val != "Not stated" and not looks_like_license(license_val):
        flags.append("LICENSE_SUSPICIOUS")

    if URL_RE.search(desc):
        flags.append("DESCRIPTION_CONTAINS_URL")

    if dataset_name.count(",") >= 3:
        flags.append("MULTIPLE_DATASETS_IN_NAME")

    if task_cat.lower().startswith("http"):
        flags.append("TASK_CATEGORY_COLUMN_ERROR")
    if license_val.lower().startswith("http") and "license" not in license_val.lower():
        flags.append("LICENSE_MAY_BE_LINK")
    if dataset_name.lower().startswith("http"):
        flags.append("DATASET_NAME_COLUMN_ERROR")

    # Modality contradiction against derived modality
    primary_modality = infer_primary_modality(row)
    if primary_modality in {"Text + Image", "Text + Audio", "Text + Video"} and multimodal == "No":
        flags.append("MODALITY_CONTRADICTION")

    return flags, primary_modality, countries, langs


def assign_review_tier(flags):
    contradiction_markers = {
        "COUNTRY_ATTRIBUTION_CONTRADICTION",
        "MULTI_COUNTRY_CONTRADICTION",
        "LANGUAGE_COUNT_CONTRADICTION",
        "MODALITY_CONTRADICTION",
        "TASK_CATEGORY_COLUMN_ERROR",
        "DATASET_NAME_COLUMN_ERROR",
        "DATASET_LINK_NOT_URL",
    }

    human_markers = {
        "MULTIPLE_DATASETS_IN_NAME"
    }

    machine_markers = {
        "LINK_MISSING",
        "LICENSE_MISSING",
        "LICENSE_SUSPICIOUS",
        "ORIGIN_TYPE_MISSING",
        "COUNTRY_ATTRIBUTION_MISSING",
        "COUNTRIES_MISSING",
        "DESCRIPTION_CONTAINS_URL",
        "LICENSE_MAY_BE_LINK",
        "MULTIPLE_CHOICE_MISSING"
    }

    if any(f in contradiction_markers for f in flags):
        return "contradiction"
    if any(f in human_markers for f in flags):
        return "human_review"
    if any(f in machine_markers for f in flags):
        return "machine_followup"
    return "auto_validated"


def main():
    df = pd.read_csv(INPUT_CSV)

    triage_rows = []

    for _, row in df.iterrows():
        flags, primary_modality, countries, langs = assign_flags(row)
        tier = assign_review_tier(flags)
        candidate_country = "; ".join(LANGUAGE_COUNTRY_MAP.get(l) for l in langs if l in LANGUAGE_COUNTRY_MAP)
            

        triage_rows.append({
            **row.to_dict(),
            "Normalized Countries": "; ".join(countries) if countries else "Not stated",
            "Normalized Languages": "; ".join(langs) if langs else "Not stated",
            "Primary Modality": primary_modality,
            "Validation Flags": "; ".join(flags) if flags else "",
            "Candidate Country From Language": candidate_country if candidate_country else "",
            "Review Tier": tier,
            "Needs Machine Followup": "Yes" if tier == "machine_followup" else "No",
            "Needs Human Review": "Yes" if tier in {"human_review", "contradiction"} else "No",
        })

    triage_df = pd.DataFrame(triage_rows)
    triage_df.to_csv(OUTPUT_TRIAGE, index=False)

    triage_df[triage_df["Review Tier"] == "machine_followup"].to_csv(OUTPUT_MACHINE, index=False)
    triage_df[triage_df["Review Tier"] == "human_review"].to_csv(OUTPUT_HUMAN, index=False)
    triage_df[triage_df["Review Tier"] == "contradiction"].to_csv(OUTPUT_CONTRADICTIONS, index=False)

    # anomaly summary
    lines = []
    lines.append(f"Total rows: {len(triage_df)}")
    lines.append("")

    lines.append("Review tier counts:")
    lines.append(triage_df["Review Tier"].value_counts(dropna=False).to_string())
    lines.append("")

    lines.append("Top validation flags:")
    flag_counts = (
        triage_df["Validation Flags"]
        .fillna("")
        .str.split("; ")
        .explode()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
    )
    lines.append(flag_counts.head(50).to_string())
    lines.append("")

    # suspicious column distributions
    lines.append("Top Country Attribution Method values:")
    lines.append(triage_df["Country Attribution Method"].value_counts(dropna=False).head(20).to_string())
    lines.append("")

    lines.append("Top Origin Type values:")
    lines.append(triage_df["Origin Type"].value_counts(dropna=False).head(20).to_string())
    lines.append("")

    lines.append("Top License values:")
    lines.append(triage_df["License"].value_counts(dropna=False).head(30).to_string())
    lines.append("")

    lines.append("Top Primary Modality values:")
    lines.append(triage_df["Primary Modality"].value_counts(dropna=False).to_string())

    lines.append("Top Multiple Choice values:")
    lines.append(triage_df["Is dataset in multiple choice format?"].value_counts(dropna=False).head(20).to_string())
    lines.append("")

    Path(OUTPUT_ANOMALY).write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUTPUT_TRIAGE}")
    print(f"Wrote {OUTPUT_MACHINE}")
    print(f"Wrote {OUTPUT_HUMAN}")
    print(f"Wrote {OUTPUT_CONTRADICTIONS}")
    print(f"Wrote {OUTPUT_ANOMALY}")


if __name__ == "__main__":
    main()