# =============================================================================
# link_based_country_inference.py — Infer producer country from dataset/paper URLs
#
# Paper section: §3.3 "Automated Dataset Expansion" (producer-country attribution)
# Role in pipeline: Step 8 — supplements author-affiliation producer country data
#   by extracting country signals from country-code TLDs in dataset_link and
#   paper_link URLs (e.g., ".jp" → Japan, ".kr" → South Korea).
#   This is a separate signal layer: used for producer-side inference only,
#   NOT for expanding content-country coverage.
#
# Run from compiled_ACL/:
#   python link_based_country_inference.py
#
# Input:  updated_master_with_normalized_languages.csv
# Output: updated_master_with_link_mapping.csv
# =============================================================================
import pandas as pd
import re
from urllib.parse import urlparse

# run: python link_based_country_inference.py

# =========================
# LOAD
# =========================

df = pd.read_csv("updated_master_with_normalized_languages.csv")

# =========================
# CONFIG
# =========================

DATASET_LINK_COL = "Dataset Link (URL)"
PAPER_LINK_COL = "Paper Link (URL)"

# Country TLD mapping (expandable)
TLD_TO_COUNTRY = {
    "jp": "Japan",
    "kr": "South Korea",
    "cn": "China",
    "in": "India",
    "br": "Brazil",
    "ru": "Russia",
    "tr": "Turkey",
    "pl": "Poland",
    "cz": "Czechia",
    "se": "Sweden",
    "fi": "Finland",
    "ro": "Romania",
    "bg": "Bulgaria",
    "vn": "Vietnam",
    "gr": "Greece",
    "dk": "Denmark",
    "ua": "Ukraine",
    "ng": "Nigeria",
    "za": "South Africa",
    "eg": "Egypt",
    "mx": "Mexico",
    "ar": "Argentina",
    "cl": "Chile",
    "co": "Colombia",
    "ph": "Philippines",
    "th": "Thailand",
    "my": "Malaysia",
    "id": "Indonesia",
    "pk": "Pakistan",
    "bd": "Bangladesh",
    "lk": "Sri Lanka",
    "ir": "Iran",
    "il": "Israel",
    "es": "Spain",
    "it": "Italy",
    "fr": "France",
    "de": "Germany",
    "nl": "Netherlands",
    "be": "Belgium",
    "ch": "Switzerland",
    "at": "Austria",
    "no": "Norway",
    "pt": "Portugal",
    "ie": "Ireland",
    "au": "Australia",
    "nz": "New Zealand",
    "ca": "Canada",
    "uk": "United Kingdom",
}

# domains we ignore
GLOBAL_DOMAINS = {
    "github.com",
    "huggingface.co",
    "drive.google.com",
    "dropbox.com",
    "zenodo.org",
    "kaggle.com",
    "figshare.com",
}

# =========================
# HELPERS
# =========================

def is_missing(x):
    if pd.isna(x):
        return True
    x = str(x).strip().lower()
    return x in ["", "not stated", "nan", "none"]

def clean_url(url):
    if is_missing(url):
        return ""
    return str(url).strip()

def extract_domain(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        domain = domain.replace("www.", "")
        return domain
    except:
        return ""

def extract_tld(domain):
    parts = domain.split(".")
    if len(parts) < 2:
        return ""
    return parts[-1]

def split_multi(val):
    if is_missing(val):
        return []
    return [v.strip() for v in str(val).split(";") if v.strip()]

def join_multi(values):
    seen = []
    for v in values:
        if v and v not in seen:
            seen.append(v)
    return "; ".join(seen)

# =========================
# APPLY LINK INFERENCE
# =========================

new_countries = []
new_methods = []
link_signal_column = []

auto_added = 0

for _, row in df.iterrows():

    existing_countries = split_multi(row.get("all_countries_covered", ""))
    existing_method = split_multi(row.get("country_attribution_method", ""))

    dataset_url = clean_url(row.get(DATASET_LINK_COL, ""))
    paper_url = clean_url(row.get(PAPER_LINK_COL, ""))

    url = dataset_url if dataset_url else paper_url

    domain = extract_domain(url)

    inferred = []
    method = None

    if domain and domain not in GLOBAL_DOMAINS:

        tld = extract_tld(domain)

        # Case 1: country TLD
        if tld in TLD_TO_COUNTRY:
            inferred_country = TLD_TO_COUNTRY[tld]
            inferred.append(inferred_country)
            method = "inferred_from_dataset_domain_tld"
            auto_added += 1

    # merge
    final_countries = existing_countries + inferred
    final_methods = existing_method.copy()

    if method and method not in final_methods:
        final_methods.append(method)

    new_countries.append(join_multi(final_countries))
    new_methods.append(join_multi(final_methods))
    link_signal_column.append(join_multi(inferred))

# =========================
# SAVE
# =========================

df["all_countries_covered"] = new_countries
df["country_attribution_method"] = new_methods
df["Candidate Country From Link"] = link_signal_column

df.to_csv("updated_master_with_link_mapping.csv", index=False)

print("Saved to updated_master_with_link_mapping.csv")
print("New rows inferred from links:", auto_added)

# =========================
# COVERAGE
# =========================

def has_country(val):
    return not is_missing(val)

coverage = df["all_countries_covered"].apply(has_country).sum()
print(f"Coverage after link mapping: {coverage} / {len(df)} ({coverage/len(df):.2%})")