# =============================================================================
# language_mapping_diagnostics.py — Inspect language→country mapping coverage
#
# Paper section: §3.3 "Automated Dataset Expansion" (language-informed attribution)
# Role in pipeline: Diagnostic utility — run after language_mapping_update_master.py
#   to understand how many previously country-less rows gained inferred country
#   attribution through language-based expansion.
#
# Run from compiled_ACL/:
#   python language_mapping_diagnostics.py
#
# Input:  language_mapping_rowlevel_diagnostics.csv (output of language_mapping_update_master.py)
# Output: printed summary to stdout (no file written)
# =============================================================================
import pandas as pd

diag = pd.read_csv("language_mapping_rowlevel_diagnostics.csv")

def nonempty(x):
    if pd.isna(x):
        return False
    return str(x).strip() != ""

# rows that had no country before
no_country_before = diag[~diag["existing_countries_before"].fillna("").astype(str).str.strip().astype(bool)]

# among those, rows that got inferred countries added
newly_covered_candidates = no_country_before[
    diag.loc[no_country_before.index, "inferred_countries_added"].fillna("").astype(str).str.strip().astype(bool)
]

print("Rows with no country before:", len(no_country_before))
print("Rows with inferred countries added:", len(newly_covered_candidates))

print("\nSample rows that should have become newly covered:")
print(
    newly_covered_candidates[
        [
            "paper_url",
            "normalized_languages_used_for_lookup",
            "auto_expand_tokens_matched",
            "inferred_countries_added",
            "final_all_countries_covered",
        ]
    ].head(20).to_string(index=False)
)



# rows that had auto-expand token matches at all
auto_expand_rows = diag[
    diag["auto_expand_tokens_matched"].fillna("").astype(str).str.strip().astype(bool)
]

# of those, how many already had country?
auto_expand_with_existing = auto_expand_rows[
    auto_expand_rows["existing_countries_before"].fillna("").astype(str).str.strip().astype(bool)
]

auto_expand_without_existing = auto_expand_rows[
    ~auto_expand_rows["existing_countries_before"].fillna("").astype(str).str.strip().astype(bool)
]

print("Rows with auto-expand token matches:", len(auto_expand_rows))
print("...of which already had country:", len(auto_expand_with_existing))
print("...of which had no country:", len(auto_expand_without_existing))