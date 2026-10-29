# WorldBench pipeline tracker

## Stage 1: Deterministic language cleanup
Script: deterministic_language_cleanup.py
Input: raw_language_frequency.csv
Output dir: language_cleanup_outputs/
Key output: normalized_token_frequency.csv
Status: completed

## Stage 2: Language -> country mapping
Script: language_country_mapping.py
Input: language_cleanup_outputs/normalized_token_frequency.csv
Output dir: language_country_mapping_outputs/
Key output: language_country_mapping_table.csv
Status: rerun after Portuguese moved to GLOBAL_OR_BROAD_LANGUAGES


## Stage 3: Apply language mapping to master
Script: language_mapping_update_master.py
Input master: country_second_pass/country_second_pass_merged/updated_master.csv
Input mapping: language_country_mapping_outputs/language_country_mapping_table.csv
Output: updated_master_with_language_mapping_v2.csv
Status: rerun after Portuguese policy change


## Stage 3.5: Write normalized languages back to master
Script: merge_normalized_language.py
Input: updated_master.csv
Output: updated_master_with_normalized_languages.csv
Status: completed; needs minor rerun after fixing Java->Javanese and title-casing issues

## Stage 4: Link inference
Script: link_based_country_inference.py
Input: updated_master_with_normalized_languages.csv
Output: updated_master_with_link_mapping.csv
Status: separate producer-signal layer, not for origin expansion


## Stage 5: 
Script: compiled_ACL/author_affiliation/compute_worldbench_country_coverage.py
Input: compiled_ACL/updated_master_with_normalized_languages.csv(instead of master_with_langmap.csv)
Output: coverage_summary.json, coverage_summary.txt
Status: rerun for stats

## Stage 6: 
Script: task_normalization.py
Input: updated_master_with_normalized_languages.csv


## Stage : 
Script: final_merge.py
Input: updated_master_with_normalized_languages.csv