# =============================================================================
# unique_languages.py — Build raw language frequency tables from master dataset
#
# Paper section: §3.1 "Annotation Framework and Schema Design" (language metadata)
# Role in pipeline: Step 7a — prerequisite for deterministic_language_cleanup.py.
#   Counts how often each language token appears across all extracted records,
#   producing frequency tables used to guide the cleanup and normalization policy.
#
# Run from compiled_ACL/:
#   python unique_languages.py
#
# Input:  country_second_pass/country_second_pass_merged/updated_master.csv
# Output: raw_language_frequency.csv       (freq table of raw language strings)
#         normalized_language_frequency.csv (freq table of normalized language strings)
#         first_200_raw_langs.csv           (least common raw tokens, for review)
#         last_150_raw_langs.csv            (most common raw tokens)
#         all_languages.txt                 (full set of raw language tokens)
# =============================================================================
import pandas as pd
import numpy as np

df = pd.read_csv("/home/jnwatu/Worldbench/compiled_ACL/country_second_pass/country_second_pass_merged/updated_master.csv")

language_list_a = list(df['Languages covered (separate using semicolon)'])
language_list_b = list(df['Normalized Languages'])


list_of_languages = []
for a in language_list_a:
    langs = a.split(',')
    list_of_languages.extend(langs)

set_of_languages = set(list_of_languages)

# file_path = 'all_languages.txt'
# with open(file_path, 'w') as file:
#     for item in set_of_languages:
#         file.write(str(item) + "\\n")

lang_freq_dict = dict()
for lang in list_of_languages:
    lang_freq_dict[lang] = lang_freq_dict.get(lang, 0) + 1

lang_df = pd.DataFrame()
lang_df['Language'] = list(lang_freq_dict.keys())
lang_df['Frequency'] = list(lang_freq_dict.values())


#--------------------------------------

list_of_languagesb = []
for b in language_list_b:
    langs = b.split(',')
    list_of_languagesb.extend(langs)

set_of_languagesb = set(list_of_languagesb)

# file_path = 'all_languages.txt'
# with open(file_path, 'w') as file:
#     for item in set_of_languages:
#         file.write(str(item) + "\\n")

lang_freq_dictb = dict()
for lang in list_of_languagesb:
    lang_freq_dictb[lang] = lang_freq_dictb.get(lang, 0) + 1

lang_dfb = pd.DataFrame()
lang_dfb['Language'] = list(lang_freq_dictb.keys())
lang_dfb['Frequency'] = list(lang_freq_dictb.values())

# ---------------------------------------

langdf = lang_df.sort_values(by='Frequency')

langdf.to_csv('raw_language_frequency.csv')
lang_dfb.to_csv('normalized_language_frequency.csv')

print('Number of rows: ' + str(len(df)))
print('Number of raw_unique langs: ' + str(len(set_of_languages)))
print('Number of normalized unique langs: ' + str(len(set_of_languagesb)))
print('Number in normalized langs: ' + str(len(language_list_b)))


langdf[:200].to_csv('first_200_raw_langs.csv')

lang__df = lang_df.sort_values(by='Frequency', ascending=False)

lang__df[:150].to_csv('last_150_raw_langs.csv')