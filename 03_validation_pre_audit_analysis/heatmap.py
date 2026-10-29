Loaded 197 valid countries from valid_countries.csv
Rows with valid country but invalid task: 4
                                    Dataset name Task Category normalized_task_category
12192                                 Not stated    Not stated                      NaN
12196                                 Not stated    Not stated                      NaN
12319                                 Not stated    Not stated                      NaN
14986  Bilingual Lexicosemantic Network of Bread    Not stated                      NaN
Missing countries: 6
['Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'Samoa', 'Tuvalu']
Countries with ≤5 datasets: 34
   total_rows  has_any_country  has_valid_country  has_valid_task  final_rows_used
0       18035            11925              11764           18017            11760
drop_reason
kept               11760
missing_country     6110
invalid_country      161
invalid_task           4
Name: count, dtype: int64
Missing tasks: {'Raw Corpus', 'AI/LLM Agents'}
WorldBench country x task matrix stats
=====================================

Input rows: 18035

Rows with valid content-side country: 11764
Rows missing valid content-side country: 6271

Rows with unmapped task labels: 0
Rows contributing to matrix: 11760

Expanded dataset-country-task rows: 21360
Unique countries in matrix: 191
Unique tasks in matrix: 30

Top 10 countries by dataset count:
       country  total_datasets
 United States            2500
         China            2091
       Germany            1361
         India            1304
        France            1020
United Kingdom             831
         Japan             684
         Spain             665
         Italy             593
        Canada             481

Top 10 tasks by dataset count:
                                                        Task Category  total_datasets
                                                     NLP Applications            2723
                   Information Extraction, Retrieval, and Text Mining            1102
                                     Dialogue and Interactive Systems            1093
                                          Natural Language Generation             879
                                                  Machine Translation             807
Computational Social Science, Cultural Analytics, NLP for social good             803
                                                   Question Answering             555
  Multimodality and Language Grounding to Vision, Robotics and Beyond             492
                                           Ethics, Bias, and Fairness             487
          Sentiment Analysis, Stylistic Analysis, and Argument Mining             401




# =============================================================================
# heatmap.py — Generate the country × task coverage heatmap (Figure 2 in paper)
#
# Paper section: §5.1 "Dataset Coverage is Highly Uneven Across Countries and Tasks"
# Produces: Figure 2 — heatmap of dataset coverage across all countries and tasks
#   (log scale), showing 59.7% of country-task pairs have no datasets.
#   Also produces top-30 and bottom-N country subsets of the heatmap.
#
# Note: Lines 1-60 above are captured stdout from task_normalization.py (run first).
#   This code uses 'pivot' from task_normalization.py — run in the same session
#   or load country_task_matrix.csv and pivot before running this block.
#
# Output PDFs/PNGs: top_30_log_scale.pdf, all_log_scale.pdf, least_30_log_scale.pdf, etc.
# =============================================================================
# ======================================
# Heatmap (log scale)
# ========================================

import numpy as np

pivot_log = np.log1p(pivot)

plt.figure(figsize=(16, 14))
plt.imshow(pivot_log, aspect='auto')
plt.colorbar(label="log(1 + dataset count)")

plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
plt.yticks(range(len(pivot.index)), pivot.index)

plt.title("WorldBench Country × Task Matrix (Log Scale)")
plt.tight_layout()
plt.show()

#==================================================
# Top 30 countries by total datasets
top_countries = pivot.sum(axis=1).sort_values(ascending=False).head(30).index

pivot_top = pivot.loc[top_countries]

pivot_top_log = np.log1p(pivot_top)

plt.figure(figsize=(14, 10))
plt.imshow(pivot_top_log, aspect='auto')
plt.colorbar(label="log(1 + dataset count)")

plt.xticks(range(len(pivot_top.columns)), pivot_top.columns, rotation=90)
plt.yticks(range(len(pivot_top.index)), pivot_top.index)

plt.title("Top 30 Countries × Task Matrix (Log Scale)")
plt.tight_layout()
plt.show()

