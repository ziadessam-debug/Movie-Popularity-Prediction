# Movie Popularity Prediction Using Big Data and Machine Learning
### Final Project Report

**Team:** 5 members (see §17) · **Tools:** PySpark 4.3, scikit-learn 1.9, Streamlit 1.67, Plotly

---

## 1. Introduction
Thousands of films are released every year, yet only a few attract large audiences. Studios, distributors
and streaming platforms want early signals of which titles are likely to become popular. This project builds
an end-to-end Big Data pipeline that processes ~45,000 movies with Apache Spark, analyses which characteristics
are associated with popularity, and trains machine learning models to estimate a movie's popularity score.

## 2. Problem Statement
*Given information that is available before a movie's release (budget, genres, runtime, release date, language,
franchise membership, production companies/countries), how accurately can we estimate its TMDB popularity
score, and which factors matter most?*

## 3. Project Objectives
1. Collect and document a real movie dataset.
2. Clean and transform it with PySpark in one reproducible notebook.
3. Perform EDA and identify data-driven patterns.
4. Train and compare regression models without data leakage; evaluate with MAE, RMSE and R².
5. Present findings, predictions and model results in an interactive dashboard.

## 4. Dataset Description
*The Movies Dataset* (R. Banik, Kaggle, CC0) — metadata from the TMDB API plus GroupLens MovieLens ratings.

| File | Rows | Columns | Role |
|------|------|---------|------|
| `movies_metadata.csv` | 45,466 | 24 | main dataset |
| `links.csv` | 45,843 | 3 | key MovieLens `movieId` ↔ TMDB `tmdbId` |
| `ratings_small.csv` | 100,004 | 4 | post-release engagement analysis |

Relevant columns: `budget`, `genres` (JSON-like list), `runtime`, `release_date`, `original_language`,
`belongs_to_collection`, `production_companies`, `production_countries`, `popularity` (target), and
post-release `revenue`, `vote_count`, `vote_average`.
**Target meaning:** TMDB popularity is a relative activity score (views, votes, favourites, watch-lists)
recorded when the data was collected in 2017 — a time-dependent snapshot rather than a fixed outcome.

## 5. Data Collection
The files were obtained from the Kaggle dataset via its public byte-identical Hugging Face mirror (no Kaggle
login required). Notebook 01 downloads them automatically if they are missing from `data/raw/`.

## 6. Data Preprocessing
Problems found during inspection (all measured in Spark): 3 malformed rows whose `id` holds a date (columns
shifted); 29 duplicated ids (30 extra rows); 36,573 budgets and 38,052 revenues equal to 0 (placeholders for
"unknown"); 452 non-released titles (rumoured, in production…); string-typed numerics whose min/max contain
file paths and titles.

| Step | Rows remaining |
|------|----------------|
| Raw file | 45,466 |
| Valid numeric id | 45,463 |
| De-duplicated by id | 45,433 |
| Released, non-adult, valid date and title | 44,899 |
| Valid popularity (> 0) | **44,861** |

Value fixes: budgets < \$1,000 → unknown (only 8,609 movies = 19% keep a budget); runtime ≤ 0 or > 400 → null,
then 1,756 values imputed with the median (95 min); revenue 0 → null; missing language → `xx`.

## 7. PySpark Implementation
All processing lives in `notebooks/01_data_processing_and_cleaning.ipynb`:

| Requirement | Spark implementation |
|-------------|----------------------|
| DataFrames & loading | `spark.read.csv` with `multiLine`, `escape`, explicit string schema |
| Inspection | `printSchema`, `show`, `count`, `describe`, null profile via `agg(sum(when(...)))` |
| Type conversion | `try_cast`, `try_to_timestamp` (invalid values become null instead of failing) |
| Filtering | malformed ids, released/non-adult scope, invalid targets, big-budget franchise query |
| Transformation | `regexp_extract_all` to parse genres/companies/countries, `array_contains` genre flags, `year`, `month`, `log1p` |
| Aggregation / grouping | `explode` + `groupBy().agg()` for genre stats; yearly trends; `percentile_approx`; `approxQuantile` imputation |
| Spark SQL | temp view `movies`; budget tiers with `CASE WHEN`; language and month statistics |
| Joins | `ratings_small` ⨝ `links` ⨝ `movies` (9,004 movies matched) |
| Window functions | `rank()` over `partitionBy(release_year)` → most popular movie per year |
| Validation | asserts on unique ids, no nulls in required columns, value ranges |
| Output | `movies_processed.parquet` (44,861 × 46), `genre_stats`, `yearly_stats`, `movie_ratings_summary` |

Example Spark SQL result — popularity by budget tier:

| Budget tier | Movies | Median popularity |
|-------------|--------|-------------------|
| unknown | 36,252 | 0.87 |
| < \$1M | 1,229 | 1.32 |
| \$1M–10M | 3,141 | 4.07 |
| \$10M–50M | 3,067 | 8.03 |
| \$50M–100M | 791 | 10.27 |
| ≥ \$100M | 381 | 14.89 |

*Why Spark:* the same declarative pipeline handles malformed multi-line CSV, nested JSON-like strings and
joins, and would scale unchanged to the full 26M-row ratings file or a cluster. Outputs are written as single
Parquet files with pyarrow (small result, simpler deployment, no Hadoop `winutils` needed on Windows).

## 8. Exploratory Data Analysis
All numbers below were computed in `02_eda.ipynb` (figures in `reports/figures/`).

| Analysis | Finding |
|----------|---------|
| Popularity distribution | Median 1.14, 90th pct 8.36, max 547.5; skewness 29.2 (0.79 after log) → log target |
| Genre | Adventure has the highest median popularity (2.98), Documentary the lowest (0.39); Drama is most common (20,043) |
| Release year | Peak in 2014 (1,960 movies); 53% of movies released since 2000 |
| Budget | Spearman budget↔popularity = 0.60 (known budgets); median 6.50 with known budget vs 0.87 unknown |
| Runtime | Median 95 min; Spearman 0.27; highest median popularity for 120–140 min films |
| Release month | July highest (1.46), January lowest (0.75, many placeholder 1 Jan dates) |
| Language | English = 71% of movies |
| Correlation | Strongest pre-release correlates: number of companies (0.47), log budget (0.47), budget known (0.45); post-release vote count 0.89 |
| Franchise | Collection movies median 3.71 vs 1.03 standalone |
| Engagement (join) | Spearman(MovieLens rating count, popularity) = 0.51 across 9,004 movies |

Suggested figures: `popularity_distribution.png`, `genre_popularity.png`, `budget_vs_popularity.png`,
`correlation_heatmap.png`.

## 9. Feature Engineering
| Feature | Type | Selected | Reason |
|---------|------|----------|--------|
| `log_budget`, `budget_known` | numeric / flag | yes | budget known pre-release; log handles skew; flag encodes missingness |
| `runtime` | numeric | yes | known pre-release |
| `release_year`, `release_month` | numeric | yes | timing; also captures snapshot recency |
| `genre_*` (20), `num_genres` | flags / count | yes | multi-label genres |
| `original_language` | categorical | yes | one-hot, rare languages grouped |
| `has_collection` | flag | yes | franchise membership is known before release |
| `num_companies`, `num_countries`, `is_us_production` | numeric / flag | yes | production scale and origin |
| `revenue`, `vote_count`, `vote_average`, MovieLens ratings | numeric | **no** | post-release → leakage |
| `overview`, `title`, `homepage` | text | no | out of scope for a simple, explainable model |

## 10. Machine Learning Methodology
- Target `log_popularity = log(1 + popularity)`; predictions are converted back with `expm1`.
- 80/20 random train/test split (35,888 / 8,973), `random_state = 42`.
- scikit-learn `Pipeline`: `StandardScaler` for numeric features, `OneHotEncoder(min_frequency=100,
  handle_unknown="infrequent_if_exist")` for language — fitted on the training set only and saved together
  with the model, so the dashboard applies identical transformations.
- Models: mean baseline, Linear Regression, Decision Tree, Random Forest, Histogram Gradient Boosting.
- Hyper-parameters tuned with 3-fold CV on the training set (Grid/Randomized search, RMSE). The best model is
  selected by CV RMSE, so the test set is used only once.

## 11. Model Evaluation
MAE = average error size; RMSE = error that penalises large misses; R² = share of variance explained
(0 = predicting the mean, 1 = perfect).

| Model | CV RMSE (log) | MAE (log) | RMSE (log) | R² (log) | MAE | RMSE | R² | Selected |
|-------|---------------|-----------|------------|----------|-----|------|----|----------|
| Baseline (mean) | 0.805 | 0.667 | 0.801 | 0.000 | 2.570 | 7.918 | −0.026 | |
| Linear Regression | 0.571 | 0.444 | 0.566 | 0.500 | 1.833 | 7.162 | 0.161 | |
| Decision Tree | 0.567 | 0.434 | 0.561 | 0.509 | 1.783 | 7.095 | 0.176 | |
| Random Forest | 0.537 | 0.409 | 0.529 | 0.563 | 1.686 | 7.038 | 0.189 | |
| **Gradient Boosting** | **0.533** | **0.403** | **0.523** | **0.573** | **1.676** | **6.992** | **0.200** | ✔ |

Best parameters: learning rate 0.05, 31 leaf nodes, min 50 samples per leaf. An MAE of 0.40 on the log
scale means predictions are typically within a factor of ≈1.5 of the true score. Permutation importance ranks
**log budget** first by a wide margin, followed by number of production companies, release year, runtime and
US production. Figures: `model_comparison.png`, `actual_vs_predicted.png`, `feature_importance.png`.

## 12. Dashboard Implementation
`dashboard/app.py` (Streamlit + Plotly, dark cinematic theme, cached data/model loading, no retraining):
1. **Overview** — KPIs (total movies, average/highest popularity, average budget, most common genre),
   release trend, popularity distribution, genre chart, computed highlights; year and genre filters.
2. **Movie Explorer** — title search, genre/language/year/budget filters, sorting, table, detail card.
3. **Popularity Prediction** — validated form → saved pipeline → predicted score, error range from test
   RMSE, dataset percentile gauge, position in distribution, comparable movies.
4. **Data Insights** — tabs for genres, budget, release trends, runtime, language and correlation with
   findings that update with the filters.
5. **Model Performance** — metric cards, comparison table and chart, actual vs predicted, residuals,
   feature importance.
Missing data/model files and invalid inputs produce clear messages. Screenshots: `reports/screenshots/`.

## 13. Results and Discussion
Pre-release metadata explains about 57% of the variance of log popularity, a clear improvement over the
baseline (R² 0) and the linear model (0.50). Budget is the dominant signal, consistent with the EDA (median
popularity rises steadily from 0.87 for unknown budgets to 14.9 for ≥ \$100M). Production scale, franchise
membership and recency also matter, while individual genres add little once budget is known. On the raw
scale R² is only 0.20 because a handful of blockbusters with scores above 100 dominate squared errors — the
model ranks movies well but cannot pin down extreme values.

## 14. Limitations
- Popularity is a 2017 activity snapshot; recently released films are inflated (release year is partly a
  proxy for recency).
- 81% of budgets are missing; runtime partly imputed.
- No marketing, cast, crew or review information; text features were not used.
- Random split mixes years; a time-based split would be a stricter pre-release test.
- Data ends in 2017; later predictions are extrapolations.

## 15. Conclusion
The project delivers a complete, reproducible Big Data workflow: Spark-based cleaning of a messy real dataset,
data-driven EDA, leakage-free regression models with honest evaluation, and a polished interactive dashboard
that is deployable on Streamlit Community Cloud. Gradient Boosting gives the best results (test R² 0.57 on the
log scale). Future work: add cast/crew from `credits.csv`, keywords and text embeddings, and a time-based
validation.

## 16. References
1. R. Banik, *The Movies Dataset*, Kaggle, 2017. https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
2. TMDB API documentation. https://developer.themoviedb.org/docs
3. F. M. Harper, J. A. Konstan, “The MovieLens Datasets: History and Context”, *ACM TiiS*, 5(4), 2015.
4. Apache Spark 4 documentation. https://spark.apache.org/docs/latest/
5. F. Pedregosa et al., “Scikit-learn: Machine Learning in Python”, *JMLR* 12, 2011.
6. Streamlit documentation. https://docs.streamlit.io

## 17. Team Contributions
| Member | Responsibility | Deliverable |
|--------|----------------|-------------|
| 1 | Dataset research & collection | dataset documentation, §4–5 |
| 2 | PySpark processing | `01_data_processing_and_cleaning.ipynb`, processed Parquet |
| 3 | EDA & visualisation | `02_eda.ipynb`, figures |
| 4 | Machine learning | `03_model_training.ipynb`, `models/` |
| 5 | Dashboard & deployment | `dashboard/app.py`, deployment |
