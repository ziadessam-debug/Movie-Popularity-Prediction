# Requirement Compliance & Testing Checklist

## University requirements → implementation
| # | Requirement (Project Requirements.md) | Where it is satisfied | Status |
|---|---------------------------------------|-----------------------|--------|
| 1 | Team of 5–7 students | 5 members, roles in README / report §17 / slide 16 (fill in names) | ✅ (names to add) |
| 2.1 | Data collection + documented source | Kaggle *The Movies Dataset* (R. Banik), auto-download cell in notebook 01, README "Dataset" | ✅ |
| 2.2 | Preprocessing: inspect, clean, missing values | Notebook 01 §4–5 (null profile, malformed rows, duplicates, invalid budget/runtime, imputation) | ✅ |
| 2.3 | Meaningful Spark usage (filter, aggregation, joins, DataFrames, Spark SQL) | Notebook 01 §3–8: DataFrames, `filter`, `groupBy/agg`, `explode`, Spark SQL, 2 joins, window `rank()` | ✅ |
| 2.4 | EDA with meaningful patterns | Notebook 02 (10 analyses, computed findings → `reports/eda_findings.json`) | ✅ |
| 2.5 | Project-specific ML + evaluation | Notebook 03: 5 models, CV tuning, MAE/RMSE/R², permutation importance | ✅ |
| 2.6 | Interactive dashboard | `dashboard/app.py` — 5 pages, KPIs, Plotly charts, filters, prediction form | ✅ |
| 4 | Dashboard: title & description, KPIs, visuals, filters, findings, model results, clear | Overview hero + KPIs; Explorer/Insights filters; Insights findings; Model Performance page | ✅ |
| 4 note | Dashboard does not replace Spark | All cleaning/feature creation in Spark notebook 01 | ✅ |
| 5.1 | Source code / notebooks | `notebooks/01–03`, `dashboard/app.py` | ✅ |
| 5.2 | Dataset or link with attribution | README, report §4–5, notebook 01 §2 | ✅ |
| 5.3 | Working dashboard | `streamlit run dashboard/app.py` (verified locally) | ✅ |
| 5.4 | Concise report | `reports/Project_Report.md` | ✅ |
| 5.5 | Presentation/demo + README | `reports/Movie_Popularity_Presentation.pptx` (16 slides), `README.md` | ✅ |

## Testing checklist (verified on 2026-11-10)
**Data** — dataset loads (45,466 rows) · required columns present · missing values handled (no nulls in
required columns, asserted) · duplicates removed (0 duplicate ids, asserted) · transformations validated
(ranges asserted).

**PySpark** — SparkSession starts (Spark 4.3.0, Java 17) · DataFrames created · all transformations execute
(notebook runs top-to-bottom without errors) · processed Parquet files saved and re-read by notebooks 02/03.

**Machine learning** — training completes · train/test indices disjoint (asserted) · no leakage (assert that
revenue / votes / popularity are not features) · metrics computed with scikit-learn · model saved, reloaded
and predictions identical (asserted) · prediction input built in training feature order (`feature_order` in
`model_metadata.json`).

**Dashboard** — all 5 pages load without server errors · title search and filters update the table · charts
use the processed data · prediction returns a score · empty genre selection shows a validation error ·
missing model/data files show an explanatory message instead of a crash.

**Deployment** — local run verified. Streamlit Community Cloud: not deployed from here (needs your GitHub
account); follow README "Online deployment". `dashboard/requirements.txt` pins the scikit-learn version used
for training.
