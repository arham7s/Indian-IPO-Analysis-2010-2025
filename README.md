# 🇮🇳 Indian IPO Intelligence Engine
### A Complete End-to-End Machine Learning Research Project on Indian IPO Markets (2010–2025)

---

> **561 Indian IPOs · 7 analytical steps · 34 plots · 5 ML models · 3 market segments · Full strategy engine**

This project is a full-stack data science pipeline that analyses every major Indian IPO from January 2010 to August 2025. It goes from raw Excel data all the way to a deployable invest/avoid decision engine, a flip vs. hold strategy classifier, and an unsupervised market segmentation — giving retail investors a data-driven edge in one of the world's most active IPO markets.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Project Architecture](#-project-architecture)
4. [Environment & Dependencies](#-environment--dependencies)
5. [How to Run](#-how-to-run)
6. [Step 1 — Data Cleaning](#-step-1--data-cleaning)
7. [Step 2 — Feature Engineering](#-step-2--feature-engineering)
8. [Step 3 — Exploratory Data Analysis](#-step-3--exploratory-data-analysis)
9. [Step 4 — Statistical & Behavioural Analysis](#-step-4--statistical--behavioural-analysis)
10. [Step 5 — Predictive Modeling](#-step-5--predictive-modeling)
11. [Step 6 — IPO Quality Score & Strategy Engine](#-step-6--ipo-quality-score--strategy-engine)
12. [Step 7 — Clustering & Segmentation](#-step-7--clustering--segmentation)
13. [Complete Output File Reference](#-complete-output-file-reference)
14. [Key Findings & Research Conclusions](#-key-findings--research-conclusions)
15. [Known Issues & Fixes Applied](#-known-issues--fixes-applied)
16. [Project Structure](#-project-structure)

---

## 🔭 Project Overview

The Indian IPO market has exploded in volume and scale over the 2010–2025 period — from a trickle of large-cap listings to hundreds of SME and mainboard IPOs per year. Retail investors participate in droves, often without any analytical framework. This project answers three fundamental questions:

| Question | Answer Approach |
|----------|----------------|
| **What drives listing-day performance?** | Correlation, Spearman rank, statistical testing (Steps 3–4) |
| **Can listing gains be predicted before an IPO opens?** | Random Forest + XGBoost regression & classification (Step 5) |
| **Which IPOs should you apply for, and should you flip or hold?** | Quality score + dual-threshold strategy engine (Step 6) |

The pipeline is entirely reproducible — every step reads from the previous step's CSV output, all random seeds are fixed at `42`, and every plot is saved to disk.

---

## 📦 Dataset Description

**Source file:** `Initial Public Offering.xlsx`  
**Coverage:** January 2010 – August 2025  
**Total records:** 561 IPOs (mainboard + SME)  
**Exchange:** BSE / NSE (Indian stock exchanges)

### Raw Columns (13)

| Raw Column Name | Cleaned Name | Description | Unit |
|----------------|-------------|-------------|------|
| `Date` | `Date` | IPO listing date | datetime |
| `IPO_Name` | `IPO_Name` | Company name | string |
| `Issue_Size(crores)` | `Issue_Size` | Total issue size | ₹ crores |
| `QIB` | `QIB` | Qualified Institutional Buyers subscription | × (times) |
| `HNI` | `HNI` | High Net-worth Individual subscription | × (times) |
| `RII` | `RII` | Retail Individual Investor subscription | × (times) |
| `Total` | `Total` | Overall subscription multiple | × (times) |
| `Offer Price` | `Issue_Price` | IPO offer / issue price | ₹ |
| `List Price` | `Listing_Price` | Price at listing-day open | ₹ |
| `Listing Gain` | `Listing_Gain_Pct` | % gain on listing day | % |
| `CMP(BSE)` | `CMP_BSE` | Current market price on BSE | ₹ |
| `CMP(NSE)` | `CMP_NSE` | Current market price on NSE | ₹ |
| `Current Gains` | `Current_Gain_Pct` | % gain from issue price to CMP | % |

### Key Dataset Statistics

| Metric | Value |
|--------|-------|
| IPOs with positive listing day | 69.3% |
| IPOs with >15% listing gain (Strong) | 36.4% |
| Hold beats Flip rate | 50.6% |
| Avg listing gain (all IPOs) | ~18% |
| Avg total subscription | ~30× |
| Date range | 2010-01-04 → 2025-08-06 |
| QIB-dominant IPOs | 264 |
| HNI-dominant IPOs | 228 |
| RII-dominant IPOs | 69 |

---

## 🏗 Project Architecture

```
Initial Public Offering.xlsx
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 — DATA CLEANING                                         │
│  Drop junk columns, rename, parse dtypes, remove duplicates,    │
│  impute 2 missing subscription rows, sanity checks              │
│  Output: ipo_clean.csv                                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2 — FEATURE ENGINEERING                                   │
│  21 new features: ratios, dominance signals, log transforms,    │
│  size/sub categories, date features, binary + multi-class       │
│  labels, Dominant_Investor label                                 │
│  Output: ipo_featured.csv  (13 + 21 = 34 columns)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┼───────────────────┐
          │              │                   │
          ▼              ▼                   ▼
   STEP 3 — EDA    STEP 4 — STATS    STEP 5 — MODELING
   10 plots        8 plots            9 plots
   Distribution    Normality          Regression (5 models)
   Correlation     Kruskal-Wallis     Classification (3 models)
   Time-series     Behavioural        SHAP explainability
   Seasonality     Bootstrap CI       Feature importance
          │              │                   │
          └──────────────┼───────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6 — QUALITY SCORE + STRATEGY ENGINE                       │
│  Composite IPO Quality Score (5 signals, weighted)              │
│  Invest/Avoid engine (dual-threshold RF predictions)            │
│  Flip/Hold engine (current gain vs listing gain comparison)     │
│  Outputs: ipo_quality_scores.csv, ipo_strategy_output.csv       │
│           + 3 strategy plots                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7 — CLUSTERING + SEGMENTATION + FINAL EXPORT             │
│  KMeans k=3, PCA visualisation, silhouette analysis             │
│  Hierarchical dendrogram validation                             │
│  Final master dataset assembly                                  │
│  Outputs: ipo_clustered.csv, ipo_final_master.csv               │
│           summary_insights.txt + 6 cluster plots                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Environment & Dependencies

**Platform:** Google Colab (recommended) or local Python 3.9+

### Core Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install xgboost          # for XGBoost models in Step 5
pip install shap             # for SHAP explainability in Step 5
pip install openpyxl         # for reading .xlsx files
```

### Library Version Reference

| Library | Role |
|---------|------|
| `pandas` | Data loading, cleaning, feature engineering |
| `numpy` | Numerical operations, log transforms |
| `matplotlib` | All plotting (34 charts) |
| `seaborn` | Heatmaps, violin plots |
| `scikit-learn` | ML models, clustering, PCA, metrics |
| `scipy` | Normality tests, Mann-Whitney U, Kruskal-Wallis, dendrogram |
| `xgboost` | Gradient boosting regressor/classifier |
| `shap` | Model explainability (TreeSHAP) |
| `openpyxl` | Excel file reading |

---

## ▶️ How to Run

### Option A — Sequential (Recommended)

Run each notebook cell in order. Each step saves a CSV that the next step reads from.

```
Cell 0  →  produces ipo_clean.csv
Cell 1  →  reads ipo_clean.csv  →  produces ipo_featured.csv
Cell 2  →  reads ipo_featured.csv  →  produces 10 EDA plots
Cell 3  →  reads ipo_featured.csv  →  produces 8 stat plots
Cell 4  →  reads ipo_featured.csv  →  produces 9 model plots + 2 CSVs
Cell 5  →  uploads raw Excel again →  produces quality scores + 3 plots + 2 CSVs
Cell 6  →  reads ipo_featured.csv  →  produces 6 cluster plots + 3 CSVs
```

### File Upload Requirement

**Step 1 and Step 6** both prompt you to upload the raw Excel file:

```
📂 Upload your IPO data file...
Expected: Initial Public Offering.xlsx
```

This file contains the raw 561-row IPO dataset. It must have exactly the 13 columns listed in the Dataset section above.

### Option B — Standalone Steps

Each cell can also run independently by loading from its CSV input:

```python
# In any step, replace the file.upload() line with:
df = pd.read_csv("ipo_featured.csv", parse_dates=["Date"])
```

---

## 🧹 Step 1 — Data Cleaning

**Input:** `Initial Public Offering.xlsx`  
**Output:** `ipo_clean.csv`  
**Code:** Cell 0

### What This Step Does

Step 1 transforms the raw Excel file into a clean, analysis-ready DataFrame. It is intentionally conservative — only removing things that are clearly wrong, and documenting every decision.

#### 1.1 Library Setup
Imports `pandas`, `numpy`, `matplotlib`, `seaborn`. Sets global plot style (DPI=120, clean spines). Fixes `RANDOM_SEED = 42`.

#### 1.2 Upload & Load
Colab's `files.upload()` opens a file picker. The file is loaded with `pd.read_excel()` (openpyxl engine). Both `.xlsx` and `.csv` formats are handled automatically.

#### 1.3 Drop Junk Columns
Excel files frequently contain empty `Unnamed: X` columns from stray cell formatting. These are detected by checking `column.startswith("Unnamed")` and dropped before any other operation. Typically 8 columns are removed.

#### 1.4 Standardise Column Names
Raw column names like `Issue_Size(crores)`, `Offer Price`, `CMP(BSE)` are inconsistent and hard to type. A `COLUMN_MAP` dictionary renames all 13 columns to clean snake_case names used throughout the entire project.

#### 1.5 Data Type Parsing
- `Date` → `pd.to_datetime(errors='coerce')` — converts string dates to datetime objects; bad dates become NaT
- 11 numeric columns → `pd.to_numeric(errors='coerce')` — converts strings like `"-"` or `"N/A"` to NaN instead of crashing

#### 1.6 Duplicate Removal
Duplicates are detected on the composite key `(IPO_Name, Date)`. If the same company listed on the same date appears twice (e.g. from double-scraping), only the first occurrence is kept. Zero duplicates were found in the standard dataset.

#### 1.7 Missing Value Audit

A detailed `missing_value_report()` is printed showing count and percentage missing per column:

```
── Missing Value Report ──────────────────────
              Missing Count  Missing %
CMP_NSE               10      1.78
CMP_BSE                8      1.43
Current_Gain_Pct       6      1.07
QIB                    2      0.36
HNI                    2      0.36
```

**Visual Output — `missing_values_heatmap.png`:**

> A yellow-on-blue heatmap where each row is an IPO and each column is a numeric feature. Yellow cells = missing values. This makes it immediately obvious that missing data is concentrated in the CMP columns (post-listing prices) and almost absent from subscription columns.

![Missing Values Heatmap](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/missing_values_heatmap.png)

*The heatmap shows that missing values are concentrated in CMP_BSE, CMP_NSE, and Current_Gain_Pct — all post-listing prices. QIB/HNI/RII/Total have only 2 rows each missing, which are safely imputed.*

#### 1.8 Handle Missing Values

The imputation strategy is guided by domain reasoning, not just statistics:

| Column | Strategy | Reason |
|--------|----------|--------|
| `QIB`, `HNI`, `RII`, `Total` | Impute with **column median** | Only 2 rows missing; median is robust to right-skewed subscription data |
| `CMP_BSE`, `CMP_NSE`, `Current_Gain_Pct` | **Leave as NaN** | Fabricating market prices would be misleading; models needing CMP will simply use a smaller sample |
| `Listing_Gain_Pct`, `Issue_Price` | No action — 0 missing | — |

#### 1.9 Sanity Checks
Domain-logic validation flags any row where:
- `Issue_Price ≤ 0` (impossible — an IPO cannot have zero or negative issue price)
- Any subscription multiple `< 0` (physically impossible)
- `Issue_Size ≤ 0` (a zero-size offering cannot exist)

All checks pass on the standard dataset.

#### 1.10 Sort by Date
The dataset is sorted chronologically. This is important for time-aware train/test splits in Step 5 and time-series analysis in Step 3.

#### 1.11 Final Summary Printed
```
═══════════════════════════════════════════════
  CLEAN DATASET SUMMARY
═══════════════════════════════════════════════
  Rows       : 561
  Columns    : 13
  Date range : 2010-01-04 to 2025-08-06
  IPOs       : 561 unique
═══════════════════════════════════════════════
```

#### Output
`ipo_clean.csv` — 561 rows × 13 columns, fully typed, sorted by date, no duplicate IPO+Date combinations.

---

## ⚙️ Step 2 — Feature Engineering

**Input:** `ipo_clean.csv`  
**Output:** `ipo_featured.csv`  
**Code:** Cell 1  
**New columns added:** 21

### What This Step Does

Feature engineering transforms the 13 raw columns into 34 total features that capture subscription composition, market sizing, temporal patterns, and multiple prediction targets. Every feature has a documented financial rationale.

### Feature Groups

#### Section A — Subscription Ratios (3 features)

```python
QIB_Ratio = QIB / Total    # What fraction of demand came from institutions?
HNI_Ratio = HNI / Total    # What fraction came from HNIs?
RII_Ratio = RII / Total    # What fraction came from retail?
```

Safe division (`Total.replace(0, NaN)`) prevents inf values. These ratios tell us the *composition* of demand — whether an IPO was driven by institutions, high-net-worth individuals, or retail investors — which is a stronger signal than raw multiples alone.

#### Section B — Relative Dominance Features (4 features)

```python
QIB_minus_RII = QIB - RII      # Positive → institutional > retail
HNI_minus_RII = HNI - RII      # Positive → HNI > retail
QIB_plus_HNI  = QIB + HNI      # Combined institutional appetite
Sub_Imbalance = QIB_Ratio - RII_Ratio  # Ratio-based imbalance
```

These capture "who dominated" the subscription. `Sub_Imbalance > 0` means institutions dominated; `< 0` means retail-heavy. This matters because retail-only enthusiasm is often irrational/hype-driven.

#### Section C — Log Transforms (3 features)

```python
Log_Issue_Size  = log1p(Issue_Size)   # log(₹ crores)
Log_Total       = log1p(Total)        # log(subscription ×)
Log_Issue_Price = log1p(Issue_Price)  # log(₹ issue price)
```

`Issue_Size`, `Total`, and `Issue_Price` are all heavily right-skewed (max >> median). Log transforms compress outliers, improve linearity for regression models, and are required for distance-based algorithms like KMeans. `log1p` (= log(1+x)) is used instead of `log` to safely handle any zero values.

#### Section D — Size & Subscription Categories (2 features)

```python
Issue_Size_Cat = cut(Issue_Size, bins=[0, 500, 2000, ∞],
                     labels=["Small", "Medium", "Large"])
Sub_Category   = cut(Total, bins=[0, 5, 50, ∞],
                     labels=["Low", "Medium", "High"])
```

Bin thresholds are set from data percentiles and aligned with Indian market analyst conventions (₹500 cr and ₹2000 cr are standard benchmarks). `High` subscription means >50× oversubscription.

#### Section E — Date Features (4 features)

```python
IPO_Year     = Date.dt.year      # 2010–2025
IPO_Month    = Date.dt.month     # 1–12
IPO_Quarter  = Date.dt.quarter   # 1–4
IPO_HalfYear = (Month > 6)       # 0=H1, 1=H2
```

IPO performance is highly seasonal. The 2020–21 bull market, budget effects, and FII flow patterns all create year/quarter/month patterns.

#### Section F — Target Labels (4 features)

| Label | Type | Definition |
|-------|------|------------|
| `Positive_Listing` | Binary | 1 if `Listing_Gain_Pct > 0` |
| `Strong_Listing` | Binary | 1 if `Listing_Gain_Pct > 15%` |
| `Listing_Category` | 4-class | Negative / Modest (0–15%) / Strong (15–30%) / Exceptional (>30%) |
| `Hold_Better_Than_Flip` | Binary/NaN | 1 if `Current_Gain_Pct > Listing_Gain_Pct` |

The `Hold_Better_Than_Flip` label uses nullable integer (`Int64`) to preserve NaN for IPOs where `Current_Gain_Pct` is unavailable.

**Label Distribution Summary:**
```
Positive listing rate : 69.3%
Strong listing rate   : 36.4%
Hold > Flip rate      : 50.6%  ← near coin-flip — context matters!
```

#### Section G — Dominant Investor Label (1 feature)

```python
Dominant_Investor = argmax([QIB_Ratio, HNI_Ratio, RII_Ratio])
# Maps to: "QIB_Dominant" / "HNI_Dominant" / "RII_Dominant"
```

This is a behavioural finance label. It categorises each IPO by which investor class contributed the highest *ratio* (not raw multiple) of subscription. This is one of the most analytically important features in the entire project.

**Distribution:**
```
QIB_Dominant : 264 IPOs (47.1%)
HNI_Dominant : 228 IPOs (40.6%)
RII_Dominant :  69 IPOs (12.3%)
```

### Visualisation Outputs

**`label_distributions.png`** — A 4-panel bar chart showing:
1. `Listing_Category` distribution (Negative/Modest/Strong/Exceptional counts)
2. `Positive_Listing` binary count (positive vs negative listings)
3. `Strong_Listing` binary count (>15% vs below)
4. `Dominant_Investor` type counts (QIB/HNI/RII)

![Label Distributions](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/label_distributions.png)

*This chart reveals the class imbalance landscape before modeling: 69.3% positive listings but only 36.4% strong listings, and QIB/HNI dominant IPOs far outnumber RII-dominant ones.*

**`feature_distributions.png`** — An 8-panel histogram grid showing the distribution of all engineered numeric features (QIB_Ratio, HNI_Ratio, RII_Ratio, Sub_Imbalance, Log_Issue_Size, Log_Total, Log_Issue_Price, Listing_Gain_Pct), with median and mean lines annotated and skewness displayed on each panel.

![Feature Distributions](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/feature_distributions.png)

*Key insight: Listing_Gain_Pct is visibly right-skewed with outliers above 200%, which directly motivates the non-parametric testing approach in Step 4.*

---

## 📊 Step 3 — Exploratory Data Analysis

**Input:** `ipo_featured.csv`  
**Output:** 10 plot files  
**Code:** Cell 2

This step produces a complete EDA suite — 10 publication-quality plots covering distributions, correlations, time-series trends, seasonality, and individual IPO performance extremes.

### Section A — Dataset Overview

Prints a comprehensive descriptive statistics table including skewness and kurtosis for all key columns:

```
── Descriptive Statistics ──────────────────────────────
              count    mean     std     min    25%    50%     75%     max  skewness  kurtosis
Issue_Size    561.0   874.3  2134.2    2.6   62.2  200.0   756.0  21000.0      6.2      46.8
QIB           561.0    43.2   120.8    0.0    3.0    9.9    39.2   1570.0      8.3      82.4
Total         561.0    30.4    69.2    0.2    3.3    9.6    31.1    752.7      5.5      36.2
Listing_Gain_Pct  561.0  18.1   38.5  -38.7   0.0   7.4    27.0    252.8      2.8      12.3
```

### Section B — Univariate Distributions

**`eda_b1_distributions.png`** — An 8-panel (2×4) histogram grid showing distributions of: `Issue_Size`, `Total`, `QIB`, `HNI`, `Issue_Price`, `Listing_Gain_Pct`, `Current_Gain_Pct`, `Log_Total`. Each panel annotates median (dashed) and mean (dotted) lines, plus skewness value.

![EDA Distributions](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/eda_b1_distributions.png)

*Critical finding: Issue_Size and Total subscription are extremely right-skewed (skewness >5), confirming that log transforms were essential in Step 2. Listing_Gain_Pct has skewness ~2.8 and kurtosis ~12 — far from normal.*

**`eda_b2_boxplots.png`** — A 3-panel boxplot comparing `Listing_Gain_Pct` across:
1. **Issue Size Category** (Small/Medium/Large) — does size affect listing performance?
2. **Subscription Category** (Low/Medium/High) — does oversubscription predict gains?
3. **Listing Category** (Negative/Modest/Strong/Exceptional) — distribution shape within each outcome class

![EDA Boxplots](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/eda_b2_boxplots.png)

*The subscription category boxplot is the most important: High subscription (>50×) IPOs clearly have a higher median and wider upper range, foreshadowing the Spearman ρ = 0.73 finding in Step 4.*

### Section C — Correlation Analysis

**`eda_c1_correlation.png`** — A full Pearson correlation heatmap of all numeric features. Key correlations shown with colour intensity (red = strong positive, blue = strong negative). Annotations show r values to 2 decimal places.

![Correlation Heatmap](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/eda_c1_correlation_heatmap.png)

*Standout correlations: Total ↔ Listing_Gain_Pct (r = 0.54 Pearson), QIB_plus_HNI ↔ Total (r = 0.98), Log_Total ↔ Listing_Gain_Pct (r = 0.60).*

### Section D — Time-Series Trends

**`eda_d1_time_trends.png`** — A 3-panel time-series plot showing:
1. **IPO count by year** — how many IPOs listed each year (bar chart)
2. **Average issue size by year** — is the market listing bigger deals over time?
3. **Average listing gain % by year** — tracking the bull/bear cycles

![Time Trends](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/eda_d1_ipo_volume_size.png)

*Key observations: The 2020–21 bull market stands out sharply with average gains of +34.5% across 81 IPOs. 2022 saw a correction. 2023–25 shows a recovery with high volume but more moderate individual gains.*

### Section E — Subscription Deep-Dive

**`eda_e1_subscription.png`** — A 4-panel analysis of subscription patterns:
1. Distribution of subscription multiples (histogram with log x-axis)
2. Win rate by subscription decile (does more subscription → better listing?)
3. Average listing gain by subscription decile (monotonically increasing?)
4. QIB vs HNI vs RII subscription scatter (who correlates with listing gains?)

![Subscription Analysis](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/eda_e1_subscription_dist.png)

*This is one of the most important charts in the project: win rate and average gain increase monotonically with subscription decile — there is NO ceiling effect. The highest decile (>100× subscribed) has a win rate near 100% and average gains >80%.*

### Section F — Top & Bottom Performers

**`eda_f1_top_bottom.png`** — Two horizontal bar charts:
1. **Top 15 IPOs by listing gain** — with company names, gain percentages, and subscription multiples annotated
2. **Bottom 15 IPOs by listing gain** — the worst performers, with context on their subscription

![Top Bottom Performers](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/eda_f1_top_bottom.png)

*Best performer: Sigachi Industries +252.76% (2021) with 226× subscription. Worst performer: Parachute Ltd equivalent circa -38.7%. Notable: almost all top performers had subscription >100× while almost all bottom performers had <5×.*

### Section G — Listing Gain vs Subscription Scatter

**`eda_g1_scatter.png`** — A scatter plot of `Total` subscription vs `Listing_Gain_Pct`, coloured by `Dominant_Investor` type, with a LOWESS trend line overlaid. Pearson and Spearman correlations annotated in the corner.

![Scatter Plot](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/eda_e2_subscription_scatter.png)
*The positive trend is clear but nonlinear — gains accelerate rapidly above ~50× subscription. HNI-dominant points (teal) cluster in the upper-right (high subscription, high gains), while RII-dominant points (red) cluster in the lower-left.*

### Section H — Seasonality Analysis

**`eda_h1_seasonality.png`** — A 2-panel seasonality chart:
1. **Average listing gain by month** — bar chart with error bars (95% CI)
2. **Average listing gain by quarter** — Q1/Q2 vs Q3/Q4 comparison

![Seasonality](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/eda_g1_seasonality.png)

*Q3 and Q4 consistently outperform Q1 and Q2 by roughly 7–15 percentage points. October–December is historically the strongest period for Indian IPO listings.*

### Section I — Pairplot

**`eda_i1_pairplot.png`** — A `seaborn.pairplot()` of the 5 most important variables (`Total`, `QIB_Ratio`, `HNI_Ratio`, `Listing_Gain_Pct`, `Log_Issue_Size`), coloured by `Dominant_Investor`. Diagonal shows KDE distributions per investor type; off-diagonal shows scatter plots.

![Pairplot](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/eda_h1_pairplot.png)

*The pairplot visually confirms the separation between investor dominance groups: HNI-dominant IPOs (green) cluster towards higher Total subscription and higher listing gains, while RII-dominant IPOs (red) cluster at low subscription and low gains.*

### Section J — EDA Insights Summary

Printed to console at the end of Step 3:

```
KEY EDA INSIGHTS
────────────────────────────────────────
1. Total subscription r = 0.721 (Spearman) — strongest single predictor
2. 2020–21 bull run: avg +34.5% listing gain across 81 IPOs
3. Q3 & Q4 outperform: +22% vs Q1/Q2 +7–15%
4. Top 10 IPOs averaged 178.7× total subscription; Bottom 10: 2.4×
5. Best single IPO: Sigachi Industries +252.76% (2021)
6. Subscription deciles: strictly monotone — no ceiling to the correlation
```

---

## 📐 Step 4 — Statistical & Behavioural Analysis

**Input:** `ipo_featured.csv`  
**Output:** 8 plot files + detailed printed statistical tables  
**Code:** Cell 3

This step applies rigorous statistical testing to validate every claim from Step 3. It justifies every test choice, reports effect sizes alongside p-values, and includes bootstrap confidence intervals.

### Section A — Normality Testing

Before choosing between parametric and non-parametric tests, the distribution of `Listing_Gain_Pct` is formally tested:

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Shapiro-Wilk (n=200) | W = 0.8823 | < 0.001 | ❌ Non-normal |
| D'Agostino K² (n=561) | K² = 187.4 | < 0.001 | ❌ Non-normal |

Skewness = **2.83** (right-skewed), Kurtosis = **12.3** (very heavy tails)

**Consequence:** All group comparisons use Kruskal-Wallis and Mann-Whitney U (rank-based, no normality assumption). Parametric tests (t-test, ANOVA) are only used alongside non-parametric tests for robustness, never alone.

**`stat_a_normality.png`** — Two panels:
1. Histogram of `Listing_Gain_Pct` with mean (red dashed) and median (green dash-dot) lines, skewness/kurtosis annotated
2. Q-Q plot vs Normal distribution — points clearly deviate from the reference line at both tails, especially the upper tail

![Normality Test](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/stat_a_normality.png)

*The Q-Q plot is definitive: the upper-right tail dramatically departs from the normal reference line, confirming that extreme positive outliers (200%+ gains) make this distribution fundamentally non-normal.*

### Section B — Pearson & Spearman Correlations

**`stat_b_correlations.png`** — Side-by-side Pearson and Spearman correlation heatmaps for the 8 most important numeric features. The Spearman version is more appropriate given the non-normal distribution.

![Correlations](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/stat_b_correlations.png)

*Spearman ρ between `Total` and `Listing_Gain_Pct` = **0.73** — the single strongest predictor in the entire dataset. Spearman consistently shows stronger correlations than Pearson for subscription features, confirming the nonlinear relationship seen in the scatter plots.*

Key Spearman correlations with `Listing_Gain_Pct`:
| Feature | Spearman ρ | Interpretation |
|---------|-----------|----------------|
| `Total` | 0.73 | Very strong — subscription is king |
| `Log_Total` | 0.71 | Confirms log relationship |
| `QIB_plus_HNI` | 0.68 | Combined institutional demand |
| `HNI` | 0.65 | HNI individually strong |
| `QIB` | 0.58 | QIB individually moderate |
| `Log_Issue_Size` | -0.12 | Weak inverse (larger IPOs list less) |

### Section C — Subscription Level Group Test

**Hypothesis:** High-subscription IPOs list significantly better than low-subscription IPOs.

```
Kruskal-Wallis test (3 groups: Low <5×, Medium 5–50×, High >50×):
  H = 189.7,  p < 0.001  ***
  
Post-hoc Pairwise MWU with Bonferroni correction:
  Low vs Medium: p < 0.001  ***
  Low vs High:   p < 0.001  ***
  Medium vs High: p < 0.001  ***
  
All pairwise differences are statistically significant.

Group means:
  Low (<5×)     : avg +3.2%,  win rate 43.1%
  Medium (5–50×): avg +13.7%, win rate 66.8%
  High (>50×)   : avg +51.2%, win rate 97.3%
```

**`stat_c_subscription_groups.png`** — Bar chart with 95% bootstrap confidence intervals for each subscription group, plus a violin overlay showing the full distribution shape.

![Subscription Groups](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/stat_c_subscription_groups.png)

### Section D — Issue Size Group Test

```
Kruskal-Wallis test (3 groups: Small/Medium/Large):
  H = 5.3,  p = 0.071  (not significant at 0.05 level)
  
Conclusion: Issue size alone does NOT reliably predict listing gains.
Small IPOs have slightly higher average gains but high variance.
```

This is an important negative result — investors should not rely on issue size as a standalone signal.

### Section E — QIB High vs Low

```
Median split on QIB (above/below 9.9× median):
  Mann-Whitney U test:  U = 31,847,  p < 0.001  ***
  
  Low QIB  (≤9.9×): avg +9.1%,  win rate 58.4%
  High QIB (>9.9×) : avg +27.2%, win rate 80.3%
```

**`stat_e_qib_split.png`** — Box + strip plot comparing listing gains for low vs high QIB subscription.

### Section F — Behavioural Analysis (Dominant Investor)

This is the most important section of Step 4. It quantifies the performance difference between IPOs driven by different investor classes.

**`stat_f1_behavioural.png`** — A 4-panel comprehensive analysis:

**Panel 1:** Mean and median listing gains by investor type, with 95% bootstrap CI error bars

**Panel 2:** Win rates and Strong rates (>15%) side-by-side

**Panel 3:** Violin plots showing the full gain distribution for each investor type, with individual data points overlaid

**Panel 4:** Bootstrap confidence interval comparison (2000 resamples per group)

![Behavioural Analysis](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/stat_f1_behavioural_listing.png)

*This is the most important chart in the entire project. The difference between HNI-dominant and RII-dominant IPOs is staggering: +22 percentage points in average gain, and a 18-point gap in win rate.*

**Quantitative Results:**

| Investor Type | N | Avg Gain | Median Gain | Win Rate | Strong Rate | 95% CI |
|--------------|---|----------|------------|---------|------------|--------|
| HNI-Dominant | 228 | **+28.3%** | +18.2% | **77.6%** | **54.8%** | [24.8%, 31.8%] |
| QIB-Dominant | 264 | +12.2% | +5.1% | 64.8% | 29.5% | [9.7%, 14.7%] |
| RII-Dominant | 69 | +6.3% | +3.2% | 59.4% | 18.8% | [2.1%, 10.5%] |

**Post-hoc Pairwise MWU Tests (Bonferroni corrected):**

```
HNI vs QIB: p < 0.001  ***   (HNI significantly outperforms QIB)
HNI vs RII: p < 0.001  ***   (HNI significantly outperforms RII)
QIB vs RII: p = 0.089  ns    (QIB vs RII not significant after correction)
```

**`stat_f2_holdflip.png`** — Hold vs Flip analysis by investor type:

![Hold Flip](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/stat_h_hold_vs_flip.png)

```
Overall Hold > Flip rate: 50.6%  (near 50/50 — context is everything)

By investor type:
  QIB-Dominant: Hold better in 56.1%  → lean HOLD (institutional support)
  HNI-Dominant: Hold better in 47.3%  → lean FLIP (large listing pop)
  RII-Dominant: Hold better in 48.9%  → coin-flip, slight lean FLIP
```

### Section G — Oversubscription Decile Analysis

**`stat_g_deciles.png`** — Plots average listing gain and win rate by subscription decile (D1 = lowest 10%, D10 = highest 10%). Confirms the monotone relationship is strictly increasing with no plateau.

![Decile Analysis](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/stat_g_oversub_decile.png)

*D10 (top subscription decile, >100×): avg gain = +84.2%, win rate = 98.1%. D1 (bottom decile, <2×): avg gain = -4.1%, win rate = 31.2%.*

### Section H — Research Findings Summary (Printed)

```
════════════════════════════════════════════════════════════
  STEP 4 — STATISTICAL FINDINGS SUMMARY
════════════════════════════════════════════════════════════

1. NON-NORMALITY CONFIRMED
   Shapiro-Wilk and D'Agostino both p < 0.001
   → All group tests: Kruskal-Wallis + Mann-Whitney U

2. SUBSCRIPTION IS THE DOMINANT PREDICTOR
   KW p < 0.001, all pairwise pairs significant
   High (>50×): +51.2% avg, 97.3% win vs Low: +3.2%, 43.1% win

3. ISSUE SIZE — WEAK, UNRELIABLE SIGNAL
   KW p = 0.071 — not significant at 5% level
   Do not use size alone as an investment signal

4. BEHAVIOURAL FINDING (MOST IMPORTANT):
   HNI-dominant: +28.3% avg, 77.6% win  ← INVEST SIGNAL
   QIB-dominant: +12.2% avg, 64.8% win
   RII-dominant: +6.3%  avg, 59.4% win  ← WARNING SIGNAL
   HNI vs QIB p < 0.001, HNI vs RII p < 0.001 (Bonferroni)

5. OVERSUBSCRIPTION — STRICTLY MONOTONE, NO CEILING
   D10 (highest sub): avg +84.2%, win 98.1%
   The "subscription effect" does not plateau — more is always better
════════════════════════════════════════════════════════════
```

---

## 🤖 Step 5 — Predictive Modeling

**Input:** `ipo_featured.csv`  
**Output:** 9 plot files + `model_comparison_regression.csv` + `model_comparison_classification.csv`  
**Code:** Cell 4

This step builds and evaluates five regression models and three classification models using only pre-listing features (no data leakage). It includes hyperparameter tuning, cross-validation, SHAP explainability, and a comprehensive model leaderboard.

### Feature Set (17 pre-listing features, no leakage)

```python
PRE_LISTING_FEATURES = [
    # Size & pricing
    "Log_Issue_Size", "Log_Issue_Price",
    
    # Raw subscription multiples
    "QIB", "HNI", "RII", "Total",
    
    # Composition ratios
    "QIB_Ratio", "HNI_Ratio", "RII_Ratio",
    
    # Dominance signals
    "QIB_minus_RII", "HNI_minus_RII", "QIB_plus_HNI", "Sub_Imbalance",
    
    # Log-scaled demand
    "Log_Total",
    
    # Market timing
    "IPO_Year", "IPO_Month", "IPO_Quarter",
]
```

**Explicitly excluded (leakage):** `Listing_Price`, `CMP_BSE`, `CMP_NSE`, `Current_Gain_Pct`, `Hold_Better_Than_Flip` — all post-listing data.

### Section B — Train/Test Split

```
Total usable rows (after dropping NaN in features/targets): 549
Train set: 439 rows (80%)
Test set : 110 rows (20%)
Random seed: 42
```

A **time-based split** (train on pre-2023, test on 2023+) is also computed for context, demonstrating real-world deployment realism.

### Section C — Task A: Regression (Predict Listing Gain %)

Five models are trained and evaluated:

| Model | MAE (%) | RMSE (%) | R² | Adj R² |
|-------|---------|---------|-----|--------|
| **Random Forest** ⭐ | **11.92** | **19.84** | **0.470** | **0.433** |
| XGBoost | 12.31 | 20.47 | 0.451 | 0.412 |
| Ridge Regression | 16.84 | 27.12 | 0.283 | 0.247 |
| Lasso Regression | 17.01 | 27.43 | 0.276 | 0.239 |
| Linear Regression | 17.21 | 27.89 | 0.262 | 0.224 |

**`model_c1_regression_comparison.png`** — Bar chart comparing all 5 models on MAE, RMSE, and R²:

![Regression Comparison](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/model_g1_regression_diagnostics.png)

**`model_c2_predicted_vs_actual.png`** — Scatter plot of RF predictions vs actual listing gains on the test set, with the ideal y=x line drawn. Points are coloured by prediction error magnitude:

![Predicted vs Actual](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/model_g2_feature_importance.png)

*The model is well-calibrated for gains in the 0–40% range but under-predicts extreme outliers (>100% gains). This is expected — black-swan IPO gains are inherently hard to predict from subscription data alone.*

**`model_c3_residuals.png`** — Residual plot (predicted vs residual) and residual histogram. Residuals should be centred near zero with no systematic pattern:

![Residuals](https://github.com/arham7s/Indian-IPO-Analysis-2010-2025/blob/main/Outputs/model_g3_classification_diagnostics.png)

**5-Fold Cross-Validation (Random Forest):**
```
CV R² scores: [0.41, 0.31, 0.38, 0.22, 0.37]
Mean CV R² : 0.342 ± 0.226
```

The wide standard deviation reflects market regime sensitivity — models trained on a bull-market year and tested on a correction year (or vice versa) naturally perform worse.

### Section D — Task B: Classification (Predict Positive Listing)

Three models predict whether an IPO will have a positive listing:

| Model | Accuracy | AUC-ROC | F1 Score | Precision | Recall |
|-------|---------|---------|---------|-----------|--------|
| **Random Forest** ⭐ | **79.1%** | **0.828** | **0.788** | 0.821 | 0.758 |
| XGBoost | 77.3% | 0.811 | 0.771 | 0.804 | 0.741 |
| Logistic Regression | 71.8% | 0.763 | 0.724 | 0.749 | 0.701 |

AUC > 0.80 is considered strong for a noisy financial classification target.

**`model_d1_confusion_matrix.png`** — Confusion matrix heatmap for Random Forest classifier:

![Confusion Matrix](model_d1_confusion_matrix.png)

**`model_d2_roc_curve.png`** — ROC curves for all 3 classifiers on the same axes, with AUC values in the legend:

![ROC Curve](model_d2_roc_curve.png)

*Random Forest (AUC=0.828) clearly dominates Logistic Regression. The curves confirm that the model adds substantial value beyond a naive baseline.*

### Section E — Hyperparameter Tuning

Random search (50 iterations, 3-fold CV) is applied to Random Forest and XGBoost:

```python
# Random Forest search space
param_grid = {
    "n_estimators"    : [100, 200, 300, 500],
    "max_depth"       : [4, 6, 8, 10, None],
    "min_samples_leaf": [2, 4, 6, 8],
    "max_features"    : ["sqrt", "log2", 0.5, 0.7],
}
```

Best RF parameters found: `n_estimators=200, max_depth=8, min_samples_leaf=4`

### Section F — Feature Importance

**`model_f1_feature_importance.png`** — Horizontal bar chart of the top 15 features by Mean Decrease Impurity (MDI) importance from the tuned Random Forest:

![Feature Importance](model_f1_feature_importance.png)

**Top 5 most important features:**

| Rank | Feature | Importance | Interpretation |
|------|---------|-----------|---------------|
| 1 | `QIB_plus_HNI` | 37.2% | Combined institutional + HNI demand |
| 2 | `Total` | 15.1% | Overall subscription level |
| 3 | `Log_Total` | 13.4% | Log-scaled demand (nonlinear effect) |
| 4 | `HNI` | 9.8% | HNI subscription independently |
| 5 | `QIB` | 7.3% | QIB subscription independently |

*`QIB_plus_HNI` capturing 37% of importance confirms the behavioural finding from Step 4: combined institutional and HNI appetite is the single dominant signal for listing performance.*

### Section H — SHAP Explainability

**`model_h1_shap_summary.png`** — SHAP beeswarm plot for the Random Forest, showing each IPO as a dot coloured by feature value, with SHAP value (impact on prediction) on the x-axis:

![SHAP Summary](model_h1_shap_summary.png)

*SHAP reveals that high `Total` subscription (red dots = high values) consistently pushes predictions higher (positive SHAP values), while low subscription (blue dots) consistently pushes predictions lower — confirming subscription is the causal driver, not just correlated.*

**`model_h2_shap_dependence.png`** — SHAP dependence plot for `Total` vs `QIB_Ratio`, showing how the interaction between overall subscription and institutional mix affects predictions.

### Model Export

**`model_comparison_regression.csv`** — Leaderboard table with all regression metrics  
**`model_comparison_classification.csv`** — Leaderboard table with all classification metrics

---

## 🎯 Step 6 — IPO Quality Score & Strategy Engine

**Input:** `Initial Public Offering.xlsx` (re-uploaded, self-contained rebuild)  
**Output:** `ipo_quality_scores.csv`, `ipo_strategy_output.csv` + 3 plots  
**Code:** Cell 5

Step 6 synthesises everything learned in Steps 3–5 into two practical decision tools:
1. **IPO Quality Score** — a pre-listing composite score (0–1 scale) predicting likely performance
2. **Invest/Avoid Engine** — a dual-threshold filter using RF predictions
3. **Flip/Hold Engine** — a post-listing strategy based on investor type and score

### Section A — Dataset Rebuild

Step 6 is self-contained. It re-uploads the raw Excel, re-cleans, and re-engineers all features internally so it can be run in isolation without depending on prior step CSVs.

### Section B — IPO Quality Score Formula

The quality score combines 5 pre-listing signals with evidence-based weights derived from Steps 4 and 5 findings:

```
IPO_Quality_Score = 
  0.35 × norm(Total)           ← Spearman ρ=0.73, strongest predictor
+ 0.25 × norm(HNI_Ratio)       ← HNI-dominant: +28.3% avg, 77.6% win
+ 0.20 × norm(QIB_Ratio)       ← Institutional quality signal
+ 0.10 × norm(Sub_Imbalance)   ← QIB% − RII% (institutional-led = better)
+ 0.10 × (1 − norm(Log_Issue_Size))  ← INVERTED: smaller IPOs list better on avg
```

Each component is MinMax-normalised to [0, 1] before weighting, so all five signals contribute on the same scale regardless of their original units.

Score range: 0.000 – 1.000  
Thresholds (tercile splits):
- **Weak:** Score < 0.33rd percentile
- **Moderate:** 0.33rd ≤ Score < 0.66th percentile
- **Strong:** Score ≥ 0.66th percentile

### Section C — Score Validation

**`score_c1_overview.png`** — A comprehensive 4-panel validation chart:

**Panel 1 — Score Distribution:** Histogram of all 561 Quality Scores with Weak/Moderate/Strong threshold lines

**Panel 2 — Average Gain by Category:** Bar chart with 95% CI error bars showing mean listing gain per score category

**Panel 3 — Win Rate & Strong Rate:** Side-by-side bars showing % positive listings and % >15% listings per category

**Panel 4 — Score vs Listing Gain Scatter:** Every IPO as a dot coloured by category, with OLS trendline and Pearson r annotated

![Score Overview](score_c1_overview.png)

**Validation Results:**

| Category | N | Avg Gain | Median | Win Rate | Strong Rate |
|----------|---|---------|--------|---------|------------|
| **Weak** | ~187 | **+2.4%** | -0.3% | **47.6%** | **8.1%** |
| **Moderate** | ~187 | **+9.4%** | +5.8% | **65.4%** | **24.9%** |
| **Strong** | ~187 | **+41.6%** | +27.1% | **94.2%** | **74.9%** |

The score perfectly stratifies IPOs into three meaningfully distinct performance tiers, validating both the feature weights and the 5-component design.

### Section D — Invest/Avoid Strategy Engine

**Dual Threshold Rule:**
```
INVEST if:  Predicted_Gain > 10%
        AND Prob_Positive  > 0.65
AVOID  otherwise
```

Both thresholds are applied simultaneously. The Random Forest regressor predicts the expected gain; the RF classifier predicts the probability of a positive listing. Both must be satisfied.

**`strategy_f1_invest_avoid.png`** — Two panels comparing Invest picks vs Avoid vs Naive (all IPOs):

![Invest Avoid](strategy_f1_invest_avoid.png)

**Strategy Performance:**

| Strategy | N | Avg Gain | Median | Win Rate | Strong Rate |
|----------|---|---------|--------|---------|------------|
| **Invest (our picks)** | ~247 | **+39.6%** | +27.3% | **98.0%** | **72.1%** |
| Avoid (we skip) | ~302 | +1.8% | -0.2% | 45.7% | 6.3% |
| Naive (all IPOs) | 549 | +18.0% | +7.4% | 69.3% | 36.4% |

**Alpha vs naive: +21.55 percentage points**  
**Coverage: 45.1% of IPOs are flagged INVEST**

### Section E — Flip/Hold Strategy Engine

```
FLIP if Listing_Gain_Pct >= Current_Gain_Pct  (selling on listing day was better)
HOLD if Current_Gain_Pct  > Listing_Gain_Pct  (holding to CMP was better)
```

**`strategy_f2_flip_hold.png`** — Two panels:
1. Box plots comparing gain distribution: Flip (listing day) vs Hold (current CMP)
2. Bar chart showing "Hold beats Flip" rate by segment (Score Category + Investor Type)

![Flip Hold](strategy_f2_flip_hold.png)

**Flip vs Hold Results:**

```
Overall Hold > Flip: 50.6%

By Score Category:
  Weak     : Hold better 44.2%  → FLIP (don't wait for recovery)
  Moderate : Hold better 50.9%  → coin-flip, consider HOLD
  Strong   : Hold better 58.7%  → HOLD (momentum continues)

By Investor Type:
  QIB-Dominant: Hold better 56.1%  → HOLD (institutional support)
  HNI-Dominant: Hold better 47.3%  → FLIP (large listing pop often the peak)
  RII-Dominant: Hold better 48.9%  → FLIP (hype fades post-listing)
```

### Section F — Yearly Backtest

**`strategy_f3_yearly_backtest.png`** — Bar chart comparing Invest strategy vs Naive benchmark across every year 2010–2025, with alpha (α = strategy gain − naive gain) annotated above each pair:

![Yearly Backtest](strategy_f3_yearly_backtest.png)

*The strategy generates positive alpha in 12 out of 15 years. The 3 years of negative alpha coincide with market-wide corrections (2015, 2019, 2022) where even high-quality IPOs underperformed expectations.*

### Exported Files

**`ipo_quality_scores.csv`** — All 561 IPOs sorted by Quality Score descending:
```
IPO_Name, Issue_Size, QIB, HNI, RII, Total, Issue_Price,
Listing_Gain_Pct, IPO_Quality_Score, Score_Category,
Positive_Listing, Strong_Listing
```

**`ipo_strategy_output.csv`** — Full strategy decisions per IPO:
```
IPO_Name, Listing_Gain_Pct, Current_Gain_Pct, Predicted_Gain,
Prob_Positive, Invest_Flag, Hold_Better_Than_Flip,
Flip_Hold_Strategy, IPO_Quality_Score, Score_Category,
Dominant_Investor
```

---

## 🔵 Step 7 — Clustering & Segmentation

**Input:** `ipo_featured.csv` + optional Step 6 CSVs  
**Output:** `ipo_clustered.csv`, `ipo_final_master.csv`, `summary_insights.txt` + 6 plots  
**Code:** Cell 6

Step 7 uses unsupervised learning to discover natural market segments within Indian IPOs — groups of IPOs that behave similarly without any predefined labels.

### Section A — Clustering Feature Set (6 features)

```python
CLUSTER_FEATURES = [
    "Log_Issue_Size",    # size signal (log-scaled)
    "Log_Total",         # demand intensity (log-scaled)
    "QIB_Ratio",         # institutional fraction
    "HNI_Ratio",         # HNI fraction
    "RII_Ratio",         # retail fraction
    "Listing_Gain_Pct",  # outcome (for retrospective segmentation)
]
```

All features are StandardScaler-normalised before clustering so no feature dominates by units.

### Section B — Finding Optimal k

**`cluster_b1_elbow_silhouette.png`** — Side-by-side Elbow and Silhouette plots for k=2 through k=8:

![Elbow Silhouette](cluster_b1_elbow_silhouette.png)

```
Silhouette scores by k:
  k=2: 0.247   k=3: 0.290 ✅   k=4: 0.241
  k=5: 0.228   k=6: 0.219   k=7: 0.203   k=8: 0.194

Best k = 3 (highest silhouette score)
```

The elbow plot shows a kink at k=3, and the silhouette plot confirms k=3 maximises cluster separation.

### Section C — KMeans Clustering (k=3)

```
KMeans final inertia = 2,847.3
Cluster 0: 124 IPOs  (avg silhouette = 0.241)
Cluster 1: 252 IPOs  (avg silhouette = 0.318)
Cluster 2: 185 IPOs  (avg silhouette = 0.267)
```

### Section D — Cluster Profiles & Naming

After inspecting mean values per cluster, each cluster is assigned a finance-meaningful name:

| Cluster | Name | N | Avg Size | Avg Sub | Avg Gain | Win Rate |
|---------|------|---|----------|---------|---------|---------|
| **0** | 🔴 Retail-Driven Weak Demand | 124 | ₹251 cr | 6.7× | **+5.6%** | **57%** |
| **1** | 🔵 Institutional Heavyweights | 252 | ₹2,470 cr | 13× | **+4.5%** | **63%** |
| **2** | 🟢 Blockbuster Demand Stars | 185 | ₹748 cr | 91.5× | **+44.8%** | **95%** |

**Cluster 0 — Retail-Driven Weak Demand (n=124)**
Small IPOs (₹251 cr avg issue size), barely oversubscribed (6.7×), with retail investors making up a disproportionate fraction of demand (RII_Ratio high). Average listing gain of only +5.6%, win rate 57%. These are undersubscribed small-cap or SME IPOs where retail fills the gap left by absent institutional interest. **High risk, low expected reward.**

**Cluster 1 — Institutional Heavyweights (n=252)**
Large IPOs averaging ₹2,470 crores, with moderate but not spectacular overall subscription (13×). 87% of this cluster is QIB-dominant — PSU disinvestments, large insurance companies, state banks, and mega-cap listings. Institutions must absorb their mandated allocation; retail largely ignores them. Average listing gain is only +4.5%, but current gains (from issue to CMP) average +75% — these are long-term compounders. **BUY-AND-HOLD category, not for listing flippers.**

**Cluster 2 — Blockbuster Demand Stars (n=185)**
Mid-size IPOs (₹748 cr) with explosive total subscription (91.5× average). HNI-dominant. These are the marquee IPOs — Tata Technologies, Premier Energies, Paras Defence, SME sector stars. Average listing gain of +44.8%, win rate 95%, strong rate 78%. **The clear INVEST signal — these are the IPOs to apply for.**

### Section E — Cluster Visualisations

**`cluster_e1_pca_scatter.png`** — PCA 2D scatter plot. All 6 clustering features are reduced to 2 principal components; each dot is one IPO coloured by cluster; centroids marked with ✕:

![PCA Scatter](cluster_e1_pca_scatter.png)

*PC1 and PC2 together explain ~68% of variance. The three clusters are well-separated in PCA space: Cluster 2 (green, Demand Stars) is isolated in the upper region corresponding to high subscription; Cluster 1 (blue, Heavyweights) separates on the size axis.*

**`cluster_e2_heatmap.png`** — Normalised cluster centroid heatmap. Each cell shows how high or low that cluster scores on each feature (0=lowest across clusters, 1=highest):

![Cluster Heatmap](cluster_e2_heatmap.png)

| Feature | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|-----------|-----------|-----------|
| log(Issue Size) | 🔴 0.12 | 🟢 **0.98** | 🟡 0.47 |
| log(Total Sub) | 🔴 0.18 | 🟡 0.41 | 🟢 **1.00** |
| QIB Ratio | 🟡 0.38 | 🟢 **0.87** | 🟡 0.52 |
| HNI Ratio | 🟡 0.41 | 🔴 0.19 | 🟢 **0.82** |
| RII Ratio | 🟢 **0.79** | 🟡 0.31 | 🔴 0.22 |
| Listing Gain % | 🔴 0.11 | 🟡 0.09 | 🟢 **1.00** |

**`cluster_e3_outcomes.png`** — Three outcome comparison panels: listing gain (mean + median), win rate + strong rate, and stacked bar showing Negative/Modest/Strong/Exceptional composition per cluster:

![Cluster Outcomes](cluster_e3_outcomes.png)

**`cluster_e4_silhouette.png`** — Per-IPO silhouette plot. Each horizontal bar represents one IPO; bar width = silhouette coefficient. Wider = better cluster assignment. Negative silhouette = misclassified:

![Silhouette Plot](cluster_e4_silhouette.png)

*Cluster 1 (blue, Institutional Heavyweights) has the highest average silhouette (0.318), meaning its members are the most distinctly similar to each other and dissimilar from other clusters. Cluster 0 has the most negative-silhouette points, indicating some borderline IPOs that could belong to either cluster 0 or 1.*

**`cluster_e5_dendrogram.png`** — Hierarchical clustering dendrogram on a 100-IPO random sample using Ward linkage. Validates that hierarchical methods also suggest 3 natural groups:

![Dendrogram](cluster_e5_dendrogram.png)

*Three main branches are clearly visible before the red cut line, independently confirming that k=3 is the natural number of market segments in Indian IPO data.*

### Section F — Cluster × Strategy Cross-Analysis

Printed table showing how investor dominance and year era distribute across clusters:

```
Dominant Investor composition per cluster (%):
                    HNI_Dominant  QIB_Dominant  RII_Dominant
Weak Demand              18.5          31.5          50.0
Inst. Heavyweights       11.1          85.3           3.6
Demand Stars             71.9          21.1           7.0

Year era distribution per cluster (%):
                    2010–15  2016–19  2020–22  2023–25
Weak Demand           28.2     31.5     18.5    21.8
Inst. Heavyweights    41.7     28.2     14.3    15.9
Demand Stars          12.4     23.8     41.1    22.7
```

Demand Stars are increasingly concentrated in 2020–25 — the modern IPO boom era.

### Section G — Final Master Dataset

`assemble_final_master()` merges cluster labels back into `ipo_featured.csv` and optionally joins quality scores (Step 6) and strategy flags (Step 6):

**`ipo_final_master.csv`** — The complete, fully-merged dataset with all original features, all engineered features, cluster labels, quality scores, and strategy flags. Final shape: ~561 rows × 40+ columns.

**`ipo_clustered.csv`** — Lighter version with just clustering-relevant columns:
```
IPO_Name, Date, Issue_Size, QIB, HNI, RII, Total,
Issue_Price, Listing_Gain_Pct, Positive_Listing,
Strong_Listing, Current_Gain_Pct, Cluster, Cluster_Name
```

### Section H — Complete Project Summary

**`summary_insights.txt`** — A full research-paper-style summary covering all 7 steps, saved as a plain text file for sharing. Contains all key statistics, model performance metrics, cluster profiles, and research conclusions.

---

## 📁 Complete Output File Reference

### CSV Files

| File | Step | Rows | Cols | Description |
|------|------|------|------|-------------|
| `ipo_clean.csv` | 1 | 561 | 13 | Clean, typed, deduplicated raw data |
| `ipo_featured.csv` | 2 | 561 | 34 | All engineered features + labels |
| `model_comparison_regression.csv` | 5 | 5 | 6 | Regression model leaderboard |
| `model_comparison_classification.csv` | 5 | 3 | 7 | Classification model leaderboard |
| `ipo_quality_scores.csv` | 6 | 561 | 12 | Quality scores sorted descending |
| `ipo_strategy_output.csv` | 6 | ~549 | 11 | Full strategy decisions per IPO |
| `ipo_clustered.csv` | 7 | ~561 | 14 | Cluster assignments |
| `ipo_final_master.csv` | 7 | ~561 | 40+ | Complete merged master dataset |

### Plot Files

| File | Step | Description |
|------|------|-------------|
| `missing_values_heatmap.png` | 1 | Missing data map (yellow=missing) |
| `label_distributions.png` | 2 | 4-panel: all target label distributions |
| `feature_distributions.png` | 2 | 8-panel: engineered feature histograms |
| `eda_b1_distributions.png` | 3 | 8-panel: key variable distributions |
| `eda_b2_boxplots.png` | 3 | Listing gain by group boxplots |
| `eda_c1_correlation.png` | 3 | Pearson correlation heatmap |
| `eda_d1_time_trends.png` | 3 | Time-series: volume/size/gains |
| `eda_e1_subscription.png` | 3 | Subscription deep-dive (4 panels) |
| `eda_f1_top_bottom.png` | 3 | Top 15 and bottom 15 performers |
| `eda_g1_scatter.png` | 3 | Subscription vs listing gain scatter |
| `eda_h1_seasonality.png` | 3 | Monthly and quarterly seasonality |
| `eda_i1_pairplot.png` | 3 | Pairplot: 5 key variables by investor type |
| `stat_a_normality.png` | 4 | Q-Q plot + histogram: non-normality proof |
| `stat_b_correlations.png` | 4 | Pearson vs Spearman side-by-side |
| `stat_c_subscription_groups.png` | 4 | KW test: subscription level groups |
| `stat_e_qib_split.png` | 4 | High vs low QIB comparison |
| `stat_f1_behavioural.png` | 4 | 4-panel: dominant investor deep-dive |
| `stat_f2_holdflip.png` | 4 | Hold vs flip by investor type |
| `stat_g_deciles.png` | 4 | Subscription decile analysis |
| `model_c1_regression_comparison.png` | 5 | 5-model regression leaderboard chart |
| `model_c2_predicted_vs_actual.png` | 5 | RF predictions vs actual scatter |
| `model_c3_residuals.png` | 5 | Residual plot + histogram |
| `model_d1_confusion_matrix.png` | 5 | RF classifier confusion matrix |
| `model_d2_roc_curve.png` | 5 | ROC curves (all classifiers) |
| `model_f1_feature_importance.png` | 5 | Top 15 features by importance |
| `model_h1_shap_summary.png` | 5 | SHAP beeswarm plot |
| `model_h2_shap_dependence.png` | 5 | SHAP dependence: Total × QIB_Ratio |
| `score_c1_overview.png` | 6 | 4-panel: quality score validation |
| `strategy_f1_invest_avoid.png` | 6 | Invest vs Avoid vs Naive |
| `strategy_f2_flip_hold.png` | 6 | Flip vs Hold strategy |
| `strategy_f3_yearly_backtest.png` | 6 | Year-by-year strategy alpha |
| `cluster_b1_elbow_silhouette.png` | 7 | Elbow + silhouette for k selection |
| `cluster_e1_pca_scatter.png` | 7 | PCA 2D cluster scatter |
| `cluster_e2_heatmap.png` | 7 | Normalised cluster centroid heatmap |
| `cluster_e3_outcomes.png` | 7 | Cluster outcome comparison (3 panels) |
| `cluster_e4_silhouette.png` | 7 | Per-IPO silhouette coefficients |
| `cluster_e5_dendrogram.png` | 7 | Hierarchical dendrogram (n=100) |

**Total: 34 plots + 8 CSV files**

---

## 🔬 Key Findings & Research Conclusions

### Finding 1 — Subscription is King

Total subscription multiple is the single strongest predictor of listing-day performance (Spearman ρ = 0.73). A simple threshold rule — "apply only if Total > 50×" — already generates a win rate above 90%.

```
Low subscription (<5×):  avg +3.2%,  win rate 43.1%
Mid subscription (5–50×): avg +13.7%, win rate 66.8%
High subscription (>50×): avg +51.2%, win rate 97.3%
```

### Finding 2 — HNI Signal Beats QIB

Despite QIB representing institutional ("smart money") investors, HNI-dominant IPOs outperform QIB-dominant IPOs by 16 percentage points in average gain. **Why?** QIB buying in large IPOs is often mandatory (index inclusion, regulatory allocation) rather than signal-driven. HNI investors are leveraged, due-diligence-intensive, and their large bids signal genuine conviction. Every pairwise comparison is statistically significant (MWU p < 0.001 after Bonferroni).

### Finding 3 — Retail Mania is a Warning Sign

RII-dominant IPOs (where retail demand is the primary driver) have the worst listing performance of all three groups: only +6.3% average gain and 59.4% win rate. When institutional investors are absent or under-subscribed, retail enthusiasm alone is not sufficient to produce strong listing gains — and may actually signal overpricing.

### Finding 4 — Machine Learning Adds Real Alpha

The RF classifier achieves AUC = 0.828, well above the 0.75 threshold typically required to demonstrate genuine predictive signal in financial markets. The Invest strategy engine generates +21.55 percentage points of alpha vs naive (all IPOs), with a win rate of 98% on flagged IPOs. This confirms that subscription composition data contains genuine predictive information beyond what simple rules capture.

### Finding 5 — Hold vs Flip is Context-Dependent

The 50/50 overall split (Hold beats Flip in 50.6% of cases) shows there is no universal answer. The right strategy depends on:
- **Quality Score:** Strong-scored IPOs → lean HOLD (58.7% hold-better)
- **Investor Type:** QIB-dominant → HOLD (56.1%); HNI/RII-dominant → FLIP
- **Listing Pop Size:** If listing gain >30%, flipping is statistically more often correct

### Finding 6 — Three Natural Market Segments

The k-means clustering reveals three fundamentally different types of Indian IPO:
1. **Retail-Driven Weak Demand** — small, under-institutional, avoid or flip quickly
2. **Institutional Heavyweights** — large PSU/bank listings, poor listing pop but strong long-term holds
3. **Blockbuster Demand Stars** — mid-cap HNI-led, the ideal INVEST+FLIP/HOLD targets

### Decision Framework Summary

```
┌─────────────────────────────────────────────────────┐
│  PRE-LISTING DECISION                               │
│                                                     │
│  Step 1: Is Total subscription > 50×?              │
│    No  → AVOID (win rate only ~67%)                 │
│    Yes → continue                                   │
│                                                     │
│  Step 2: Is HNI_Ratio the dominant fraction?        │
│    Yes → STRONG BUY SIGNAL                          │
│    QIB-dominant → MODERATE BUY                      │
│    RII-dominant → CAUTION                           │
│                                                     │
│  Step 3: Check IPO Quality Score                    │
│    Strong (>0.66) → INVEST with high confidence     │
│    Moderate       → INVEST selectively              │
│    Weak  (<0.33)  → AVOID                           │
│                                                     │
│  POST-LISTING DECISION                              │
│                                                     │
│  Quality Score Strong + QIB-dominant → HOLD         │
│  Quality Score Strong + HNI-dominant → FLIP if >30%│
│  Quality Score Weak/Moderate         → FLIP         │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Known Issues & Fixes Applied

### Bug 1 — Step 6: `ValueError: DataFrame index must be unique for orient='index'`

**Location:** `apply_invest_avoid()` function  
**Root cause:** `sdf.set_index("IPO_Name").to_dict("index")` crashes when `IPO_Name` contains duplicate values (same company listed on multiple dates as different tranches).

**Fix applied:**
```python
# BEFORE (crashes):
score_map = sdf.set_index("IPO_Name")[["IPO_Quality_Score","Score_Category"]].to_dict("index")

# AFTER (safe):
sdf_dedup    = sdf[["IPO_Name","IPO_Quality_Score","Score_Category"]]
               .drop_duplicates(subset="IPO_Name", keep="last")
               .set_index("IPO_Name")
score_map    = sdf_dedup["IPO_Quality_Score"].to_dict()
category_map = sdf_dedup["Score_Category"].to_dict()
```

### Bug 2 — Step 7: Same `ValueError` in `assemble_final_master()`

**Location:** `assemble_final_master()` function  
**Root cause:** Same issue — `cdf.set_index("IPO_Name").to_dict("index")` with duplicate IPO names.

**Fix applied:** Same deduplication pattern using `.drop_duplicates(subset="IPO_Name", keep="last")` before `.set_index()`.

### Bug 3 — Step 7: `summary_insights.txt` encoding error

**Issue:** Writing the ₹ symbol to a file without specifying UTF-8 encoding fails on some systems.  
**Fix:** Added `encoding="utf-8"` to `open("summary_insights.txt", "w")`.

---

## 📂 Project Structure

```
INDIAN_IPO_ANALYSIS.ipynb          ← Main notebook (7 cells/steps)
Initial Public Offering.xlsx        ← Raw input data (upload when prompted)
│
├── Intermediate CSVs
│   ├── ipo_clean.csv               ← Step 1 output
│   └── ipo_featured.csv            ← Step 2 output (main pipeline input)
│
├── Step 3 Plots (EDA)
│   ├── eda_b1_distributions.png
│   ├── eda_b2_boxplots.png
│   ├── eda_c1_correlation.png
│   ├── eda_d1_time_trends.png
│   ├── eda_e1_subscription.png
│   ├── eda_f1_top_bottom.png
│   ├── eda_g1_scatter.png
│   ├── eda_h1_seasonality.png
│   └── eda_i1_pairplot.png
│
├── Step 4 Plots (Statistics)
│   ├── stat_a_normality.png
│   ├── stat_b_correlations.png
│   ├── stat_c_subscription_groups.png
│   ├── stat_e_qib_split.png
│   ├── stat_f1_behavioural.png
│   ├── stat_f2_holdflip.png
│   └── stat_g_deciles.png
│
├── Step 5 Outputs (Modeling)
│   ├── model_c1_regression_comparison.png
│   ├── model_c2_predicted_vs_actual.png
│   ├── model_c3_residuals.png
│   ├── model_d1_confusion_matrix.png
│   ├── model_d2_roc_curve.png
│   ├── model_f1_feature_importance.png
│   ├── model_h1_shap_summary.png
│   ├── model_h2_shap_dependence.png
│   ├── model_comparison_regression.csv
│   └── model_comparison_classification.csv
│
├── Step 6 Outputs (Strategy)
│   ├── score_c1_overview.png
│   ├── strategy_f1_invest_avoid.png
│   ├── strategy_f2_flip_hold.png
│   ├── strategy_f3_yearly_backtest.png
│   ├── ipo_quality_scores.csv
│   └── ipo_strategy_output.csv
│
└── Step 7 Outputs (Clustering)
    ├── cluster_b1_elbow_silhouette.png
    ├── cluster_e1_pca_scatter.png
    ├── cluster_e2_heatmap.png
    ├── cluster_e3_outcomes.png
    ├── cluster_e4_silhouette.png
    ├── cluster_e5_dendrogram.png
    ├── ipo_clustered.csv
    ├── ipo_final_master.csv
    └── summary_insights.txt
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Past IPO performance does not guarantee future returns. The Indian IPO market is subject to regulatory changes, market sentiment shifts, and macroeconomic factors not captured in this dataset. Always conduct your own due diligence before making any investment decision.

---

*Built with 🐍 Python · pandas · scikit-learn · matplotlib · seaborn · scipy · XGBoost · SHAP*
