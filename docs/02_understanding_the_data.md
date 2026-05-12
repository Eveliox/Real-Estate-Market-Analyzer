# 02 — Understanding the Data

> *Before you clean, model, or visualize anything, you need to know exactly what each column means and what kinds of values it can take. This doc is the map.*

---

## 1. What we have

This project uses **two complementary datasets**:

| Dataset           | Granularity                   | Span              | Used in              |
| ----------------- | ----------------------------- | ----------------- | -------------------- |
| **Ames Housing**  | One row per *home sale*       | 2006–2010 (Ames, IA) | Phases 2–4, 6 — regression |
| **Zillow ZHVI**   | One row per *US ZIP code*, one column per *month* | 2000–today (US-wide) | Phases 5, 7 — forecasting + heatmap |

They are **different shapes for different jobs**. Ames is *wide and rich* per home (~80 columns of detail) which makes it perfect for learning regression. Zillow is *aggregate and longitudinal* per ZIP, which is what you need for forecasting trends and drawing maps.

---

## 2. Why these two specifically?

We considered alternatives:

| Dataset                          | Pros                                       | Cons (for this project)                     |
| -------------------------------- | ------------------------------------------ | ------------------------------------------- |
| **Ames** ✅                       | Gold-standard learning, ~80 features, public, well-documented | No lat/lon, no time series                  |
| **Zillow ZHVI** ✅                | Long monthly history, every US ZIP, free   | Aggregate (no per-home features)            |
| Kaggle "House Prices" competition | Same as Ames (it IS Ames)                   | The competition truncates rows; we use the original. |
| Realtor.com USA Listings         | Has lat/lon and current prices             | Snapshot only, weaker feature set            |
| Boston Housing                   | Famous                                     | Tiny (506 rows), ethically controversial (used a race-coded feature) — actively retired from sklearn. |
| Redfin Data Center               | Detailed metro stats                       | Less ZIP-level depth than Zillow            |

The hybrid gives us both *richness* and *time/geography* without paying for any data.

> 📘 **Why we don't use Boston Housing.** It used to be the default. It was removed from scikit-learn 1.2+ because one of its features (`B`) is a transformation of the racial composition of a neighborhood, with no ethically defensible role in a price model. We treat Ames as the modern replacement.

---

## 3. Where the files live

After running `python -m src.data_loader`:

```
data/raw/
├── ames_housing.tsv               # Ames (tab-separated values)
├── ames_data_dictionary.txt       # Full column-by-column definitions from De Cock
└── zillow_zhvi_zip_monthly.csv    # Zillow ZHVI (CSV)
```

The dictionary file is part of the deliverable — it is the **authoritative description** of every Ames column, including the meaning of every coded value. **Open it.**

---

## 4. Ames — column groups

The ~82 columns of Ames fall into logical groups. You don't need to memorize them all; you need to know *what kinds of columns exist* so you can plan preprocessing.

| Group                | Examples                                                | Type        | Notes |
| -------------------- | ------------------------------------------------------- | ----------- | ----- |
| **Identifiers**      | `PID`, `Order`                                          | Numeric (don't model!) | These uniquely tag a row. **Drop before training** or the model will "predict" using a tax ID. |
| **Location**         | `Neighborhood`, `MS Zoning`                              | Categorical | Strong predictors. `Neighborhood` alone explains a lot of variance. |
| **Lot**              | `Lot Area`, `Lot Shape`, `Lot Frontage`                  | Mixed       | `Lot Frontage` is often missing — typical real-world annoyance. |
| **Building type**    | `Bldg Type`, `House Style`, `Year Built`, `Year Remod/Add` | Mixed | `Year Built` is more useful as `age_at_sale = YrSold - YearBuilt` (we'll engineer this in Phase 3). |
| **Quality ratings**  | `Overall Qual`, `Overall Cond`, `Kitchen Qual`           | Ordinal (encoded as numbers 1-10 or "Ex", "Gd", "TA", "Fa", "Po") | Ordinal means *ordered* — Ex > Gd > TA. Don't one-hot-encode these. |
| **Sizes**            | `Gr Liv Area`, `1st Flr SF`, `2nd Flr SF`, `Total Bsmt SF` | Numeric | Strong predictors. Watch for outliers (huge homes). |
| **Rooms & baths**    | `Bedroom AbvGr`, `Full Bath`, `Half Bath`, `Kitchen AbvGr` | Numeric | Often integer counts. |
| **Garage**           | `Garage Type`, `Garage Yr Blt`, `Garage Cars`, `Garage Area` | Mixed | If `Garage Type` is NA, all garage columns are NA — they're *structurally* missing (no garage). |
| **Basement**         | `Bsmt Qual`, `Bsmt Cond`, `BsmtFin SF 1`, `Total Bsmt SF` | Mixed   | Same structural-missingness story. |
| **Sale info**        | `Mo Sold`, `Yr Sold`, `Sale Type`, `Sale Condition`       | Mixed       | `Sale Condition = Abnorml` flags non-arms-length sales — usually filtered out. |
| **Target**           | `SalePrice`                                              | Numeric (USD) | What we predict in Phase 4. |

> ⚠️ **Beginner pitfall: ordinal vs. nominal.** A column like `Kitchen Qual` with values `Ex > Gd > TA > Fa > Po` is **ordinal** — the order matters. A column like `Neighborhood` is **nominal** — the categories are unordered. They need *different* encoding strategies. We'll cover this in [03_data_cleaning_explained.md](03_data_cleaning_explained.md).

---

## 5. Ames — the meaning of "NA"

A peculiarity worth flagging: in the Ames dataset, many `NA` values mean *"this feature does not apply"* rather than *"unknown"*.

| Column           | What `NA` actually means                  |
| ---------------- | ------------------------------------------ |
| `PoolQC`         | Home has no pool                           |
| `Fireplace Qu`   | Home has no fireplace                      |
| `Garage Type`    | Home has no garage                         |
| `Bsmt Qual`      | Home has no basement                       |
| `Misc Feature`   | No miscellaneous feature (e.g., elevator)  |
| `Lot Frontage`   | Truly unknown (not "no frontage")          |
| `Mas Vnr Type`   | Truly unknown — or no masonry, the dictionary is fuzzy |

The decision rule:

- **Structural NA** ("no pool") → replace with the string `"None"` (a meaningful category).
- **True missing** ("we don't know the lot frontage") → impute (e.g., median for numeric, mode for categorical).

If you do the same imputation for both, you erase a real signal: "has a pool" is itself a feature, and you just replaced it with the median pool quality.

---

## 6. Zillow ZHVI — anatomy

ZHVI = **Zillow Home Value Index**: a smoothed estimate of the *typical* (33rd–67th percentile) home value in a region, for each month.

```
ZIP-level CSV layout:
┌──────────┬───────────┬──────────┬─────────┬──────────┬──────────┬─────┬───────────┐
│ RegionID │ RegionName│ RegionType│ State  │ City     │ Metro    │ ... │ 2000-01-31│ 2000-02-29 │ ... │ 2024-12-31 │
├──────────┼───────────┼───────────┼────────┼──────────┼──────────┼─────┼───────────┼────────────┼─────┼────────────┤
│  61148   │   10001   │   zip     │   NY   │ New York │ NY-NJ-PA │     │  255100   │  255800    │     │   ...      │
└──────────┴───────────┴───────────┴────────┴──────────┴──────────┴─────┴───────────┴────────────┴─────┴────────────┘
```

Each *row* = one ZIP. Each *date column* = a monthly value (USD) for that ZIP. This is called **wide format**.

For most analysis we'll **reshape it to long format**:

```
RegionName | date       | zhvi_usd
10001      | 2000-01-31 | 255100
10001      | 2000-02-29 | 255800
...
```

We'll handle that in Phase 5 with `pd.melt(...)`.

### Important caveats about ZHVI

- **It's a smoothed index**, not a transaction price. Two homes in the same ZIP that sold last month don't appear; only the modeled trend does.
- **Older months are revised** over time as Zillow's models improve. Re-downloading the file changes historical numbers slightly. We'll snapshot ours.
- **Not every ZIP has every month**. Brand-new ZIPs have NaN before they started reporting. Always check `dropna()` before plotting.

---

## 7. Why we *can* join these two datasets

They both contain ZIP-level information (Ames implicitly via Neighborhood → ZIP lookup; Zillow explicitly). But the granularity differs:

- Ames is per-home, snapshot 2006–2010.
- Zillow is per-ZIP, monthly, 2000–today.

So joining is **augmentation**, not row-by-row alignment: we attach the *prevailing typical home value in this ZIP at the time of sale* to each Ames row as an extra feature. That gives the regression model a sense of "market conditions" beyond the home itself.

We'll formalize this join in Phase 3 (Feature Engineering).

---

## 8. Data types — a primer

The columns you'll see fall into a few statistical types. Each gets different treatment.

| Type        | What it is                       | Example                  | Encoding for ML           |
| ----------- | -------------------------------- | ------------------------ | ------------------------- |
| **Numeric (continuous)** | Real number, infinite values | `Lot Area`, `SalePrice` | Use directly (sometimes scale). |
| **Numeric (count)**      | Non-negative integer        | `Bedroom AbvGr`         | Use directly. |
| **Ordinal**              | Categories with a meaningful order | `Overall Qual` (1–10), `Kitchen Qual` (Ex/Gd/...) | Map to integers preserving order. |
| **Nominal categorical**  | Categories with no order    | `Neighborhood`, `Bldg Type` | One-hot encode or target-encode. |
| **Datetime**             | A point in time              | `Yr Sold`, `Mo Sold`     | Decompose to year/month/dayofweek, or compute deltas. |
| **Identifier**           | Unique tag, no semantic meaning | `PID`, `Order`          | **Drop** before modeling. |

Phase 2 will codify these decisions for every Ames column.

---

## 9. Missingness — kinds and treatments

Three flavors of missing data:

| Kind                       | What it means                                  | Example                                | Treatment                       |
| -------------------------- | ---------------------------------------------- | -------------------------------------- | ------------------------------- |
| **MCAR** (missing completely at random) | No pattern — bug or random dropout | A few `Garage Yr Blt` values randomly NA | Median/mode imputation is fine. |
| **MAR** (missing at random, given other columns) | Depends on observed features        | `Lot Frontage` more often missing for irregular lots | Model-based imputation (use other columns to predict). |
| **MNAR** (missing not at random)   | Missingness *itself* carries info             | `PoolQC = NA` means "no pool"        | Add a flag column OR replace with a sentinel category. |

We will treat Ames `NA` values as MNAR by default unless the dictionary tells us otherwise.

---

## 10. Common pitfalls when first meeting data

- **Trusting the column name.** `1st Flr SF` could mean "first floor square footage" (it does), but you don't *know* that until you check the dictionary or its distribution. Always verify.
- **Assuming numeric means continuous.** `Mo Sold` is stored as 1–12 but is **categorical/cyclic**, not a quantity. Treating it as continuous tells the model "December (12) is 11 more than January (1)", which is nonsense for months.
- **Skipping the dictionary.** With Ames, the data dictionary is your most important file. Open it. Search it. Quote it.
- **Joining at the wrong granularity.** Ames-per-home joined to Zillow-per-ZIP-per-month means *every Ames row picks a single Zillow point*. You'd never want to *aggregate Zillow up to per-home* — that would invert the join.
- **Mixing snapshot and historical thinking.** Ames is a snapshot 2006-2010. Predictions on Ames are *not* "what will this house sell for in 2026" — they're "what would this house have sold for in 2006-2010 given these features". Phase 5 (forecasting on Zillow) is where we get to talk about the future.

---

## 11. Further reading

- **Ames dataset paper** — Dean De Cock, *"Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project"*, Journal of Statistics Education, 2011. (URL is in `src/data_loader.py`.)
- **Ames data dictionary** — `data/raw/ames_data_dictionary.txt` after Phase 1 runs.
- **Zillow Research methodology** — [zillow.com/research/data/](https://www.zillow.com/research/data/) (read the ZHVI methodology PDF if you want to know exactly what is being smoothed).
- **pandas dtypes overview** — [pandas.pydata.org/docs/user_guide/basics.html#dtypes](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes).

---

## 12. Self-test quiz

1. Why does Ames provide a **data dictionary**, and what's an example of a question only the dictionary can answer?
2. In Ames, what is the difference between an **ordinal** categorical (like `Kitchen Qual`) and a **nominal** categorical (like `Neighborhood`)? Why does it matter for encoding?
3. The Zillow ZHVI file is in **wide format**. Describe in one sentence what its shape looks like, and what "long format" would look like instead.
4. In Ames, `PoolQC = NA` does *not* mean "missing — please impute". What does it mean, and what's the right way to handle it?
5. Why would `Mo Sold` (month sold) be a bad column to feed to a model **as a raw integer 1–12**?
6. The two datasets are at different *granularities*. When we join them in Phase 3, what does each Ames row pick up from Zillow — one number per home, or many?

<details>
<summary>Show suggested answers</summary>

1. The dictionary defines the meaning of each column and the meaning of each coded value (e.g., `Ex/Gd/TA/Fa/Po` for quality). Example: it tells you that `Bsmt Qual = NA` means "no basement", not "unknown".
2. Ordinal has a meaningful order (Ex > Gd > TA > Fa > Po); nominal doesn't (Crawfor and NAmes aren't "more" than each other). Ordinals should be mapped to ordered integers preserving the rank; nominals should be one-hot or target-encoded so the model doesn't infer a fake order.
3. Wide: one row per ZIP, one column per month. Long: one row per (ZIP, month) pair, with a single `value` column. Long is friendlier for time-series math and for grouping.
4. It means the home has no pool. Replace `NA` with a meaningful sentinel like `"None"`, or add a binary `has_pool` flag — don't impute it as the "typical" pool quality.
5. Because the integer encoding implies December (12) is *11 more than* January (1). Month is *cyclic*: December and January are adjacent. Better: convert to two columns `sin(2π·month/12)`, `cos(2π·month/12)` or one-hot encode.
6. Each Ames row picks up **one** Zillow value: the typical home value in that ZIP at the time the home sold. The join collapses Zillow's time dimension down to a single per-row number based on `Yr Sold` / `Mo Sold`.

</details>
