# 05 — Feature Engineering

> *Algorithms tell you what kind of model you're using. Features tell you what your model can actually see.*

---

## 1. What feature engineering is

**Feature engineering** is the act of *creating new input columns* for your model from the data you already have. A few real examples:

| Raw columns                         | Engineered feature                       | Why it's better                                       |
| ----------------------------------- | ---------------------------------------- | ----------------------------------------------------- |
| `1st_flr_sf`, `2nd_flr_sf`, `total_bsmt_sf` | `total_sf` (sum)                  | One signal of "how big" instead of three pieces.       |
| `year_built`, `yr_sold`             | `home_age = yr_sold - year_built`        | Centered at zero, model-friendly.                      |
| `mo_sold` (integer 1–12)            | `mo_sold_sin`, `mo_sold_cos`             | Encodes that Dec is adjacent to Jan, not 11 units away. |
| `garage_area`                       | `has_garage = garage_area > 0`           | Surfaces the binary "garage / no garage" cleanly.      |
| (no column for it!)                 | `ames_market_index` joined from Zillow   | Adds a market-conditions signal the raw data lacks.    |

Different from cleaning — cleaning *fixes* what's already there. Feature engineering *adds* signals the model could not otherwise see.

> 📘 **Why it matters.** Across countless Kaggle competitions and industry studies, the gap between a beginner's model and a top model has *more to do with features than algorithms*. A boring linear regression with good features routinely beats a sloppy XGBoost on raw columns.

---

## 2. The six families used in this project

Every engineered feature in `src/features.py` falls into one of these groups. Each function in that file lives in exactly one box.

### 2.1 Aggregation features

Collapse a set of correlated columns into one informative total.

```python
df["total_sf"] = df["1st_flr_sf"] + df["2nd_flr_sf"] + df["total_bsmt_sf"]
```

**Why it helps.** A single, monotonically-meaningful column is easier for a linear model and shorter to split on for a tree. The parts were redundant; the sum is one clean signal.

**Examples added in this project:** `total_sf`, `total_bath`, `total_porch_sf`.

### 2.2 Temporal features

Transform raw years/dates into "how long ago?" framings.

```python
df["home_age"] = df["yr_sold"] - df["year_built"]
df["years_since_remodel"] = df["yr_sold"] - df["year_remod_add"]
df["is_remodeled"] = (df["year_remod_add"] != df["year_built"]).astype(int)
df["is_new"] = (df["yr_sold"] == df["year_built"]).astype(int)
```

**Why it helps.** "Home age 13 years" is more interpretable than "built in 1995 and sold in 2008". Linear models especially benefit from features that are *centered* (have a meaningful zero).

### 2.3 Cyclic encoding

For variables that wrap around (months, hours, days of week, compass bearings), encode them as two trigonometric components:

```python
import numpy as np
month_rad = 2 * np.pi * df["mo_sold"] / 12
df["mo_sold_sin"] = np.sin(month_rad)
df["mo_sold_cos"] = np.cos(month_rad)
```

**Why this works.** Each month maps to a point on the unit circle:

```
        Jan (1, 0)
   Oct *   *   *  Apr
    *           *
    *   (0, 0)  *
    *           *
   Jul *   *   * Apr
   (-1, 0)        (0, 1)
```

December's `(cos, sin) ≈ (1, -0.5)` is *close* to January's `(1, 0)` — they're neighbors on the circle. The model now sees that adjacency, where the raw integer encoding hid it.

> 🧠 **Geometric intuition.** The integer encoding lives on a line: 1, 2, 3, …, 12. On a line, December and January are 11 units apart. The sin/cos encoding lives on a circle. On a circle, December and January are *one step* apart, the same as January and February.

### 2.4 Binary "has-it" flags

For features where *existence* matters more than *size*:

```python
df["has_pool"] = (df["pool_area"] > 0).astype(int)
df["has_fireplace"] = (df["fireplaces"] > 0).astype(int)
df["has_basement"] = (df["total_bsmt_sf"] > 0).astype(int)
```

**Why it helps.** A buyer searches "homes with a fireplace" — not "homes with at least one of any size fireplace". The model can have both the *count/area* and the *flag*; it picks which to use per decision.

### 2.5 Target transformation

When the target is heavily skewed, transform it before training:

```python
df["log_saleprice"] = np.log1p(df["saleprice"])
```

At prediction time you undo the transform: `dollars = np.expm1(prediction)`.

**Why it helps.**
- **Linear regression** assumes residuals are normally distributed. With a right-skewed target, residuals on big homes get systematically larger. Log-transforming evens this out.
- **Tree-based models** don't have the same residual assumption, but a compressed target range gives them a more uniform error landscape across the data.
- **Reported errors** in log-space are roughly *percentage errors* in dollar-space, which is often what stakeholders actually care about ("we're off by ~10%", not "we're off by $25k").

> 🧠 **Math note — log1p vs. log.** `log1p(x) = log(1 + x)`. The "+1" lets us safely log values including zero (since `log(0) = -∞`). For `saleprice`, all values are positive, so it's barely different from plain `log`. Convention: use `log1p` unless you're certain there are no zeros.

### 2.6 Cross-dataset features

Join external data sources to enrich each row with context.

```python
df = add_market_index(df, zhvi)
# adds ames_market_index = avg ZHVI for Ames ZIPs at (yr_sold, mo_sold)
```

**Why it helps.** The base Ames data has no measure of *market conditions* at the time of sale. A home sold for $200k in 2007 was the same home worth $180k in 2009 — without a market signal, the model has to learn the whole housing crash from `yr_sold` alone.

The Zillow ZHVI provides exactly that signal: monthly typical home values per ZIP, dating back to 2000. We aggregate Ames-area ZIPs into a single time series and attach the right month's value to each sale.

> ⚠️ **Gotcha — granularity mismatch.** Ames is per-home; Zillow is per-ZIP-per-month. The join is *augmentation*: every home in March 2008 gets the same Ames-wide market index for that month. This is a loss of resolution we accept because Ames doesn't have per-home ZIPs.

---

## 3. The cardinal sin: target leakage

If a feature secretly contains information about the target, your model "predicts" perfectly during training and *catastrophically* on new data. The most insidious flavor of bug in ML.

**Examples of target leakage in real estate features:**

| Feature                                       | Why it leaks                                          |
| --------------------------------------------- | ----------------------------------------------------- |
| `price_per_sqft = saleprice / total_sf`       | Contains the target directly. Model just inverts.    |
| `discount_vs_neighborhood_median`             | Computed using *this row's* saleprice.               |
| `was_above_asking = saleprice > list_price`   | Future information (you don't know at listing time).  |
| Mean SalePrice per Neighborhood (naively)     | Includes the current row's SalePrice in its mean.    |

The "price per square foot" example is the one beginners reach for most often. It looks innocent — divide two numbers you already have — but the result *contains the target*. Any model that sees it learns to multiply back.

### The clean way to do neighborhood-level features

If you really want "average SalePrice in this home's neighborhood" (a useful feature!), do it with **out-of-fold target encoding**: for each row, compute the mean of *every other row in the same neighborhood except this one*. Or compute it on the training set only and apply that lookup table to the test set.

We'll use this pattern carefully in Phase 6 if at all. For now, we avoid any feature derived from the target.

---

## 4. Other pitfalls

- **Multicollinearity.** Adding `total_sf` while keeping the three parts gives you four columns that are perfectly linearly dependent. Tree models shrug; linear models get unstable coefficients. Phase 4 will introduce feature selection / regularization.
- **Date arithmetic.** `home_age` can go *negative* if a sale year predates a build year (data entry quirk). We clamp at zero. Always handle the edge case.
- **One-hot explosion.** A "neighborhood_size" feature with thousands of distinct neighborhoods one-hot encoded creates thousands of columns. For high cardinality, use frequency encoding or target encoding (with leakage guards).
- **Forgetting the test set.** Any aggregation that uses *training row counts or medians* must be applied to test data using *the values learned on training*. Pipelines (Phase 4) automate this.
- **Engineering for the wrong model.** A polynomial feature `total_sf**2` is great for linear regression and irrelevant for trees (they already model nonlinearities by splitting). Match the feature to the algorithm.
- **The "more is always better" fallacy.** Each extra feature is more capacity to overfit and more time to train. Add features that *should* help; cut features that *don't*.

---

## 5. How to evaluate whether a new feature is "working"

Three signals to look at, in order of strength:

1. **Cross-validated model score improves** with the feature included vs. excluded. Gold standard.
2. **Feature importance** (from a tree-based model — Phase 6) ranks the new feature above noise.
3. **Correlation with target** is meaningfully nonzero. Weakest, but the cheapest to compute.

Phase 4 will give us cross-validation and Phase 6 will give us SHAP / permutation importance. In Notebook 03 we only have signal #3 — useful as a sanity check, not as proof.

> 📘 **Rule of thumb.** A feature with `|corr| < 0.05` is unlikely to help. A feature with `|corr| ≥ 0.1` is worth keeping for the model to decide.

---

## 6. Concepts and vocabulary

- **Feature interaction** — when the effect of one feature on the target *depends* on the value of another. E.g., "high quality" matters more in a big house than a small one. Tree models capture interactions automatically; linear models need explicit interaction terms (`quality * sqft`).
- **Polynomial feature** — adding `x²`, `x³`, etc. so linear models can fit curves. Common with linear regression; pointless with trees.
- **Target encoding** — replacing a category with the mean target value for that category. Powerful but leaky if done carelessly.
- **Frequency encoding** — replacing a category with how often it appears. A weaker but leakage-proof alternative to target encoding.
- **Embedding** — a learned dense representation of a high-cardinality category, popular in deep learning. Out of scope for this tabular project.
- **Time-derived features** — day-of-week, is-weekend, days-since-event. The relatives of our `home_age` and `is_new`.
- **Lag features** — for time-series, the value at t-1, t-7, t-30. We'll meet these in Phase 5.

---

## 7. Further reading

- **Casari & Zheng, *Feature Engineering for Machine Learning* (2018, O'Reilly).** The book most ML practitioners point at for tabular feature design.
- **scikit-learn — Preprocessing data:** [scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html)
- **Featuretools** ([featuretools.com](https://www.featuretools.com)) — a Python library that automates a wide class of feature derivations. Worth a look once you understand the manual versions.
- **Kaggle's "Feature Engineering" course** ([kaggle.com/learn/feature-engineering](https://www.kaggle.com/learn/feature-engineering)). Short, practical, free.
- **"A Few Useful Things to Know About Machine Learning" — Pedro Domingos.** Classic short paper. The line *"feature engineering is the key"* lives in it.

---

## 8. Self-test quiz

1. We engineered `total_sf` even though we already have `gr_liv_area`. What's the difference, and why might `total_sf` win for a *linear* model in particular?
2. `home_age` and `year_built` carry the same information. Give one concrete reason a model could still benefit from having *both*.
3. Explain in your own words why integer-encoded months `1–12` are a bad input to a regression model, and what the sin/cos encoding fixes.
4. Define **target leakage** in your own words. Why is `price_per_sqft = saleprice / total_sf` a leakage feature, and what would you do instead to capture "is this home well-priced for its size"?
5. The log-transform of a right-skewed target tends to help both linear and tree models. Why does the *reason* it helps each kind of model differ?
6. The `ames_market_index` was constructed by averaging ZHVI over five Ames ZIPs. What information do we *lose* by averaging, and when would you re-engineer this if the Ames dataset gave you per-home ZIP codes?
7. (Trick question.) Adding a feature *almost always* improves training-set accuracy. Why is training-set accuracy a terrible way to decide whether a feature is "useful"?

<details>
<summary>Show suggested answers</summary>

1. `gr_liv_area` is above-grade (1st + 2nd floor) living area; it excludes basement. `total_sf` adds the basement. For a linear model that assumes "each extra square foot adds roughly the same dollar value", `total_sf` is the more correct *single* signal because basement square footage also adds value. Trees can derive the sum from the parts, but for linear models the direct sum gives them a cleaner coefficient to fit.

2. Tree-based models often split better on the original year columns (e.g., "is `year_built >= 2000`"), while linear models prefer `home_age` because it's centered. Having both lets each algorithm pick what suits it. (Acceptable alternative answer: `is_new` and `is_remodeled` are derived features that capture qualitative facts the raw years don't surface even though they contain the information.)

3. Integer encoding puts months on a line: 1, 2, …, 12. December (12) is then "11 units away" from January (1), but they are actually adjacent calendar months. The sin/cos encoding maps each month onto a point on a unit circle, where December and January are next to each other. A model that uses the two trigonometric components can now see the cyclic adjacency.

4. Target leakage is when a feature secretly contains information about the target that would not be available at prediction time. `price_per_sqft = saleprice / total_sf` includes `saleprice` in the numerator — the model just multiplies by `total_sf` to recover it. To safely capture "is this home well-priced for its size", you'd compute a *neighborhood median price-per-sqft* using only training-set rows, then apply that lookup to the test set; or use out-of-fold target encoding.

5. Linear regression assumes residuals are roughly normal and homoscedastic (equal variance); a right-skewed target produces residuals that grow on big homes, violating both. Log-transforming the target evens out the residual distribution and stabilizes variance. Trees don't have those distributional assumptions — but they benefit because a compressed target range yields more uniform loss across splits, and tree losses (like squared error) on log-space are equivalent to percentage-error losses in dollar-space, which usually matches the business objective.

6. Averaging hides ZIP-level variation. ZIP A may have crashed harder than ZIP B during the recession; we paint them with the same brush. If we had per-home ZIP we'd do a one-to-one lookup (each home gets its own ZIP's market index for that month), preserving the local effect.

7. Adding a feature gives the model more capacity. With enough capacity, the model memorizes the training set, including its noise. So training accuracy goes up whether the feature is useful or just noise. The honest test is held-out (test set or cross-validation) accuracy, where overfitting is penalized. We'll formalize this in Phase 4.

</details>
