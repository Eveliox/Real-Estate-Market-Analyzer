# 03 — Data Cleaning Explained

> *"Real data is dirty data."* Cleaning is not optional, and it's where most ML projects live or die. This doc breaks down every transformation in [`src/preprocessing.py`](../src/preprocessing.py) — what it does, *why* we chose it, and the traps it dodges.

---

## 1. What "cleaning" actually means

In a textbook example, you load a CSV and start modeling. In real life, raw data has all of the following:

| Problem                    | Looks like…                                    | Why it kills models                              |
| -------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| Missing values             | `NaN`, empty cells, `"?"`, `"-"`, `"N/A"`     | Most algorithms refuse to train on NaN.          |
| Wrong types                | A numeric column stored as string             | Math breaks; comparisons are alphabetical.       |
| Inconsistent names         | `"MS SubClass"` vs `"ms_subclass"`             | Joins silently fail; bugs appear later.          |
| Encoded categories         | `Ex`, `Gd`, `TA` strings                       | Models can't compute on strings.                 |
| Outliers                   | A $755k home with 5,642 sq ft                  | Pulls regression lines, inflates loss.           |
| Duplicate rows             | Same home appears twice                        | Over-weights those rows in training.             |
| Identifier columns         | `PID`, `customer_id`, row index               | Models memorize them instead of learning.        |
| Mixed conventions          | Half rows in USD, half in thousands of USD     | Garbage. Mostly silent.                          |

The Ames dataset has several of these (missing values, encoded categories, name inconsistencies, two flavors of identifier columns, a handful of outliers). Walking through it teaches you to spot them anywhere.

> 📘 **Mental model.** Cleaning is like prep cooking. The actual cooking (modeling) is glamorous and fast. The chopping, washing, deboning is 80% of the work and *makes the meal possible*.

---

## 2. The order matters

Our [`clean_ames()`](../src/preprocessing.py) function applies these steps in this exact order:

```
1. standardize_column_names
2. drop_identifier_columns
3. remove_known_outliers       (only if drop_outliers=True)
4. fill_structural_nas
5. impute_missing
6. encode_ordinals
7. encode_nominals             (only if one_hot=True)
```

Each step **depends on the ones above it**:

- Step 2 looks up `"pid"` and `"order"` — those names only exist *after* step 1 standardizes them.
- Step 5 must come *after* step 4. If we imputed before filling structural NAs, the imputer would see `pool_qc = NaN` and replace it with `"TA"` (typical quality), turning every poolless home into "has a typical-quality pool" — a lie.
- Step 7 must come *after* step 6, or one-hot would expand quality columns into 5+ dummy columns each, hiding the ranking the model could otherwise learn from.

> ⚠️ **Pitfall — accidentally cleaning twice.** If you re-run `clean_ames()` on an already-cleaned DataFrame, some steps no-op (good) and some misbehave (bad). Always start from the *raw* DataFrame in your pipeline. The `data/processed/ames_clean.csv` file exists precisely so downstream phases never re-clean.

---

## 3. Standardizing column names

```python
out.columns = (
    out.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("/", "_")
    .str.lower()
)
```

**Why:** "MS SubClass" → `ms_subclass` because:
- Python can't use dot-access on a name with a space: `df.MS SubClass` is a syntax error.
- Joining two DataFrames where one source has "Lot Frontage" and another has "lot_frontage" produces a silent zero-row join. Painful to diagnose.
- It looks better.

**Why this approach and not `df.rename(columns={...})`?** A rename map requires you to enumerate every column. For 82 columns that's tedious and error-prone. A rule that transforms all names uniformly is shorter and self-consistent.

> 📘 **Convention — snake_case.** Most Python communities use `snake_case` (lowercase with underscores). It's not enforced — but matching the convention makes your code feel native to readers.

---

## 4. Dropping identifier columns

```python
IDENTIFIER_COLS = ["pid", "order"]
```

**Why:** `pid` (parcel ID) is a 9-digit tax number unique to each home. Imagine training a model that learns "homes with `pid` starting with `52` cost ~$200k". On the training set this looks like a feature. On *new* data — a home in a different parcel — the model's `pid` lookup is meaningless.

This is the simplest form of **data leakage**: the model exploits a column that *correlates with the target* but isn't *causally related*.

**Beginner trap:** "But it correlates! It must be useful." Correlation in the training set ≠ generalization. The most insidious leakage is the kind that *looks great* during training and falls apart on new data. Drop IDs early, with prejudice.

---

## 5. Three kinds of missing values

A blanket "fill all NaN with the mean" is **wrong** in subtle ways. Missingness has flavors:

| Kind                                          | What it means                              | Right treatment                                       |
| --------------------------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| **MCAR** — Missing Completely At Random       | No pattern. Random hiccup.                 | Simple imputation (median/mode).                      |
| **MAR** — Missing At Random (given other cols) | Depends on observed features              | Model-based imputation (predict from other columns).  |
| **MNAR** — Missing Not At Random              | Missingness *itself* carries information   | Replace with a sentinel category or add a flag column. |

In Ames:

- `pool_qc = NaN` → **MNAR**. The home has *no pool*. We replace with the string `"None"`. The model will learn that homes coded `pool_qc=None` are different from homes with a real pool quality.
- `lot_frontage = NaN` → **MCAR / MAR** (probably MAR — irregular lots are slightly more likely to be missing). We impute with the median.

This split is what `fill_structural_nas` (handles MNAR) and `impute_missing` (handles MCAR/MAR) accomplish — in that order.

### Why median, not mean?

For `lot_frontage`, the median is **robust to outliers**. One unusually huge lot would pull the mean up, biasing every imputed cell. The median is the middle value — unaffected by tail values.

> 🧠 **Math note.** For a perfectly symmetric distribution, mean = median. For a right-skewed distribution (most real-world prices, areas, incomes), mean > median because the long tail pulls the mean. Use median unless you have a specific reason not to.

### Why mode for categoricals?

For a categorical, "central tendency" means "the most common value". If 75% of homes have `electrical = "SBrkr"`, imputing missing ones with `"SBrkr"` is the lowest-information-loss choice.

---

## 6. Outlier handling — guided by domain, not statistics

Naive outlier removal: drop anything more than 3 standard deviations from the mean.

**Why we don't do this:** Real housing data has fat tails. A $750k home in Ames is *not* noise — it's a real $750k home. Cutting it removes signal you genuinely want the model to learn.

**What we do instead:** Apply the *specific* outliers the dataset's author flagged:

> "I would recommend removing any houses with more than 4000 square feet from the data set… these may have undue influence on regression results."
> — De Cock, 2011

Only 5 rows match. Two are partial sales (parents transferring to children — not market deals), two are oddly-cheap mansions whose listing details suggest data-entry errors. Removing them is *domain-justified*, not statistics-justified.

> 📘 **Rule of thumb.** Before removing a row, you should be able to write one sentence explaining *why this row is unrepresentative of what we're trying to predict*. If you can't, leave it in.

---

## 7. Ordinal encoding — preserving the rank

`kitchen_qual`: `Ex > Gd > TA > Fa > Po`. Five categories, but they have a clear order. If we one-hot encode this, we get 5 binary columns and the model has to *learn* that "Ex" is better than "Gd" via the data — wasting capacity.

Manual mapping:

```python
QUALITY_SCALE = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
df["kitchen_qual"] = df["kitchen_qual"].map(QUALITY_SCALE)
```

Now `kitchen_qual` is a single integer column, and the model gets the ranking *for free*. A tree splits on "kitchen_qual ≥ 4" with one operation.

> ⚠️ **Pitfall — assuming order from numbers.** `month` is integer-valued 1–12, but it is **not** ordinal in the usual sense — December and January are *adjacent* (cyclic), not far apart. Encoding cyclic values as integers gives the model the wrong topology. We'll fix `mo_sold` in Phase 3 with sin/cos encoding.

### What about the gaps between values?

Mapping {Po:1, Fa:2, TA:3, Gd:4, Ex:5} implicitly says "Excellent is 5x as good as Poor, and the gap from TA to Gd equals the gap from Gd to Ex". That's not strictly true — it's a simplification. In practice it works because:
1. Tree-based models don't care about the spacing, only the *order*.
2. Linear models care, but the simplification rarely hurts much and is the standard treatment.

If you wanted to be precise: use a "target-encoded" mean price per category, or learn a smooth monotonic transformation. These are advanced; the simple integer map is the right starting point.

---

## 8. Nominal encoding — one-hot

`neighborhood`: 28 distinct neighborhood names with **no inherent order**. Crawford and NAmes are different, not greater/lesser.

One-hot encoding turns one column into N indicator columns:

```
Before:                   After:
neighborhood              neighborhood_Crawfor   neighborhood_NAmes   ...
"Crawfor"                 1                      0
"NAmes"                   0                      1
"Crawfor"                 1                      0
```

`drop_first=True` removes one category per encoded column. Why? Because if you keep all 28 dummies, one is a perfect linear combination of the other 27 ("if all others are 0, this one is 1"). Linear regression algorithms with that *perfect multicollinearity* fail or produce unstable coefficients (the "dummy variable trap"). Tree models are unaffected — they don't care. Dropping the first is the safe default for either.

> 📘 **Cardinality watch.** One-hot is great for `neighborhood` (28 categories). It's awful for `customer_id` (100,000 categories — your DataFrame explodes). For high-cardinality columns, look up "target encoding" or "frequency encoding" — those topics live in Phase 5+ territory.

---

## 9. Why we will redo this in Phase 4 (sklearn Pipelines)

This file does cleaning correctly *if you only ever do it once*. But in Phase 4 we'll do train/test splits, and the rule is:

> **Compute imputation values (medians, modes) on the TRAINING set only.** Then apply *those* values to the test set. If you compute them on the full dataset, your test set is no longer an honest assessment — its values influenced the imputer.

This is called **data leakage through preprocessing**, and it's by far the most common silent bug in beginner ML projects.

The fix: wrap preprocessing in a sklearn `Pipeline`. The pipeline `.fit()` on training data learns the medians; `.transform()` on test data *applies* them. We'll switch over in Phase 4 — the function-based version in this phase exists to make the *steps* visible. The pipeline version will make the *correctness* automatic.

---

## 10. Common pitfalls (collected)

1. **Imputing then splitting.** Always split first, then fit imputer on train only.
2. **Dropping NaN rows blindly.** `df.dropna()` deletes any row with *any* missing value. On Ames that wipes 50%+ of the dataset.
3. **Using LabelEncoder on multiple categorical columns.** `sklearn.preprocessing.LabelEncoder` is designed for the *target* variable, not features. For features, use `OrdinalEncoder` (for ordinals) or `OneHotEncoder` / `pd.get_dummies` (for nominals).
4. **Forgetting to one-hot the test set the same way.** If train has `neighborhood_Edwards` and test doesn't, you'll get a column-count mismatch at predict time. Pipelines or `pd.get_dummies(columns=..., dummy_na=...)` with consistent column lists fix this.
5. **Scaling before split.** Same trap as imputation. Scalers learn min/max or mean/std from data — they must learn it from *train only*.
6. **Encoding cyclic values as integers.** Months, hours, days of week — use sin/cos encoding.
7. **Re-cleaning already-clean data.** Read from `data/processed/ames_clean.csv` downstream, not from raw.

---

## 11. Further reading

- **scikit-learn — Imputing missing values:** [scikit-learn.org/stable/modules/impute.html](https://scikit-learn.org/stable/modules/impute.html)
- **scikit-learn — Encoding categorical features:** [scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
- **Dean De Cock's original Ames paper** (PDF): [http://jse.amstat.org/v19n3/decock.pdf](http://jse.amstat.org/v19n3/decock.pdf) — read at least the section on data anomalies and outliers.
- **"Tidy Data" by Hadley Wickham** (the philosophy of clean tabular data): [vita.had.co.nz/papers/tidy-data.pdf](http://vita.had.co.nz/papers/tidy-data.pdf)
- **"On the Stability of Feature Selection"** — for when you start worrying about imputation choices affecting model interpretation.

---

## 12. Self-test quiz

1. We standardize column names *first*. What goes wrong if we try to drop the column `"PID"` (uppercase, with no underscore) after applying our standardizer?
2. Explain in your own words the difference between *structural* missingness (MNAR) and *true* missingness (MCAR/MAR), with an Ames example of each.
3. Why is the **median** preferred to the **mean** for imputing right-skewed numeric columns?
4. We drop only 5 outliers, even though dozens of homes are >3 standard deviations from the mean SalePrice. Why are those 5 special and the rest not?
5. `kitchen_qual` ends up as a single integer column. `neighborhood` ends up as ~27 binary columns. Why the asymmetry? What property of each column drives the choice?
6. What would have gone wrong if we ran `impute_missing` *before* `fill_structural_nas`?
7. (Looking ahead.) Why is the function-based cleaner in this file *not* safe to use the same way once we have a train/test split?

<details>
<summary>Show suggested answers</summary>

1. Our standardizer produces `"pid"` (lowercase). Trying to drop `"PID"` (uppercase) raises a KeyError, because that name no longer exists in the DataFrame.
2. Structural missingness means "no such attribute exists" — e.g., `pool_qc = NaN` means the home has no pool. True missingness means "the attribute exists, we just don't know its value" — e.g., `lot_frontage = NaN` for a few rows where the data collector didn't measure it. The first should be filled with a meaningful sentinel ("None"); the second should be imputed.
3. The mean is sensitive to outliers — one tail value can shift it noticeably. The median is the 50th percentile and is unaffected by extreme values. For right-skewed columns (prices, areas, incomes), the median is a more representative "central" value.
4. De Cock specifically flagged those 5 in the dataset paper as partial sales / data-entry oddities — they aren't representative of arms-length market transactions. The other tail observations are real, expensive homes; removing them would discard signal the model should learn.
5. `kitchen_qual` is *ordinal* — its values have a meaningful order (Ex > Gd > TA > Fa > Po), so we map them to ranked integers and let the model use the ordering. `neighborhood` is *nominal* — Crawfor isn't "more" than NAmes, just different — so we one-hot encode and let the model learn an independent coefficient per neighborhood.
6. `impute_missing` would see e.g. `pool_qc = NaN` and replace it with the modal pool quality. Every home without a pool would then look like it had a "typical" pool, and the "no pool" signal would vanish entirely. Filling structural NAs first preserves that signal.
7. The function uses `df.median()`/`df.mode()` computed across the whole DataFrame. If we feed it the full dataset and then split into train/test, the test set's values influenced the imputer — that's **data leakage**. The sklearn `Pipeline` version we'll build in Phase 4 fixes this by `.fit()`-ing on train only and `.transform()`-ing the test set with the learned values.

</details>
