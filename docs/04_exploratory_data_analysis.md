# 04 — Exploratory Data Analysis (EDA)

> *EDA is the conversation you have with your data before you ask it to predict the future. Skip it and your model will surprise you in painful ways.*

---

## 1. What EDA is, and why it comes before modeling

**Exploratory Data Analysis** is the practice of summarizing, visualizing, and questioning a dataset *before* you train any model on it. The term was coined by statistician John Tukey in 1977. The goal is to **build intuition**:

- What does the *target* look like? Symmetric, skewed, bimodal?
- Which *features* look related to the target? Strongly? Linearly?
- Are there *patterns of missingness*, *outliers*, or *clusters* you didn't expect?
- Are there *redundancies* — features that say the same thing twice?

EDA is the source of the questions that drive feature engineering (Phase 3) and model choice (Phase 4). Skipping it means modeling blind.

> 📘 **Tukey's framing.** *"Exploratory data analysis is an attitude, a state of flexibility, a willingness to look for those things that we believe are not there, as well as for those things we believe might be there."*

---

## 2. The four pillars

Every EDA pass touches four kinds of question:

| Pillar          | Question it answers                            | Tools in `eda.py`                       |
| --------------- | ---------------------------------------------- | --------------------------------------- |
| **Univariate**  | What does each column look like on its own?    | `numeric_summary`, `plot_distribution`  |
| **Bivariate**   | How does each feature relate to the target?    | `plot_numeric_vs_target`, `plot_categorical_vs_target` |
| **Multivariate**| Which features are redundant with each other?  | `plot_correlation_heatmap`, `top_correlations` |
| **Missingness** | Where are the gaps, and is there a pattern?    | `missingness_table`                      |

You usually go top to bottom: target distribution → feature distributions → feature/target relationships → feature/feature relationships.

---

## 3. Univariate analysis

### Histograms and KDEs

A **histogram** bins the data and counts how many values fall in each bin. A **Kernel Density Estimate (KDE)** is a smooth curve over the same data that gives you the shape without bin choices.

```python
eda.plot_distribution(df["saleprice"], log=True)
```

Reading a histogram, ask:

| Shape                | What it tells you                                  | Action               |
| -------------------- | -------------------------------------------------- | -------------------- |
| Symmetric bell       | Roughly normal — most models love this.            | Use as-is.           |
| Right-skewed tail    | Long upper tail (a few very big values).           | Consider log/Box-Cox transform. |
| Left-skewed tail     | Long lower tail. Less common in price data.        | Consider reflect + log. |
| Bimodal (two peaks)  | Two underlying populations mixed together.        | Often you should split & model separately. |
| Spikes at zero       | A "zero-inflated" distribution.                    | Add an "is zero" flag feature. |
| Plateau / uniform    | No central tendency — likely a categorical you forgot to encode. | Re-check the column type. |

### Skewness — a number for the shape

Skewness measures asymmetry:

- `skew ≈ 0`  → symmetric (normal-ish)
- `skew > 1`  → strongly right-skewed
- `skew < -1` → strongly left-skewed

For `saleprice` on Ames, `skew ≈ 1.7` raw — clearly right-skewed. After `log1p`, it drops to roughly 0.1 — almost normal. This is why **log-transforming a right-skewed target often makes regression easier**: linear models assume normally distributed residuals, and trees benefit from a more compressed range.

> 🧠 **Math note — log1p.** `log1p(x) = log(1 + x)`. The `+1` lets you safely log values that include zero (which `log(0) = -∞` would break). For SalePrice, every value is positive, so the difference between `log` and `log1p` is negligible. Convention: use `log1p` for "log-like" transforms unless you know zeros are impossible.

---

## 4. Bivariate analysis

### Numeric vs. numeric (scatter + regression line)

```python
eda.plot_numeric_vs_target(df, "gr_liv_area", target="saleprice", log_y=True)
```

What to look for:

1. **Direction** — does the line go up, down, or flat?
2. **Tightness** — points hugging the line means strong relationship; cloud means weak.
3. **Curvature** — if the cloud bends, the relationship is nonlinear. Linear regression will struggle; tree-based models will not.
4. **Heteroskedasticity** — fancy word for "more spread in some regions than others". Visible as a cone-shaped scatter. Often log-transforming the target fixes it.
5. **Outliers** — points way off the trend, especially if domain-suspicious.

### Numeric vs. categorical (box plots)

```python
eda.plot_categorical_vs_target(df, feature="overall_qual", target="saleprice")
```

A **box plot** for each category shows:

```
   ┌────┐
   │    │  ← upper quartile (75th percentile)
   │    │
   │ ── │  ← median (50th percentile)
   │    │
   │    │  ← lower quartile (25th percentile)
   └────┘
   │
   ↓
  whiskers extend to non-outlier extremes
  
  • outliers (beyond 1.5 × IQR)
```

What to look for:

- **Median ladder** — do the medians rise/fall monotonically across categories? That's signal.
- **Box overlap** — if the boxes for two categories overlap heavily, that feature is *not* discriminating between them.
- **Spread per category** — wider boxes mean more within-category variance; the feature only explains *some* of the variation.

> ⚠️ **Pitfall — alphabetical ordering.** By default pandas/seaborn order categories alphabetically. For an ordinal-ish feature (or one with many categories) that buries the trend. `eda.plot_categorical_vs_target` orders by median to make the ranking visible.

---

## 5. Multivariate analysis — correlation

### Pearson correlation

The **Pearson correlation coefficient** *r* measures the *linear* relationship between two numeric variables:

- `r =  1`  → perfect positive linear
- `r =  0`  → no linear relationship
- `r = -1`  → perfect negative linear

Pearson's *r* is the most common "do these move together?" number you'll see. Caveats:

1. It only measures **linear** relationships. Two variables can be perfectly *related* (e.g., y = x²) and have Pearson *r* near zero.
2. It's sensitive to outliers. One extreme point can move *r* dramatically.
3. **Correlation ≠ causation.** Ice cream sales and drownings are positively correlated. Neither causes the other; summer causes both.

### Spearman correlation

**Spearman's *ρ*** uses *ranks* instead of raw values. It measures *monotonic* relationships (does y always go up when x goes up, even if not linearly?). For nonlinear-but-ordered features (most quality ratings), Spearman often catches signal Pearson misses.

`top_correlations(df, method="spearman")` swaps the math. Try both in the notebook.

### Correlation heatmap — feature-vs-feature

The heatmap is **not** there to find good features (the per-feature-vs-target column does that). Its real job: spot **multicollinearity**. Two features highly correlated with each other AND with the target are saying the same thing twice:

- For *linear* models, multicollinearity makes coefficients unstable (huge variance). They're hard to interpret and may even flip signs depending on the train split.
- For *tree-based* models, it's harmless — trees just pick whichever feature splits first and ignore the redundant one (which then *looks* unimportant in feature-importance plots, even though it's perfectly informative).

For Ames you'll see e.g. `garage_cars` and `garage_area` near +0.9 to each other. They measure the same thing in different units. Phase 3 may collapse them into one feature.

> 📘 **Quick mental model for *r*:**
> - `|r| > 0.7` strong
> - `0.3 < |r| < 0.7` moderate
> - `|r| < 0.3` weak (but not necessarily useless — context matters)

---

## 6. Reading findings into decisions

Every chart you make should produce a decision (or a question to investigate). Some examples for Ames:

| Finding                                                                            | Decision / question                                                               |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `saleprice` right-skewed (skew ≈ 1.7).                                             | In Phase 3, train models on `log1p(saleprice)` and exponentiate predictions back. |
| `overall_qual` has Pearson *r ≈ 0.80* with `saleprice`.                            | Strong predictor — keep it, definitely.                                            |
| `gr_liv_area` and `1st_flr_sf + 2nd_flr_sf` are perfectly correlated.              | Redundant. Drop one, or engineer a `total_living_area = 1st + 2nd + bsmt`.        |
| `garage_cars` and `garage_area` near +0.9.                                          | Redundant for linear models. Either drop one, or model with trees.                 |
| Several quality columns are heavy on `TA` (typical).                               | Low variance → low information. Watch in feature importance later.                |
| `mo_sold` correlation with target ≈ 0.                                              | Don't trust this — month is cyclic. Re-encode with sin/cos in Phase 3 and re-check. |

The discipline is: *every chart causes either a feature engineering idea or a "leave it alone" decision*. Charts without consequences waste time.

---

## 7. EDA on the Zillow time series

For Zillow ZHVI, the framing flips:

- The "rows" are ZIPs, not transactions.
- The "columns" are dates, not features.
- The right first plot is a **line chart of one ZIP's monthly value over time**, not a histogram.

We touched this in Notebook 01 (the ZIP 50010 plot). The deeper time-series-specific EDA (autocorrelation, seasonality, trend) belongs in Phase 5 and gets its own doc, [10_time_series_forecasting.md](10_time_series_forecasting.md).

---

## 8. Visualization principles (a preview of doc 12)

A few rules to make your charts read clearly without trying:

1. **One idea per chart.** A scatter that "also shows" three other things via color, size, and shape rarely communicates any of them.
2. **Sort categories meaningfully.** Alphabetical is usually wrong. Sort by median, frequency, or domain order.
3. **Label your axes.** `x: gr_liv_area (sq ft)` not `x: x`.
4. **Pick log axes when ranges span orders of magnitude.** A scatter where 90% of points are clumped on the left is hiding the action.
5. **Mind your color choices.** Default rainbow palettes mislead about ordering (rainbow is *not* perceptually uniform). seaborn's `colorblind` palette is a safe default.
6. **The chart vs. the table.** If three numbers will do, *use three numbers*. Charts are for shape, comparisons, and relationships — not for "displaying" data.

Edward Tufte's *The Visual Display of Quantitative Information* is the canonical book on this. We'll dive in for the Power BI prep doc.

---

## 9. Common EDA pitfalls

1. **Confusing correlation with causation.** Pearson's *r = 0.8* between `square_footage` and `saleprice` doesn't prove sqft *causes* price — they could share a confounder (e.g., neighborhood).
2. **Trusting summary statistics alone.** Famous example: **Anscombe's quartet** — four datasets with identical mean, variance, and correlation but wildly different shapes when plotted. *Always plot too.*
3. **Looking at correlations in the presence of outliers.** Even one extreme point can flip *r* from 0.1 to 0.8. Run `r` with the outliers removed as a sanity check.
4. **One-shot conclusions.** "Bedroom count has *r = 0.4* with price, so beds matter." Maybe — but beds also correlate with sqft. Once you control for sqft, beds may have *r ≈ 0*. Multivariate analysis tells you what marginal analysis can't.
5. **Forgetting to sanity-check categorical encodings.** If `kitchen_qual` shows correlations *after* our ordinal encoding, those numbers come from the *integer mapping* we chose. A different mapping → different correlations. Be aware.
6. **EDA never ending.** EDA is exploratory, but you can spend a week on it. Time-box: do enough to make a feature engineering plan, then go build.

---

## 10. Further reading

- **John Tukey, *Exploratory Data Analysis* (1977).** The book that named the field. Still readable.
- **Anscombe's quartet** on Wikipedia — go look at the four scatter plots once.
- **seaborn tutorial gallery:** [seaborn.pydata.org/examples/index.html](https://seaborn.pydata.org/examples/index.html). When you need a chart type, this is the fastest way to find the right `sns.xyz()` call.
- **"Visualizing Categorical Data" — Friendly, Meyer.** If you want to dig deeper into the categorical/box plot side.
- **"Storytelling with Data" — Cole Nussbaumer Knaflic.** The bridge from "I have a chart" to "I can communicate a finding". Useful for the Power BI phase.

---

## 11. Self-test quiz

1. What does it mean for a distribution to have **skewness ≈ 2**? What would you typically do before fitting a linear model on a column with that skewness?
2. The Pearson correlation between `overall_qual` and `saleprice` is around 0.80, but the Spearman correlation is even higher. What does that suggest about the shape of the relationship?
3. You see two features with Pearson *r = 0.95* with each other. Why is that a *bigger* problem for a linear regression model than for a random forest?
4. Why is a box plot ordered by category *median* more useful than one ordered alphabetically?
5. What is **Anscombe's quartet** and what's its main lesson?
6. After EDA, you have a hunch that `month_sold` matters but its raw integer correlation with `saleprice` is near zero. What encoding change should you try before discarding it?
7. Give one decision (feature engineering or model choice) you could make right now, based purely on Notebook 02's EDA output.

<details>
<summary>Show suggested answers</summary>

1. The distribution has a long right tail — a few very large values pull the mean above the median. Before linear regression, you'd typically log-transform the column (e.g., `log1p`) so its distribution becomes closer to symmetric/normal.
2. Spearman being higher means the relationship is *monotonic* (rank-preserving) but not perfectly *linear*. Quality matters consistently across the scale, but the *jumps* aren't uniform — going from "Gd → Ex" probably maps to a bigger price jump than "TA → Gd". Trees and ranks handle this fine; pure linear models lose information.
3. Linear regression's coefficient estimates become unstable under multicollinearity — small changes in the data flip signs and inflate the variance of estimates, hurting both interpretability and out-of-sample stability. A random forest doesn't care: it just picks one of the two redundant features at each split and ignores the other.
4. Alphabetical ordering scatters the trend (Ex, Fa, Gd, Po, TA puts "Ex" first and "Po" fourth even though they're at opposite ends of quality). Median ordering makes the ranking visible — the chart reads as "left = worst, right = best" automatically.
5. Anscombe's quartet is four datasets with identical means, variances, and correlations but wildly different shapes when plotted. The lesson: summary statistics alone can hide the truth — always plot.
6. Month is *cyclic*: December (12) is adjacent to January (1), not 11 units away. Re-encode it as two columns `sin(2π·month/12)` and `cos(2π·month/12)` so the model sees the topology correctly.
7. Many valid answers. Some examples: (a) Train on `log1p(saleprice)` because of the right skew. (b) Engineer a `total_sf = 1st_flr_sf + 2nd_flr_sf + total_bsmt_sf` feature because the three are correlated and a sum may be more interpretable. (c) Consider tree-based models as a first try because of the visible nonlinearity in `gr_liv_area` vs. price and the multicollinearity between several features.

</details>
