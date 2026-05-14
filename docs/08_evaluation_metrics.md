# 08 — Evaluation Metrics: RMSE, MAE, R², and friends

> *A metric is the question you are asking. Pick the wrong metric and you'll happily optimize the wrong thing.*

---

## 1. The pipeline: residuals → loss → metric

For every prediction your model makes, three derived quantities matter:

```
   y_true (actual)     y_pred (predicted)
        │                     │
        └─────── - ───────────┘
                 │
              residual r_i = y_true_i - y_pred_i
                 │
                 ▼
          aggregate over all n rows
                 │
                 ▼
              one number  ← THE METRIC
```

A **metric** is a single summary number computed from the residuals. Different aggregations answer different questions.

---

## 2. The three metrics we use

### 2.1 MAE — Mean Absolute Error

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |r_i| = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**Plain English:** the average magnitude of the errors. Each row contributes its mistake equally.

**Units:** same as `y`. If `y` is dollars, MAE is in dollars.

**Reading it:** "On average, the model is off by *MAE* dollars per home."

**Robustness:** MAE is **robust to outliers**. One $200k mistake on a $1M home adds 200,000 to the sum — the same as the average of two 100,000 mistakes. No outsized penalty for tail mistakes.

### 2.2 RMSE — Root Mean Squared Error

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} r_i^2}
$$

**Plain English:** the *quadratic-average* magnitude of errors. Each row's contribution is *squared* before averaging, then we take the square root to get back to the original units.

**Units:** same as `y` (squaring then square-rooting cancels).

**Reading it:** "Roughly the size of the *typical* error, but with extra weight given to big mistakes."

**Why squared?** A $20k mistake counts 4× as bad as a $10k mistake (because 20² = 400, 10² = 100). RMSE *penalizes large errors disproportionately*. If you'd rather avoid one very large mistake than ten small ones, RMSE is your metric.

> 🧠 **Mathematical note.** RMSE is the metric that vanilla linear regression is *literally* minimizing. The "least squares" in OLS is the sum of squared residuals — which is RMSE squared times n. Most ML algorithms default to RMSE-shaped losses.

### 2.3 R² — Coefficient of Determination

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

**Plain English:** the fraction of variance in `y` explained by the model. The denominator is the variance you'd have if you naively predicted the *mean* every time; the numerator is the variance left over after the model.

**Range:**
- `R² = 1` → perfect predictions.
- `R² = 0` → the model is no better than predicting the mean.
- `R² < 0` → the model is *worse* than predicting the mean. (Yes, this can happen, and yes, it's embarrassing.)

**Reading it:** "Our model explains 95% of the variation in log SalePrice." A unitless, intuitive single number — most stakeholders understand it.

> ⚠️ **Pitfall — R² hides absolute size.** A model with `R² = 0.99` on a target ranging from $50k to $50,001 explains 99% of essentially nothing. R² is great for *comparing* models on the same target but tells you nothing about *practical* accuracy. Always look at RMSE/MAE alongside.

---

## 3. RMSE vs. MAE — when each wins

| Question                                                           | Metric to pick |
| ------------------------------------------------------------------- | -------------- |
| "How big is a typical mistake?"                                    | MAE            |
| "How bad are my worst mistakes?"                                   | RMSE           |
| "Are there outliers I should not let dominate my evaluation?"      | MAE            |
| "I really hate big mistakes and want the model to focus on avoiding them." | RMSE      |
| "I want a metric that matches what the model is *literally* optimizing." | RMSE (most algorithms) |

**Rule of thumb.** Always report **both**. The *ratio* `RMSE / MAE` tells you about the error distribution:
- `RMSE ≈ MAE` → errors are roughly uniform.
- `RMSE >> MAE` → a few large errors are dragging RMSE up.

In our project, dollar-space `RMSE ≈ $18.7k`, `MAE ≈ $11.9k`. The ratio is ~1.6× — there are some larger mistakes (likely on expensive homes) that RMSE is penalizing.

---

## 4. Log-space metrics vs. dollar-space metrics

We train on `log_saleprice` = `log1p(saleprice)`. After prediction we exponentiate back to dollars with `np.expm1(pred)`.

We report metrics in **both** spaces. Why bother?

| Metric type        | What it measures                                                      |
| ------------------ | --------------------------------------------------------------------- |
| `log_rmse`         | Typical error in log-space — roughly a *percentage* error in dollar-space. |
| `log_r2`           | Variance explained in log-space (where the model is fit).             |
| `dollar_rmse`      | Typical error in dollars (penalizing big absolute mistakes).          |
| `dollar_mae`       | Median-ish error in dollars (robust to expensive-home tail).          |
| `dollar_r2`        | Variance explained in raw dollar-space.                              |

**Why log metrics are useful even when the boss wants dollars:** an `log_rmse` of 0.10 corresponds to roughly ±10% error on most homes regardless of price level. A $10k mistake on a $100k home (10%) is "equivalent" to a $50k mistake on a $500k home (10%) in log-space. RMSE in dollars would weight the second one as 25× worse; log-space treats them as the same.

> 📘 **Mental conversion.** If `log_rmse = 0.10`, then a typical prediction is multiplied by something between `e^-0.10 ≈ 0.91` and `e^0.10 ≈ 1.11` — so within roughly ±10% of the true price.

---

## 5. The predicted-vs-actuals diagnostic plot

Numbers don't tell you *what kind* of mistakes the model makes. A scatter plot does.

```
              ↑ predicted
              │       •  ← cluster on diagonal = good
              │     •
              │   •  •
              │ • • •     • ← above diagonal = over-predict
              │•• ←        • • below diagonal = under-predict
              │•  •
              ─────────────────→ actual
```

What to look for:

1. **Tight cluster on the diagonal** → good model.
2. **Systematic bias** (whole cloud above or below the diagonal) → the model is consistently over- or under-predicting.
3. **Heteroskedasticity** (cone shape: wider scatter at one end) → the model is more accurate in some price ranges than others.
4. **Outliers** (points way off) → individual homes the model badly mispredicts. Worth investigating: bad data, or a missing feature?

The Phase 4 notebook plots this on the held-out test set in dollar space. Look at yours: where's the model good, where's it bad?

---

## 6. The residuals histogram

The **residual** is `y_true - y_pred`. Plotting their distribution tells you about the model's biases:

| Histogram shape        | What it means                                                  |
| ---------------------- | --------------------------------------------------------------- |
| Symmetric bell, centered on 0 | The model is well-calibrated, no systematic bias.      |
| Centered to the right of 0    | Model systematically *under-predicts* (residuals positive). |
| Centered to the left of 0     | Model systematically *over-predicts*.                  |
| Long right tail               | The model occasionally massively *under-predicts* a few outliers. |
| Long left tail                | The model occasionally massively *over-predicts*.      |
| Two peaks                     | There are two populations in your data the model handles differently. |

Healthy residuals are roughly normal, centered on zero. Phase 4's notebook plots this; check yours.

---

## 7. Other metrics worth knowing

We don't use these in this project, but you'll see them elsewhere.

| Metric                            | Formula                                                       | When                                                  |
| --------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------- |
| **MSE** (Mean Squared Error)      | `mean((y-y_hat)^2)`                                            | Same as RMSE² but in squared units — only used as a loss internally. |
| **MAPE** (Mean Absolute Pct Error) | `mean(|y - y_hat| / |y|) * 100%`                              | Reporting in % when stakeholders prefer it. Caveat: blows up on `y ≈ 0`. |
| **MdAPE** (Median APE)             | median of the same                                            | More robust version of MAPE.                          |
| **SMAPE** (Symmetric MAPE)         | `mean(2|y-y_hat| / (|y|+|y_hat|)) * 100%`                     | Avoids the asymmetry MAPE has between over- and under-prediction. |
| **Quantile loss / Pinball loss**   | Asymmetric loss                                                | When over- and under-prediction have different costs. |
| **Huber loss**                     | MAE for big residuals, MSE for small                          | Robust regression — combines RMSE and MAE benefits.  |

The right metric is the one that matches what the **business** cares about. Predicting house prices for an investor? They might care more about *not over-paying* than *not under-bidding* — that's an asymmetric problem and a symmetric metric like RMSE is the wrong choice.

---

## 8. Common pitfalls

1. **R² without RMSE/MAE.** R² hides absolute scale. Always pair them.
2. **Trusting one metric on one split.** Use cross-validation and report mean ± std.
3. **Comparing metrics across different targets.** RMSE on `log_y` and RMSE on `y` are not interchangeable. State which space you're in.
4. **Confusing MSE with RMSE.** MSE has *squared* units (dollars²). Always report RMSE for human readers.
5. **MAPE on data with zero values.** `MAPE` divides by `y`. If `y` is ever zero or near-zero, MAPE explodes. Use MAE or MdAPE instead.
6. **Reporting train-set metrics.** Train metrics measure memorization, not generalization. Report CV or test metrics.
7. **Choosing a metric *after* seeing model results.** "I'll use whichever metric makes my model look best" is fraud. Pick the metric before evaluation.

---

## 9. How metrics interact with the loss function

When you call `RidgeRegressor.fit(X, y)`, the algorithm *literally minimizes* an RMSE-shaped loss internally. When you evaluate it with RMSE, you're checking how well it did at its own goal.

What if you evaluate with MAE instead? You're judging the model on a slightly different objective than it was optimizing for. Usually the discrepancy is small, but for outlier-heavy data the gap can be meaningful.

For XGBoost, you can change the *training* loss via the `objective` parameter:
- `"reg:squarederror"` (default) — optimizes RMSE.
- `"reg:absoluteerror"` — optimizes MAE.
- `"reg:pseudohubererror"` — optimizes Huber loss (compromise).

If your *evaluation* metric is MAE, sometimes training with `"reg:absoluteerror"` gives a small boost. Worth trying once you've exhausted feature engineering.

> 📘 **Rule of thumb.** Match training loss to evaluation metric *when possible*, but don't waste time on it before squeezing every drop out of features and standard hyperparameters.

---

## 10. Further reading

- **scikit-learn — Model evaluation:** [scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html). The reference for every metric sklearn exposes.
- **scikit-learn — Regression metrics specifically:** [scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
- **Hastie, Tibshirani, Friedman, *Elements of Statistical Learning*, Ch. 7** — model assessment and selection. The classical treatment.
- **Cross Validated (stats.stackexchange.com)** — search "RMSE vs MAE", "interpreting R²", etc. Excellent intuition-builders.

---

## 11. Self-test quiz

1. Compute both metrics by hand. For predictions `[100, 200, 300]` and actuals `[110, 195, 350]`:
   - What is the MAE?
   - What is the RMSE?
   - Why is RMSE larger than MAE here, and what does that tell you about the error distribution?
2. A model gets `R² = 0.95` on a held-out test set. Why is that *not enough information* to declare it useful?
3. Explain in your own words why `log_rmse` is roughly a "percentage error" in dollar-space.
4. You see `RMSE = 2 × MAE` on your test predictions. What does this ratio suggest about the kind of mistakes the model is making?
5. When would you choose MAE over RMSE as your *training* loss? Why?
6. A residual histogram has a long right tail (lots of positive residuals far from zero). Is the model systematically over-predicting or under-predicting? What might you do about it?
7. You're modeling delivery times in minutes. Most rows are 10–60 minutes, but some are 0 (failed deliveries). Should you use MAPE here? Why or why not?

<details>
<summary>Show suggested answers</summary>

1. Residuals are `[-10, 5, -50]`.
   - MAE = (10 + 5 + 50) / 3 = 65 / 3 ≈ **21.67**.
   - RMSE = √((100 + 25 + 2500) / 3) = √(2625/3) = √875 ≈ **29.58**.
   - RMSE > MAE because the squared term blew up the $50 mistake (50² = 2500) into something that dominates. The ratio ~1.36 tells you one of the three errors is noticeably bigger than the others; the squared error metric is paying more attention to the worst case.
2. R² is unitless and hides absolute scale. The model could be off by $50k on average ("RMSE ≈ $50k") while still explaining 95% of the *variation* if the price range is wide enough. Without RMSE/MAE you can't tell whether the model is useful in practice.
3. `log_rmse = 0.10` means a typical residual in log-space is 0.10. Exponentiating: `exp(0.10) ≈ 1.105`, so predictions are typically multiplied by something between ~0.90 and ~1.11 — that's ±10% in dollar-space. The log-space RMSE roughly tracks the model's *percentage* accuracy regardless of price level.
4. `RMSE ≈ 2 × MAE` is a heavy-tailed error pattern — most errors are small, but a small number of mistakes are large enough to drag the squared average way above the absolute average. Likely the model is mostly accurate but has a handful of outlier failures worth investigating.
5. Choose MAE as the training loss when (a) your evaluation metric is MAE; (b) the dataset has known outliers you don't want to dominate the fit; (c) all mistakes have equal real-world cost regardless of magnitude. RMSE-based loss is mathematically more convenient and is the right default otherwise.
6. Long right tail in `y_true - y_pred` = positive residuals = `y_true > y_pred` = the model is *under-predicting* on those rows. To fix, look at *which* rows have the big positive residuals — they often share a feature pattern the model isn't using. Engineering a feature that captures that pattern, or switching to a more expressive model, may help.
7. No. MAPE divides by `y`. With `y = 0` rows, MAPE either crashes or returns ∞. Use MAE (in minutes) or MdAPE if you want a percentage view.

</details>
