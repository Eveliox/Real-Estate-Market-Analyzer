# 07 — Model Selection: Ridge vs. Random Forest vs. XGBoost

> *Three model families cover ~95% of tabular regression problems you'll meet in real life. This doc explains what each one is doing under the hood, when it shines, and how to read a comparison.*

---

## 1. Why three models and not just one?

Different algorithms make different *assumptions* about the world. Picking the wrong assumption costs you accuracy. Trying three families, comparing them honestly, then picking the best — that's how serious ML projects start.

The three we cover:

| Family             | Assumption it makes about the world                                    | Best when…                                                  |
| ------------------ | ----------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Linear (Ridge)** | The target is a weighted sum of features (with regularization).         | Data is roughly linear in the features; interpretability matters; small data. |
| **Random Forest**  | The target is well-explained by averaging many *independent* decision trees on bootstrap samples. | Nonlinear data, robust default, low tuning cost.            |
| **Gradient Boosting (XGBoost)** | The target is well-explained by a *sequence* of trees each fixing the previous ones' mistakes. | Same as RF but you have a little time to tune. Typically the best on tabular data. |

The rest of this doc dives into each one.

---

## 2. Linear regression — and why we use Ridge

### 2.1 Plain linear regression (OLS)

The most ancient model in the ML toolbox. It assumes:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
$$

A weighted sum of features. The "training" finds the coefficients $\beta_0 \ldots \beta_p$ that minimize the sum of squared residuals:

$$
\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

This is called **Ordinary Least Squares** (OLS). The math has a beautiful closed-form solution; no iterative training needed.

**When it works:** the relationship between features and target is roughly *additive* and *linear*. Doubling square footage adds roughly $X to the price, and that $X doesn't depend on whether the house has a garage.

### 2.2 Why pure OLS struggles on Ames

With 230+ features and visible multicollinearity (recall Phase 2 EDA: `garage_cars` and `garage_area` are ~95% correlated), pure OLS has two problems:

1. **Unstable coefficients.** Two features carrying the same signal split the coefficient between them in unstable ways — flipping signs on small data perturbations. The *predictions* are still OK but the coefficients become meaningless.
2. **Sensitivity to outliers.** Squared error penalizes a $100k mistake 10,000× more than a $1k mistake. One bad row can dominate the fit.

### 2.3 Ridge regression — adding a regularizer

Ridge fixes the multicollinearity issue by adding an **L2 penalty** to the loss:

$$
\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2
$$

The new term penalizes large coefficients. `α` (alpha) is the **regularization strength** hyperparameter:

- `α = 0` → identical to OLS
- `α → ∞` → all coefficients shrink to zero (predicts the mean of y)
- `α` chosen by cross-validation → balances fit vs. stability

The intuition: when two features carry the same signal, OLS can't decide how to split the coefficient. Ridge gently nudges *both* toward zero (and toward each other), giving stable, near-identical estimates.

> 🧠 **Why "L2"?** The penalty is the *sum of squared* coefficients — same as the L2 (Euclidean) norm. Lasso uses **L1** regularization (sum of absolute values) which has a different geometric effect: it actually *zeros out* some coefficients, doing implicit feature selection.

### 2.4 Why Ridge needs scaled features

The penalty $\alpha \sum \beta_j^2$ applies *uniformly* across all features. If one feature is in dollars (10⁵-scale) and another is 0/1, the dollar feature's coefficient must be tiny (10⁻⁵) just to keep the prediction in range. The penalty on that tiny coefficient is negligible — so the regularization effectively *ignores* the dollar feature.

The fix is `StandardScaler`: center every feature on zero with unit variance. Then the penalty applies fairly. **All linear models with regularization need scaled inputs.** (Trees do not.)

### 2.5 When Ridge wins

- Genuinely linear relationships.
- Small datasets (where complex models would overfit).
- When you need **interpretable** coefficients (e.g., "each extra bathroom adds $X").
- As a strong baseline to compare against fancier models.

---

## 3. Decision trees and Random Forests

### 3.1 What a decision tree does

A decision tree learns a sequence of yes/no questions to partition the data into homogeneous groups.

```
                    [overall_qual >= 7?]
                   /                    \
                  no                    yes
                  |                      |
       [total_sf >= 2200?]    [total_sf >= 2500?]
        /          \            /            \
      no          yes          no            yes
      |            |            |             |
   $135k        $185k        $250k         $410k
```

To predict for a new home: walk the tree, follow the splits, output the leaf value.

**Why splits help:** they capture *nonlinearity* and *interactions* for free. The effect of `total_sf` is different in low-quality homes than in high-quality ones — a linear model can't see this without explicit interaction terms; a tree builds the interaction by splitting.

**Why a single tree fails:** it overfits trivially. A deep enough tree can memorize the training set (one leaf per row). Pruning (limiting depth) helps but trades accuracy.

### 3.2 Random Forest — bagging trees

A **Random Forest** trains many trees, each on:
- A *bootstrap sample* of the training rows (sample with replacement; ~63% unique rows per tree).
- A *random subset* of the features at each split.

Then averages their predictions. This trick — called **bagging** (bootstrap aggregating) — dramatically reduces variance. Each individual tree is noisy; their average is stable.

**Key hyperparameters:**

| Hyperparameter         | What it does                                            | Typical value          |
| ---------------------- | ------------------------------------------------------- | ---------------------- |
| `n_estimators`         | How many trees                                          | 200–1000 (more is better, slower) |
| `max_depth`            | Max tree depth (None = grow until pure)                 | None or 10–30          |
| `min_samples_leaf`     | Minimum samples in a leaf                               | 1–10                   |
| `max_features`         | Features considered per split                           | `"sqrt"` (classification), `1.0` (regression) |

### 3.3 When Random Forest wins

- A solid **default** model for tabular data — works well with default hyperparameters.
- Nonlinear or interaction-heavy data.
- When you can't be bothered to tune carefully.
- When you want a robust baseline that handles ugly data gracefully.

It rarely *wins* against tuned XGBoost on tabular data but loses by a small margin and takes way less effort.

---

## 4. Gradient Boosting (XGBoost)

### 4.1 The boosting idea

Where Random Forest builds trees **in parallel** (each one independent), gradient boosting builds them **sequentially**:

1. Start with a baseline prediction (e.g., the mean of y).
2. Compute the residuals (errors).
3. Train a tree to *predict the residuals*.
4. Add (a fraction of) that tree's predictions to the running total.
5. Re-compute residuals and repeat.

Each new tree corrects the residual error left by the previous trees. The result is an *additive* ensemble: prediction = baseline + tree₁_correction × learning_rate + tree₂_correction × learning_rate + …

### 4.2 Why "gradient" boosting

The "gradient" comes from generalizing this to arbitrary loss functions. For squared-error loss, the gradient is just the residual. For other losses (e.g., classification log-loss), the gradient is something else but the algorithm is the same: train each new tree on the negative gradient of the loss.

### 4.3 What XGBoost adds beyond textbook gradient boosting

XGBoost is *gradient boosting with engineering polish*. Key additions:

- **Built-in L1/L2 regularization** on tree weights — controls overfitting.
- **Second-order optimization** (uses Hessian, not just gradient) — converges faster.
- **Smart handling of missing values** — splits learn which side missing values go.
- **Histogram-based splitting** — bin continuous features into ~256 bins for fast split-finding.
- **Built-in early stopping** — quit when validation loss stops improving.
- **Excellent parallelism** — uses all CPU cores well.

It is, broadly, the strongest single-model algorithm on tabular data as of the mid-2020s.

### 4.4 The hyperparameters that matter

Only a handful actually drive performance. Don't tune the rest until these are good.

| Hyperparameter            | Effect                                                    | Typical range          |
| ------------------------- | ---------------------------------------------------------- | ---------------------- |
| `learning_rate` (`eta`)   | How much each tree contributes. Lower = slower + better.  | 0.01–0.10              |
| `n_estimators`            | Number of trees.                                          | 200–2000 (paired w/ learning_rate) |
| `max_depth`               | Tree depth. Deeper = more interactions captured, more overfit risk. | 3–8 (this dataset: 4) |
| `subsample`               | Row sampling per tree (like RF bagging).                  | 0.6–1.0                |
| `colsample_bytree`        | Feature sampling per tree.                                | 0.6–1.0                |
| `reg_lambda`              | L2 regularization on leaf weights.                        | 0–10                   |
| `min_child_weight`        | Min samples (weighted) per leaf.                          | 1–10                   |

**The headline tradeoff:** `learning_rate × n_estimators` is roughly your "budget". Halving the learning rate and doubling the trees usually gives slightly better accuracy at 2× the compute.

### 4.5 When XGBoost wins

- **Almost always on medium-to-large tabular data** (10³–10⁶ rows).
- When you have a few minutes for tuning.
- When you have time to think about the loss function (XGBoost lets you pick).

It is overkill for very small datasets (<500 rows) where Ridge is more robust.

---

## 5. Side-by-side mental model

```
Ridge:           y_hat = sum( w_i * feature_i )
                 + assumption: world is linear in features
                 + penalty:    don't let any w_i get too big

Random Forest:   y_hat = mean( T_1(X), T_2(X), ..., T_N(X) )
                 each T_k trained on a random subset of rows & features

XGBoost:         y_hat = baseline + lr * T_1(X) + lr * T_2(X) + ...
                 each T_k trained to predict the previous mistakes
```

---

## 6. Reading the comparison table

Notebook 04 produces a CV comparison like this (your numbers may vary slightly):

```
        model  rmse_mean  rmse_std  mae_mean  r2_mean
      xgboost     0.123     0.006     0.079    0.907
        ridge     0.131     0.003     0.085    0.895
random_forest     0.140     0.009     0.091    0.880
```

How to read it:

- **`rmse_mean`** — average 5-fold validation RMSE in **log-space**. Lower is better.
- **`rmse_std`** — fold-to-fold variation. Lower is more stable.
- **`mae_mean`** — second metric for robustness check.
- **`r2_mean`** — variance explained. Higher is better.

Notes on this specific result:
- XGBoost wins on mean RMSE, as expected.
- **Ridge is unusually competitive** — only ~6% behind XGBoost. That's a sign your *feature engineering* (Phase 3) gave the linear model enough to work with. On raw Ames without engineering, Ridge typically lags much further.
- Random Forest is behind XGBoost — RF rarely beats well-tuned gradient boosting on tabular data.
- **`rmse_std`** for Ridge is the smallest (0.003). Linear models are typically the *most stable* across folds. If you need a model that "predicts about the same accuracy on every kind of input", Ridge is the safer pick.

---

## 7. When to *not* use these three

The three models above are excellent for *tabular* regression. They're a poor fit for:

| Problem type                        | Better tools                                    |
| ----------------------------------- | ----------------------------------------------- |
| Text / NLP                          | Transformers (HuggingFace ecosystem)            |
| Images                              | CNNs / Vision Transformers (PyTorch, Keras)     |
| Time series with strong seasonality | Prophet, ARIMA, ETS — **see Phase 5**           |
| Many sequential events per "row"    | RNNs, Transformers                              |
| Graph-structured data               | Graph Neural Networks                           |
| Huge data (10⁹+ rows)               | Spark MLlib, distributed gradient boosting (e.g. LightGBM on Dask) |

For 95% of business analytics work, the three in this doc are still the right starting point.

---

## 8. Common pitfalls

1. **Comparing models with different preprocessing.** Always wrap each in a pipeline so the comparison is apples-to-apples.
2. **Skipping the linear baseline.** "I'll just use XGBoost" — but if Ridge gets within 5% with much less compute and full interpretability, that may be the better business choice.
3. **Trusting one fold's score.** A single train/test split can mislead. Use CV.
4. **Tuning before EDA.** Hyperparameters are downstream of feature quality. Better features beat better hyperparameters almost always.
5. **Over-tuning Random Forest.** RF's strength is "good out of the box". If you're spending hours tuning RF, consider XGBoost instead.
6. **Pure OLS without scaling on multicollinear data.** Coefficients flip signs run-to-run. Use Ridge.

---

## 9. Further reading

- **scikit-learn — Ensemble methods:** [scikit-learn.org/stable/modules/ensemble.html](https://scikit-learn.org/stable/modules/ensemble.html)
- **XGBoost docs — Introduction to Boosted Trees:** [xgboost.readthedocs.io/en/latest/tutorials/model.html](https://xgboost.readthedocs.io/en/latest/tutorials/model.html). The single best intro to gradient boosting.
- **Tianqi Chen & Carlos Guestrin, *XGBoost: A Scalable Tree Boosting System* (2016).** The original paper. Surprisingly readable.
- **Hastie, Tibshirani, Friedman, *Elements of Statistical Learning*, Ch. 3 (linear), Ch. 9-10 (trees, boosting).** Free PDF at the authors' Stanford page.
- **LightGBM / CatBoost** — XGBoost's main competitors. Often comparable or faster. Worth knowing about.

---

## 10. Self-test quiz

1. Why does Ridge regression require scaled features, while Random Forest does not?
2. Explain in one sentence each: what does **bagging** do, and what does **boosting** do? Which one is Random Forest? Which one is XGBoost?
3. What does the `learning_rate` hyperparameter control in XGBoost, and what's the headline tradeoff it shares with `n_estimators`?
4. Ridge with engineered features came within ~6% of XGBoost on this dataset. What does that suggest about Phase 3's feature engineering?
5. You're modeling a dataset with 200 rows and 20 features. Which of the three model families would you reach for first, and why?
6. Random Forest's `n_estimators` is "more is better, slower". Why does adding more trees not cause Random Forest to overfit (the way it would for a single tree)?
7. Pure OLS on multicollinear data gives unstable coefficients but reasonable predictions. Why is that distinction important if you only care about *predictions*?

<details>
<summary>Show suggested answers</summary>

1. Ridge's L2 penalty applies the same penalty weight to every coefficient regardless of the underlying feature's scale. A feature in dollars has tiny coefficients, so the penalty barely affects it, defeating the regularization. Scaling puts all features on equal footing. Random Forest makes scale-invariant splits ("feature > threshold"), so feature scale doesn't matter to it.
2. Bagging trains many models *in parallel* on bootstrap samples and averages their predictions to reduce variance — that's Random Forest. Boosting trains models *sequentially*, each one correcting the previous ones' residuals — that's XGBoost.
3. `learning_rate` controls how much each new tree contributes to the running prediction (multiplied in). Lower learning_rate = each tree corrects gently, so you need more trees (`n_estimators`) to fit. The tradeoff: lower learning_rate × more trees usually gives slightly better accuracy at the cost of more compute.
4. It suggests the engineered features captured most of the nonlinear signal explicitly (total_sf, home_age, has_pool flags, etc.). The linear model didn't need to *learn* those nonlinearities because we *gave* them to it. This is a hallmark of good feature engineering.
5. Ridge. With only 200 rows, tree ensembles will overfit quickly, and the noise in the CV scores will dwarf any tuning gains. Ridge is robust to small data and produces interpretable coefficients you can defend to a stakeholder.
6. Each Random Forest tree is trained on a different bootstrap sample of the data and a random subset of features. Adding more trees adds more diverse votes; the average becomes more stable, not more fit to noise. The variance of the mean of N independent models goes down as N grows, while the bias stays the same.
7. If you ever need to *explain* the model — "what features matter, by how much, in what direction" — unstable coefficients (sign flipping on small data changes) destroy interpretability. For pure prediction you can ignore it, but ML rarely stays in pure-prediction mode once stakeholders get involved.

</details>
