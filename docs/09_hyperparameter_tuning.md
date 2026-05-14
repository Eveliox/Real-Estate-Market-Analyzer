# 09 — Hyperparameter Tuning

> *Defaults are someone else's best guess for an "average" dataset. Your dataset is not average. Tuning is the process of finding the right knob positions for **your** specific problem.*

---

## 1. Parameter vs. hyperparameter

Two words that beginners confuse, and that you must keep straight.

| Term                | What it is                                            | Who sets it?              | Example                       |
| ------------------- | ----------------------------------------------------- | ------------------------- | ----------------------------- |
| **Parameter**       | A value the model learns from training data          | The algorithm, during `.fit()` | Coefficients in Ridge; split thresholds in a tree |
| **Hyperparameter**  | A configuration *knob on the model itself*           | You, before `.fit()`      | `alpha` in Ridge; `max_depth` in XGBoost |

The model learns parameters; you tune hyperparameters.

A Ridge model with 230 features has 231 parameters (one per feature plus an intercept) — all learned automatically. It has one main hyperparameter, `alpha`. Tuning means finding the value of `alpha` that gives the best generalization.

---

## 2. Why default hyperparameters aren't enough

The defaults shipped with scikit-learn and XGBoost are reasonable starting points designed to *not embarrass the library* on most datasets. They are rarely *optimal* for your specific problem.

Three reasons tuning matters:

1. **Data size matters.** What works on 100k rows underfits 1k rows and overfits 10M rows.
2. **Feature space matters.** A dataset with 5 features tunes very differently from one with 500.
3. **The right complexity depends on signal-to-noise.** Noisy data needs simpler models; clean data benefits from complex ones.

Typical lift from tuning: 5–20% improvement in RMSE over defaults. Sometimes more, rarely less. Worth doing once feature engineering is done.

---

## 3. The validation pattern

You can't tune on the **test set** — that would peek (doc 06 §4). You can't tune on the **training set alone** — that picks whichever hyperparameters memorize the training data hardest.

The fix is a **validation set** — a chunk of data you hold back from training *but check repeatedly* during tuning.

Three common arrangements:

### 3.1 Simple holdout

```
[ TRAIN (60%) ][ VAL (20%) ][ TEST (20%) ]
```

You fit candidates on TRAIN, score on VAL, pick the best, then check it once on TEST.

Pros: fast.  
Cons: VAL is small (and noisy) on small datasets — your "best" pick is unreliable.

### 3.2 K-fold cross-validation on training data

```
TRAIN (80%):  split into 5 folds, score each candidate by averaging 5 fold-validations
TEST (20%):   touched only at the end
```

Pros: every training row gets a "validation" turn — more data-efficient.  
Cons: 5× more compute per candidate.

This is what we use in this project — for ~3000 rows, CV beats single-holdout by a meaningful margin.

### 3.3 Nested CV (advanced)

For very small datasets where even the test set is noisy. Two layers of CV — outer for performance estimation, inner for tuning. Computationally heavy and overkill here.

---

## 4. Grid search

The textbook approach: enumerate every combination of hyperparameter values you want to try.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__max_depth":     [3, 4, 5, 6, 7],
    "model__n_estimators":  [200, 400, 600, 800],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.10],
}
search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_root_mean_squared_error")
search.fit(X_train, y_train)
```

`GridSearchCV` evaluates *every combination* with 5-fold CV. The grid above is 5 × 4 × 4 = 80 combinations × 5 folds = 400 model fits.

### When grid search makes sense

- The search space is small (≤2 hyperparameters with ≤10 values each).
- You want completeness — "I checked every combination".
- Compute is cheap.

### Why it doesn't scale

With 6 hyperparameters at 4 values each, you have 4⁶ = 4,096 combinations × 5 folds = 20,480 fits. On XGBoost that's hours. And the problem is worse than slow: **most of those combinations are equivalent**. You waste compute exploring all combinations of unimportant dimensions.

---

## 5. Random search — better than grid for >2 hyperparameters

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "model__max_depth":     [3, 4, 5, 6, 7],
    "model__n_estimators":  [200, 400, 600, 800, 1200],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.10],
    "model__subsample":     [0.7, 0.8, 0.9, 1.0],
    "model__reg_lambda":    [0.5, 1.0, 2.0, 5.0],
}
search = RandomizedSearchCV(
    pipeline, param_dist,
    n_iter=30,   # try 30 random combinations
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
)
```

Random search picks `n_iter` combinations *at random* from the joint space. With 30 samples across 5 dimensions, each individual dimension gets ~30 distinct values explored — almost as much coverage as if you'd put all 30 samples on that one dimension.

### The Bergstra & Bengio result

In a famous 2012 paper, Bergstra and Bengio showed:

> **For most hyperparameter spaces, random search finds equally good or better configurations than grid search using a fraction of the compute.**

The reason: in any real ML problem, only a few hyperparameters drive most of the gains. Grid search wastes evaluations on the unimportant ones; random search distributes evaluations uniformly across the important ones.

We use **random search** in this project — `n_iter=30` typically lands within 1–2% of the global optimum.

---

## 6. Smarter alternatives (mentioned, not used)

- **Bayesian optimization** (`optuna`, `scikit-optimize`, `hyperopt`): models the objective function and picks the next candidate based on what it has learned. Much fewer evaluations needed than random search. Use for serious tuning budgets.
- **Halving search** (`HalvingRandomSearchCV` in sklearn ≥0.24): evaluates many candidates on a *small* subset first, then promotes the survivors to a larger subset. Cheaper than vanilla random search.
- **Population-based training (PBT)**: used in deep learning. Multiple models train in parallel, periodically swapping hyperparameters with the best performers.

For tabular regression at our data size, Random Search is the right tool. For massive deep learning problems, Bayesian is the standard.

---

## 7. Which XGBoost hyperparameters actually matter

Of XGBoost's many knobs, only a handful drive >80% of the performance gains. Tune these first:

| Hyperparameter           | Typical search range  | Why it matters                                          |
| ------------------------ | --------------------- | ------------------------------------------------------- |
| `learning_rate`          | 0.01 – 0.10           | Step size for each new tree. Lower = slower + better. |
| `n_estimators`           | 200 – 2000            | Number of trees. Paired with learning_rate.            |
| `max_depth`              | 3 – 8                 | Tree depth. Deeper = more interactions captured.       |
| `subsample`              | 0.6 – 1.0             | Row sampling per tree. Lower = less overfit.           |
| `colsample_bytree`       | 0.6 – 1.0             | Feature sampling per tree. Lower = less overfit.       |
| `reg_lambda`             | 0 – 10                | L2 regularization on leaf weights.                     |

Beyond these, `min_child_weight`, `gamma`, `reg_alpha` (L1) give marginal additional gains and aren't worth tuning until the above are settled.

This is the search space `tune_xgboost` in [`src/models.py`](../src/models.py) uses.

---

## 8. Reading tuning output

`RandomizedSearchCV` returns rich results. The most useful summary is:

```
best_params:
  model__n_estimators:     800
  model__learning_rate:    0.05
  model__max_depth:        4
  model__subsample:        0.9
  model__colsample_bytree: 0.9
  model__reg_lambda:       1.0

top 5 configurations (by mean RMSE on held-out folds):
   mean_rmse  std    params
0  0.1185     0.005  {n_estimators=800, learning_rate=0.05, ...}
1  0.1192     0.006  {...}
2  0.1198     0.007  {...}
...
```

Three things to look at:

1. **The gap between #1 and #5.** If it's tiny (e.g., 0.0005), your search is in a flat region — further tuning won't help much. If it's big (e.g., 0.01), you may want more iterations.
2. **The chosen `learning_rate × n_estimators` product.** If the best config has the smallest `learning_rate` *and* the largest `n_estimators`, your grid is bottlenecking you — extend it.
3. **Boundary values.** If the best `max_depth` is at the maximum of your grid, the model wants deeper trees than you allowed. Extend the grid and re-run.

---

## 9. Tuning rules of thumb

1. **Tune features before hyperparameters.** A 5% feature improvement beats a 1% hyperparameter improvement, and is more durable across data shifts.
2. **Tune the impactful hyperparameters only.** 4–6 hyperparameters is plenty. More just adds search noise.
3. **Pair `learning_rate` and `n_estimators`.** They trade off — let the search vary both.
4. **Use random search for >2 dimensions.** Bergstra & Bengio.
5. **Set `random_state` everywhere.** Without it, tuning runs aren't reproducible.
6. **Watch the std across folds.** A configuration with low mean RMSE but high std might be lucky on this fold split, not actually better.
7. **Re-tune when you change features.** New features can shift the optimal hyperparameters.
8. **Stop when the marginal gain is below noise.** If 2 hours of tuning yields a 0.001 improvement in CV RMSE, walk away.

---

## 10. Anti-patterns

1. **Tuning on the test set.** Every "small adjustment based on what helps test score" leaks. Use CV.
2. **One-at-a-time grid search.** Tuning `max_depth` to its best value, then *fixing* it and tuning `learning_rate`, etc. This misses interactions. Random search jointly.
3. **Reporting tuned-model performance on training data.** Of course it's great. The training data is what you optimized for.
4. **Cherry-picking the best fold.** "The model got RMSE = 0.10 on fold 3!" — irrelevant; report the *mean* across folds, with the std.
5. **Tuning a model that's underfitting.** If your model is too simple to capture the data, no hyperparameter combination will save it. Add features or pick a more flexible model first.
6. **Tuning until you've memorized the validation set.** With enough random search iterations, you'll find a configuration that overfits to the *validation* set specifically. Practical safeguards: a separate test set, or *nested* CV.

---

## 11. Further reading

- **Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization* (2012):** the canonical paper on why random ≥ grid. Free PDF: [jmlr.csail.mit.edu/papers/v13/bergstra12a.html](https://jmlr.csail.mit.edu/papers/v13/bergstra12a.html).
- **scikit-learn — Tuning the hyperparameters of an estimator:** [scikit-learn.org/stable/modules/grid_search.html](https://scikit-learn.org/stable/modules/grid_search.html).
- **Optuna** ([optuna.org](https://optuna.org)): the most popular Bayesian tuning library for Python. Drop-in replacement for `RandomizedSearchCV` with much smarter sampling.
- **XGBoost — Parameters reference:** [xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html). Every knob, what it does, valid ranges.

---

## 12. Self-test quiz

1. In one sentence, what's the difference between a **parameter** and a **hyperparameter**? Give an example of each from this project.
2. Why is it *forbidden* to tune hyperparameters on the test set? What's the recommended pattern instead?
3. Bergstra and Bengio (2012) showed that random search beats grid search for most ML hyperparameter spaces. Summarize the *reason* in your own words.
4. You run a grid search over 4 hyperparameters with 4 values each, using 5-fold CV. How many total model fits does this require?
5. Your best XGBoost configuration sits at `max_depth=8`, the maximum value in your grid. What does this suggest, and what should you do?
6. Why pair `learning_rate` and `n_estimators` in the search rather than tuning them independently?
7. After tuning, the top 5 configurations differ in mean RMSE by less than 0.002. What does that tell you about whether more tuning iterations will help?

<details>
<summary>Show suggested answers</summary>

1. A *parameter* is learned by the model from data during `.fit()` (e.g., a Ridge coefficient). A *hyperparameter* is a configuration knob set by you before training (e.g., `alpha` in Ridge, `max_depth` in XGBoost).
2. Tuning on the test set means every "try this and check" peek leaks information about the test set into your decision-making. After many tries, your chosen hyperparameters are partially fit to the test set, and your reported test score is optimistically biased. Use cross-validation on the training set instead, and only check the test set with the *final* selected model.
3. In a high-dimensional hyperparameter space, only a few dimensions actually drive most of the performance gains. Grid search wastes evaluations exploring all combinations of unimportant dimensions (covering them densely but redundantly). Random search distributes evaluations more uniformly across the important dimensions, finding good configurations faster.
4. 4⁴ × 5 = 256 × 5 = **1,280 model fits**.
5. The best configuration is at the *boundary* of your search range. The model probably wants deeper trees than you allowed; the truly-optimal max_depth might be 9 or 10. Extend your grid (e.g., to `[3, 4, 5, 6, 7, 8, 10, 12]`) and re-run the search.
6. They trade off directly: lower learning_rate means each tree contributes less, so you need more trees to fit. Tuning learning_rate alone with n_estimators fixed (or vice versa) misses configurations where both move together. A joint search captures the tradeoff.
7. Your search has converged to a flat region of the objective landscape — many configurations work about equally well. More iterations won't meaningfully improve the best model. Better to stop here and either (a) accept this performance, or (b) revisit feature engineering for a bigger lever.

</details>
