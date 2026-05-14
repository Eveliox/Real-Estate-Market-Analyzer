# 06 — Machine Learning Basics

> *Read this once before opening Notebook 04. Everything else in Phase 4 assumes the vocabulary in this doc.*

---

## 1. What "machine learning" is, in one paragraph

**Machine learning** is the practice of writing programs that *learn rules from data* rather than being told the rules directly. You hand the algorithm a table of past examples (rows of features + a known outcome) and it figures out a function that maps features → outcome. You then apply that function to new rows whose outcome you don't yet know.

The "learning" is just numerical optimization: the algorithm starts with a guess (e.g., random coefficients), measures how wrong it is on the training data, and adjusts to be less wrong. Repeat until the wrongness stops decreasing.

> 📘 **Tom Mitchell's definition (1997):** *"A computer program is said to learn from experience E with respect to some task T and performance measure P, if its performance at task T, as measured by P, improves with experience E."*
> - **Task** = predict SalePrice
> - **Experience** = the labeled Ames rows
> - **Performance measure** = RMSE on held-out data

---

## 2. The three flavors

| Flavor              | What you give the algorithm                                | What it returns                            | Example                          |
| ------------------- | ----------------------------------------------------------- | ------------------------------------------ | -------------------------------- |
| **Supervised**      | Features + labels (each row has a known outcome)            | A function `f(features) -> label`           | Predict SalePrice from a house's attributes (this project) |
| **Unsupervised**    | Features only (no labels)                                   | Structure (clusters, components, anomalies) | Group neighborhoods by similarity |
| **Reinforcement**   | An environment, actions, and a reward signal                | A *policy* mapping states to actions       | Train a bot to play chess         |

This project is **supervised**. Notebook 04 trains supervised models from now on.

### Two kinds of supervised problem

| Problem type      | The label is…       | Example                              | Models for it                |
| ----------------- | ------------------- | ------------------------------------ | ---------------------------- |
| **Regression**    | A number            | "What will this house sell for?" (USD) | Ridge, Random Forest, XGBoost, … |
| **Classification**| A category          | "Will this house sell within 30 days?" (yes/no) | Logistic regression, Random Forest, XGBoost, … |

Our problem is **regression**. Same algorithms, slightly different machinery (the model emits a number, not a probability over classes).

---

## 3. Features, labels, X, and y

Every supervised dataset has the same shape:

```
   features (X)              label (y)
   ───────────────────       ──────────
   row1: x1, x2, x3, …       y1
   row2: x1, x2, x3, …       y2
   row3: x1, x2, x3, …       y3
   ...
```

Conventions you'll see *everywhere*:

- **`X`** — the *feature matrix*. Capital X because it's a matrix (2-D).
- **`y`** — the *target vector*. Lowercase y because it's a vector (1-D).
- **`n`** — number of rows (samples).
- **`p`** or **`d`** — number of columns (features).

In our project: `X` has shape `(2925, 230)` and `y` has shape `(2925,)`. We use `log_saleprice` as `y` rather than raw `saleprice` (see Phase 3 doc).

---

## 4. The train / test split

Imagine you "evaluate" a student by giving them the answer key during the test. They'll score 100%. Did they learn? Unknown.

A model evaluated on the data it was trained on has the same problem — it has *memorized* the answers, not necessarily *learned* the underlying patterns. The fix is to hide some rows from the model during training, then measure performance on those hidden rows. That's the **train/test split**.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```

Key vocabulary:

- **Training set** (`X_train`, `y_train`): what the model *sees*. Typically 70–80% of the data.
- **Test set** (`X_test`, `y_test`): the held-out rows. The model never sees these during training. Touched **once**, at the very end, for an honest performance estimate.
- **`random_state`**: a fixed seed for reproducibility. Without it, every run produces a different split — and slightly different scores.

> ⚠️ **Pitfall — peeking at the test set.** If you compare 50 models on the test set and pick the one with the best score, you've used the test set as a training signal. Your "test" RMSE is now optimistically biased. The cure: use *cross-validation* for model selection, and only look at the test set with the final model.

### Validation set vs. test set vs. cross-validation

Three names for related ideas, often confused:

| Name              | Used for                                            | When                                                        |
| ----------------- | ---------------------------------------------------- | ----------------------------------------------------------- |
| **Test set**      | One-time *final* evaluation                          | Only at the very end, after model and hyperparameters are chosen. |
| **Validation set**| Comparing model candidates / picking hyperparameters | Whenever you'd otherwise be tempted to peek at the test set. |
| **Cross-validation (CV)** | A way to **simulate** many validation sets from limited data | Anytime you're using validation. We use 5-fold CV in this project. |

For this project we **don't** create an explicit validation set — we use 5-fold CV on the training set instead, which is more data-efficient when you have ~3000 rows.

---

## 5. K-fold cross-validation

Cross-validation extracts more information from a fixed training set by *repeating* the train/validate split many ways and averaging.

```
Fold 1:  train: rows 587-2925       validate: rows 0-586
Fold 2:  train: rows 0-585, 1175-2925   validate: rows 586-1174
Fold 3:  train: rows 0-1174, 1762-2925  validate: rows 1175-1761
Fold 4:  train: rows 0-1761, 2349-2925  validate: rows 1762-2348
Fold 5:  train: rows 0-2348              validate: rows 2349-2925
```

For each fold:
1. Fit a fresh copy of the model on the train portion.
2. Score it on the validate portion.

The 5 scores get averaged. That average is a less noisy estimate of out-of-sample performance than any single train/test split.

**Why 5 folds?** Convention. 3 is too noisy; 10 takes twice as long for marginally better estimates. 5 is the sweet spot. Use `KFold(shuffle=True)` so the folds aren't just contiguous slices of a possibly-ordered file.

---

## 6. Overfitting and underfitting

The two failure modes of a model:

| Mode             | Training error | Test error | What's happening                                                  |
| ---------------- | -------------- | ---------- | ----------------------------------------------------------------- |
| **Underfitting** | High           | High       | Model is too simple to capture the patterns. Add features, capacity. |
| **Sweet spot**   | Low            | Low        | Generalizes well.                                                 |
| **Overfitting**  | Low            | High       | Model memorized noise. Reduce capacity, regularize, get more data. |

The classic visual is the **bias-variance tradeoff**:

```
        ↑
        │     underfit         overfit
  Error │      \_                ___/
        │        \___          _/
        │            \_______/        ← total test error
        │             min ↑           
        │___________________________→
                   model complexity
```

- **Bias** = systematic error from the model being too simple (line through curved data).
- **Variance** = sensitivity to the specific training set (memorizing noise).

You can almost always *decrease bias* by making the model more complex, but variance goes up. Cross-validation is the empirical tool for finding the sweet spot.

> 📘 **Mental model.** Underfitting is "I haven't studied enough". Overfitting is "I memorized the practice problems verbatim and can't generalize". The remedy for each is opposite.

---

## 7. The scikit-learn API

Every model in scikit-learn (and most other ML libraries) follows the same three-method contract:

```python
model.fit(X_train, y_train)      # learn from training data
predictions = model.predict(X_new)   # apply learned rules to new data
score = model.score(X_test, y_test)  # built-in default scoring
```

This uniformity is *huge*. Once you know the API, swapping `Ridge()` for `XGBRegressor()` is a one-line change. The rest of the code is identical.

For finer control:

```python
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
```

---

## 8. Pipelines — the leakage shield

Recall from doc 03: if you compute imputation values or scaling stats on the *full* dataset and then split into train/test, your test set's values influenced the preprocessing. That's **data leakage**.

The sklearn `Pipeline` is the structural fix:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge()),
])

pipe.fit(X_train, y_train)   # scaler learns from X_train only
pipe.predict(X_test)         # scaler APPLIES learned values to X_test
```

The pipeline guarantees: *every step inside it sees only the training data during `.fit()`*. Cross-validation respects this — each fold's pipeline learns its scaler from that fold's training portion.

You can put many steps in a pipeline:
- `StandardScaler` → scale numeric features
- `SimpleImputer` → fill missing values
- `OneHotEncoder` → encode categoricals
- `Ridge` / `RandomForestRegressor` / `XGBRegressor` → the actual model

In our project, the data is already cleaned/encoded by Phase 2 — so our pipelines are short: one scaler (for Ridge) plus the model.

---

## 9. Concepts you'll meet in the other Phase 4 docs

- **Hyperparameter** — a knob on the model itself (e.g., `max_depth` in a tree, `alpha` in Ridge). Different from a *parameter*, which is something the model learns (the coefficients themselves).
- **Regularization** — a penalty added to the loss to discourage extreme parameter values. Ridge uses L2 regularization (penalty proportional to the *squared* coefficients).
- **Ensemble** — a model made up of many smaller models whose outputs are combined. Random Forest and XGBoost are both ensembles.
- **Boosting** vs. **Bagging** — two ensembling philosophies. Random Forest *bags* (averages many independent trees). XGBoost *boosts* (each new tree corrects the previous tree's mistakes).
- **Generalization** — performance on data the model has not seen. The thing we actually care about.

Each of these is unpacked in [`07_model_selection.md`](07_model_selection.md), [`08_evaluation_metrics.md`](08_evaluation_metrics.md), and [`09_hyperparameter_tuning.md`](09_hyperparameter_tuning.md).

---

## 10. Common pitfalls (collected)

1. **Evaluating on the training set.** Reports memorization, not learning.
2. **Peeking at the test set during model selection.** Optimism creep — every peek leaks a bit of test information into your decisions.
3. **Forgetting to set `random_state`.** Non-reproducible splits make tiny differences look like meaningful improvements.
4. **Imputing or scaling before splitting.** Data leakage through preprocessing. Use a Pipeline.
5. **Using `model.score(X_train, y_train)` and trusting it.** Returns training-set R²; the more complex the model, the more this fools you.
6. **Cross-validation without shuffling.** If the original file is ordered (e.g., by date), contiguous folds violate the i.i.d. assumption. Use `KFold(shuffle=True)` unless you have a *specific* reason to keep order (like time series — Phase 5).
7. **Comparing models with different preprocessing.** If Ridge sees scaled features and XGBoost doesn't, the comparison isn't apples-to-apples. Wrap both in pipelines so the preprocessing is part of the model.

---

## 11. Further reading

- **scikit-learn — Getting started:** [scikit-learn.org/stable/getting_started.html](https://scikit-learn.org/stable/getting_started.html)
- **scikit-learn — Cross-validation:** [scikit-learn.org/stable/modules/cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html)
- **scikit-learn — Pipelines:** [scikit-learn.org/stable/modules/compose.html](https://scikit-learn.org/stable/modules/compose.html)
- **James, Witten, Hastie, Tibshirani, *An Introduction to Statistical Learning* (ISL).** Chapters 2 (statistical learning) and 5 (resampling). PDF free at [statlearning.com](https://www.statlearning.com).
- **Andrej Karpathy, "A Recipe for Training Neural Networks":** [karpathy.github.io/2019/04/25/recipe](http://karpathy.github.io/2019/04/25/recipe/). Neural-net focused but the *discipline* applies to all ML.

---

## 12. Self-test quiz

1. In one sentence each, define **bias**, **variance**, and explain the *tradeoff* between them.
2. Why is a model's score on its **training data** an unreliable estimate of how well it will do on new data?
3. What problem is a **train/test split** trying to solve, and why do we *also* use **cross-validation** on top of it?
4. Explain in your own words why putting preprocessing **inside** a `Pipeline` (rather than calling it before the split) prevents a kind of data leakage.
5. What does `random_state=42` do, and why is fixing it important *before* you compare two models?
6. Define **hyperparameter** and contrast it with **parameter**. Give an example of each from this project.
7. You compare 10 models by looking at their test-set RMSE and pick the lowest. Why is the chosen model's reported RMSE probably *optimistically biased*, and what's the right way to do this comparison?

<details>
<summary>Show suggested answers</summary>

1. *Bias* is systematic error from the model being too simple to capture the truth (e.g., a straight line through curved data). *Variance* is sensitivity to the particular training set — a complex model fits noise as well as signal. The *tradeoff*: increasing complexity reduces bias but increases variance; the sweet spot minimizes total error and is found empirically (CV).
2. The model can simply memorize the training labels — its training score reflects memorization, not generalization. Performance on held-out data is the only honest estimate of how it'll do on new examples.
3. The split protects against memorization-based optimism. CV further extracts more signal from a fixed dataset by repeating the train/validate process multiple times and averaging — important when total data is limited (~3000 rows here) and a single split's score is noisy.
4. The pipeline ensures that any *learning* step (computing means, scales, medians, etc.) happens during `.fit()` on the training portion only. During `.predict()` or cross-validation folds, the same parameters are *applied*, never re-learned. Doing the preprocessing manually before splitting lets the test-set's values influence the learned scales, which leaks information.
5. `random_state=42` fixes the pseudo-random seed used by the splitter (or any randomized algorithm). Without it, every run produces different splits, and a 1% difference in score between two models could easily come from the split rather than the model. Fixing the seed makes comparisons honest.
6. A *parameter* is something the model learns from data (e.g., a coefficient in Ridge, or a split threshold in a tree). A *hyperparameter* is a knob on the model that *you* choose before training (e.g., `alpha` in Ridge, `max_depth` in XGBoost). The model fits parameters; you tune hyperparameters.
7. Each test-set check leaks a tiny amount of information about the test set into your decision-making. Repeating it 10 times means your choice is partly fit to the test set, so its reported score is biased low. The right pattern: compare models with cross-validation on the training set, choose the best, *then* check the test set once.

</details>
