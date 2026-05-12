# 01 — Data Science Fundamentals

> *Read this once before you write any code. Come back to it whenever a later phase feels disconnected from "the big picture".*

---

## 1. What is Data Science, really?

**Plain English.** Data science is the practice of turning raw data into decisions. That's it. The reason it has its own name (rather than being called "statistics" or "programming") is that doing it well requires three skills at once:

- **Domain knowledge** — you have to know what the numbers *mean*. Predicting house prices without knowing what a `BsmtFinSF1` is will make you cry.
- **Statistics & math** — you have to know which patterns are real and which are noise.
- **Engineering** — you have to actually load, clean, transform, model, and ship the data using code.

A data scientist is someone fluent enough in all three to move between them.

**Analogy.** Think of a chef. Domain knowledge = recipes & ingredients. Statistics = taste-testing & food chemistry. Engineering = the actual cooking & plating. A great chef can't drop any of the three.

---

## 2. The Data Science Lifecycle

Most projects flow through this cycle (sometimes multiple times):

```
       ┌──────────────┐
       │ 1. Problem   │  What decision are we trying to support?
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │ 2. Data      │  Find it, fetch it, store it.
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │ 3. Clean     │  Missing values, types, dedupe.
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │ 4. Explore   │  EDA: stats + plots to build intuition.
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │ 5. Engineer  │  Create features that help the model.
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │ 6. Model     │  Train, tune, validate.
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │ 7. Evaluate  │  Did we actually solve the problem?
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │ 8. Deliver   │  Dashboards, reports, predictions to a downstream system.
       └──────────────┘
```

The arrows in real life are not straight. You will discover at step 7 that step 3 missed something, and have to loop. **That iteration is the work.**

> 📘 **Common pitfall.** Beginners want to spend 90% of their time on **step 6 (Model)** because it's the glamorous part. Pros spend 60–80% of their time on steps 2–5. The model is downstream of how good the data is — *garbage in, garbage out*.

---

## 3. How this project maps to the lifecycle

| Project phase | Lifecycle step(s) | Notebook / Module                                                     |
| ------------- | ----------------- | --------------------------------------------------------------------- |
| Phase 1       | 1, 2              | [`src/data_loader.py`](../src/data_loader.py), `notebooks/01_data_exploration.ipynb` |
| Phase 2       | 3, 4              | `src/preprocessing.py`, `src/eda.py`, `notebooks/02_eda.ipynb`        |
| Phase 3       | 5                 | `notebooks/03_features.ipynb`                                          |
| Phase 4       | 6, 7              | `src/models.py`, `notebooks/04_modeling.ipynb`                         |
| Phase 5       | 6 (time-series)   | `src/forecasting.py`, `notebooks/05_forecasting.ipynb`                |
| Phase 6       | 7 (interpretation)| `notebooks/06_interpretation_and_roi.ipynb`                            |
| Phase 7       | 8                 | `src/export.py`, Power BI                                              |

By the end you will have **lived the full cycle once**, on real data, with real friction.

---

## 4. Two kinds of questions data science answers

This matters because it dictates which **algorithm family** is appropriate.

| Question type | Example                                         | Algorithm family       | We use it in… |
| ------------- | ----------------------------------------------- | ---------------------- | ------------- |
| **Predict a number** | "What will this house sell for?"        | *Regression*           | Phases 4 & 6  |
| **Predict a category** | "Will this house sell within 30 days?" | *Classification*       | (not this project) |
| **Predict the future** | "What's the median price in this ZIP next year?" | *Time-series forecasting* | Phase 5 |
| **Find structure** | "Are there 'types' of neighborhoods?"    | *Clustering / unsupervised* | Touched in Phase 3 |
| **Explain a result** | "*Why* did the model predict $500k?"   | *Interpretability* (SHAP) | Phase 6 |

If you ever feel lost about *what* you're trying to do at a given moment, ask yourself which row of this table you're on.

---

## 5. Key concepts you'll meet in this project

These will each get their own deep-dive doc later, but here are the one-paragraph teasers so the vocabulary doesn't blindside you:

- **Feature** — a column used as input to the model (e.g., `LotArea`). The target is *not* a feature — it's what we're predicting.
- **Target / Label** — the thing we want to predict (`SalePrice`). Sometimes also called `y` while features are `X`.
- **Training set vs. Test set** — we split the data into two pieces, fit the model on the first, score it on the second. This is the only honest way to measure "does it generalize?".
- **Overfitting** — model memorizes the training data instead of learning the underlying patterns. It scores great on the training set and terribly on the test set. Half of practical ML is fighting overfitting.
- **Bias-Variance tradeoff** — simple models miss the pattern (high bias). Complex models also fit the noise (high variance). You have to dial in the sweet spot.
- **Cross-validation (CV)** — instead of one train/test split, do many and average the scores. More honest estimate of generalization.
- **Hyperparameter** — a knob on the model itself (e.g., tree depth). Different from a *parameter*, which is something the model learns (e.g., a weight). Tuning means searching the knob space.
- **Metric** — a number that summarizes how good the model is. For regression: RMSE, MAE, R². Doc 08 will pull these apart.
- **Pipeline** — a sequence of steps (impute → scale → encode → model) wrapped together so the *same* sequence is applied to training and prediction data. Prevents data leakage.
- **Data leakage** — the model accidentally sees information at training time that it wouldn't have at prediction time. Silent killer; produces models that look amazing in testing and crash in production.

You don't need to memorize these yet. They will keep coming back and gradually click.

---

## 6. The math you'll actually use

You do **not** need a math degree for this project. You do need to be comfortable with:

- **Means, medians, percentiles, standard deviation.** If "the 75th percentile of `LotArea` is 11,600 sq ft" doesn't mean something concrete to you, take 20 minutes on Khan Academy.
- **Distributions** — what does it mean for data to be "skewed"? What's a normal/bell curve?
- **Correlation** — how to read a correlation coefficient (-1 to +1) and what "correlated does not equal caused" means.
- **Linear algebra (just a taste)** — what a matrix is, what "fitting a line" means in 2D. We won't multiply matrices by hand.

Each deep-dive doc adds a *Math/Statistics* section that explains anything specific to that topic.

---

## 7. Tools we use and why

| Tool             | Why                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------- |
| **Python**       | Industry standard for data science; huge ecosystem.                                  |
| **pandas**       | DataFrames — table-like data with named columns. Think Excel with code.              |
| **NumPy**        | The fast numerical array under pandas. We use it for math, not for tables.           |
| **scikit-learn** | The unified ML API: same `.fit() / .predict()` across dozens of algorithms.          |
| **XGBoost**      | Gradient-boosted trees. The "competition winner" algorithm for tabular data.         |
| **Prophet / statsmodels** | Time-series forecasting (Phase 5). Prophet is friendly; statsmodels is classical. |
| **matplotlib / seaborn / plotly** | Plotting. matplotlib is the base; seaborn is prettier defaults; plotly is interactive. |
| **Jupyter**      | Notebooks for **exploration**. Great for thinking. Bad for production — that's why we *also* have `src/*.py`. |
| **SQLite**       | A serverless SQL database in a single file. Power BI connects to it easily.          |
| **Power BI**     | The dashboard layer. End users never run Python — they open Power BI.                |

Why two homes for code (notebooks and `src/*.py`)? Notebooks are for *deciding what to do*; `.py` files are *the thing you do*, reliably, the same way every time. The notebook calls into the `.py` so they stay in sync.

---

## 8. How to learn from this project

A repeated trap when learning from a tutorial repo: you read it, nod along, run it, and remember nothing. Things that actually work:

1. **Run every cell yourself.** Don't trust mine — print the shapes, the heads, the dtypes.
2. **Break things on purpose.** Change a parameter, see how the result moves. Hypothesis-test the code.
3. **Re-explain the concept in your own words** at the end of each phase, out loud or in a note. If you can't, you didn't get it yet.
4. **Use the quiz at the bottom of each doc.** No peeking.

---

## 9. Further reading

- **Free / online**
  - *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani) — the friendliest serious ML textbook. PDF is free at [statlearning.com](https://www.statlearning.com/).
  - *Python Data Science Handbook* (Jake VanderPlas) — free at [jakevdp.github.io/PythonDataScienceHandbook/](https://jakevdp.github.io/PythonDataScienceHandbook/).
  - scikit-learn user guide — surprisingly readable: [scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html).
- **Paid / classic**
  - *The Elements of Statistical Learning* — the heavier sibling of ISL. Same authors. Free PDF too.
  - *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Géron) — best modern practitioner book.

---

## 10. Self-test quiz

Cover the answers and write yours down first.

1. In one sentence, what's the difference between **regression** and **classification**?
2. Why do we hold out a **test set** instead of evaluating on the data we trained on?
3. What does it mean for a model to **overfit**?
4. List three steps in the data-science lifecycle that typically take more time than the "modeling" step in real projects.
5. Why do we keep production code in `.py` files separate from `.ipynb` notebooks?

<details>
<summary>Show suggested answers</summary>

1. Regression predicts a *number* (e.g., a price); classification predicts a *category* (e.g., spam vs. not).
2. Because evaluating on the training data tells you how well the model *memorized*, not how well it *generalizes* to new inputs.
3. The model fits the training data so closely that it captures noise as well as signal — high training score, low test score.
4. Any three of: data acquisition, cleaning, EDA, feature engineering, evaluation, delivery. Modeling itself is often <20% of the work.
5. Reliability and reuse. Notebooks are for thinking; `.py` modules give you a single, importable, testable source of truth that both notebooks and downstream systems can call.

</details>
