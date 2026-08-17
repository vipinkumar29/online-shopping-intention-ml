# Online Shoppers Purchasing Intention — Classification

This project trains five classification models to predict whether an online
shopping session ends in a purchase, and serves them through a small Streamlit
web app where you can upload test data, pick a model, and see how it performs.

- **GitHub repository:** https://github.com/vipinkumar29/online-shopping-intention-ml
- **Live Streamlit app:** https://online-shopping-intention-ml-fxzhhvgvc3j3iufkxv8zap.streamlit.app/

## a. Problem statement

An online retailer wants to know whether a visitor's browsing session will end in
a purchase. Predicting this helps the business target undecided visitors, plan
inventory, and estimate conversions. Given the behavioural and session-level
features of a visit, the task is to predict the target `Revenue`
(`True` = purchase, `False` = no purchase). This is a **binary classification**
problem.

## b. Dataset description

- **Source:** UCI Machine Learning Repository — *Online Shoppers Purchasing
  Intention Dataset* (ID 468):
  https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- **Instances:** 12,330 sessions collected over a one-year period.
- **Features:** 17 (10 numeric + 7 categorical). Target: `Revenue`.
- **Class balance:** imbalanced — 1,908 purchases (**15.5%**) and 10,422 non-purchases.

| Type | Columns |
|------|---------|
| Numeric (10) | `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay` |
| Categorical (7) | `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend` |

**Preprocessing.** Numeric columns are standardised with `StandardScaler` (needed
for kNN and Logistic Regression, which are sensitive to feature scale) and the
categorical columns are one-hot encoded. Both steps sit inside each model's
pipeline, so the same transformation is applied during training and when the app
scores new data. The data is split 80/20 with stratification on the target to
keep the 15.5% purchase rate in both parts. Since the classes are imbalanced,
`class_weight="balanced"` is used for the models that support it.

## c. GitHub repository link

https://github.com/vipinkumar29/online-shopping-intention-ml

```
online-shopping-intention-ml/
├── app.py                      # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv               # 20% test split used by the app
├── data/
│   └── online_shoppers_intention.csv   # full dataset (training only)
├── model/
│   ├── common.py               # shared preprocessing, split and metrics helpers
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   ├── metrics.json            # scores for the comparison table
│   └── *.pkl                   # the five saved models
└── notebooks/
    └── training.ipynb          # same training, as a notebook (run on BITS Lab)
```

## d. Models used

Each model has its own script under `model/`. They all import `model/common.py`
so they share the same preprocessing and the same train/test split, which means
every model is trained and evaluated on identical data.

1. **Logistic Regression** — linear baseline.
2. **Decision Tree** — depth- and leaf-limited to avoid overfitting.
3. **k-Nearest Neighbors** — distance based, `k ≈ √N`.
4. **Naive Bayes (Gaussian)** — generative probabilistic model.
5. **Random Forest** — bagged ensemble of trees.

Run a model to train it and save its `.pkl`, for example:

```bash
python model/random_forest.py
```

### Comparison table (held-out test set, 2,466 sessions)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8410 | 0.8932 | 0.4913 | 0.7435 | 0.5917 | 0.5145 |
| Decision Tree | 0.8362 | 0.9231 | 0.4836 | 0.8482 | 0.6160 | 0.5548 |
| kNN | 0.8735 | 0.8507 | 0.7823 | 0.2539 | 0.3834 | 0.3990 |
| Naive Bayes | 0.2729 | 0.7334 | 0.1726 | 0.9738 | 0.2933 | 0.1289 |
| Random Forest (Ensemble) | **0.8747** | 0.9220 | 0.5714 | 0.7644 | **0.6540** | **0.5886** |

Numbers come from `model/metrics.json` (random_state = 42). The app recomputes
them live on the uploaded test data.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A solid baseline with a strong AUC (0.89) and good recall (0.74). The linear boundary picks up the dominant `PageValues` / exit-rate signal. Balancing the classes trades precision (0.49) for recall, which is a reasonable choice when we care about catching buyers. |
| Decision Tree | Best AUC (0.92) and highest recall (0.85), but low precision (0.48), so it flags a lot of non-buyers as buyers. A single tree captures non-linear splits the linear model misses but is a high-variance model, which is exactly what the ensemble improves on. |
| kNN | Highest precision (0.78) but very low recall (0.25) — it only spots purchases that fall in dense buyer neighbourhoods. After one-hot encoding the feature space is wide, so distances become less meaningful and most true buyers get missed. |
| Naive Bayes | Very low accuracy (0.27) and near-total recall (0.97): it labels almost everything as a purchase. The feature-independence and Gaussian assumptions don't hold here (one-hot dummies aren't Gaussian and the duration/rate columns are correlated), so its probabilities are poorly calibrated. |
| Random Forest (Ensemble) | Best overall. Averaging many de-correlated trees fixes the single tree's variance and gives the best MCC (0.59) and F1 (0.65) with a strong AUC (0.92) and the highest accuracy (0.87). It balances precision and recall better than any other model. |
| **Overall winner for this dataset** | **Random Forest** — highest MCC (0.59), F1 (0.65) and accuracy (0.87), with a near-top AUC. MCC is used as the tie-breaker because it stays reliable under the 15.5% class imbalance. If the goal were to catch as many buyers as possible (recall), the Decision Tree or Logistic Regression would be worth a look. |

## How to run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) retrain the models — regenerates model/*.pkl, metrics.json, test_data.csv
python model/logistic_regression.py
python model/decision_tree.py
python model/knn.py
python model/naive_bayes.py
python model/random_forest.py

# 3. Launch the app
streamlit run app.py
```

Open the local URL Streamlit prints, upload `test_data.csv` (or use the bundled
default), pick a model, and view its metrics and confusion matrix.

## Streamlit app features

- **CSV upload** of test data (defaults to the bundled `test_data.csv`).
- **Model dropdown** to switch between the five models.
- **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC.
- **Confusion matrix** heatmap and a full **classification report**.
- A table comparing all five models on the held-out test set.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. New app → select this repo → branch `main` → main file `app.py` → Deploy.
4. Add the resulting public URL to the *Live Streamlit app* link above.

## Tech stack

Python, scikit-learn, pandas, numpy, matplotlib, seaborn, Streamlit.
