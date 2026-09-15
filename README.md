# Heart Disease Prediction

Machine learning model to predict the risk of heart disease from clinical data.

## About

Binary classification: given a patient's clinical measurements, predict whether they have heart disease (1) or not (0).

The focus is on **recall for the positive class** — in a medical context, missing a sick patient is far more dangerous than a false alarm.

## Dataset

[Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) — UCI Machine Learning Repository.

- 303 patients, 14 clinical features
- Target: presence of heart disease

## Approach

- **Preprocessing:** removed 2 rows with invalid `thal` values, one-hot encoded categorical features
- **Scaling:** `StandardScaler` (required for logistic regression convergence)
- **Class imbalance:** used `class_weight="balanced"` — dataset has 72% healthy / 28% sick
- **Threshold tuning:** final prediction threshold lowered to **0.3** (instead of default 0.5) to maximize recall

## Results

Evaluated on a held-out test set (61 patients):

| Metric | Value |
|--------|-------|
| Accuracy | 86.9% |
| **Recall (sick)** | **0.88** |
| Precision (sick) | 0.71 |
| F1 (sick) | 0.79 |

The model correctly identifies **15 out of 17 sick patients**, missing only 2.

For comparison, a standard logistic regression (threshold 0.5, no class balancing) achieves 84% accuracy but only **0.53 recall** — missing 8 of 17 sick patients.

## Key Takeaways

- Accuracy alone is misleading on imbalanced medical data
- `class_weight="balanced"` significantly improves recall on the minority class
- Threshold tuning is a powerful, cheap way to trade precision for recall
- The final model must be chosen with a **doctor**, not by the data scientist alone

## Limitations

- Small test set (61 patients) — differences of 2–5 cases are not statistically strong
- Dataset is from 1988 — results may differ on modern data
- Model is a **decision-support tool**, not a replacement for medical judgment

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- Google Colab

## How to Run

Open the notebook in Google Colab (no local setup required):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bendarbasil-svg/heart-disease-prediction/blob/main/heart-disease-prediction.ipynb)

## Author

[Basil Bendarsky](https://github.com/bendarbasil-svg)
