This repository showcases three end‑to‑end machine learning projects built with a consistent, production‑minded workflow. Each project includes clean preprocessing, careful validation, strong baselines, and scalable models.

Movie Genre Classification (multi‑label NLP)

Credit Card Fraud Detection (highly imbalanced tabular)

Customer Churn Prediction (customer analytics and uplift focus)

I used Perplexity and Gemini for code assistance, debugging, and quick literature checks to accelerate development and validation.

Features at a Glance
Reproducible pipelines with consistent preprocessing in train/inference

Strong baselines (Logistic Regression) and advanced models (Decision Trees, Gradient Boosting)

Robust evaluation: stratified CV, class weights, calibrated thresholds

Clear metrics aligned to business goals (PR‑AUC, F1, Lift, Gains)

Clean, modular code structure ready for deployment

Repository Structure
/genre-classification

data/ (not included; instructions to fetch)

notebooks/

src/

data_prep.py

fe_text.py

model_train.py

predict.py

utils.py

models/ (artifacts, .gitignored)

README.md

/fraud-detection

data/

notebooks/

src/

fe_tabular.py

model_train.py

thresholding.py

utils.py

models/

README.md

/churn-prediction

data/

notebooks/

src/

fe_customer.py

model_train.py

evaluation.py

utils.py

models/

README.md

/common

config/

metrics/

utils/

environment.yml or requirements.txt

Makefile (optional)

LICENSE

Setup
Clone
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

Environment

Option A: Conda
conda env create -f environment.yml
conda activate ml-portfolio

Option B: pip
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Data

Place raw data under each project’s data/ directory or follow the mini‑README in each project to auto‑download public datasets.

Do not commit raw data.

Tech Stack
Python, NumPy, Pandas, Scikit‑learn

NLP: TF‑IDF, basic word embeddings

Models: Logistic Regression, Decision Trees, Gradient Boosting (e.g., GradientBoostingClassifier or XGBoost/LightGBM if enabled)

Evaluation: scikit‑learn metrics, PR‑AUC, ROC‑AUC, F1, Lift/Gains

Experimentation: notebooks + scripts; configuration via YAML

Project 1: Movie Genre Classification
Goal

Predict one or more genres from a movie plot summary (multi‑label classification).

Data & Features

Text normalization: lowercasing, punctuation/stopword removal.

Vectorization: TF‑IDF (word and bi‑grams), experiments with embeddings for semantic context.

Models

Baseline: One‑vs‑Rest Logistic Regression on TF‑IDF.

Comparators: Decision Trees, Gradient Boosting.

Tuning: C (regularization), class weights, threshold per label.

Evaluation

Macro/Micro F1, exact‑match accuracy, per‑label precision/recall.

Stratified CV; consistent preprocessing in training and inference.

How to run

Training: python genre-classification/src/model_train.py --config config/genre.yaml

Inference: python genre-classification/src/predict.py --input data/sample_plots.csv --output preds.csv

Project 2: Credit Card Fraud Detection
Goal

Flag fraudulent transactions under extreme class imbalance.

Data & Features

Standardize numeric fields.

Time‑window features: transaction velocity, amount ratios, recent declines/chargebacks.

Class weighting; careful handling of any resampling to avoid leakage.

Models

Baseline: Logistic Regression (interpretable, fast).

Advanced: Decision Trees, Gradient Boosting (often best PR‑AUC).

Threshold calibration to meet recall/precision targets.

Evaluation

PR‑AUC as primary, recall@fixed precision, ROC‑AUC, cost‑sensitive metrics.

Stratified splits/CV; auditability for fraud‑ops review.

How to run

Training: python fraud-detection/src/model_train.py --config config/fraud.yaml

Threshold tuning: python fraud-detection/src/thresholding.py --metric pr_at_precision --target 0.90

Inference: python fraud-detection/src/predict.py --input data/transactions_test.csv --output flags.csv

Project 3: Customer Churn Prediction
Goal

Identify customers at risk of leaving to target retention actions.

Data & Features

Behavioral and relationship features: usage frequency/recency, support interactions, tenure, plan type, payment history, rolling trends and ratios.

Models

Baseline: Logistic Regression to understand feature contributions.

Advanced: Decision Trees, Gradient Boosting for non‑linear interactions.

Evaluation

ROC‑AUC, PR‑AUC; Lift and Cumulative Gains in top deciles for campaign sizing.

Simple what‑if analyses to inform retention offers and budget allocation.

How to run

Training: python churn-prediction/src/model_train.py --config config/churn.yaml

Evaluation: python churn-prediction/src/evaluation.py --report reports/churn_eval.json

Inference: python churn-prediction/src/predict.py --input data/customers_scored.csv --output risk_scores.csv

Results Summary
Genre: Strong micro/macro F1 with TF‑IDF + Logistic Regression; threshold tuning improved rare‑genre recall.

Fraud: Gradient Boosting achieved best PR‑AUC; calibrated threshold met recall targets at acceptable precision.

Churn: Gradient Boosting improved Lift in top 10–20% risk segments, enabling efficient, budget‑aware outreach.

Reproducibility and Quality
Deterministic seeds for CV and training where possible.

Train/validation splits that preserve label distribution.

No leakage between feature engineering and splits.

Consistent preprocessing artifacts (vectorizers/scalers) saved with models.

Deployment Notes
Save and version model + preprocessing pipeline (e.g., joblib).

Provide a single predict(input_df) interface for services/Batch.

Monitor drift: input feature stats, output score stability, and calibration.

Retrain cadence: time‑based or performance‑triggered.

Roadmap
Swap basic embeddings with modern sentence embeddings for the genre task.

Calibrated probabilities with Platt/Isotonic where helpful.

Add SHAP/feature importance dashboards for churn and fraud.

Optional: migrate Gradient Boosting to LightGBM/XGBoost with GPU support.

How to Contribute
Open an issue with a clear description and minimal reproducible example.

Use conventional commits and submit PRs to a feature/* branch.

Add tests for new utilities and keep notebooks lightweight.

Acknowledgments
Coding assistance and quick research support from Perplexity and Gemini helped speed iteration and improve code quality.
