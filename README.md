---
title: Churn MLOps
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Churn Prediction MLOps Pipeline

An end-to-end machine learning pipeline that predicts telecom customer churn — covering data preprocessing, model training and hyperparameter tuning, experiment tracking, containerized serving, CI/CD, and a live demo UI.

**Live demo:** `https://YOUR_USERNAME-churn-mlops.hf.space/ui`
**API docs:** `https://YOUR_USERNAME-churn-mlops.hf.space/docs`

---

## What it does

Given a telecom customer's account details (tenure, contract type, monthly charges, services subscribed, etc.), the model returns a churn probability and a risk label. The project's focus is less on the model itself and more on the full lifecycle around it: reproducible training, tracked experiments, and a deployable, testable, monitorable service — mirroring how churn-prediction systems run in production.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — IBM Sample Data, CC-BY-4.0 license. ~7,000 customers, ~20 features, binary target (`Churn`: Yes/No). Class distribution is imbalanced (~74% No / ~26% Yes), which directly shaped the modeling and evaluation choices below.

## Architecture

```
Data (CSV)
   → Preprocessing (pandas: TotalCharges cleanup, sklearn ColumnTransformer:
      StandardScaler + OneHotEncoder, fit only on training folds)
   → Training (GridSearchCV, 5-fold stratified CV, scored on F1):
      Logistic Regression | Random Forest | XGBoost | PyTorch MLP (GPU)
   → MLflow tracking (params, metrics, models logged per run)
   → Best model exported as a single joblib pipeline (preprocessing + model
      bundled together — no separate encoder files to keep in sync)
   → FastAPI serves /predict, /health, /model-info, and a static /ui form
   → Dockerized, built and tested via GitHub Actions on every push
   → Deployed on Hugging Face Spaces (Docker SDK)
```

## Why F1, not accuracy

With ~74% of customers not churning, a model that always predicts "no churn" scores ~74% accuracy while catching zero actual churners. Every model here is evaluated primarily on **F1 and recall for the churn class**, with accuracy and ROC-AUC reported alongside for context — F1 was also the metric `GridSearchCV` optimized for during hyperparameter search, not accuracy.

| Model | Accuracy | F1 | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (class-weighted) | — | — | — | — |
| Random Forest (class-weighted) | — | — | — | — |
| XGBoost (scale_pos_weight) | — | — | — | — |
| PyTorch MLP (GPU, weighted loss) | — | — | — | — |

*(Fill in from your MLflow comparison run before publishing — screenshot the MLflow UI table as a second reference.)*

## Tech stack

| Layer | Tool |
|---|---|
| Data / training | pandas, scikit-learn, XGBoost, PyTorch (CUDA) |
| Experiment tracking | MLflow |
| Serving | FastAPI + Pydantic |
| Frontend | Vanilla HTML/JS, served as static files via FastAPI |
| Packaging | Docker |
| CI/CD | GitHub Actions |
| Deployment | Hugging Face Spaces (Docker SDK) |

## Project structure

```
churn-mlops/
├── src/
│   ├── preprocess.py      # data loading, ColumnTransformer pipeline
│   ├── train.py            # trains & tunes all 4 models, logs to MLflow
│   └── torch_model.py       # sklearn-compatible PyTorch MLP wrapper
├── api/
│   ├── main.py              # FastAPI app: /predict, /health, /model-info, /ui
│   └── static/
│       └── index.html       # prediction form UI
├── artifacts/               # exported champion model + metadata (used at inference)
├── tests/
│   └── test_api.py
├── .github/workflows/
│   └── ci.yml                # runs tests + builds Docker image on push
├── Dockerfile
├── requirements.txt
└── README.md
```

## Running locally

```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train (optional — artifacts/ already contains an exported model)
python src/train.py
mlflow ui   # view experiment comparisons at localhost:5000

# Serve
uvicorn api.main:app --reload
```
Visit `http://localhost:8000/ui` for the form, `http://localhost:8000/docs` for the API.

## Running with Docker

```bash
docker build -t churn-mlops .
docker run -p 8000:8000 churn-mlops
```

## Key design decisions

- **Single-pipeline artifact**: preprocessing (scaler + encoder) and the model are bundled into one sklearn `Pipeline` and saved as one joblib file, not separate encoder/scaler files. This eliminates train/inference mismatch and guarantees the API applies the exact transformation the model was trained on.
- **No data leakage**: the preprocessor is fit inside `GridSearchCV`, meaning it's refit on only the training fold at every cross-validation split — never on data it will later be evaluated against.
- **`TotalCharges` missing values filled with 0, not the median**: investigated first — all missing values belonged to customers with `tenure = 0` (brand-new customers who haven't been billed yet), so 0 is the factually correct value, not an imputed guess.
- **Class imbalance handled via `class_weight="balanced"` / `scale_pos_weight` / weighted loss**, not naive oversampling, and models are compared on F1/recall rather than accuracy.
- **Decision threshold stored in `champion_meta.json`**, not hardcoded to 0.5 — the deployed API uses whatever threshold was actually tuned during training.

## Known limitations

- Free-tier hosting means the service may take 30–50 seconds to respond on first request after inactivity (cold start).
- Input validation checks types but not sane ranges (e.g. negative tenure isn't rejected) — a production version would add stricter bounds.
- Monitoring is minimal (request logging only); no automated drift detection is deployed yet.
