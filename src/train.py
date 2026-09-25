"""
Trains and tunes 4 models on the churn dataset, tracks every run in
MLflow, and registers the best one by F1 (chosen over accuracy -- see
README for why, given the ~74/26 class imbalance).

Each model is a full sklearn Pipeline (preprocessor + classifier), so:
  - GridSearchCV refits the preprocessor per CV fold -> no leakage
  - the saved model is one artifact that does raw-input -> prediction,
    nothing extra to keep in sync at inference time

Model selection uses the CV f1 score (search.best_score_), never the
test-set score. The test set is evaluated for every model so the final
table is honest, but it plays no part in choosing the winner -- using
test performance to pick between models is the same leak as tuning
hyperparameters on it, just one level up.
"""

import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocess import ALL_FEATURE_COLS, build_preprocessor, load_data, split_features_target
from torch_model import TorchMLPClassifier

RANDOM_STATE = 42


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def run_grid_search(name, pipeline, param_grid, X_train, y_train, X_test, y_test,
                    n_jobs=1, code_paths=None, skops_trusted_types=None,
                    serialization_format=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # scoring="f1" (not accuracy) -- selects the model config that best
    # catches churners on an imbalanced target, matching the metric
    # that actually reflects the business problem
    search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=cv, n_jobs=n_jobs)

    with mlflow.start_run(run_name=name):
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        metrics = evaluate(best_model, X_test, y_test)
        metrics["cv_best_f1"] = float(search.best_score_)  # selection metric

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        # code_paths: only needed for TorchMLPClassifier, a custom class
        # defined in torch_model.py. Without it, mlflow.pyfunc.load_model
        # in another environment (e.g. the API container) fails with
        # ModuleNotFoundError because it can't find that class to unpickle.
        #
        # skops_trusted_types: mlflow's sklearn flavor serializes with skops,
        # which refuses to load a file containing an sklearn.tree._tree.Tree
        # object (the node storage used by every tree-based estimator) unless
        # that type is explicitly marked trusted -- it stores raw array
        # indices with no bounds checking, so skops treats it as a narrow
        # deserialization risk for files from an untrusted source. This is
        # our own model from our own training run, so trusting it here is
        # fine; only pass it for estimators that actually contain trees
        # (RandomForest, GradientBoosting, etc.) rather than blanket-trusting
        # everything.
        #
        # serialization_format: left at mlflow's default (skops) for every
        # model except pytorch_mlp. TorchMLPClassifier wraps live torch
        # internals (tensors, nn.Sequential, nn.Linear, an OrderedDict state
        # dict) that skops has no concept of, so its trusted-type audit would
        # likely surface several more unfamiliar types one at a time instead
        # of one clean list like sklearn's Tree or xgboost's Booster did.
        # cloudpickle has no trusted-type gate and is the right format for a
        # custom estimator like this.
        log_kwargs = dict(code_paths=code_paths, skops_trusted_types=skops_trusted_types)
        if serialization_format is not None:
            log_kwargs["serialization_format"] = serialization_format
        mlflow.sklearn.log_model(best_model, "model", **log_kwargs)

        print(f"\n{name}")
        print("  best params:", search.best_params_)
        print("  cv f1 (selection metric):", round(metrics["cv_best_f1"], 4))
        print("  test metrics:", {k: round(v, 4) for k, v in metrics.items()
                                   if k != "cv_best_f1"})

        return best_model, metrics


def main():
    mlflow.set_experiment("churn-prediction")

    df = load_data("data/churn.csv")
    X, y = split_features_target(df)

    # Split BEFORE any fitting happens -- test set stays completely
    # untouched until final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # imbalance ratio, ~2.77

    results = {}

    # 1. Logistic Regression
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear",
                                    class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    grid = {"clf__C": [0.01, 0.1, 1, 10]}
    results["logreg"] = run_grid_search("logreg", pipe, grid, X_train, y_train,
                                        X_test, y_test, n_jobs=1)

    # 2. Random Forest
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [6, 10, None],
        "clf__min_samples_leaf": [1, 5],
    }
    results["random_forest"] = run_grid_search("random_forest", pipe, grid, X_train, y_train,
                                               X_test, y_test, n_jobs=1,
                                               skops_trusted_types=["sklearn.tree._tree.Tree"])

    # 3. XGBoost
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, tree_method="hist",
        )),
    ])
    grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.05, 0.1],
    }
    results["xgboost"] = run_grid_search("xgboost", pipe, grid, X_train, y_train,
                                         X_test, y_test, n_jobs=1,
                                         skops_trusted_types=["xgboost.core.Booster",
                                                              "xgboost.sklearn.XGBClassifier"])

    # 4. PyTorch MLP (GPU if available)
    # n_jobs=1 here is not optional: GridSearchCV's n_jobs=-1 forks worker
    # processes, and a CUDA context does not survive being forked, so
    # this one must stay single-process regardless of how fast your CPU
    # models ran above.
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", TorchMLPClassifier(pos_weight=pos_weight)),
    ])
    grid = {"clf__lr": [1e-3, 5e-4], "clf__epochs": [30, 50]}
    torch_model_path = str(Path(__file__).resolve().parent / "torch_model.py")
    results["pytorch_mlp"] = run_grid_search("pytorch_mlp", pipe, grid, X_train, y_train,
                                             X_test, y_test, n_jobs=1,
                                             code_paths=[torch_model_path],
                                             serialization_format="cloudpickle")

    # Pick best by CV f1 (the selection metric every GridSearchCV already
    # optimized for) -- NOT by test f1. Test metrics are printed for every
    # model below so you get an honest final table, but they never decide
    # the winner.
    print("\n=== Summary ===")
    best_name, best_cv_f1 = None, -1
    for name, (_, metrics) in results.items():
        print(f"{name:15s} cv_f1={metrics['cv_best_f1']:.4f}  "
              f"test_f1={metrics['f1']:.4f}  recall={metrics['recall']:.4f}  "
              f"roc_auc={metrics['roc_auc']:.4f}  accuracy={metrics['accuracy']:.4f}")
        if metrics["cv_best_f1"] > best_cv_f1:
            best_name, best_cv_f1 = name, metrics["cv_best_f1"]

    print(f"\nBest model by CV F1: {best_name} ({best_cv_f1:.4f})")
    print(f"  (its test F1 was {results[best_name][1]['f1']:.4f} -- reported, not used to select)")
    print("Register this run in the MLflow UI as 'churn-model' for the tracking history.")

    # Also export the champion as a plain file. Render (or any host) can't
    # reach the MLflow server on your laptop, so api/main.py loads this
    # instead of hitting models:/churn-model/latest at serving time.
    # NOTE: if pytorch_mlp ever wins, this joblib file -- and therefore your
    # Docker image -- will need torch installed to load it. Worth checking
    # which model won before writing the Dockerfile.
    project_root = Path(__file__).resolve().parent.parent
    artifact_dir = project_root / "artifacts"
    artifact_dir.mkdir(exist_ok=True)

    champion_model, champion_metrics = results[best_name]
    joblib.dump(champion_model, artifact_dir / "champion_model.joblib")

    meta = {
        "model_name": best_name,
        "decision_threshold": 0.5,  # matches evaluate(); revisit if you add threshold tuning
        "features": ALL_FEATURE_COLS,
        "cv_f1": best_cv_f1,
        "test_metrics": {k: round(v, 4) for k, v in champion_metrics.items()},
    }
    (artifact_dir / "champion_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nExported for serving: {artifact_dir / 'champion_model.joblib'}")
    print(f"                       {artifact_dir / 'champion_meta.json'}")


if __name__ == "__main__":
    main()