"""
Data loading and preprocessing pipeline construction.

Design decision: preprocessing is expressed as a sklearn ColumnTransformer
rather than manual pandas transforms + separately saved encoders. This
guarantees:
  1. No train/test leakage: when wrapped in a Pipeline and passed to
     GridSearchCV, the transformer is re-fit on ONLY the training fold
     at every CV split, automatically -- not once on the full dataset.
  2. No train/inference mismatch: the exact same fitted object is reused
     at prediction time (loaded from the saved model), so there is no
     risk of the API applying a differently-ordered or differently-fit
     encoding than what the model was trained on.
  3. Consistent column order: ColumnTransformer always applies transforms
     to named columns, not positional indices, so column reordering in
     incoming data can never silently misalign features.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Churn"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

# OneHotEncoder(drop="if_binary") handles 2-class columns as a single 0/1
# column and multi-class columns as full one-hot -- one encoder, no need
# to separately maintain LabelEncoders for binary columns.
#
# sparse_output=False is explicit on purpose: OneHotEncoder defaults to
# sparse output, and a ColumnTransformer with mixed sparse/dense
# transformers only densifies the combined result if the one-hot density
# happens to clear sklearn's internal sparse_threshold (0.3). That's an
# implicit, data-shape-dependent behavior. TorchMLPClassifier calls
# np.asarray(X, dtype=np.float32) on whatever this returns, which breaks
# silently on a sparse matrix, so we pin the output type instead of
# depending on how sparse today's category counts happen to be.
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def load_data(path: str = "data/churn.csv") -> pd.DataFrame:
    """Load raw CSV and apply deterministic, distribution-independent
    cleaning only (safe to do before train/test split since it does not
    depend on any statistic of the data -- no leakage risk)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # guard against stray whitespace

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges has blank strings for brand-new customers (tenure=0,
    # confirmed by inspecting the 11 affected rows). Filling with 0 is
    # factually correct here, not an approximation -- they've paid
    # nothing so far, not "average" or "unknown".
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


def split_features_target(df: pd.DataFrame):
    X = df[ALL_FEATURE_COLS].copy()
    y = df[TARGET_COL].apply(lambda v: 1 if v == "Yes" else 0)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Returns an unfit ColumnTransformer. Fit only inside a Pipeline /
    GridSearchCV so it is always refit per-fold on training data only."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )