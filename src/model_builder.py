import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from src.config import TEST_SIZE, RANDOM_STATE


def build_model(df):

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first"), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    model = RandomForestClassifier(random_state=RANDOM_STATE)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),  # Moved here
        ("model", model)
    ])

    param_dist = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        cv=5,
        scoring="roc_auc",
        n_iter=4,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(
        search.best_estimator_,
        method="sigmoid",
        cv=5
    )

    calibrated.fit(X_train, y_train)

    y_pred = calibrated.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    print(f"Final ROC-AUC: {auc:.4f}")

    return calibrated
