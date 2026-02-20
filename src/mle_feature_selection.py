import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.config import ALPHA


def calculate_vif(X_df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_df.values, i)
        for i in range(X_df.shape[1])
    ]
    return vif_data


def mle_logistic_selection(df, target):
    y = df[target].map({"No": 0, "Yes": 1})
    X = pd.get_dummies(df.drop(target, axis=1), drop_first=True)
    X = X.astype(float)
    print("\nCalculating VIF...")
    vif_df = calculate_vif(X)
    print(vif_df.sort_values("VIF", ascending=False))
    high_vif_features = vif_df[vif_df["VIF"] > 10]["feature"].tolist()

    if high_vif_features:
        print("\nDropping high VIF features:", high_vif_features)
        X = X.drop(columns=high_vif_features)

    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False)

    summary = model.summary2().tables[1]
    significant = summary[summary["P>|z|"] < ALPHA]
    selected_features = significant.index.tolist()

    return selected_features, model


def pseudo_r2(model):
    return 1 - (model.llf / model.llnull)
