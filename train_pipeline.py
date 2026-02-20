from src.data_loader import load_data
from src.statistical_tests import numerical_tests, categorical_tests
from src.mle_feature_selection import mle_logistic_selection, pseudo_r2
from src.model_builder import build_model
from src.save_model import save_model

def main():

    df = load_data()

    print("Running hypothesis tests...")
    num_selected = numerical_tests(df, "Churn")
    cat_selected = categorical_tests(df, "Churn")

    print("Numerical Significant:", num_selected)
    print("Categorical Significant:", cat_selected)

    print("Running MLE logistic selection...")
    selected_features, logit_model = mle_logistic_selection(df, "Churn")
    print("MLE Significant:", selected_features)
    print("Pseudo R2:", pseudo_r2(logit_model))

    model = build_model(df)

    save_model(model)

if __name__ == "__main__":
    main()
