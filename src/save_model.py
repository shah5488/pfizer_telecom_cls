import joblib
from src.config import MODEL_PATH

def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print("Model saved successfully")
