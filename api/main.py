from fastapi import FastAPI
import pandas as pd
from api.load_model import model

app = FastAPI(title="Telecom Churn API")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}
