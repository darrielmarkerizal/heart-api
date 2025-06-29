from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load model once at startup
model = joblib.load("heart_model_xgb.pkl")

class HeartData(BaseModel):
    age: int
    gender: int  # 1 = Male, 0 = Female
    heart_rate: int
    systolic_bp: int
    diastolic_bp: int
    blood_sugar: float
    ckmb: float
    troponin: float

@app.get("/")
def home():
    return {"message": "API for Heart Attack Prediction is active"}

@app.post("/predict")
def predict(data: HeartData):
    flag_ckmb = 1 if data.ckmb > 50 else 0
    flag_trop = 1 if data.troponin > 0.4 else 0

    input_array = np.array([[data.age, data.gender, data.heart_rate,
                             data.systolic_bp, data.diastolic_bp,
                             data.blood_sugar, data.ckmb, data.troponin,
                             flag_ckmb, flag_trop]])
    result = model.predict(input_array)[0]
    return {
        "prediction": int(result),
        "risk": "High" if result == 1 else "Low"
    }
