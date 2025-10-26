from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

from fastapi.middleware.cors import CORSMiddleware



app= FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # or ["*"] for all origins (dev only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model_path = os.path.join(os.path.dirname(__file__), 'insurance_cost_model.pkl')
model = joblib.load(model_path)

# Define input schema
class InsuranceInput(BaseModel):
    age: float
    bmi: float
    children: int
    sex_male: int
    smoker_yes: int

@app.post("/predict")
def predict(data: InsuranceInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    axtual_cost=np.exp(prediction)
    return {"predicted_cost": round(axtual_cost, 2)}


