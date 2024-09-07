from fastapi import FastAPI
from src.make_prediction import make_prediction
import uvicorn

# App init
app = FastAPI()

# Home
@app.get("/")
async def home():
    return {"Message": "My first API"}

@app.get("/healthy")
async def status():
    return {"Message": "Healthy"}

# Predict method
@app.post("/predict")
async def predict(
    sepal_length: float, 
    sepal_width: float, 
    petal_length: float, 
    petal_width: float):

    label = make_prediction(sepal_length, sepal_width, petal_length, petal_width)

    return {"label": label}