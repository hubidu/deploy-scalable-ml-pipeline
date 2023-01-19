import pandas as pd
from pydantic import BaseModel, Field

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starter.ml.model import inference
from starter.ml.data import process_data
from joblib import load

app = FastAPI()


class PersonData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True


model = None
encoder = None
scaler = None


@app.on_event("startup")
def startup_event(): 
    print("ON STARTUP")
    global model, encoder, scaler
    model = load("./model/census_model.joblib")
    encoder = load("./model/encoder.joblib")
    scaler = load("./model/scaler.joblib")


@app.get("/")
def get_root():
    return "Welcome to the salary prediction api"


@app.post("/predict")
def predict(person_data: PersonData):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    df = pd.DataFrame(data=person_data.dict(by_alias=True), index=[0])

    X, _, _, _, _ = process_data(
        df, cat_features, None, training=False, encoder=encoder, scaler=scaler
    )

    preds = inference(model, X)

    return JSONResponse(content={'predictions': int(preds[0])})
