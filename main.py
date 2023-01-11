import pandas as pd
from pydantic import BaseModel

from fastapi import FastAPI
from starter.ml.model import inference
from joblib import load

app = FastAPI()


class PersonData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    marital_status: str 
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
def get_root():
    return "Welcome to the salary prediction api"


@app.post("/predict")
def predict(person_data: PersonData):
    all_cols = [
                "age", "workclass", "fnlgt", "education", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", 
                "capital-loss", "hours-per-week", "native-country"
               ]
    model = load("./model/census_model.joblib")

    preds = inference(model, pd.DataFrame(data=[
                                        person_data.age, 
                                        person_data.workclass, 
                                        person_data.fnlgt, 
                                        person_data.education, 
                                        person_data.marital_status, 
                                        person_data.occupation, 
                                        person_data.relationship, 
                                        person_data.race, 
                                        person_data.sex, 
                                        person_data.capital_gain, 
                                        person_data.capital_loss, 
                                        person_data.hours_per_week, 
                                        person_data.native_country
                                        ], columns=all_cols))

    return {'predictions': preds}
