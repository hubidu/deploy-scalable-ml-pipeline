"""
    Train the machine learning model
"""
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model

data = pd.read_csv('./data/trimmed_census.csv')

train, test = train_test_split(data, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

model = train_model(X_train, y_train)

dump(model, './model/census_model.joblib')
