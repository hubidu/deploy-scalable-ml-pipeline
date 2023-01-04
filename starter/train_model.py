"""
    Train the machine learning model
"""
import pandas as pd
import json
from joblib import dump
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

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


def dump_slice_metrics(model, test_data, encoder, lb, slice_column):
    slice_values = test_data[slice_column].unique()
    slice_metrics = []

    for slice_value in slice_values:
        slice_filter = test_data[slice_column] == slice_value

        X_test, y_test, _, _ = process_data(
            test_data[slice_filter],
            categorical_features=cat_features, label="salary", training=False,
            encoder=encoder, lb=lb
        )

        y_preds = inference(model, X_test)

        precision, recall, fbeta, accuracy = compute_model_metrics(
            y_test, y_preds
        )

        slice_metrics.append({
                'slice': f'{slice_column}={slice_value}',
                'samples': y_test.shape[0],
                'precision': precision,
                'recall': recall,
                'fbeta': fbeta,
                'accuracy': accuracy,
            })

    with open('slice_output.txt', 'w') as f:
        f.write(json.dumps(slice_metrics, indent=2))


data = pd.read_csv('./data/trimmed_census.csv')

train, test = train_test_split(data, test_size=0.20)


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

model = train_model(X_train, y_train)

dump_slice_metrics(model, test, encoder, lb, 'education')

dump(model, './model/census_model.joblib')
