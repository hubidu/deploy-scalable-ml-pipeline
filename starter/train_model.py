"""
    Train the machine learning model
"""
import pandas as pd
import json
from joblib import dump
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import logging

logging.basicConfig(filename='training.log', encoding='utf-8', level=logging.INFO)

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


def dump_slice_metrics(model, test_data, encoder, lb, slice_columns):
    slice_metrics = []
    for slice_column in slice_columns:
        slice_values = test_data[slice_column].unique()

        for slice_value in slice_values:
            slice_filter = test_data[slice_column] == slice_value

            X_test, y_test, _, _ = process_data(
                test_data[slice_filter],
                categorical_features=cat_features, label="salary", training=False,
                encoder=encoder, lb=lb
            )

            y_preds = inference(model, X_test)

            precision, recall, fbeta = compute_model_metrics(
                y_test, y_preds
            )

            slice_metrics.append({
                    'slice': f'{slice_column}={slice_value}',
                    'samples': y_test.shape[0],
                    'precision': precision,
                    'recall': recall,
                    'fbeta': fbeta,
                })

    with open('slice_output.txt', 'w') as f:
        f.write(json.dumps(slice_metrics, indent=2))

logging.info('Reading training data ./data/trimmed_census.csv')
data = pd.read_csv('./data/trimmed_census.csv')

train, test = train_test_split(data, test_size=0.30, random_state=123)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

logging.info('Training the model')
model = train_model(X_train, y_train)

logging.info('Computing slice metrics')
dump_slice_metrics(model, test, encoder, lb, ['sex', 'occupation', 'race'])

logging.info('Saving model to ./model/census_model.joblib')
dump(model, './model/census_model.joblib')

logging.info('Success.')