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

logging.basicConfig(filename='training.log', level=logging.DEBUG)
logger = logging.getLogger("train_model")
logger.setLevel(logging.DEBUG)

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


def dump_slice_metrics(model, test_data, encoder, lb, scaler, slice_columns):
    slice_metrics = []
    for slice_column in slice_columns:
        slice_values = test_data[slice_column].unique()

        for slice_value in slice_values:
            slice_filter = test_data[slice_column] == slice_value

            X_test, y_test, _, _, _ = process_data(
                test_data[slice_filter],
                categorical_features=cat_features, label="salary", 
                training=False,
                encoder=encoder, lb=lb, scaler=scaler
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


def do_train_model():
    """
    Read the input csv, split the data into training and test sets and train the model on the training data.
    Serialize the trained model to disk. 
    """
    logger.info('Reading training data ./data/trimmed_census.csv')
    data = pd.read_csv('./data/trimmed_census.csv')

    train, test = train_test_split(
        data, test_size=0.30, stratify=data["salary"], random_state=123
    )

    X_train, y_train, encoder, lb, scaler = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    logger.info('Training the model')
    model = train_model(X_train, y_train)

    y_pred = inference(model, X_train)

    precision, recall, beta = compute_model_metrics(y_train, y_pred)
    logger.info(f"Training set metrics: Precision={precision}, Recall={recall}, Beta={beta}")

    train["labels"] = y_pred.tolist()
    logger.info(train.head())

    logger.info('Computing slice metrics')
    dump_slice_metrics(model, test, encoder, lb, scaler, ['sex', 'occupation', 'race'])

    logger.info('Saving model to ./model/census_model.joblib')
    dump(model, './model/census_model.joblib')
    dump(encoder, './model/encoder.joblib')
    dump(scaler, './model/scaler.joblib')

    logger.info('Success.')


if __name__ == "__main__":
    do_train_model()