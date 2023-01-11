import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.model import train_model, inference, compute_model_metrics
from starter.ml.data import process_data


@pytest.fixture
def training_data():
	df = pd.read_csv('./data/trimmed_census.csv')
	train = df

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

	X, y, _, _ = process_data(
		train, categorical_features=cat_features, label="salary", training=True
	)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

	return (X_train, y_train, X_test, y_test)

@pytest.fixture
def trained_model(training_data):
	X_train, y_train, _, _ = training_data

	model = train_model(X_train, y_train)
	return model


@pytest.fixture
def y_preds(training_data, trained_model):
	_, _, X_test, _ = training_data

	y_preds = inference(trained_model, X_test)

	return y_preds


def test_train_model(trained_model):
	assert trained_model is not None


def test_inference(training_data, y_preds):
	_, _, _, y_test = training_data
	assert y_preds is not None
	assert y_preds.shape[0] == y_test.shape[0]


def test_compute_model_metrics(training_data, y_preds):
	_, _, _, y_test = training_data

	precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

	assert precision > 0.72
	assert recall > 0.6
	assert fbeta > 0.65
