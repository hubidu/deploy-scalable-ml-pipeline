import requests


def to_label(label_value):
	if label_value == 0:
		return "<= 50K"
	else:
		return "> 50K"


url = "https://stefan-ml-pipeline.herokuapp.com/predict" 
person_data = {
	'age': 39, 
	'workclass': "State-gov",
	'fnlgt': 77516, 
        'education': "Master", 
	'marital_status': "Never-married",
        'occupation': "Adm-clerical", 
	'relationship': "Husband",
        'race': "White", 
	'sex': "Male", 
	'capital_gain': 50000, 
	'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': "United-States"
}

print(f"Calling prediction api at {url}")
response = requests.post(url, json=person_data)

prediction = response.json()

print(f"Response status is {response.status_code}")
print(f"Predicting '{to_label(prediction['predictions'])}' for data '{person_data}'")
