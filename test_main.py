from fastapi.testclient import TestClient

from main import app, PersonData


client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.text


def test_predict_under_50K_salary():
    person = PersonData(
        age=39, workclass="State-gov", fnlgt=77516, 
        education="Bachelors", marital_status="Never-married",
        occupation="Adm-clerical", relationship="Not-in-family",
        race="White", sex="Male", capital_gain=2174, capital_loss=0,
        hours_per_week=40,
        native_country="United-States"
    )
    response = client.post("predict", json=person.dict())
    assert response.status_code == 200
    assert {'predictions': [0]} == response.json()


def test_predict_over_50K_salary():
    person = PersonData(
        age=49, workclass="Federal-gov", fnlgt=125892, 
        education="Bachelors", marital_status="Married-civ-spouse",
        occupation="Exec-managerial", relationship="Husband",
        race="White", sex="Male", capital_gain=0, capital_loss=0,
        hours_per_week=40,
        native_country="United-States"
    )
    response = client.post("predict", json=person.dict())
    assert response.status_code == 200
    assert {'predictions': [1]} == response.json()

 