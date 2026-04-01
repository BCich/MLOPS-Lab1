from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_welcome_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML API"}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_setosa():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] == "setosa"


def test_predict_versicolor():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 6.0,
            "sepal_width": 2.7,
            "petal_length": 4.5,
            "petal_width": 1.5,
        },
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "versicolor"


def test_predict_virginica():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 7.2,
            "sepal_width": 3.2,
            "petal_length": 6.1,
            "petal_width": 2.3,
        },
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "virginica"


def test_predict_invalid_missing_field():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
        },
    )
    assert response.status_code == 422


def test_predict_invalid_wrong_type():
    response = client.post(
        "/predict",
        json={
            "sepal_length": "not_a_number",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert response.status_code == 422
