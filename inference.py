import joblib

IRIS_CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_model(file_path: str = "model.joblib"):
    model = joblib.load(file_path)
    return model


def predict(model, input_data: dict) -> str:
    features = [
        input_data["sepal_length"],
        input_data["sepal_width"],
        input_data["petal_length"],
        input_data["petal_width"],
    ]

    prediction_array = model.predict([features])
    predicted_index = prediction_array[0]

    return IRIS_CLASS_NAMES.get(predicted_index, "unknown")
