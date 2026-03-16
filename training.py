from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib


def load_data():
    return load_iris()


def train_model():
    data = load_data()
    model = RandomForestClassifier()
    model.fit(data.data, data.target)
    return model


def save_model(model, path: str = "model.joblib") -> None:
    joblib.dump(model, path)


if __name__ == "__main__":
    model = train_model()
    save_model(model)
    print("Model trained and saved!")
