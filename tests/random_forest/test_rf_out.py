import pytest
import pandas as pd
import mlflow


@pytest.fixture(scope="module")
def model() -> mlflow.pyfunc.PyFuncModel:
    mlflow.set_tracking_uri("sqlite:///./mlruns/mlflow.db")
    model_name = "random_forest"
    model_version = 2
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )


def test_model_out(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'Pregnancies': 0,
        'Glucose': 100,
        'BloodPressure': 88,
        'SkinThickness': 60,
        'Insulin': 110,
        'BMI': 46.8,
        'DiabetesPedigreeFunction': 0.962,
        'Age': 31
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 0

def test_model_out_shape(model: mlflow.pyfunc.PyFuncModel):
    data_path = './data/diabetes_test.csv'
    df = pd.read_csv(data_path)
    input = df.head(1).drop("Outcome", axis=1)
    prediction = model.predict(data=input)
    assert prediction.shape == (1, )