import pandas as pd
import yaml
import pickle
import mlflow
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

## Load DagsHub/MLflow credentials from .env
load_dotenv()

## Load parameters from yaml file
params = yaml.safe_load(open("params.yaml"))["train"]


def evaluate(model_path, data_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/nameerakhan1105/MLpipeline.mlflow")

    ## Load the model
    model = pickle.load(open(model_path, "rb"))
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    ## log metrics to mlflow
    mlflow.log_metric("test_accuracy", accuracy)
    print(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    evaluate(params["model"], params["data"])
