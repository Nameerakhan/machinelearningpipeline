import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
from mlflow.models import infer_signature
import os
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

## Load DagsHub/MLflow credentials from .env
load_dotenv()


def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search
 

## Load parameters from yaml file
params = yaml.safe_load(open("params.yaml"))["train"]


def train(data_path, model_output_path, random_state, n_estimator, max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/nameerakhan1105/MLpipeline.mlflow")

    ## start mlflow run
    with mlflow.start_run():
        ## split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=random_state
        )
        signature = infer_signature(X_train, y_train)

        ## Define hyperparameters grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        ## perform hyperparameter tuning
        search_results = hyperparameter_tuning(X_train, y_train, param_grid)

        ## get the best model 
        
        best_model = search_results.best_estimator_

        ## predict on test data
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy}")

        ## log metrics and best params to mlflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_n_estimators", search_results.best_params_["n_estimators"])
        mlflow.log_param("best_max_depth", search_results.best_params_["max_depth"])
        mlflow.log_param("best_min_samples_split", search_results.best_params_["min_samples_split"])
        mlflow.log_param("best_min_samples_leaf", search_results.best_params_["min_samples_leaf"])

        ## log confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        ## log the model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best Model")
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        ## save the model locally
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        with open(model_output_path, "wb") as f:
            pickle.dump(best_model, f)

        print(f"Model saved to {model_output_path}")


if __name__ == "__main__":
    train(
        params["data"],
        params["model"],
        params["random_state"],
        params["n_estimators"],
        params["max_depth"],
    )
