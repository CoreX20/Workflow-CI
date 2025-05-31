import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import time
import warnings


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    df = pd.read_csv("MLProject/Student_Performance_preprocessing.csv")

    # Pisahkan fitur dan target
    X = df.drop(["Performance Index"], axis=1)
    y = df["Performance Index"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standarization
    features = [
        "Hours Studied",
        "Previous Scores",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ]

    scaler = StandardScaler()
    X_train[features] = scaler.fit_transform(X_train[features])

    X_test.loc[:, features] = scaler.transform(X_test[features]).astype(float)

    knn = KNeighborsRegressor()

    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree"],
    }

    grid_search = GridSearchCV(knn, param_grid, cv=3, scoring="r2", n_jobs=-1)

    start_time = time.time()
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Hitung metrik performa
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    training_time = time.time() - start_time

    input_example = X_train[0:5]

    # Logging manual ke MLflow
    with mlflow.start_run():
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("training_time", training_time)

        mlflow.sklearn.log_model(best_model, "model", input_example=input_example)
