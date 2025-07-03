import os
import time
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

EXPERIMENT_NAME = "Paddy Yield Forecasting-Tuning"
PREPROCESSED_DATA_PATH = "padi_preprocessing/preprocessing_padi.csv"
LABEL_ENCODER_PATH = "padi_preprocessing/label_encoder.joblib"
SCALER_PATH = "padi_preprocessing/scaler.joblib"  
TARGET_COLUMN = "Produksi"

if os.getenv("MLFLOW_RUN_ID") is None:
    mlflow.set_experiment(EXPERIMENT_NAME)

CI_MODE = os.getenv("CI", "false").lower() == "true"

if CI_MODE:
    param_grid = {
        'n_estimators': [50],
        'max_depth': [10],
        'min_samples_split': [2]
    }
    n_jobs_setting = 1
else:
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    n_jobs_setting = 1

with mlflow.start_run(run_name="RandomForest_Tuning_Run_Final"):
    master_table = pd.read_csv(PREPROCESSED_DATA_PATH)

    features = master_table.drop(TARGET_COLUMN, axis=1)
    target = master_table[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs_setting
    )
    grid_search.fit(X_train, y_train)

    duration = time.time() - start_time

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mlflow.log_params(best_params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    model_local_path = "optimized_model"
    mlflow.sklearn.save_model(best_model, model_local_path)
    mlflow.log_artifacts(model_local_path, artifact_path="model")

    mlflow.log_artifact(SCALER_PATH, artifact_path="preprocessor")
    mlflow.log_artifact(LABEL_ENCODER_PATH, artifact_path="preprocessor")
