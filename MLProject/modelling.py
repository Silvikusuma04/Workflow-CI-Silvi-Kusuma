import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

EXPERIMENT_NAME = "Paddy Yield Forecasting"
PREPROCESSED_DATA_PATH = "padi_preprocessing/preprocessing_padi.csv"
LABEL_ENCODER_PATH = "padi_preprocessing/label_encoder.joblib"
SCALER_PATH = "padi_preprocessing/scaler.joblib"  
TARGET_COLUMN = "Produksi"

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="RandomForest_Baseline"):
    mlflow.sklearn.autolog(log_models=True, log_datasets=False)

    master_table = pd.read_csv(PREPROCESSED_DATA_PATH)

    features = master_table.drop(TARGET_COLUMN, axis=1)
    target = master_table[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    regressor_model = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor_model.fit(X_train, y_train)

    predictions = regressor_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Log artefak preprocessor (encoder dan scaler)
    mlflow.log_artifact(LABEL_ENCODER_PATH, artifact_path="preprocessor")
    mlflow.log_artifact(SCALER_PATH, artifact_path="preprocessor")

    print(f"Mean Squared Error: {mse}")
    print("Pelatihan model dasar selesai. Eksperimen tercatat di MLflow.")