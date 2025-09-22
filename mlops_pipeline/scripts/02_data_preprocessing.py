import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow
from pathlib import Path
import os

def preprocess(file_path, output_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Input file not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    target_col = "2urvived"
    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in dataset!")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_processed = pd.DataFrame(X_scaled, columns=X.columns)

    processed_df = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)
    processed_df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing done. Saved to {output_path}")

    # ตั้ง MLflow URI - ใช้ environment variable หรือ default
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    # สร้าง experiment ถ้ายังไม่มี
    try:
        experiment_id = mlflow.create_experiment("MLOps_Pipeline")
    except Exception:
        experiment_id = mlflow.get_experiment_by_name("MLOps_Pipeline").experiment_id
    
    mlflow.set_experiment(experiment_id=experiment_id)
    
    with mlflow.start_run(run_name="data_preprocessing"):
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_artifact(str(output_path))

if __name__ == "__main__":
    input_data_path = Path("train_and_test2.csv")
    output_data_path = Path("train_and_test2_preprocessed.csv")
    
    if not input_data_path.exists():
        raise FileNotFoundError(f"❌ Input data file not found: {input_data_path}")
    
    preprocess(input_data_path, output_data_path)
