import pandas as pd
import mlflow
from pathlib import Path

def validate_data(file_path):
    df = pd.read_csv(file_path)
    
    print("ขนาดข้อมูล:", df.shape)
    print("คอลัมน์:", df.columns.tolist())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)
    
    # ตั้ง MLflow URI ให้เป็น folder เขียนได้
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    with mlflow.start_run(run_name="data_validation"):
        mlflow.log_metric("num_rows", df.shape[0])
        mlflow.log_metric("num_cols", df.shape[1])
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            mlflow.log_metric(f"missing_{col}", missing_count)

if __name__ == "__main__":
    path = Path("train_and_test2.csv")
    validate_data(path)
