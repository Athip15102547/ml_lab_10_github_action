import pandas as pd
import mlflow

def validate_data(file_path):
    df = pd.read_csv(file_path)

    print("✅ ขนาดข้อมูล:", df.shape)
    print("✅ คอลัมน์:", df.columns.tolist())
    print("\n📌 Missing Values:")
    print(df.isnull().sum())
    print("\n📌 Data Types:")
    print(df.dtypes)

    # Log metrics to MLflow
    with mlflow.start_run(run_name="data_validation"):
        mlflow.log_metric("num_rows", df.shape[0])
        mlflow.log_metric("num_cols", df.shape[1])
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            mlflow.log_metric(f"missing_{col}", missing_count)

if __name__ == "__main__":
    path = "C:/Users/User/Downloads/LAB10/train_and_test2.csv"  # ปรับ path ตามจริง
    validate_data(path)
