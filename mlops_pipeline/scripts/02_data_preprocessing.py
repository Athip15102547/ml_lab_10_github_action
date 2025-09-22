import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow
import os

def preprocess(file_path, output_path):
    # ตรวจสอบว่าไฟล์ต้นฉบับมีอยู่จริงหรือไม่
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Input file not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    target_col = "2urvived"  # หรือชื่อที่ตรงกับในไฟล์จริง
    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in dataset!")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_processed = pd.DataFrame(X_scaled, columns=X.columns)

    processed_df = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)

    # บันทึกไฟล์ที่ถูก preprocess แล้ว
    processed_df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing done. Saved to {output_path}")

    # Log artifact and params to MLflow
    with mlflow.start_run(run_name="data_preprocessing"):
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_artifact(output_path)

if __name__ == "__main__":
    # ใช้พาธแบบสัมพันธ์
    input_data_path = "train_and_test2.csv"
    output_data_path = "train_and_test2_preprocessed.csv"
    preprocess(input_data_path, output_data_path)
