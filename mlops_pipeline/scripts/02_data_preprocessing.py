import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow

def preprocess(file_path):
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

    output_path = "C:/Users/User/Downloads/LAB10/train_and_test2_preprocessed.csv"  
    # ปรับ path ตามต้องการ
    processed_df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing done. Saved to {output_path}")

    # Log artifact and params to MLflow
    with mlflow.start_run(run_name="data_preprocessing"):
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_artifact(output_path)

if __name__ == "__main__":
    preprocess("C:/Users/User/Downloads/LAB10/train_and_test2.csv")
