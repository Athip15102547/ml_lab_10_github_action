import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
import os

def load_and_predict(model_uri, input_csv, target_col=None):
    # ตรวจสอบว่าไฟล์ input มีอยู่จริง
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"❌ Input CSV file not found: {input_csv}")

    # ตั้ง MLflow URI - ใช้ environment variable หรือ default
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    # สร้าง experiment ถ้ายังไม่มี (สำหรับการโหลดโมเดล)
    try:
        experiment_id = mlflow.create_experiment("MLOps_Pipeline")
    except Exception:
        experiment_id = mlflow.get_experiment_by_name("MLOps_Pipeline").experiment_id
    
    mlflow.set_experiment(experiment_id=experiment_id)

    # โหลดโมเดลจาก mlflow
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Model loaded successfully from: {model_uri}")
    except Exception as e:
        print(f"❌ Failed to load model from {model_uri}: {e}")
        raise

    # โหลดข้อมูล csv
    df = pd.read_csv(input_csv)

    # ลบคอลัมน์ target ถ้ามี และกําหนดชื่อ
    if target_col and target_col in df.columns:
        df = df.drop(target_col, axis=1)

    # ทํา prediction
    predictions = model.predict(df)

    print("📄 Input data preview:")
    print(df.head())
    print("\n✅ Predictions:")
    print(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_uri", type=str, required=True,
                        help="MLflow model URI")
    # แก้ไขพาธเริ่มต้นให้เป็นพาธแบบสัมพันธ์ที่ถูกต้อง
    parser.add_argument("--input_csv", type=str,
                        default="train_and_test2_preprocessed.csv", help="Input CSV file path")
    # เพิ่มการกำหนดค่า default สำหรับ target_col ให้ชัดเจน
    parser.add_argument("--target_col", type=str, default="2urvived",
                        help="Name of target column to drop from input if exists")
    args = parser.parse_args()

    load_and_predict(args.model_uri, args.input_csv, args.target_col)