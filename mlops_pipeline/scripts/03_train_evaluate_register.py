import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from pathlib import Path
import os

def train():
    file_path = Path("train_and_test2_preprocessed.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Input file not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    target_col = "2urvived"
    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in dataset!")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("file:///tmp/mlruns")
    with mlflow.start_run(run_name="train_random_forest"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model", "Lab09Model"
        )

        print(f"✅ Accuracy: {acc}")
        print("✅ Model saved and registered.")

if __name__ == "__main__":
    train()
