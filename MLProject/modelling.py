import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
import os
import shutil

if __name__ == "__main__":
    print("Memuat dataset bersih...")
    df = pd.read_csv('data_bersih.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print("Melatih model untuk CI/CD Pipeline...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Nama folder tempat model akan disimpan
    model_path = "model_output"
    
    # Hapus folder jika sebelumnya sudah ada agar tidak error
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    print("Menyimpan model ke dalam folder lokal...")
    # Menyimpan model dengan format MLflow murni
    mlflow.sklearn.save_model(model, model_path)
    print("✅ Pelatihan selesai dan model berhasil disimpan!")