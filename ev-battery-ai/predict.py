import os
import joblib
import pandas as pd

MODEL_FILE = "ev_battery_model.joblib"
DATASET_FILE = "dataset/ev battery_failure prediction Dataset.csv"

def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Model dosyası ({MODEL_FILE}) bulunamadı. Lütfen önce 'python3 train.py' çalıştırın.")
        exit(1)
    return joblib.load(MODEL_FILE)

def test_random_samples(pipeline, n_samples=5, threshold=0.2):
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Veri seti dosyası ({DATASET_FILE}) bulunamadı.")
        return

    df = pd.read_csv(DATASET_FILE)
    
    # Sağlam ve Arızalı örnekler seç
    healthy_samples = df[df["battery_failure"] == 0].sample(n=n_samples, random_state=None)
    failing_samples = df[df["battery_failure"] == 1].sample(n=n_samples, random_state=None)
    test_df = pd.concat([healthy_samples, failing_samples]).sample(frac=1)

    X_test = test_df.drop(columns=["battery_failure", "vehicle_id", "battery_serial", "predicted_remaining_life_cycles"], errors="ignore")
    y_actual = test_df["battery_failure"].tolist()
    vehicle_ids = test_df.get("vehicle_id", pd.Series(["N/A"] * len(test_df))).tolist()

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "="*78)
    print(f" 🔋 ELEKTRİKLİ ARAÇ BATARYA ARIZA TAHMİNİ (Eşik Değeri: %{threshold*100:.0f})")
    print("="*78)
    print(f"{'Araç ID':<14} | {'Gerçek Durum':<14} | {'Arıza İhtimali':<18} | {'Tahmin Edilen Risk':<15}")
    print("-" * 78)

    for vid, actual, prob in zip(vehicle_ids, y_actual, probabilities):
        actual_str = "❌ ARIZALI" if actual == 1 else "✅ SAĞLAM"
        risk_str = "🚨 YÜKSEK RİSK" if prob >= threshold else "🟢 DÜŞÜK RİSK"
        print(f"{str(vid):<14} | {actual_str:<14} | %{prob*100:>6.2f}            | {risk_str:<15}")

    print("="*78 + "\n")

if __name__ == "__main__":
    print("🤖 Model yükleniyor...")
    pipeline = load_model()
    test_random_samples(pipeline)
