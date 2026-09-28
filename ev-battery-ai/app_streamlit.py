import os
import joblib
import pandas as pd
import streamlit as st

# Sayfa Yapılandırması
st.set_page_config(
    page_title="EV Battery AI - Arıza Tahmin Platformu",
    page_icon="🔋",
    layout="wide"
)

MODEL_FILE = "ev_battery_model.joblib"
DATASET_FILE = "dataset/ev battery_failure prediction Dataset.csv"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_dataset():
    if not os.path.exists(DATASET_FILE):
        return None
    return pd.read_csv(DATASET_FILE)

model = load_model()
df = load_dataset()

# Başlık ve Açıklama
st.title("🔋 EV Battery AI - Batarya Arıza Tahmin Platformu")
st.markdown("""
Bu platform, Elektrikli Araç (EV) batarya verilerinizi girerek makine öğrenmesi modeli ile **Arıza Riski Tahmini** yapmanızı sağlar.
Sağ menüden veya veri giriş formundan kendi batarya ölçümlerinizi girebilir veya örnek senaryoları yükleyebilirsiniz.
""")

if model is None:
    st.error("❌ Model dosyası bulunamadı! Lütfen önce terminalde `python3 train.py` çalıştırın.")
    st.stop()

# X varsayılan sütunlarını ve medyan değerlerini hazırla
if df is not None:
    X_base = df.drop(columns=["battery_failure", "vehicle_id", "battery_serial", "predicted_remaining_life_cycles"], errors="ignore")
    default_values = X_base.median(numeric_only=True).to_dict()
    # Categorical defaults
    for col in X_base.select_dtypes(include=['object', 'category', 'string']).columns:
        default_values[col] = X_base[col].mode()[0] if not X_base[col].mode().empty else "N/A"
else:
    default_values = {}

# Yan Menü (Sidebar) - Hazır Senaryolar
st.sidebar.header("🎯 Hazır Örnek Senaryolar")
preset = st.sidebar.radio("Senaryo Seçin:", ["Özel Veri Girişi", "🟢 Sağlam Batarya Örneği", "🚨 Arızalı Batarya Örneği"])

# Varsayılan Değerleri Ayarla
if preset == "🟢 Sağlam Batarya Örneği":
    preset_vals = {
        "thermal_health_score": 98.14,
        "thermal_runaway_risk": 4.88,
        "cell_temperature_max": 27.87,
        "capacity_loss_percent": 19.4,
        "internal_resistance": 0.506,
        "cell_temperature_avg": 24.5,
        "battery_health_percent": 96.5,
        "state_of_health": 97.0,
        "previous_faults": 0,
        "battery_stress_index": 22.0,
        "charge_efficiency": 97.5,
        "discharge_efficiency": 96.8,
        "vehicle_age_years": 2.0,
        "aging_score": 18.0,
        "BMS_warning_count": 0
    }
elif preset == "🚨 Arızalı Batarya Örneği":
    preset_vals = {
        "thermal_health_score": 52.24,
        "thermal_runaway_risk": 59.3,
        "cell_temperature_max": 68.61,
        "capacity_loss_percent": 41.67,
        "internal_resistance": 0.877,
        "cell_temperature_avg": 54.2,
        "battery_health_percent": 68.0,
        "state_of_health": 65.0,
        "previous_faults": 4,
        "battery_stress_index": 78.5,
        "charge_efficiency": 82.0,
        "discharge_efficiency": 80.5,
        "vehicle_age_years": 7.0,
        "aging_score": 75.0,
        "BMS_warning_count": 5
    }
else:
    preset_vals = {}

def get_val(key, default):
    return float(preset_vals.get(key, default_values.get(key, default)))

tab1, tab2 = st.tabs(["📝 Manuel Veri Girişi", "📁 Toplu CSV Tahmini"])

with tab1:
    st.subheader("⚙️ Batarya Ölçüm Parametreleri")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌡️ Termal & Sıcaklık")
        thermal_health_score = st.slider("Termal Sağlık Skoru (0-100)", 0.0, 100.0, get_val("thermal_health_score", 85.0))
        thermal_runaway_risk = st.slider("Termal Kaçak (Runaway) Riski", 0.0, 100.0, get_val("thermal_runaway_risk", 15.0))
        cell_temp_max = st.number_input("Max Hücre Sıcaklığı (°C)", 0.0, 120.0, get_val("cell_temperature_max", 35.0))
        cell_temp_avg = st.number_input("Ortalama Hücre Sıcaklığı (°C)", 0.0, 100.0, get_val("cell_temperature_avg", 30.0))

    with col2:
        st.markdown("### ⚡ Kapasite & İç Direnç")
        capacity_loss_pct = st.number_input("Kapasite Kaybı (%)", 0.0, 100.0, get_val("capacity_loss_percent", 15.0))
        internal_resistance = st.number_input("İç Direnç (Ohm/mΩ)", 0.0, 5.0, get_val("internal_resistance", 0.45))
        battery_health_pct = st.slider("Batarya Sağlık Durumu (SoH %)", 0.0, 100.0, get_val("battery_health_percent", 90.0))
        battery_stress_idx = st.slider("Batarya Stres İndeksi", 0.0, 100.0, get_val("battery_stress_index", 30.0))

    with col3:
        st.markdown("### 📊 Verimlilik & Geçmiş")
        charge_eff = st.slider("Şarj Verimliliği (%)", 50.0, 100.0, get_val("charge_efficiency", 95.0))
        discharge_eff = st.slider("Deşarj Verimliliği (%)", 50.0, 100.0, get_val("discharge_efficiency", 94.0))
        previous_faults = st.number_input("Geçmiş Arıza Sayısı", 0, 50, int(get_val("previous_faults", 0)))
        bms_warnings = st.number_input("BMS Uyarı Sayısı", 0, 50, int(get_val("BMS_warning_count", 0)))
        vehicle_age = st.number_input("Araç Yaşı (Yıl)", 0.0, 30.0, get_val("vehicle_age_years", 3.0))

    threshold = st.slider("🚨 Risk Karar Eşiği (Threshold %)", 5, 50, 20) / 100.0

    if st.button("🚀 Batarya Arıza Riskini Hesapla", type="primary"):
        # Test input dict
        input_data = default_values.copy()
        input_data.update({
            "thermal_health_score": thermal_health_score,
            "thermal_runaway_risk": thermal_runaway_risk,
            "cell_temperature_max": cell_temp_max,
            "cell_temperature_avg": cell_temp_avg,
            "capacity_loss_percent": capacity_loss_pct,
            "internal_resistance": internal_resistance,
            "battery_health_percent": battery_health_pct,
            "battery_stress_index": battery_stress_idx,
            "charge_efficiency": charge_eff,
            "discharge_efficiency": discharge_eff,
            "previous_faults": previous_faults,
            "BMS_warning_count": bms_warnings,
            "vehicle_age_years": vehicle_age
        })

        input_df = pd.DataFrame([input_data])
        
        # Prediction
        prob = model.predict_proba(input_df)[:, 1][0]
        risk_pct = prob * 100

        st.markdown("---")
        st.subheader("📈 Tahmin Sonuçları")

        m1, m2, m3 = st.columns(3)
        m1.metric("Tahmini Arıza Olasılığı", f"%{risk_pct:.2f}")

        if prob >= threshold:
            m2.error("🚨 YÜKSEK ARIZA RİSKİ")
            m3.warning(f"Olasılık %{risk_pct:.1f} (Eşik %{threshold*100:.0f}'i aştı)")
            st.error(f"⚠️ Bataryada kritik arıza riski tespit edildi! Tahmini arıza olasılığı: %{risk_pct:.2f}")
        elif prob >= 0.10:
            m2.warning("🟡 ORTA RİSK (İzlenmeli)")
            m3.info(f"Olasılık %{risk_pct:.1f}")
            st.warning(f"⚡ Bataryada orta düzey risk var. Düzenli bakım önerilir.")
        else:
            m2.success("🟢 DÜŞÜK RİSK (Sağlam)")
            m3.success(f"Olasılık %{risk_pct:.1f}")
            st.success(f"✅ Batarya oldukça sağlıklı görünüyor.")

with tab2:
    st.subheader("📁 Toplu CSV Dosyası Yükleyip Tahmin Alma")
    uploaded_file = st.file_uploader("Bir CSV Veri Seti Yükleyin", type=["csv"])
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("Yüklenen Veri Seti (Önizleme):", user_df.head())
        
        X_user = user_df.drop(columns=["battery_failure", "vehicle_id", "battery_serial", "predicted_remaining_life_cycles"], errors="ignore")
        probs = model.predict_proba(X_user)[:, 1]
        user_df["Ariza_Olasiligi_%"] = (probs * 100).round(2)
        user_df["Risk_Tahmini"] = ["🚨 YÜKSEK RİSK" if p >= 0.20 else "🟢 DÜŞÜK RİSK" for p in probs]

        st.write("🎯 Tahmin Eklenmiş Veri Seti:", user_df[["vehicle_id", "Ariza_Olasiligi_%", "Risk_Tahmini"] + [c for c in user_df.columns if c not in ["vehicle_id", "Ariza_Olasiligi_%", "Risk_Tahmini"]]].head(20))
        
        csv_data = user_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Sonuçları CSV Olarak İndir", csv_data, "batarya_ariza_tahminleri.csv", "text/csv")
