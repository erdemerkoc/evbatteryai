import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# 1. DATASET
# ==========================================

file_path = "dataset/ev battery_failure prediction Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset:", df.shape)


# ==========================================
# 2. TARGET
# ==========================================

target = "battery_failure"

X = df.drop(columns=[target])
y = df[target]


# ==========================================
# 3. ID SÃœTUNLARINI Ã‡IKAR
# ==========================================

X = X.drop(columns=["vehicle_id", "battery_serial", "predicted_remaining_life_cycles"])


# ==========================================
# 4. NUMERIC / CATEGORICAL SÃœTUNLAR
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ==========================================
# 5. NUMERIC PIPELINE
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# ==========================================
# 6. CATEGORICAL PIPELINE
# ==========================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# ==========================================
# 7. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ==========================================
# 8. MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ==========================================
# 9. PIPELINE
# ==========================================

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])


# ==========================================
# 10. TRAIN / TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining:", X_train.shape)
print("Testing :", X_test.shape)


# ==========================================
# 11. TRAIN
# ==========================================

print("\nModel training baÅŸladÄ±...")

pipeline.fit(X_train, y_train)

print("Training tamamlandÄ±!")


# ==========================================
# 12. PREDICTION
# ==========================================

y_pred = pipeline.predict(X_test)


# ==========================================
# 13. RESULTS
# ==========================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(classification_report(y_test, y_pred))


print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(confusion_matrix(y_test, y_pred))
# ==========================================
# 14. FEATURE IMPORTANCE
# ==========================================

importances = pipeline.named_steps["model"].feature_importances_

feature_names = pipeline.named_steps[
    "preprocessing"
].get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\n==============================")
print("TOP 20 FEATURE IMPORTANCE")
print("==============================")

print(importance_df.head(20).to_string(index=False))


# ==========================================
# 15. FAILURE PROBABILITIES
# ==========================================

y_probability = pipeline.predict_proba(X_test)[:, 1]

print("\n==============================")
print("SAMPLE FAILURE PROBABILITIES")
print("==============================")

for i in range(20):
    print(
        f"Gerçek: {y_test.iloc[i]} | "
        f"Failure probability: {y_probability[i]:.2%}"
    )



# ==========================================
# 16. THRESHOLD ANALYSIS
# ==========================================

from sklearn.metrics import precision_score, recall_score, f1_score

print("\n==============================")
print("THRESHOLD ANALYSIS")
print("==============================")

thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]

for threshold in thresholds:

    y_threshold = (y_probability >= threshold).astype(int)

    precision = precision_score(y_test, y_threshold, zero_division=0)
    recall = recall_score(y_test, y_threshold, zero_division=0)
    f1 = f1_score(y_test, y_threshold, zero_division=0)

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )


# ==========================================
# 17. CONFUSION MATRIX - THRESHOLD 0.4
# ==========================================

threshold = 0.4

y_threshold = (y_probability >= threshold).astype(int)

print("\n==============================")
print("CONFUSION MATRIX - THRESHOLD 0.4")
print("==============================")

print(confusion_matrix(y_test, y_threshold))


# ==========================================
# 18. DATA LEAKAGE CHECK
# ==========================================

print("\n==============================")
print("DATA LEAKAGE CHECK")
print("==============================")

correlations = df.select_dtypes(
    include=["int64", "float64"]
).corr()["battery_failure"].sort_values(
    ascending=False
)

print("\nTop correlations with battery_failure:")
print(correlations.head(15).to_string())


# ==========================================
# 19. FEATURE DISTRIBUTION CHECK
# ==========================================

print("\n==============================")
print("FEATURE DISTRIBUTION BY FAILURE")
print("==============================")

check_features = [
    "thermal_runaway_risk",
    "capacity_loss_percent",
    "internal_resistance",
    "cell_temperature_max",
    "battery_stress_index"
]

print(
    df.groupby("battery_failure")[check_features]
      .mean()
      .T
      .to_string()
)


# ==========================================
# 20. MODEL KAYDETME (JOB-LIB)
# ==========================================

import joblib

model_filename = "ev_battery_model.joblib"
joblib.dump(pipeline, model_filename)
print(f"\n✅ Eğitilmiş model başarıyla kaydedildi: '{model_filename}'")


