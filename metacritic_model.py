import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# --- CSV 불러오기
data = pd.read_csv("metacritic_data.csv")

# --- 기본 전처리
data = data.dropna(subset=["metacritic_score", "user_score", "genre", "platform", "release_year"])
data["metacritic_score"] = pd.to_numeric(data["metacritic_score"], errors="coerce")
data["user_score"] = pd.to_numeric(data["user_score"], errors="coerce")
data["release_year"] = pd.to_numeric(data["release_year"], errors="coerce")
data = data.dropna()

# --- 입력(X), 출력(Y)
features = ["genre", "platform", "user_score", "release_year"]
X = data[features]
y = data["metacritic_score"]

# --- 파이프라인
categorical = ["genre", "platform"]
numerical = ["user_score", "release_year"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# --- 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# --- 평가
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# --- 모델 저장
joblib.dump(model, "metacritic_predictor.pkl")
print("✅ 모델 저장 완료: metacritic_predictor.pkl")
