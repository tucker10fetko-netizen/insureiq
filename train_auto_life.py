import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os

os.makedirs('models', exist_ok=True)

def evaluate(name, y_test, y_pred):
    print(f"\n── {name} Results ──────────────────────")
    print(f"  R²:   {r2_score(y_test, y_pred):.4f}")
    print(f"  MAE:  ${mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

def save_model(obj, filename):
    with open(f'models/{filename}', 'wb') as f:
        pickle.dump(obj, f)
    print(f"  Saved: models/{filename}")

# ── 1. AUTO MODEL ─────────────────────────────────────────────
print("\n[1/2] Training auto model...")

df_a = pd.read_csv(r'C:\Users\tucke\Downloads\car_insurance_premium_dataset.csv')

X_a = df_a.drop('Insurance Premium ($)', axis=1)
y_a = df_a['Insurance Premium ($)']

X_train, X_test, y_train, y_test = train_test_split(X_a, y_a, test_size=0.2, random_state=42)

auto_model = RandomForestRegressor(n_estimators=200, max_depth=20,
                                    min_samples_leaf=2, n_jobs=-1, random_state=42)
auto_model.fit(X_train, y_train)
evaluate("Auto", y_test, auto_model.predict(X_test))
save_model(auto_model, 'auto_model.pkl')
save_model(list(X_a.columns), 'auto_features.pkl')

# ── 2. LIFE MODEL ─────────────────────────────────────────────
print("\n[2/2] Training life model...")

df_l = pd.read_csv(r'C:\Users\tucke\Downloads\Kaggle.csv')

# Keep useful columns and drop rows with missing premium
keep = ['ENTRY AGE', 'SEX', 'POLICY TYPE 1', 'PAYMENT MODE',
        'BENEFIT', 'SUBSTANDARD RISK', 'Policy Year', 'Premium']
df_l = df_l[keep].dropna()
df_l = df_l[df_l['Premium'] > 0]

for col in df_l.select_dtypes(include='object').columns:
    df_l[col] = LabelEncoder().fit_transform(df_l[col].astype(str))

X_l = df_l.drop('Premium', axis=1)
y_l = df_l['Premium']

X_train, X_test, y_train, y_test = train_test_split(X_l, y_l, test_size=0.2, random_state=42)

life_model = RandomForestRegressor(n_estimators=200, max_depth=20,
                                    min_samples_leaf=4, n_jobs=-1, random_state=42)
life_model.fit(X_train, y_train)
evaluate("Life", y_test, life_model.predict(X_test))
save_model(life_model, 'life_model.pkl')
save_model(list(X_l.columns), 'life_features.pkl')

print("\n✓ Both models trained and saved.")