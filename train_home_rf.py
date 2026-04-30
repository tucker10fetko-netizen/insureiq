import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os

os.makedirs('models', exist_ok=True)

print("Training improved home model with Random Forest...")

df = pd.read_csv(r'C:\Users\tucke\Downloads\home_insurance.csv')

keep_cols = [
    'P1_EMP_STATUS', 'BUS_USE', 'AD_BUILDINGS', 'RISK_RATED_AREA_B',
    'SUM_INSURED_BUILDINGS', 'NCD_GRANTED_YEARS_B', 'AD_CONTENTS',
    'RISK_RATED_AREA_C', 'SUM_INSURED_CONTENTS', 'NCD_GRANTED_YEARS_C',
    'CONTENTS_COVER', 'BUILDINGS_COVER', 'P1_MAR_STATUS', 'P1_SEX',
    'APPR_ALARM', 'APPR_LOCKS', 'BEDROOMS', 'ROOF_CONSTRUCTION',
    'WALL_CONSTRUCTION', 'FLOODING', 'NEIGH_WATCH', 'OCC_STATUS',
    'OWNERSHIP_TYPE', 'PROP_TYPE', 'SAFE_INSTALLED', 'SUBSIDENCE',
    'YEARBUILT', 'CLAIM3YEARS', 'LAST_ANN_PREM_GROSS'
]

df = df[keep_cols].dropna()
df = df[df['LAST_ANN_PREM_GROSS'] > 0]

for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop('LAST_ANN_PREM_GROSS', axis=1)
y = df['LAST_ANN_PREM_GROSS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=4,
                            n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"  R²:   {r2_score(y_test, y_pred):.4f}")
print(f"  MAE:  ${mean_absolute_error(y_test, y_pred):,.2f}")
print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

with open('models/home_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('models/home_features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("\n✓ Improved home model saved.")