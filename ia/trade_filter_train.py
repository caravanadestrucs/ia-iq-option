"""Script ligero para entrenar `TradeFilter` a partir de `ia/trade_filter_data.csv`.
Genera `ia/models/trade_filter.pkl`.

Uso:
    py ia/trade_filter_train.py

"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

DATA_CSV = os.environ.get('TRADE_FILTER_CSV', 'ia/trade_filter_data.csv')
MODEL_OUT = os.environ.get('TRADE_FILTER_MODEL', 'ia/models/trade_filter.pkl')

REGIME_MAP = {
    None: 0,
    'unknown': 0,
    'trend': 1,
    'range': 2,
    'volatile': 3,
    'flat': 4,
    'mixed': 5
}

FEATURES = [
    'model_votes_call', 'model_votes_put', 'indicator_votes_call', 'indicator_votes_put',
    'model_votes_total', 'model_votes_in_favor', 'indicator_votes_in_favor',
    'regime', 'volatility', 'price_change_1m', 'asset_winrate', 'bet_size', 'wm_confidence', 'minutes_to_expiry'
]

if not os.path.exists(DATA_CSV):
    print('No se encontró el CSV de entrenamiento:', DATA_CSV)
    raise SystemExit(1)

df = pd.read_csv(DATA_CSV)
if df.empty:
    print('CSV de entrenamiento vacío')
    raise SystemExit(1)

# filtrar solo trades con resultado conocido (win/loss)
df = df[df['result'].isin(['win','loss'])].copy()
if df.empty:
    print('No hay ejemplos con label "win"/"loss" en el CSV')
    raise SystemExit(1)

# target: 1=win, 0=loss
df['y'] = (df['result'] == 'win').astype(int)

# preparar X
for c in ['model_votes_call','model_votes_put','indicator_votes_call','indicator_votes_put',
          'model_votes_total','model_votes_in_favor','indicator_votes_in_favor']:
    if c not in df.columns:
        df[c] = 0

# encode regime
df['regime_enc'] = df['regime'].map(REGIME_MAP).fillna(0).astype(int)

X = pd.DataFrame()
X['model_votes_call'] = df['model_votes_call'].fillna(0).astype(float)
X['model_votes_put'] = df['model_votes_put'].fillna(0).astype(float)
X['indicator_votes_call'] = df['indicator_votes_call'].fillna(0).astype(float)
X['indicator_votes_put'] = df['indicator_votes_put'].fillna(0).astype(float)
X['model_votes_total'] = df['model_votes_total'].fillna(0).astype(float)
X['model_votes_in_favor'] = df['model_votes_in_favor'].fillna(0).astype(float)
X['indicator_votes_in_favor'] = df['indicator_votes_in_favor'].fillna(0).astype(float)
X['regime'] = df['regime_enc']
X['volatility'] = df['volatility'].fillna(0.0).astype(float)
X['price_change_1m'] = df['price_change_1m'].fillna(0.0).astype(float)
X['asset_winrate'] = df['asset_winrate'].fillna(0.0).astype(float)
X['bet_size'] = df['bet_size'].fillna(0.0).astype(float)
X['wm_confidence'] = df['wm_confidence'].fillna(0.0).astype(float)
X['minutes_to_expiry'] = df['minutes_to_expiry'].fillna(0).astype(float)

y = df['y'].values

# quick train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print('Accuracy test:', accuracy_score(y_test, pred))
try:
    print('AUC test:', roc_auc_score(y_test, proba))
except Exception:
    pass
print(classification_report(y_test, pred))

# cross-val
scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print('CV AUC (5-fold):', scores.mean(), scores)

os.makedirs(os.path.dirname(MODEL_OUT) or '.', exist_ok=True)
joblib.dump(model, MODEL_OUT)
print('Modelo guardado en', MODEL_OUT)
