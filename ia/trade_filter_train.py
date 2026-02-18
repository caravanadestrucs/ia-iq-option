"""Entrena un RandomForest para el TradeFilter usando `ia/trade_filter_data.csv`.
Genera `ia/models/trade_filter.pkl`.
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

CSV = os.environ.get('TRADE_FILTER_CSV', 'ia/trade_filter_data.csv')
OUT = os.environ.get('TRADE_FILTER_MODEL', 'ia/models/trade_filter.pkl')

if not os.path.exists(CSV):
    print('No se encontró CSV de entrenamiento:', CSV)
    raise SystemExit(1)

df = pd.read_csv(CSV)
# mantener solo ejemplos con label win/loss
df = df[df['result'].isin(['win','loss'])].copy()
if df.empty:
    print('No hay ejemplos con label win/loss en CSV')
    raise SystemExit(1)

X = pd.DataFrame()
X['model_votes_call'] = df['model_votes_call'].fillna(0).astype(float)
X['model_votes_put'] = df['model_votes_put'].fillna(0).astype(float)
X['indicator_votes_call'] = df['indicator_votes_call'].fillna(0).astype(float)
X['indicator_votes_put'] = df['indicator_votes_put'].fillna(0).astype(float)
X['model_votes_total'] = df['model_votes_total'].fillna(0).astype(float)
X['model_votes_in_favor'] = df['model_votes_in_favor'].fillna(0).astype(float)
X['indicator_votes_in_favor'] = df['indicator_votes_in_favor'].fillna(0).astype(float)
X['regime'] = df['regime'].map({'unknown':0,'trend':1,'range':2,'volatile':3,'flat':4,'mixed':5}).fillna(0).astype(int)
X['volatility'] = df['volatility'].fillna(0.0).astype(float)
X['price_change_1m'] = df['price_change_1m'].fillna(0.0).astype(float)
X['asset_winrate'] = df['asset_winrate'].fillna(0.0).astype(float)
X['bet_size'] = df['bet_size'].fillna(0.0).astype(float)
X['wm_confidence'] = df['wm_confidence'].fillna(0.0).astype(float)
X['minutes_to_expiry'] = df['minutes_to_expiry'].fillna(0).astype(float)

y = (df['result'] == 'win').astype(int).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_val)
proba = model.predict_proba(X_val)[:,1]
print(classification_report(y_val, pred))
try:
    print('AUC:', roc_auc_score(y_val, proba))
except Exception:
    pass

os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
joblib.dump(model, OUT)
print('Modelo guardado en', OUT)
