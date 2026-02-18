import os
import csv
import joblib
import numpy as np
from typing import Dict, Any

from configuraciones.config import INDICATOR_CONFIRMATIONS

FEATURE_COLUMNS = [
    "asset",
    "direction",
    "model_votes_call",
    "model_votes_put",
    "indicator_votes_call",
    "indicator_votes_put",
    "model_votes_total",
    "model_votes_in_favor",
    "indicator_votes_in_favor",
    "regime",
    "volatility",
    "price_change_1m",
    "asset_winrate",
    "bet_size",
    "wm_confidence",
    "minutes_to_expiry",
]

REGIME_MAP = {
    None: 0,
    'unknown': 0,
    'trend': 1,
    'range': 2,
    'volatile': 3,
    'flat': 4,
    'mixed': 5
}


class TradeFilter:
    """Meta‑validador de operaciones.

    - predict(features) -> {'approve': bool, 'prob': float}
      Usa un modelo guardado si existe; si no, aplica una regla fallback simple.
    - append_example(features, label) -> guarda ejemplo a CSV para entrenamiento posterior.
    """

    def __init__(self, model_path: str = "ia/models/trade_filter.pkl", threshold: float = 0.62,
                 csv_path: str = "ia/trade_filter_data.csv"):
        self.model_path = model_path
        self.threshold = float(threshold)
        self.csv_path = csv_path
        self._model = None

    def _load(self):
        if self._model is None and os.path.exists(self.model_path):
            try:
                self._model = joblib.load(self.model_path)
            except Exception:
                self._model = None

    def _encode_regime(self, r: Any) -> int:
        return REGIME_MAP.get(r, 0)

    def _features_vector(self, features: Dict) -> np.ndarray:
        # Build numeric vector in a stable order; missing -> 0.0
        v = []
        v.append(0.0)  # asset (ignored numerically)
        # direction: call=1, put=0 (keep as numeric hint)
        d = features.get('direction')
        v.append(1.0 if d == 'call' else 0.0)
        v.append(float(features.get('model_votes_call', 0)))
        v.append(float(features.get('model_votes_put', 0)))
        v.append(float(features.get('indicator_votes_call', 0)))
        v.append(float(features.get('indicator_votes_put', 0)))
        v.append(float(features.get('model_votes_total', 0)))
        v.append(float(features.get('model_votes_in_favor', 0)))
        v.append(float(features.get('indicator_votes_in_favor', 0)))
        v.append(float(self._encode_regime(features.get('regime'))))
        v.append(float(features.get('volatility', 0.0)))
        v.append(float(features.get('price_change_1m', 0.0)))
        v.append(float(features.get('asset_winrate') or 0.0))
        v.append(float(features.get('bet_size', 0.0)))
        v.append(float(features.get('wm_confidence', 0.0)))
        v.append(float(features.get('minutes_to_expiry', 0.0)))
        return np.array(v).reshape(1, -1)

    def predict(self, features: Dict) -> Dict[str, Any]:
        """Devuelve dict {'approve': bool, 'prob': float}.
        - Si existe modelo entrenado, usa su probabilidad.
        - Si no, aplica una regla fallback simple y conservadora.
        """
        self._load()
        x = self._features_vector(features)
        if self._model is not None:
            try:
                prob = float(self._model.predict_proba(x)[:, 1][0])
            except Exception:
                prob = 0.0
        else:
            # fallback: exigir al menos 2 modelos y confirmación de indicadores
            mv_total = int(features.get('model_votes_total', 0))
            ind_in_favor = int(features.get('indicator_votes_in_favor', 0))
            vol = float(features.get('volatility', 1.0))
            # regla conservadora: 2+ modelos + al menos INDICATOR_CONFIRMATIONS indicadores + volatilidad moderada
            rule_ok = (mv_total >= 2) and (ind_in_favor >= INDICATOR_CONFIRMATIONS) and (vol < 0.01)
            prob = 0.65 if rule_ok else 0.0
        approve = prob >= self.threshold
        return {"approve": approve, "prob": prob}

    def append_example(self, features: Dict, label: str):
        """Anexa un ejemplo (features + label) al CSV de entrenamiento.
        Crea el CSV con encabezado si no existe.
        """
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
        header = FEATURE_COLUMNS + ["result", "profit", "entry_balance", "exit_balance"]
        write_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        # normalizar fila
        row = {}
        for c in FEATURE_COLUMNS:
            v = features.get(c)
            if c == 'regime':
                row[c] = features.get(c)
            else:
                row[c] = v if v is not None else ''
        row.update({"result": label, "profit": features.get('profit', ''),
                    "entry_balance": features.get('entry_balance', ''),
                    "exit_balance": features.get('exit_balance', '')})
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            # no debe romper la ejecución del bot si falla el logging
            pass
