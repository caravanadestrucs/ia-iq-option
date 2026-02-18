import os
import csv
import joblib
import numpy as np
from typing import Dict, Any

from configuraciones.config import INDICATOR_CONFIRMATIONS, MAX_VOLATILITY_FOR_TRADING

CSV_PATH_DEFAULT = "ia/trade_filter_data.csv"
MODEL_PATH_DEFAULT = "ia/models/trade_filter.pkl"

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
    """Validador "win-confidence" que puede usar un modelo guardado o una regla "fallback".

    - predict(features) -> {'approve': bool, 'prob': float}
    - append_example(features, label) -> guarda ejemplo en CSV para entrenamiento
    """

    def __init__(self, model_path: str = MODEL_PATH_DEFAULT, threshold: float = 0.62, csv_path: str = CSV_PATH_DEFAULT):
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

    def _vector(self, features: Dict) -> np.ndarray:
        # Vector simple: keep stable order for the model training later
        v = []
        v.append(1.0 if features.get('direction') == 'call' else 0.0)
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
        """Devuelve {'approve': bool, 'prob': float}.
        Usa modelo si existe; si no, aplica regla conservadora como fallback.
        """
        self._load()
        x = None
        try:
            x = self._vector(features)
        except Exception:
            x = None

        prob = 0.0
        if self._model is not None and x is not None:
            try:
                prob = float(self._model.predict_proba(x)[:, 1][0])
            except Exception:
                prob = 0.0
        else:
            # fallback rule: require >=2 model votes, at least INDICATOR_CONFIRMATIONS indicators
            # and not too high volatility
            mv_total = int(features.get('model_votes_total', 0))
            ind_in_favor = int(features.get('indicator_votes_in_favor', 0))
            vol = float(features.get('volatility', 0.0))
            rule_ok = (mv_total >= 2) and (ind_in_favor >= INDICATOR_CONFIRMATIONS) and (vol < max(0.0, float(MAX_VOLATILITY_FOR_TRADING)))
            prob = 0.70 if rule_ok else 0.0

        approve = prob >= self.threshold
        return {"approve": approve, "prob": prob}

    def append_example(self, features: Dict, label: str):
        """Anexa features + resultado al CSV para entrenamiento posterior.
        No debe lanzar excepciones.
        """
        try:
            os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
            header = [
                'asset','direction','model_votes_call','model_votes_put','indicator_votes_call','indicator_votes_put',
                'model_votes_total','model_votes_in_favor','indicator_votes_in_favor','regime','volatility',
                'price_change_1m','asset_winrate','bet_size','wm_confidence','minutes_to_expiry',
                'result','profit','entry_balance','exit_balance'
            ]
            write_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
            row = {
                'asset': features.get('asset',''),
                'direction': features.get('direction',''),
                'model_votes_call': features.get('model_votes_call',''),
                'model_votes_put': features.get('model_votes_put',''),
                'indicator_votes_call': features.get('indicator_votes_call',''),
                'indicator_votes_put': features.get('indicator_votes_put',''),
                'model_votes_total': features.get('model_votes_total',''),
                'model_votes_in_favor': features.get('model_votes_in_favor',''),
                'indicator_votes_in_favor': features.get('indicator_votes_in_favor',''),
                'regime': features.get('regime',''),
                'volatility': features.get('volatility',''),
                'price_change_1m': features.get('price_change_1m',''),
                'asset_winrate': features.get('asset_winrate',''),
                'bet_size': features.get('bet_size',''),
                'wm_confidence': features.get('wm_confidence',''),
                'minutes_to_expiry': features.get('minutes_to_expiry',''),
                'result': label,
                'profit': features.get('profit',''),
                'entry_balance': features.get('entry_balance',''),
                'exit_balance': features.get('exit_balance','')
            }
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            # non-blocking
            pass
