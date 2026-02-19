import os
import json
import math

class WeightManager:
    """Gestor de pesos simple (online logistic-style update).

    - predictors: lista de nombres (modelos + indicadores)
    - lr: learning rate para actualizaciones online
    - guarda pesos en un JSON (WEIGHT_FILE)
    """
    def __init__(self, predictors, lr=0.05, weight_file='ia/weights.json', regime_alpha=0.6):
        self.predictors = list(predictors)
        self.lr = lr
        self.weight_file = weight_file
        # pesos iniciales neutrales (0.0) y bias (global)
        # Antes: 1.0 favorecía implícitamente 'call' en empates; usar 0.0 para empezar neutral
        self.weights = {p: 0.0 for p in self.predictors}
        self.bias = 0.0
        # pesos específicos por régimen (aprendizaje online separado)
        self.regime_weights = {}  # e.g. {'trend': {'rf':1.0,...}, 'range': {...}}
        self.regime_bias = {}
        # blending factor para combinar global vs regime-specific en predict
        self.regime_alpha = regime_alpha
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.weight_file):
                with open(self.weight_file, 'r') as f:
                    data = json.load(f)
                    self.weights.update(data.get('weights', {}))
                    self.bias = data.get('bias', self.bias)
                    # cargar pesos por régimen si existen
                    self.regime_weights.update(data.get('regime_weights', {}))
                    self.regime_bias.update(data.get('regime_bias', {}))
        except Exception:
            pass

    def _save(self):
        try:
            with open(self.weight_file, 'w') as f:
                json.dump({'weights': self.weights, 'bias': self.bias,
                           'regime_weights': self.regime_weights, 'regime_bias': self.regime_bias}, f)
        except Exception:
            pass

    def _features(self, model_votes, indicator_votes):
        """Construye vector de features: +1 call, -1 put, 0 none"""
        feats = {}
        for p in self.predictors:
            v = None
            if p in model_votes:
                v = model_votes.get(p)
            elif p in indicator_votes:
                v = indicator_votes.get(p)
            if v == 'call':
                feats[p] = 1.0
            elif v == 'put':
                feats[p] = -1.0
            else:
                feats[p] = 0.0
        return feats

    def predict_proba(self, model_votes, indicator_votes):
        feats = self._features(model_votes, indicator_votes)
        s = self.bias
        for k, val in feats.items():
            s += self.weights.get(k, 0.0) * val
        # sigmoid
        try:
            p = 1.0 / (1.0 + math.exp(-s))
        except OverflowError:
            p = 0.0 if s < 0 else 1.0
        return p

    def predict_regime(self, model_votes, indicator_votes, regime=None, threshold=0.5):
        """Predict usando mezcla de pesos globales y pesos específicos del régimen si existen."""
        feats = self._features(model_votes, indicator_votes)
        # s_global
        s_global = self.bias
        for k, val in feats.items():
            s_global += self.weights.get(k, 0.0) * val
        # s_regime (si existe)
        if regime and regime in self.regime_weights:
            s_reg = self.regime_bias.get(regime, 0.0)
            for k, val in feats.items():
                s_reg += float(self.regime_weights.get(regime, {}).get(k, 0.0)) * val
            # mezclar
            s = (1.0 - self.regime_alpha) * s_global + self.regime_alpha * s_reg
        else:
            s = s_global
        try:
            p = 1.0 / (1.0 + math.exp(-s))
        except OverflowError:
            p = 0.0 if s < 0 else 1.0
        # usar '>' para que p==threshold no favorezca automáticamente 'call'
        return ("call" if p > threshold else "put", p)

    def predict(self, model_votes, indicator_votes, threshold=0.5):
        p = self.predict_proba(model_votes, indicator_votes)
        # evitar que p==threshold devuelva automáticamente 'call'
        return ("call" if p > threshold else "put", p)

    def predict_with_adjustments(self, model_votes, indicator_votes, weight_multipliers=None, threshold=0.5):
        """Predicción usando multiplicadores temporales de pesos (no persiste cambios).
        - weight_multipliers: dict predictor_name->multiplier. Special key 'all' aplica a todos.
        """
        feats = self._features(model_votes, indicator_votes)
        s = self.bias
        for k, val in feats.items():
            base_w = self.weights.get(k, 0.0)
            mult = 1.0
            if weight_multipliers:
                mult = weight_multipliers.get(k, weight_multipliers.get('all', 1.0))
            s += base_w * mult * val
        try:
            p = 1.0 / (1.0 + math.exp(-s))
        except OverflowError:
            p = 0.0 if s < 0 else 1.0
        # usar comparación estricta para evitar sesgo en empates
        return ("call" if p > threshold else "put", p)

    def update(self, model_votes, indicator_votes, outcome, regime: str = None):
        """Actualización online por gradiente simple (cross-entropy).
        - outcome: 'call' o 'put' — resultado real del trade (ganador)
        - regime: si se provee, actualiza también los pesos específicos de ese régimen
        """
        feats = self._features(model_votes, indicator_votes)
        # actualizar global
        p = self.predict_proba(model_votes, indicator_votes)
        y = 1.0 if outcome == 'call' else 0.0
        error = y - p
        for k, val in feats.items():
            self.weights[k] = self.weights.get(k, 0.0) + self.lr * error * val
        self.bias += self.lr * error

        # actualizar específico por régimen (si aplica)
        if regime:
            if regime not in self.regime_weights:
                # inicializar con copia de pesos globales
                self.regime_weights[regime] = {p: float(self.weights.get(p, 0.0)) for p in self.predictors}
                self.regime_bias[regime] = float(self.bias)
            # calcular probabilidad usando solo los pesos del régimen
            s_reg = self.regime_bias.get(regime, 0.0)
            for k, val in feats.items():
                s_reg += float(self.regime_weights[regime].get(k, 0.0)) * val
            try:
                p_reg = 1.0 / (1.0 + math.exp(-s_reg))
            except OverflowError:
                p_reg = 0.0 if s_reg < 0 else 1.0
            error_reg = y - p_reg
            for k, val in feats.items():
                self.regime_weights[regime][k] = float(self.regime_weights[regime].get(k, 0.0) + self.lr * error_reg * val)
            self.regime_bias[regime] = float(self.regime_bias.get(regime, 0.0) + self.lr * error_reg)

        self._save()
    def get_weights(self):
        return dict(self.weights)
