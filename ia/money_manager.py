from configuraciones.config import VOLATILITY_SIZE_REDUCTION, MIN_BET, MONTO, RISK_PCT_HIGH, RISK_PCT_MED, RISK_PCT_LOW

# Multiplicadores por régimen (ajustables)
REGIME_WEIGHT_MULTIPLIERS = {
    'trend': {'lstm': 1.8, 'rf': 1.2, 'xgb': 1.2},
    'range': {'rsi': 1.6, 'stochastic': 1.6, 'macd': 1.2},
    'volatile': {'all': 0.7}
}


def compute_bet_size(winrate=None, regime=None):
    """Devuelve el `monto` ajustado por régimen.

    - Base = MONTO (el valor que tú configuras).
    - Si `regime == 'volatile'` => reduce a (1 - VOLATILITY_SIZE_REDUCTION) * MONTO (p.ej. 70%).
    - Ignora winrate para sizing (tal como pediste).
    - Retorna al menos `MIN_BET`.
    """
    try:
        base = float(MONTO)
        if regime == 'volatile':
            base = round(base * (1.0 - VOLATILITY_SIZE_REDUCTION), 2)
        return max(MIN_BET, round(base, 2))
    except Exception:
        return max(MIN_BET, float(MONTO))