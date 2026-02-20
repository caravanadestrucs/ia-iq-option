import math
from ia.regime_detector import detect_regime


def make_candles_from_prices(prices, spread=0.001):
    out = []
    for i, p in enumerate(prices):
        o = prices[i-1] if i > 0 else p
        high = p + abs(p) * spread
        low = p - abs(p) * spread
        out.append({'open': o, 'close': p, 'high': high, 'low': low})
    return out


def test_detect_low_volatility():
    prices = [100.0 + (0.0001 * (i % 2)) for i in range(120)]
    candles = make_candles_from_prices(prices, spread=0.00005)
    info = detect_regime(candles)
    assert info['regime'] == 'low_volatility'


def test_detect_strong_trend():
    prices = [100.0 + 0.05 * i for i in range(220)]
    candles = make_candles_from_prices(prices, spread=0.001)
    info = detect_regime(candles)
    assert info['regime'] in ('strong_trend', 'weak_trend')
    assert info['trend_dir'] == 'up'


def test_detect_high_volatility():
    # simulate large ranges -> high ATR%
    prices = []
    p = 100.0
    for i in range(150):
        p += (1.0 if i % 2 == 0 else -0.8)
        prices.append(p)
    candles = make_candles_from_prices(prices, spread=0.02)
    info = detect_regime(candles)
    assert info['regime'] == 'high_volatility'
