import pandas as pd

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

# RSI oversold/overbought
def vote(cierres, period=14, low=30, high=70):
    s = pd.Series(cierres)
    rsi_val = _rsi(s, period=period).iloc[-1]
    if rsi_val < low:
        return "call"
    if rsi_val > high:
        return "put"
    return None
