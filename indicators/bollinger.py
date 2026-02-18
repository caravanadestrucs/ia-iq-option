import pandas as pd

# Bollinger Bands simple signal (touch + revert)
def vote(cierres, period=20, dev=2.0):
    s = pd.Series(cierres)
    ma = s.rolling(period).mean()
    sd = s.rolling(period).std()
    upper = ma + dev * sd
    lower = ma - dev * sd

    if s.iloc[-1] < lower.iloc[-1] and s.iloc[-2] >= lower.iloc[-2]:
        return "call"
    if s.iloc[-1] > upper.iloc[-1] and s.iloc[-2] <= upper.iloc[-2]:
        return "put"
    return None
