import pandas as pd

# Rate of Change
def vote(cierres, period=12):
    s = pd.Series(cierres)
    roc = s.pct_change(periods=period) * 100
    if roc.iloc[-1] > 0 and roc.iloc[-2] <= 0:
        return "call"
    if roc.iloc[-1] < 0 and roc.iloc[-2] >= 0:
        return "put"
    return None
