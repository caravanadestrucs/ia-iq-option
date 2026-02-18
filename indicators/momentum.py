import pandas as pd

# Momentum simple (diff)
def vote(cierres, period=3):
    s = pd.Series(cierres)
    diff = s.diff(periods=period)
    if diff.iloc[-1] > 0:
        return "call"
    if diff.iloc[-1] < 0:
        return "put"
    return None
