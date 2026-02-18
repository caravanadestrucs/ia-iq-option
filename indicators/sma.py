import pandas as pd

# SMA crossover (short/long)
def vote(cierres, short=5, long=20):
    s = pd.Series(cierres)
    sma_s = s.rolling(short).mean()
    sma_l = s.rolling(long).mean()
    if sma_s.iloc[-1] > sma_l.iloc[-1] and sma_s.iloc[-2] <= sma_l.iloc[-2]:
        return "call"
    if sma_s.iloc[-1] < sma_l.iloc[-1] and sma_s.iloc[-2] >= sma_l.iloc[-2]:
        return "put"
    return None
