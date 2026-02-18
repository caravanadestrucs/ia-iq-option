import pandas as pd

# Stochastic (%K / %D) - signals in oversold/overbought zones
def vote(cierres, k_period=14, d_period=3, low=20, high=80):
    s = pd.Series(cierres)
    low_k = s.rolling(window=k_period).min()
    high_k = s.rolling(window=k_period).max()
    k = 100 * (s - low_k) / (high_k - low_k + 1e-9)
    d = k.rolling(window=d_period).mean()

    if k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2] and k.iloc[-1] < low:
        return "call"
    if k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2] and k.iloc[-1] > high:
        return "put"
    return None
