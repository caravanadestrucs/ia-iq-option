import pandas as pd

# EMA crossover (short/long)
def vote(cierres, short=8, long=21):
    s = pd.Series(cierres)
    ema_s = s.ewm(span=short, adjust=False).mean()
    ema_l = s.ewm(span=long, adjust=False).mean()

    if ema_s.iloc[-1] > ema_l.iloc[-1] and ema_s.iloc[-2] <= ema_l.iloc[-2]:
        return "call"
    if ema_s.iloc[-1] < ema_l.iloc[-1] and ema_s.iloc[-2] >= ema_l.iloc[-2]:
        return "put"
    return None
