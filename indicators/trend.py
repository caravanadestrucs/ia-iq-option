import numpy as np

# Trend slope over last `period` bars
def vote(cierres, period=14, slope_threshold=1e-6):
    y = np.array(cierres[-period:])
    x = np.arange(len(y))
    if len(y) < 2:
        return None
    slope = np.polyfit(x, y, 1)[0]
    if slope > slope_threshold:
        return "call"
    if slope < -slope_threshold:
        return "put"
    return None
