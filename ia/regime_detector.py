"""Detector de régimen simple basado en ADX, EMA200 slope, ATR y relación cuerpo/rango.
Devuelve una etiqueta (`regime`) y métricas auxiliares.
"""
from typing import List, Dict
import pandas as pd
import numpy as np

from configuraciones.config import ADX_TREND_THRESHOLD, ADX_RANGE_THRESHOLD, EMA_SLOPE_THRESHOLD, ATR_VOL_THRESHOLD, REGIME_ADX_STRONG, REGIME_ADX_WEAK, REGIME_ATR_LOW, REGIME_ATR_HIGH


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    # normalizar nombres (orígenes de API pueden variar)
    if 'high' not in df.columns and 'max' in df.columns:
        df['high'] = df['max']
    if 'low' not in df.columns and 'min' in df.columns:
        df['low'] = df['min']
    if 'open' not in df.columns and 'from' in df.columns:
        df['open'] = df['from']
    # close ya se usa en el proyecto
    return df


def detect_regime(candles: List[Dict], adx_period: int = 14) -> Dict:
    """Devuelve dict: {'regime': str, 'adx': float, 'ema_slope': float, 'atr_pct': float, 'body_ratio': float}
    Entrada: lista de velas (cada vela: dict con keys open, close, max/min o high/low).
    """
    if not candles or len(candles) < 20:
        return {'regime': 'unknown', 'adx': None, 'ema_slope': None, 'atr_pct': None, 'body_ratio': None}

    df = pd.DataFrame(candles)
    df = _ensure_cols(df)

    # asegurar columnas mínimas
    if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns or 'open' not in df.columns:
        return {'regime': 'unknown', 'adx': None, 'ema_slope': None, 'atr_pct': None, 'body_ratio': None}

    # TR y ATR
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=adx_period, min_periods=1).mean()
    atr_last = float(atr.iloc[-1]) if not atr.isnull().all() else 0.0
    atr_pct = atr_last / float(df['close'].iloc[-1]) if df['close'].iloc[-1] else 0.0

    # DM+ / DM- para ADX
    up_move = df['high'].diff()
    down_move = df['low'].shift(1) - df['low']
    dm_plus = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # suavizado simple (rolling sum aproximado a Wilder smoothing)
    dm_plus_s = dm_plus.rolling(window=adx_period, min_periods=1).sum()
    dm_minus_s = dm_minus.rolling(window=adx_period, min_periods=1).sum()
    atr_s = atr.rolling(window=adx_period, min_periods=1).mean()

    di_plus = 100.0 * (dm_plus_s / (atr_s + 1e-9))
    di_minus = 100.0 * (dm_minus_s / (atr_s + 1e-9))
    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    adx = dx.rolling(window=adx_period, min_periods=1).mean()
    adx_last = float(adx.iloc[-1]) if not adx.isnull().all() else 0.0

    # EMA200 pendiente (usar tantos puntos como haya si <200)
    span = 200 if len(df) >= 200 else max(10, len(df))
    ema200 = df['close'].ewm(span=span, adjust=False).mean()
    # comparar último valor con el valor hace 10 velas
    look = 10 if len(ema200) > 10 else max(1, len(ema200)-1)
    prev_ema = ema200.iloc[-look]
    ema_slope = (ema200.iloc[-1] - prev_ema) / (prev_ema + 1e-9)

    # relación cuerpo/rango (promedio)
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, np.nan)
    body_ratio = (body / rng).rolling(window=adx_period, min_periods=1).mean().iloc[-1]
    body_ratio = float(np.nan_to_num(body_ratio))

    # reglas mejoradas para etiquetar régimen (más granular)
    # Prioridad: low-vol, high-vol, strong-trend, weak-trend, lateral (range), mixed
    regime = 'mixed'

    # baja/alta volatilidad (por ATR %)
    if atr_pct <= REGIME_ATR_LOW:
        regime = 'low_volatility'
    elif atr_pct >= REGIME_ATR_HIGH:
        regime = 'high_volatility'
    else:
        # tendencia fuerte / débil según ADX + pendiente EMA
        if adx_last >= REGIME_ADX_STRONG and abs(ema_slope) >= EMA_SLOPE_THRESHOLD:
            regime = 'strong_trend'
        elif adx_last >= REGIME_ADX_WEAK and abs(ema_slope) >= (EMA_SLOPE_THRESHOLD * 0.5):
            regime = 'weak_trend'
        # lateral / rango
        elif adx_last < ADX_RANGE_THRESHOLD and body_ratio < 0.6:
            regime = 'lateral'
        elif abs(ema_slope) < EMA_SLOPE_THRESHOLD and adx_last < ADX_RANGE_THRESHOLD:
            regime = 'lateral'
        else:
            regime = 'mixed'

    # dirección de tendencia (útil para aplicar reglas de 'seguir tendencia')
    trend_dir = 'up' if ema_slope > 0 else ('down' if ema_slope < 0 else 'flat')

    return {
        'regime': regime,
        'trend_dir': trend_dir,
        'adx': round(adx_last, 3),
        'ema_slope': round(float(ema_slope), 6),
        'atr_pct': round(float(atr_pct), 6),
        'body_ratio': round(float(body_ratio), 3)
    }