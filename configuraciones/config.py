# Configuración centralizada para el bot
import os

EMAIL = os.environ.get('EMAIL', "dariocasoca@gmail.com")
PASSWORD = os.environ.get('PASSWORD', "dario2000")

ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "EURJPY"]
ASSETS_OTC = ["EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "AUDUSD-OTC", "EURJPY-OTC"]
INCLUDE_OTC = True

TIMEFRAME = 900      # 5 minutos
# horizonte usado por los modelos de largo plazo (3 velas de 5m = 15 minutos)
HORIZON_CANDLES = 3
EXPIRATION = 5       # expiración en minutos
MONTO = float(os.environ.get('MONTO', 20))

LOOKBACK = 90 # velas usadas para indicadores (ajustable, p.ej. 900 para 5h de datos en TF=5m)
WINDOW_SIZE = 500
RETRAIN_HOURS = 3

# Selección de mercados (operar solo top-N por 'condiciones')
MAX_ACTIVE_MARKETS = 5
VOLATILITY_LOOKBACK = 100  # velas usadas para medir volatilidad/condiciones

# Investigación automática de trades con resultado 'unknown' (segundos)
INVESTIGATION_INTERVAL_SECONDS = 300
# Política para resolver unknown: 'inferred' usa WeightManager, 'assume_loss' marca como loss
UNKNOWN_RESOLUTION_POLICY = 'inferred'  # options: 'inferred', 'assume_loss'


# Indicadores habilitados (nombres de archivos en carpeta indicators/)
ENABLED_INDICATORS = ["ema", "rsi", "macd", "bollinger", "stochastic", "sma", "roc", "momentum", "trend"]

# Número mínimo de indicadores que deben confirmar la misma dirección
INDICATOR_CONFIRMATIONS = 1

# Ruta del archivo SQLite para el registro de operaciones
LOG_DB_PATH = os.environ.get('LOG_DB_PATH', "ia/trades.db")

# Gestor de pesos (IA ligera)
WEIGHT_LR = 0.05
WEIGHT_FILE = "ia/weights.json"
WM_CONFIDENCE_THRESHOLD = float(os.environ.get('WM_CONFIDENCE_THRESHOLD', 0.90))
# Umbral para apostar mitad cuando la confianza es moderada
WM_HALF_CONFIDENCE_THRESHOLD = float(os.environ.get('WM_HALF_CONFIDENCE_THRESHOLD', 0.75))

# Money management & regime detection (user-configurable)
# El tamaño base se toma directamente desde `MONTO` (configurado por el usuario).
# El code calcula reducciones en función del winrate y del régimen; no se usa una variable `CAPITAL`.
WINRATE_LOOKBACK = 100          # número de trades para calcular winrate por-asset (ajustado a 100)
VOLATILITY_SIZE_REDUCTION = 0.30  # reducir tamaño en 30% si régimen='volatile'
MIN_BET = 1.0                   # apuesta mínima permitida

# Máxima volatilidad histórica aceptable para abrir operaciones (std de retornos).
# Si la volatilidad calculada > este valor, el TradeFilter/las reglas pueden rechazar la operación.
MAX_VOLATILITY_FOR_TRADING = float(os.environ.get('MAX_VOLATILITY_FOR_TRADING', 0.01))

# Sizing relativo a MONTO (porcentajes)
# Escenario A (winrate >= 65%) -> mantener 2 => 2/20 = 10%  => RISK_PCT_HIGH = 0.10
# Escenario B (winrate 58-60%) -> 1.5 => 7.5% => RISK_PCT_MED = 0.075
# Escenario C (winrate <=55%) -> 1 => 5% => RISK_PCT_LOW = 0.05
RISK_PCT_HIGH = 0.10
RISK_PCT_MED = 0.075
RISK_PCT_LOW = 0.05

# Regime-detector thresholds (ajustables)
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
EMA_SLOPE_THRESHOLD = 0.0005    # pendiente relativa de EMA200 para considerar 'inclinada'
ATR_VOL_THRESHOLD = 0.002       # ATR% sobre precio para considerar 'volatility'

# Regime classifier thresholds (más granular)
REGIME_ADX_STRONG = float(os.environ.get('REGIME_ADX_STRONG', 35))     # ADX >= -> strong trend
REGIME_ADX_WEAK = float(os.environ.get('REGIME_ADX_WEAK', ADX_TREND_THRESHOLD))  # ADX >= -> weak trend
REGIME_ATR_LOW = float(os.environ.get('REGIME_ATR_LOW', 0.0006))      # ATR% <= -> low volatility
REGIME_ATR_HIGH = float(os.environ.get('REGIME_ATR_HIGH', ATR_VOL_THRESHOLD))    # ATR% >= -> high volatility

# Acción por defecto para baja volatilidad (si True, no operar — puedes cambiar a False)
REGIME_LOW_VOL_SKIP = bool(int(os.environ.get('REGIME_LOW_VOL_SKIP', '1')))
# Reducción de tamaño en alta volatilidad (usa VOLATILITY_SIZE_REDUCTION por defecto)
REGIME_HIGH_VOL_SIZE_REDUCTION = float(os.environ.get('REGIME_HIGH_VOL_SIZE_REDUCTION', VOLATILITY_SIZE_REDUCTION))

# Trend veto minimum weight (aplica veto solo si WeightManager asigna al 'trend' >= este valor)
# P.ej. 0.0 -> cualquier peso positivo activa el veto; 0.1 exige peso moderado.
TREND_VETO_MIN_WEIGHT = 0.9  # aumentado para reducir vetos de trend y abrir un poco más de operaciones

# --------------------------------------------------
# Win‑confidence / exports
# - USE_WIN_CONFIDENCE_AS_DECISION: si True, la decision de abrir/size la toma el "win confidence" (TradeFilter)
# - WIN_CONFIDENCE_MODEL_PATH: ruta al modelo serializado del TradeFilter
# - DOWNLOADS_DIR: directorio donde se guardan los .zip exportados
# --------------------------------------------------
# Volvemos al sistema ponderado clásico: desactivar Win‑confidence como criterio principal
USE_WIN_CONFIDENCE_AS_DECISION = False
WIN_CONFIDENCE_MODEL_PATH = os.environ.get('WIN_CONFIDENCE_MODEL_PATH', 'ia/models/trade_filter.pkl')
DOWNLOADS_DIR = os.environ.get('DOWNLOADS_DIR', 'downloads')

# Umbrales para que los modelos emitan voto (reduce votos "forzados" que causan sesgo)
# - MODEL_VOTE_CONFIDENCE_THRESHOLD: probabilidad mínima que exige RF/XGB para votar firmemente
# - LSTM_VOTE_MIN_REL_DELTA: cambio relativo mínimo entre pred_price y último precio para votar LSTM
MODEL_VOTE_CONFIDENCE_THRESHOLD = float(os.environ.get('MODEL_VOTE_CONFIDENCE_THRESHOLD', 0.60))
LSTM_VOTE_MIN_REL_DELTA = float(os.environ.get('LSTM_VOTE_MIN_REL_DELTA', 0.00025))


