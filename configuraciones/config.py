# Configuración centralizada para el bot
import os

EMAIL = os.environ.get('EMAIL', "dariocasoca@gmail.com")
PASSWORD = os.environ.get('PASSWORD', "dario2000")

ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "EURJPY"]
ASSETS_OTC = ["EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "AUDUSD-OTC", "EURJPY-OTC"]
INCLUDE_OTC = True

TIMEFRAME = 900      # 5 minutos
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

# Trend veto minimum weight (aplica veto solo si WeightManager asigna al 'trend' >= este valor)
# P.ej. 0.0 -> cualquier peso positivo activa el veto; 0.1 exige peso moderado.
TREND_VETO_MIN_WEIGHT = 0.0

