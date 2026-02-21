# Silenciar warnings y reducir verbosidad de librerías ruidosas
import os
# TensorFlow/oneDNN/CPU logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # 0=DEBUG,1=INFO,2=WARNING,3=ERROR (suprime INFO)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'      # desactiva mensaje oneDNN

import warnings
warnings.filterwarnings('ignore')               # suprime warnings de Python

import logging
logging.basicConfig(level=logging.ERROR)        # nivel por defecto = ERROR
# ajustar logs de librerías conocidas por ser ruidosas
for _n in ('tensorflow', 'urllib3', 'matplotlib', 'numba', 'sklearn', 'xgboost', 'keras', 'h5py', 'iqoptionapi'):
    logging.getLogger(_n).setLevel(logging.ERROR)

# FILTRO ADICIONAL: suprime mensajes que contienen '**warning**' o patrones repetidos por iqoptionapi
class _HideIQWarnings(logging.Filter):
    def filter(self, record):
        try:
            msg = record.getMessage()
            if not isinstance(msg, str):
                return True
            # suprime mensajes marcados por iqoptionapi y otros ruidosos
            if '**warning**' in msg or 'get_all_init_v2' in msg or 'get_digital_underlying_list_data' in msg:
                return False
        except Exception:
            return True
        return True

# añadir filtro al logger root y al logger específico de iqoptionapi
logging.getLogger().addFilter(_HideIQWarnings())
logging.getLogger('iqoptionapi').addFilter(_HideIQWarnings())

import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from iqoptionapi.stable_api import IQ_Option
import time
import sys

# Cargar configuraciones externas
from configuraciones.config import *

# gestor de pesos (IA ligera)
from ia.weight_manager import WeightManager
from ia.trade_logger import TradeLogger
from ia.trade_filter import TradeFilter
from ia.regime_detector import detect_regime

# (la conexión con IQ_Option se hace más abajo en la sección 'CONEXIÓN' usando EMAIL/PASSWORD del config)

# =============================
# CONFIGURACIÓN (importada desde configuraciones/config.py)
# =============================
# (valores cargados desde configuraciones.config)
# =============================
# CONEXIÓN
# =============================

# Monkeypatch defensivo: asegurar que llamadas internas no devuelvan None/estructuras inválidas
import iqoptionapi.stable_api as _stable_api
_orig_get_digital = getattr(_stable_api.IQ_Option, 'get_digital_underlying_list_data', None)

def _safe_get_digital_underlying_list_data(self, *a, **k):
    try:
        if _orig_get_digital:
            r = _orig_get_digital(self, *a, **k)
        else:
            r = None
        if not r or not isinstance(r, dict) or 'underlying' not in r:
            return {'underlying': {}}
        return r
    except Exception:
        return {'underlying': {}}

_stable_api.IQ_Option.get_digital_underlying_list_data = _safe_get_digital_underlying_list_data

# Instanciar y conectar (ahora con parche aplicado)
q_obj = IQ_Option(EMAIL, PASSWORD)
q_obj.connect()

# reasignar nombre 'iq' usado en el resto del script
iq = q_obj

if not iq.check_connect():
    print("Error conexión")
    exit()

iq.change_balance("PRACTICE")

print("Bot Optimizado 15M iniciado")

# =============================
# MODELOS
# =============================

# horizonte de predicción a largo plazo (velas de TIMEFRAME); configurado en config
from configuraciones.config import HORIZON_CANDLES

scaler = MinMaxScaler()
# modelos para horizonte inmediato
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=3)

# modelos para horizonte 15‑min
rf_model_h15 = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=43)
xgb_model_h15 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=3)

def crear_lstm():
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(LOOKBACK,1)))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

lstm_model = crear_lstm()
# LSTM independiente para 15‑min
lstm_model_h15 = crear_lstm()

last_retrain_time = None

# cargar indicadores dinámicamente
import importlib, pkgutil, os, json

def load_indicator_modules():
    pkg_name = "indicators"
    pkg_path = os.path.join(os.path.dirname(__file__), pkg_name)
    modules = []
    if not os.path.isdir(pkg_path):
        return modules
    for _, name, _ in pkgutil.iter_modules([pkg_path]):
        try:
            mod = importlib.import_module(f"{pkg_name}.{name}")
            if hasattr(mod, "vote") and callable(mod.vote):
                modules.append(mod)
        except Exception as e:
            print(f"Error cargando indicador {name}: {e}")
    print("Indicadores cargados:", [m.__name__.split('.')[-1] for m in modules])
    return modules

indicator_modules = load_indicator_modules()

# instanciar gestor de pesos (IA ligera)
# ahora hay cuatro 'modelos' principales: rf, xgb, lstm y el voto a 15min (h15)
wm_predictors = ["rf", "xgb", "lstm", "h15"] + [m.__name__.split('.')[-1] for m in indicator_modules]
weight_manager = WeightManager(wm_predictors, lr=WEIGHT_LR)

# inicializar logger SQLite
trade_logger = TradeLogger(LOG_DB_PATH)
# win‑confidence validator (TradeFilter)
trade_filter = TradeFilter(model_path=WIN_CONFIDENCE_MODEL_PATH, threshold=WM_CONFIDENCE_THRESHOLD)

# =============================
# FUNCIONES
# =============================

def preparar_datos(cierres):
    df = pd.DataFrame(cierres, columns=["close"])
    df = df.tail(WINDOW_SIZE)

    df["return"] = df["close"].pct_change()
    df["target"] = (df["return"].shift(-1) > 0).astype(int)
    df = df.dropna()

    X_ml = df[["return"]].values
    y_ml = df["target"].values

    scaled = scaler.fit_transform(df["close"].values.reshape(-1,1))

    X_lstm = []
    y_lstm = []

    # ahora también generamos etiquetas a HORIZON_CANDLES velas de distancia (15min)
    y_ml_h15 = []
    X_lstm_h15 = []
    y_lstm_h15 = []

    # para ML los features son idénticos; sólo cambia la etiqueta
    for i in range(len(X_ml)):
        # i corresponde a la fila del DataFrame original de df tras dropna
        # la etiqueta inmediatamente siguiente está en y_ml; la de 15min la calculamos manualmente
        # df está indexado de forma continua tras dropna, podemos obtenerla usando iloc
        target_idx = i + 1 + (HORIZON_CANDLES - 1)
        if target_idx < len(df):
            # comparación simple de cierres
            close_now = df.iloc[i]["close"]
            close_h15 = df.iloc[target_idx]["close"]
            y_ml_h15.append(1 if (close_h15 - close_now) > 0 else 0)
        else:
            y_ml_h15.append(y_ml[i])  # rellenar con etiqueta inmediata para mantener longitud

    for i in range(LOOKBACK, len(scaled)-1):
        X_lstm.append(scaled[i-LOOKBACK:i])
        y_lstm.append(scaled[i+1])
        if i + HORIZON_CANDLES < len(scaled):
            X_lstm_h15.append(scaled[i-LOOKBACK:i])
            # para horizonte 15m usamos el valor escalado a HORIZON_CANDLES pasos
            y_lstm_h15.append(scaled[i+HORIZON_CANDLES])
        else:
            # rellenar con valor inmediato para conservar forma
            X_lstm_h15.append(scaled[i-LOOKBACK:i])
            y_lstm_h15.append(scaled[i+1])

    return (np.array(X_lstm), np.array(y_lstm),
            np.array(X_lstm_h15), np.array(y_lstm_h15),
            X_ml, y_ml, np.array(y_ml_h15))

def entrenar_si_necesario(cierres):
    global last_retrain_time

    now = datetime.datetime.now()

    if last_retrain_time is None or \
       (now - last_retrain_time).total_seconds() > RETRAIN_HOURS * 3600:

        print("Reentrenando modelos...")

        # obtener también datos para horizonte 15m
        X_lstm, y_lstm, X_lstm_h15, y_lstm_h15, X_ml, y_ml, y_ml_h15 = preparar_datos(cierres)

        if len(X_ml) > 100:
            rf_model.fit(X_ml, y_ml)
            xgb_model.fit(X_ml, y_ml)
            # entrenar modelos 15m con misma entrada X_ml
            rf_model_h15.fit(X_ml, y_ml_h15)
            xgb_model_h15.fit(X_ml, y_ml_h15)

        if len(X_lstm) > 100:
            early_stop = EarlyStopping(monitor='loss', patience=2)
            lstm_model.fit(
                X_lstm,
                y_lstm,
                epochs=3,
                batch_size=32,
                verbose=0,
                callbacks=[early_stop]
            )
            # también entrenar LSTM 15m
            lstm_model_h15.fit(
                X_lstm_h15,
                y_lstm_h15,
                epochs=3,
                batch_size=32,
                verbose=0,
                callbacks=[early_stop]
            )

        last_retrain_time = now
        print("Modelos actualizados")
    else:
        print("No es momento de reentrenar")

def predecir(cierres):
    # se evalúan cuatro «IA»: rf, xgb, lstm (inmediato) y un predictor de 15min
    # la operación sólo se abre si al menos tres modelos coinciden y el voto de
    # horizonte 15min (h15) no contradice la dirección final.
    ultimo_precio = cierres[-1]

    retorno = (cierres[-1] - cierres[-2]) / cierres[-2]
    X_input = np.array([[retorno]])

    # probabilidades para RF/XGB inmediato
    try:
        pred_rf_proba = float(rf_model.predict_proba(X_input)[0, 1])
    except Exception:
        pred_rf_proba = 1.0 if rf_model.predict(X_input)[0] == 1 else 0.0
    try:
        pred_xgb_proba = float(xgb_model.predict_proba(X_input)[0, 1])
    except Exception:
        pred_xgb_proba = 1.0 if xgb_model.predict(X_input)[0] == 1 else 0.0

    # probabilidades para RF/XGB 15m
    try:
        pred_rf_proba_h15 = float(rf_model_h15.predict_proba(X_input)[0, 1])
    except Exception:
        pred_rf_proba_h15 = 1.0 if rf_model_h15.predict(X_input)[0] == 1 else 0.0
    try:
        pred_xgb_proba_h15 = float(xgb_model_h15.predict_proba(X_input)[0, 1])
    except Exception:
        pred_xgb_proba_h15 = 1.0 if xgb_model_h15.predict(X_input)[0] == 1 else 0.0

    scaled = scaler.transform(cierres[-WINDOW_SIZE:].reshape(-1,1))
    X_lstm = scaled[-LOOKBACK:].reshape(1,LOOKBACK,1)
    pred_lstm = lstm_model.predict(X_lstm, verbose=0)
    pred_price = scaler.inverse_transform(pred_lstm)[0][0]

    pred_lstm_h15 = lstm_model_h15.predict(X_lstm, verbose=0)
    pred_price_h15 = scaler.inverse_transform(pred_lstm_h15)[0][0]

    model_votes = {}
    votos_call = 0
    votos_put = 0

    # LSTM inmediato
    rel_delta = (pred_price - ultimo_precio) / (ultimo_precio + 1e-12)
    if rel_delta >= LSTM_VOTE_MIN_REL_DELTA:
        lstm_vote = 'call'
        votos_call += 1
    elif rel_delta <= -LSTM_VOTE_MIN_REL_DELTA:
        lstm_vote = 'put'
        votos_put += 1
    else:
        lstm_vote = None
    model_votes['lstm'] = lstm_vote

    # RF
    if pred_rf_proba >= MODEL_VOTE_CONFIDENCE_THRESHOLD:
        rf_vote = 'call'
        votos_call += 1
    elif pred_rf_proba <= (1.0 - MODEL_VOTE_CONFIDENCE_THRESHOLD):
        rf_vote = 'put'
        votos_put += 1
    else:
        rf_vote = None
    model_votes['rf'] = rf_vote

    # XGB
    if pred_xgb_proba >= MODEL_VOTE_CONFIDENCE_THRESHOLD:
        xgb_vote = 'call'
        votos_call += 1
    elif pred_xgb_proba <= (1.0 - MODEL_VOTE_CONFIDENCE_THRESHOLD):
        xgb_vote = 'put'
        votos_put += 1
    else:
        xgb_vote = None
    model_votes['xgb'] = xgb_vote

    # --- 15min horizon vote ------------------------------------------------
    model_votes['h15'] = None
    # LSTM-15
    rel_delta_h15 = (pred_price_h15 - ultimo_precio) / (ultimo_precio + 1e-12)
    if rel_delta_h15 >= LSTM_VOTE_MIN_REL_DELTA:
        vote_h15 = 'call'
        votos_call += 1
    elif rel_delta_h15 <= -LSTM_VOTE_MIN_REL_DELTA:
        vote_h15 = 'put'
        votos_put += 1
    else:
        vote_h15 = None
    model_votes['h15'] = vote_h15

    # RF-15
    if pred_rf_proba_h15 >= MODEL_VOTE_CONFIDENCE_THRESHOLD:
        if model_votes['h15'] is None:
            model_votes['h15'] = 'call'
        votos_call += 1
    elif pred_rf_proba_h15 <= (1.0 - MODEL_VOTE_CONFIDENCE_THRESHOLD):
        if model_votes['h15'] is None:
            model_votes['h15'] = 'put'
        votos_put += 1

    # XGB-15
    if pred_xgb_proba_h15 >= MODEL_VOTE_CONFIDENCE_THRESHOLD:
        if model_votes['h15'] is None:
            model_votes['h15'] = 'call'
        votos_call += 1
    elif pred_xgb_proba_h15 <= (1.0 - MODEL_VOTE_CONFIDENCE_THRESHOLD):
        if model_votes['h15'] is None:
            model_votes['h15'] = 'put'
        votos_put += 1

    # Señal agregada: exigir al menos 3 votos a favor y que el voto de horizonte coincida si existe
    señal = None
    if votos_call >= 3 and votos_call > votos_put:
        señal = 'call'
    elif votos_put >= 3 and votos_put > votos_call:
        señal = 'put'

    # si hay voto de horizon 15 y difiere, invalida la señal
    if señal and model_votes.get('h15') and model_votes.get('h15') != señal:
        señal = None

    return señal, votos_call, votos_put, model_votes

def get_indicator_votes(cierres, modules):
    votes = {}
    counts = {"call": 0, "put": 0}
    for m in modules:
        try:
            v = m.vote(cierres)
        except Exception as e:
            logging.error("Error en indicador %s: %s", m.__name__, e)
            v = None
        votes[m.__name__.split('.')[-1]] = v
        if v in counts:
            counts[v] += 1
    return votes, counts


def safe_get_candles(asset, timeframe, size, to_ts):
    try:
        # comprobar que el asset está registrado por la API antes de solicitar velas
        abiertos = iq.get_all_open_time() or {}
        abiertos_bin = abiertos.get('binary', {}) if isinstance(abiertos, dict) else {}
        if asset not in abiertos_bin:
            logging.warning("Asset %s no registrado por la API — omitiendo solicitud de velas", asset)
            return None

        velas = iq.get_candles(asset, timeframe, size, to_ts)
        if not velas:
            return None
        return velas
    except Exception as e:
        logging.error("get_candles error %s: %s", asset, e)
        # intentar reconectar una vez
        try:
            iq.connect()
        except Exception:
            pass
        return None


def mercados_abiertos(assets, include_otc=False, assets_otc=None):
    try:
        abiertos = iq.get_all_open_time()
        if not abiertos or not isinstance(abiertos, dict):
            logging.error("get_all_open_time devolvió None — intentando reconectar")
            try:
                iq.connect()
            except Exception:
                pass
            abiertos = iq.get_all_open_time() or {}
    except Exception as e:
        logging.error("Error en get_all_open_time(): %s", e)
        try:
            iq.connect()
        except Exception:
            pass
        return []

    abiertos_binarias = abiertos.get("binary", {}) if isinstance(abiertos, dict) else {}
    activos = list(assets)
    if include_otc and assets_otc:
        activos.extend(assets_otc)
    return [a for a in activos if abiertos_binarias.get(a, {}).get("open", False)]


def seleccionar_mejores_mercados(candidates, top_n=3, lookback=VOLATILITY_LOOKBACK):
    """Evalúa cada candidato por volatilidad histórica (std de retornos) y devuelve top_n."""
    scores = []
    for a in candidates:
        velas = safe_get_candles(a, TIMEFRAME, lookback, time.time())
        if not velas or len(velas) < max(10, lookback//2):
            continue
        closes = np.array([v['close'] for v in velas])
        rets = np.diff(closes) / closes[:-1]
        vol = float(np.nanstd(rets))
        scores.append((a, vol))
    if not scores:
        return []
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [s[0] for s in scores[:top_n]]
    logging.info("Mercados evaluados (vol): %s", scores)
    return selected

from ia.money_manager import compute_bet_size, REGIME_WEIGHT_MULTIPLIERS

# =============================
# LOOP PRINCIPAL
# =============================

# =============================
# LOOP PRINCIPAL
# =============================

last_investigation_time = None


def investigate_unknown_trades():
    """Intenta resolver trades con resultado 'unknown'."""
    unknowns = trade_logger.query_trades(result='unknown', limit=1000)
    if not unknowns:
        return 0
    resolved = 0
    for t in unknowns:
        tid = t.get('id')
        platform_id = t.get('platform_id')
        print(f"Investigando trade id={tid} asset={t.get('asset')} platform_id={platform_id}")

        # 1) intentar API
        api_result = None
        if platform_id:
            for _ in range(5):
                try:
                    api_result = iq.check_win_v4(platform_id)
                except Exception:
                    api_result = None
                if api_result is not None:
                    break
                time.sleep(1)

        # parse api_result
        parsed_result = None
        parsed_profit = None
        if api_result is not None:
            if isinstance(api_result, (int, float)):
                parsed_profit = float(api_result)
                parsed_result = 'win' if parsed_profit > 0 else ('loss' if parsed_profit < 0 else 'draw')
            elif isinstance(api_result, bool):
                parsed_result = 'win' if api_result else 'loss'
            elif isinstance(api_result, dict):
                if 'profit' in api_result and api_result.get('profit') is not None:
                    try:
                        parsed_profit = float(api_result.get('profit'))
                        parsed_result = 'win' if parsed_profit > 0 else ('loss' if parsed_profit < 0 else 'draw')
                    except Exception:
                        parsed_result = None
                elif 'win' in api_result:
                    v = api_result.get('win')
                    if isinstance(v, bool):
                        parsed_result = 'win' if v else 'loss'

        if parsed_result:
            trade_logger.update_trade_result(trade_id=tid if tid else None, platform_id=platform_id,
                                             result=parsed_result, profit=parsed_profit,
                                             exit_balance=t.get('exit_balance'),
                                             exit_time=t.get('exit_time') or datetime.datetime.utcnow().isoformat(),
                                             resolved_by='api')
            resolved += 1
            continue

        # 2) intentar balance si está disponible
        entry_bal = t.get('entry_balance')
        exit_bal = t.get('exit_balance')
        if exit_bal is not None and entry_bal is not None:
            try:
                profit = float(exit_bal) - float(entry_bal)
                rlabel = 'win' if profit > 0 else ('loss' if profit < 0 else 'draw')
                trade_logger.update_trade_result(trade_id=tid, result=rlabel, profit=profit,
                                                 exit_balance=exit_bal, exit_time=t.get('exit_time'),
                                                 resolved_by='balance')
                resolved += 1
                continue
            except Exception:
                pass

        # 3) inferir con WeightManager usando model_votes + indicator_votes
        try:
            mv = t.get('model_votes')
            iv = t.get('indicator_votes')
            mv = json.loads(mv) if mv and isinstance(mv, str) else (mv or {})
            iv = json.loads(iv) if iv and isinstance(iv, str) else (iv or {})
            pred_dir, pred_conf = weight_manager.predict(mv, iv)
            trade_dir = t.get('direction')
            if pred_conf >= WM_CONFIDENCE_THRESHOLD:
                # si la predicción coincide con la dirección de la operación, asumimos win
                assumed_result = 'win' if pred_dir == trade_dir else 'loss'
                trade_logger.update_trade_result(trade_id=tid, result=assumed_result, profit=None,
                                                 exit_time=t.get('exit_time') or datetime.datetime.utcnow().isoformat(),
                                                 resolved_by='inferred_wm')
                resolved += 1
                continue
        except Exception as e:
            logging.error('Error inferiendo trade %s: %s', tid, e)

        # 4) política por defecto: marcar como loss (conservador)
        if UNKNOWN_RESOLUTION_POLICY == 'assume_loss':
            trade_logger.update_trade_result(trade_id=tid, result='loss', profit=None,
                                             exit_time=t.get('exit_time') or datetime.datetime.utcnow().isoformat(),
                                             resolved_by='assume_loss')
            resolved += 1
            continue

        # no resuelto -> dejar para revisión manual
        print(f"Trade id={tid} no resuelto automáticamente; dejar como 'unknown'.")

    return resolved


def reconcile_open_trades():
    """Reconciliar trades con status 'open' llamando a la API para cerrarlos si ya finalizaron."""
    opens = trade_logger.query_trades(status='open', limit=1000)
    if not opens:
        return 0
    resolved = 0
    for t in opens:
        tid = t.get('id')
        platform_id = t.get('platform_id')
        asset = t.get('asset')
        print(f"Reconciling open trade id={tid} platform_id={platform_id} asset={asset}")
        if not platform_id:
            continue
        api_result = None
        for _ in range(5):
            try:
                api_result = iq.check_win_v4(platform_id)
            except Exception:
                api_result = None
            if api_result is not None:
                break
            time.sleep(1)
        profit = None
        result_label = None
        if isinstance(api_result, (int, float)):
            profit = float(api_result)
            result_label = 'win' if profit > 0 else ('loss' if profit < 0 else 'draw')
        elif isinstance(api_result, bool):
            result_label = 'win' if api_result else 'loss'
        elif isinstance(api_result, dict):
            if 'profit' in api_result and api_result.get('profit') is not None:
                try:
                    profit = float(api_result.get('profit'))
                    result_label = 'win' if profit > 0 else ('loss' if profit < 0 else 'draw')
                except Exception:
                    pass
            elif 'win' in api_result:
                v = api_result.get('win')
                if isinstance(v, bool):
                    result_label = 'win' if v else 'loss'
        # si obtuvimos resultado -> cerrar trade en DB
        if result_label is not None:
            exit_balance = None
            try:
                exit_balance = iq.get_balance()
            except Exception:
                exit_balance = None
            trade_logger.close_trade(platform_id=str(platform_id), result=result_label, profit=profit, exit_balance=exit_balance, exit_time=datetime.datetime.utcnow().isoformat())
            resolved += 1
            # actualizar weight manager
            model_votes = t.get('model_votes')
            ind_votes = t.get('indicator_votes')
            try:
                mv = json.loads(model_votes) if model_votes and isinstance(model_votes, str) else (model_votes or {})
                iv = json.loads(ind_votes) if ind_votes and isinstance(ind_votes, str) else (ind_votes or {})
                # result_label es 'win'|'loss' — convertir a outcome 'call'|'put' usando la dirección ejecutada
                if result_label in ('win','loss'):
                    trade_dir = t.get('direction')
                    outcome = trade_dir if result_label == 'win' else ('put' if trade_dir == 'call' else 'call')
                    # intentar inferir régimen actual para la actualización del WM (best-effort)
                    try:
                        velas_r = safe_get_candles(asset, TIMEFRAME, VOLATILITY_LOOKBACK, time.time())
                        rinfo = detect_regime(velas_r) if velas_r else {}
                        r = rinfo.get('regime')
                    except Exception:
                        r = None
                    weight_manager.update(mv, iv, outcome, regime=r)
            except Exception as e:
                logging.error('Error updating WM for reconciled trade %s: %s', tid, e)
    return resolved

# control de tiempos para tareas de mantenimieno
last_investigation_time = None
last_reconcile_time = None

# Soporte para ejecución 'una sola vez' vía variable de entorno (útil para investigar/reconciliar manualmente)
if os.environ.get("IA_ACTION") == "investigate":
    try:
        updated = investigate_unknown_trades()
        reconciled = reconcile_open_trades()
        print(f"Ejecución única: investigados={updated} unknowns, reconciliados={reconciled} open trades")
    except Exception as _e:
        print("Error durante ejecución única de investigación/reconciliación:", _e)
    sys.exit(0)

# CLI: export DB + latest weights JSON
if "--export-artifacts" in sys.argv:
    try:
        from ia.exporter import export_db_and_weights
        out = export_db_and_weights()
        print(f"Artifacts exportados: {out}")
    except Exception as _e:
        print("Error exportando artifacts:", _e)
    sys.exit(0)

while True:
    try:
        # run investigation periódica
        now = datetime.datetime.now()
        # investigar unknowns periódicamente
        if (last_investigation_time is None) or ((now - last_investigation_time).total_seconds() > INVESTIGATION_INTERVAL_SECONDS):
            n = investigate_unknown_trades()
            if n:
                print(f"Investigación: {n} trades actualizados desde 'unknown'.")
            last_investigation_time = now
        # reconciliar abiertos periódicamente
        if (last_reconcile_time is None) or ((now - last_reconcile_time).total_seconds() > INVESTIGATION_INTERVAL_SECONDS):
            r = reconcile_open_trades()
            if r:
                print(f"Reconcilación: {r} trades abiertos actualizados.")
            last_reconcile_time = now

        activos = mercados_abiertos(ASSETS, INCLUDE_OTC, ASSETS_OTC)

        if not activos:
            print("No hay mercados abiertos. Esperando...")
            time.sleep(30)
            continue

        # seleccionar solo los mercados con mejores condiciones (volatilidad) entre los abiertos
        candidatos = activos
        seleccionados = seleccionar_mejores_mercados(candidatos, top_n=MAX_ACTIVE_MARKETS, lookback=VOLATILITY_LOOKBACK)

        if not seleccionados:
            print("No se encontraron mercados con datos suficientes. Esperando...")
            time.sleep(30)
            continue

        for ASSET in seleccionados:
            velas = safe_get_candles(ASSET, TIMEFRAME, WINDOW_SIZE, time.time())
            if not velas or len(velas) < (LOOKBACK + 2):
                print(f"{ASSET} -> velas insuficientes o error API → salto")
                time.sleep(1)
                continue

            cierres = np.array([v["close"] for v in velas])

            entrenar_si_necesario(cierres)

            señal, votos_call, votos_put, model_votes = predecir(cierres)
            ind_votes, ind_counts = get_indicator_votes(cierres, indicator_modules)

            # detectar régimen y ajustar tamaño/pesos
            regime_info = detect_regime(velas)
            regime = regime_info.get('regime')
            trend_dir = regime_info.get('trend_dir')  # 'up'|'down'|'flat'

            asset_winrate = trade_logger.get_winrate(asset=ASSET, lookback=WINRATE_LOOKBACK, include_draws=True)
            bet_size = compute_bet_size(asset_winrate, regime=regime)

            # aplicar reducción de tamaño en alta volatilidad si procede
            if regime == 'high_volatility':
                bet_size = max(MIN_BET, round(bet_size * (1.0 - REGIME_HIGH_VOL_SIZE_REDUCTION), 2))

            # predecir con ajustes de pesos según régimen (temporal) o usando pesos aprendidos por régimen
            multipliers = REGIME_WEIGHT_MULTIPLIERS.get(regime)
            if multipliers:
                wm_pred, wm_conf = weight_manager.predict_with_adjustments(model_votes, ind_votes, weight_multipliers=multipliers)
            elif regime:
                wm_pred, wm_conf = weight_manager.predict_regime(model_votes, ind_votes, regime=regime)
            else:
                wm_pred, wm_conf = weight_manager.predict(model_votes, ind_votes)

            # calcular dirección agregada de indicadores (disponible para reglas de 'lateral')
            indicator_direction = None
            if ind_counts.get('call', 0) > ind_counts.get('put', 0):
                indicator_direction = 'call'
            elif ind_counts.get('put', 0) > ind_counts.get('call', 0):
                indicator_direction = 'put'

            # Reglas por régimen (filtrado conservador para mejorar calidad)
            if regime == 'low_volatility' and REGIME_LOW_VOL_SKIP:
                print(f"{ASSET} -> SKIP (low_volatility)")
                time.sleep(1)
                continue

            # si 'strong_trend' solo seguir la dirección de la tendencia
            if regime == 'strong_trend' and señal is not None:
                required = 'call' if trend_dir == 'up' else 'put' if trend_dir == 'down' else None
                if required and señal != required:
                    print(f"{ASSET} -> SKIP (strong_trend) — señal no sigue tendencia {required}")
                    time.sleep(1)
                    continue

            # en 'weak_trend' exigir mayor confirmación (seguir tendencia con más pruebas)
            if regime == 'weak_trend' and señal is not None:
                required = 'call' if trend_dir == 'up' else 'put' if trend_dir == 'down' else None
                if not required or señal != required or wm_conf < WM_HALF_CONFIDENCE_THRESHOLD or (votos_call + votos_put) < 2:
                    print(f"{ASSET} -> SKIP (weak_trend) — requiere mayor confirmación")
                    time.sleep(1)
                    continue

            # en 'lateral' priorizar contratendencia / mean-reversion (exigir indicadores)
            if regime == 'lateral':
                if indicator_direction is None:
                    print(f"{ASSET} -> SKIP (lateral) — sin señal contraria de indicadores")
                    time.sleep(1)
                    continue
                # permitir solo operaciones que vayan en la dirección indicada por indicadores (contrarian)
                if señal is None or señal != indicator_direction:
                    print(f"{ASSET} -> SKIP (lateral) — sólo contratendencia con indicadores")
                    time.sleep(1)
                    continue

            print(f"{ASSET} -> Señal: {señal} | modelos: call={votos_call} put={votos_put} | indicadores: {ind_votes} | wm={wm_conf:.2f} | regime={regime} | winrate={asset_winrate} | bet={bet_size}")

            # Trend veto removed — el indicador 'trend' ya no bloquea operaciones.
            # (Antes: si 'trend' contradecía la señal y su peso excedía TREND_VETO_MIN_WEIGHT, se saltaba el trade.)

            # Resolver conflictos: si modelos e indicadores discrepan, dejar que WeightManager decida
            indicator_direction = None
            if ind_counts.get('call', 0) > ind_counts.get('put', 0):
                indicator_direction = 'call'
            elif ind_counts.get('put', 0) > ind_counts.get('call', 0):
                indicator_direction = 'put'

            # Si hay conflicto y WM tiene suficiente confianza, usar wm_pred como señal
            if indicator_direction and señal and indicator_direction != señal:
                # Trend veto on WM proposals removed — WM can resolve conflicts even if it disagrees with 'trend'.
                # (Antes: la propuesta del WM se saltaba si contradicía 'trend' y el peso de 'trend' era suficiente.)

                if wm_conf >= WM_HALF_CONFIDENCE_THRESHOLD:
                    print(f"{ASSET} -> CONFLICT: models={señal} indicators={indicator_direction} -> WM resolves={wm_pred} (conf={wm_conf:.2f})")
                    señal = wm_pred
                    # recompute model/indicator counts for the new direction
                    modelos_a_favor = votos_call if señal == 'call' else votos_put
                    indicadores_a_favor = ind_counts.get(señal, 0)
                else:
                    print(f"{ASSET} -> CONFLICT and WM_conf too low ({wm_conf:.2f}) -> skip")
                    time.sleep(1)
                    continue

            if señal is None:
                time.sleep(1)
                continue

            # decidir monto: ahora la decisión principal la toma el "win_confidence" (TradeFilter).
            # Si USE_WIN_CONFIDENCE_AS_DECISION == False, se usa la lógica clásica por votos/WM como fallback.
            monto = None
            confirmed = False

            # dirección propuesta por los modelos (señal ya viene de predecir)
            direction = señal
            modelos_a_favor = votos_call if direction == "call" else votos_put
            indicadores_a_favor = ind_counts.get(direction, 0)

            # stake base ajustado por régimen (compute_bet_size ya devuelve MONTO o MONTO*0.7)
            adjusted_stake = bet_size

            # construir features para el TradeFilter (se registran para entrenamiento)
            volatility = float(np.nanstd(np.diff(cierres) / (cierres[:-1] + 1e-9))) if len(cierres) > 1 else 0.0
            price_change_1m = float((cierres[-1] - cierres[-2]) / (cierres[-2] + 1e-9)) if len(cierres) > 1 else 0.0
            decision_features = {
                "asset": ASSET,
                "direction": direction,
                "model_votes_call": votos_call,
                "model_votes_put": votos_put,
                "indicator_votes_call": ind_counts.get('call', 0),
                "indicator_votes_put": ind_counts.get('put', 0),
                "model_votes_total": votos_call + votos_put,
                "model_votes_in_favor": modelos_a_favor,
                "indicator_votes_in_favor": indicadores_a_favor,
                "regime": regime,
                "volatility": volatility,
                "price_change_1m": price_change_1m,
                "asset_winrate": asset_winrate or 0.0,
                "bet_size": adjusted_stake,
                "wm_confidence": wm_conf,
                "minutes_to_expiry": EXPIRATION
            }

            # usar win‑confidence como criterio principal si está activado
            if USE_WIN_CONFIDENCE_AS_DECISION:
                tf_res = trade_filter.predict(decision_features)
                prob = float(tf_res.get('prob', 0.0))
                print(f"{ASSET} -> TradeFilter prob={prob:.2f} (thresholds: full={WM_CONFIDENCE_THRESHOLD}, half={WM_HALF_CONFIDENCE_THRESHOLD})")
                if prob >= WM_CONFIDENCE_THRESHOLD:
                    monto = adjusted_stake
                    confirmed = True
                elif prob >= WM_HALF_CONFIDENCE_THRESHOLD:
                    monto = max(MIN_BET, round(adjusted_stake / 2.0, 2))
                    confirmed = True
                else:
                    confirmed = False
            else:
                # fallback: lógica previa basada en votos + WeightManager
                if modelos_a_favor >= 3:
                    if indicadores_a_favor >= (INDICATOR_CONFIRMATIONS + 1) and wm_pred == direction and wm_conf >= WM_HALF_CONFIDENCE_THRESHOLD:
                        if wm_conf >= WM_CONFIDENCE_THRESHOLD:
                            monto = adjusted_stake
                        else:
                            monto = max(MIN_BET, round(adjusted_stake / 2.0, 2))
                        confirmed = True
                    else:
                        print(f"{ASSET} -> 3 modelos a favor pero indicadores={indicadores_a_favor} o WM no tiene confianza suficiente (wm={wm_conf:.2f}) → salto")
                elif modelos_a_favor == 2:
                    if indicadores_a_favor >= (INDICATOR_CONFIRMATIONS) and wm_pred == direction and wm_conf >= WM_HALF_CONFIDENCE_THRESHOLD:
                        monto = max(MIN_BET, round(adjusted_stake / 2.0, 2))
                        confirmed = True
                    else:
                        print(f"{ASSET} -> 2 modelos a favor pero condiciones no satisfechas (wm={wm_conf:.2f}) → salto")
                else:
                    print(f"{ASSET} -> Menos de 2 votos de modelos ({votos_call} call / {votos_put} put) → salto")

            if not confirmed:
                time.sleep(1)
                continue

            check, id_op = iq.buy(monto, ASSET, señal, EXPIRATION)

            if check:
                print(f"{ASSET} -> Operación abierta | monto={monto} | platform_id={id_op}")

                # registrar operación en DB (estado=open)
                entry_balance = None
                try:
                    entry_balance = iq.get_balance()
                except Exception:
                    entry_balance = None

                db_row = trade_logger.insert_trade(
                    platform_id=str(id_op),
                    asset=ASSET,
                    is_otc=1 if ASSET.endswith("-OTC") else 0,
                    amount=monto,
                    direction=señal,
                    entry_time=datetime.datetime.utcnow().isoformat(),
                    entry_balance=entry_balance,
                    model_votes=model_votes,
                    indicator_votes=ind_votes,
                    wm_confidence=wm_conf,
                    expiration=EXPIRATION,
                    timeframe=TIMEFRAME
                )

                # esperar expiración y obtener resultado (reintentos para evitar None)
                time.sleep(EXPIRATION * 60)

                resultado = None
                for _ in range(6):
                    try:
                        resultado = iq.check_win_v4(id_op)
                    except Exception as e:
                        logging.error("check_win_v4 error for %s: %s", id_op, e)
                        resultado = None
                    if resultado is not None:
                        break
                    time.sleep(1)

                profit = None
                result_label = None

                # interpretar distintos formatos de respuesta
                if isinstance(resultado, (int, float)):
                    profit = float(resultado)
                    result_label = 'win' if profit > 0 else ('loss' if profit < 0 else 'draw')
                elif isinstance(resultado, bool):
                    result_label = 'win' if resultado else 'loss'
                elif isinstance(resultado, str):
                    lr = resultado.lower()
                    if 'win' in lr or 'true' in lr:
                        result_label = 'win'
                    elif 'loss' in lr or 'false' in lr:
                        result_label = 'loss'
                    elif 'draw' in lr or 'equal' in lr:
                        result_label = 'draw'
                elif isinstance(resultado, dict):
                    # prioritizar profit si existe
                    if 'profit' in resultado and resultado.get('profit') is not None:
                        try:
                            profit = float(resultado.get('profit'))
                            result_label = 'win' if profit > 0 else ('loss' if profit < 0 else 'draw')
                        except Exception:
                            pass
                    if result_label is None:
                        if 'win' in resultado:
                            v = resultado.get('win')
                            if isinstance(v, bool):
                                result_label = 'win' if v else 'loss'
                            elif isinstance(v, (int, float)):
                                result_label = 'win' if v > 0 else 'loss'
                        elif 'result' in resultado:
                            r = str(resultado.get('result')).lower()
                            if 'win' in r:
                                result_label = 'win'
                            elif 'loss' in r:
                                result_label = 'loss'
                            elif 'draw' in r:
                                result_label = 'draw'

                # fallback: usar balance para estimar profit cuando no hay información directa
                exit_balance = None
                try:
                    exit_balance = iq.get_balance()
                except Exception:
                    exit_balance = None

                if result_label is None and exit_balance is not None and entry_balance is not None:
                    try:
                        profit = float(exit_balance) - float(entry_balance)
                        result_label = 'win' if profit > 0 else ('loss' if profit < 0 else 'draw')
                    except Exception:
                        pass

                # último recurso: marcar como 'unknown' (se intentó todo)
                if result_label is None:
                    result_label = 'unknown'

                # actualizar DB (cerrar trade)
                trade_logger.close_trade(platform_id=str(id_op),
                                         result=result_label,
                                         profit=profit,
                                         exit_balance=exit_balance,
                                         exit_time=datetime.datetime.utcnow().isoformat())

                # registrar sample para el TradeFilter (no bloqueante)
                try:
                    sample = dict(decision_features) if 'decision_features' in locals() else {}
                    sample.update({
                        'result': result_label,
                        'profit': profit if profit is not None else None,
                        'entry_balance': entry_balance,
                        'exit_balance': exit_balance
                    })
                    trade_filter.append_example(sample, result_label)
                except Exception as _e:
                    logging.error('Error registrando sample para TradeFilter: %s', _e)

                print(f"{ASSET} -> Resultado: {result_label} profit={profit}")

                # interpretar resultado y actualizar gestor de pesos (solo si está determinado)
                if result_label in ('win', 'loss'):
                    win = (result_label == 'win')
                    outcome = señal if win else ('put' if señal == 'call' else 'call')
                    # actualizar WM usando el régimen detectado para este asset (si existe)
                    try:
                        weight_manager.update(model_votes, ind_votes, outcome, regime=regime)
                    except Exception:
                        weight_manager.update(model_votes, ind_votes, outcome)

            time.sleep(1)

        print("Esperando siguiente ronda...")
        time.sleep(60)

    except Exception as e:
        print("Error:", e)
        time.sleep(15)
