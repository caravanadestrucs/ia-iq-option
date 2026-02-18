
from flask import Flask, render_template, request, Response, url_for, redirect, flash
import os
from ia.trade_logger import TradeLogger
from configuraciones.config import LOG_DB_PATH, WINRATE_LOOKBACK, TIMEFRAME, VOLATILITY_LOOKBACK, EMAIL, PASSWORD
from ia.regime_detector import detect_regime
from ia.money_manager import compute_bet_size
import csv, io, json, time

# helper: intentar obtener velas via IQ_Option (best-effort)
try:
    from iqoptionapi.stable_api import IQ_Option
except Exception:
    IQ_Option = None

def _fetch_candles_for_asset(asset, timeframe, size=200):
    """Intenta recuperar velas usando IQ_Option; devuelve lista de velas o None."""
    if IQ_Option is None:
        return None
    try:
        q = IQ_Option(EMAIL, PASSWORD)
        q.connect()
        if not q.check_connect():
            return None
        velas = q.get_candles(asset, timeframe, size, time.time())
        try:
            q.close()
        except Exception:
            pass
        return velas
    except Exception:
        return None

app = Flask(__name__, template_folder="templates")
app.secret_key = 'dev-secret'  # necesario para flash (cámbialo en producción)
logger = TradeLogger(LOG_DB_PATH)

@app.route('/')
def index():
    filters = request.args.to_dict()
    trades = logger.query_trades(
        asset=filters.get('asset'),
        status=filters.get('status'),
        result=filters.get('result'),
        date_from=filters.get('from'),
        date_to=filters.get('to'),
        limit=int(filters.get('limit', 200))
    )
    stats = logger.get_stats()
    assets = sorted({t['asset'] for t in logger.list_trades(1000)})
    unknown_exists = bool(logger.query_trades(result='unknown', limit=1))
    return render_template('index.html', trades=trades, stats=stats, assets=assets, filters=filters, unknown_exists=unknown_exists)

@app.route('/export')
def export_csv():
    filters = request.args.to_dict()
    trades = logger.query_trades(
        asset=filters.get('asset'),
        status=filters.get('status'),
        result=filters.get('result'),
        date_from=filters.get('from'),
        date_to=filters.get('to'),
        limit=10000
    )
    si = io.StringIO()
    writer = csv.writer(si)
    if trades:
        writer.writerow(trades[0].keys())
        for t in trades:
            row = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in t.values()]
            writer.writerow(row)
    return Response(si.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=trades_export.csv'})


@app.route('/delete_unknowns', methods=['POST'])
def delete_unknowns():
    # elimina trades cuyo result = 'unknown'
    deleted = logger.delete_unknown_trades()
    flash(f"{deleted} trades 'unknown' eliminados.")
    return redirect(url_for('index'))


@app.route('/delete_trade', methods=['POST'])
def delete_trade():
    trade_id = request.form.get('trade_id')
    try:
        tid = int(trade_id)
    except Exception:
        flash('ID inválido')
        return redirect(url_for('index'))
    deleted = logger.delete_trade(tid)
    if deleted:
        flash(f'Trade {tid} eliminado')
    else:
        flash(f'Trade {tid} no encontrado')
    return redirect(url_for('index'))

@app.route('/stats')
def stats():
    return logger.get_stats()


@app.route('/statistics')
def statistics():
    overall = logger.get_stats()
    by_asset = logger.get_stats_by_asset()
    assets = sorted({t['asset'] for t in logger.list_trades(1000)})

    # enriquecer por-asset con winrate (últimas N) y suggested bet
    assets_info = []
    for a in assets:
        wr = logger.get_winrate(asset=a, lookback=WINRATE_LOOKBACK, include_draws=True)
        bet = compute_bet_size(wr, regime=None)
        assets_info.append({'asset': a, 'winrate': wr, 'suggested_bet': bet, 'regime': 'n/a'})

    return render_template('statistics.html', overall=overall, by_asset=by_asset, assets=assets, assets_info=assets_info)


@app.route('/investigate')
def investigate_endpoint():
    # intenta inferir trades marcados como 'unknown' usando el WeightManager y política de config
    import json
    from ia.weight_manager import WeightManager
    from configuraciones.config import WM_CONFIDENCE_THRESHOLD, UNKNOWN_RESOLUTION_POLICY, ENABLED_INDICATORS, WEIGHT_LR

    wm_preds = ["rf", "xgb", "lstm"] + ENABLED_INDICATORS
    wm = WeightManager(wm_preds, lr=WEIGHT_LR)

    unknowns = logger.query_trades(result='unknown', limit=1000)
    updated = []
    for t in unknowns:
        try:
            mv = t.get('model_votes')
            iv = t.get('indicator_votes')
            mv = json.loads(mv) if mv and isinstance(mv, str) else (mv or {})
            iv = json.loads(iv) if iv and isinstance(iv, str) else (iv or {})
            pred_dir, pred_conf = wm.predict(mv, iv)
            trade_dir = t.get('direction')
            if pred_conf >= WM_CONFIDENCE_THRESHOLD:
                assumed_result = 'win' if pred_dir == trade_dir else 'loss'
                logger.update_trade_result(trade_id=t['id'], result=assumed_result, profit=None, resolved_by='inferred_wm')
                updated.append({'id': t['id'], 'asset': t['asset'], 'result': assumed_result, 'conf': pred_conf})
            elif UNKNOWN_RESOLUTION_POLICY == 'assume_loss':
                logger.update_trade_result(trade_id=t['id'], result='loss', profit=None, resolved_by='assume_loss')
                updated.append({'id': t['id'], 'asset': t['asset'], 'result': 'loss', 'conf': pred_conf})
        except Exception as e:
            continue
    unknowns = logger.query_trades(result='open', limit=1000)
    updated = []
    for t in unknowns:
        try:
            mv = t.get('model_votes')
            iv = t.get('indicator_votes')
            mv = json.loads(mv) if mv and isinstance(mv, str) else (mv or {})
            iv = json.loads(iv) if iv and isinstance(iv, str) else (iv or {})
            pred_dir, pred_conf = wm.predict(mv, iv)
            trade_dir = t.get('direction')
            if pred_conf >= WM_CONFIDENCE_THRESHOLD:
                assumed_result = 'win' if pred_dir == trade_dir else 'loss'
                logger.update_trade_result(trade_id=t['id'], result=assumed_result, profit=None, resolved_by='inferred_wm')
                updated.append({'id': t['id'], 'asset': t['asset'], 'result': assumed_result, 'conf': pred_conf})
            elif UNKNOWN_RESOLUTION_POLICY == 'assume_loss':
                logger.update_trade_result(trade_id=t['id'], result='loss', profit=None, resolved_by='assume_loss')
                updated.append({'id': t['id'], 'asset': t['asset'], 'result': 'loss', 'conf': pred_conf})
        except Exception as e:
            continue
    return {'updated': updated, 'count': len(updated)}


@app.route('/regime')
def regime_endpoint():
    """Devuelve régimen detectado para un asset (best-effort)."""
    asset = request.args.get('asset')
    if not asset:
        return {'error': 'asset required'}, 400
    velas = _fetch_candles_for_asset(asset, TIMEFRAME, size=VOLATILITY_LOOKBACK)
    if not velas:
        return {'error': 'no data / unable to fetch candles'}, 500
    info = detect_regime(velas)
    return info


if __name__ == '__main__':
    # Ejecutar desde la raíz del proyecto: python server.py
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5001))
    app.run(host=host, port=port, debug=False)
