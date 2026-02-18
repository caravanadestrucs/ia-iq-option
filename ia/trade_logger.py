import sqlite3
import json
import os
import datetime
from typing import Optional, Dict

class TradeLogger:
    def __init__(self, db_path: str = "ia/trades.db"):
        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        sql = """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform_id TEXT,
            asset TEXT,
            is_otc INTEGER,
            amount REAL,
            direction TEXT,
            entry_time TEXT,
            exit_time TEXT,
            entry_balance REAL,
            exit_balance REAL,
            result TEXT,
            profit REAL,
            status TEXT,
            expiration INTEGER,
            timeframe INTEGER,
            model_votes TEXT,
            indicator_votes TEXT,
            wm_confidence REAL,
            resolved_by TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
        conn = self._connect()
        conn.execute(sql)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_platform_id ON trades(platform_id);")
        conn.commit()
        # asegurarse de que la columna resolved_by exista (compatibilidad con versiones previas)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(trades)")
        cols = [r[1] for r in cur.fetchall()]
        if 'resolved_by' not in cols:
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN resolved_by TEXT")
                conn.commit()
            except Exception:
                pass
        conn.close()

    def insert_trade(self, platform_id: str, asset: str, is_otc: bool, amount: float,
                     direction: str, entry_time: Optional[str] = None,
                     entry_balance: Optional[float] = None,
                     model_votes: Optional[Dict] = None,
                     indicator_votes: Optional[Dict] = None,
                     wm_confidence: Optional[float] = None,
                     expiration: Optional[int] = None, timeframe: Optional[int] = None) -> int:
        entry_time = entry_time or datetime.datetime.utcnow().isoformat()
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trades (platform_id, asset, is_otc, amount, direction, entry_time,
                                entry_balance, status, expiration, timeframe, model_votes,
                                indicator_votes, wm_confidence)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            platform_id, asset, 1 if is_otc else 0, amount, direction, entry_time,
            entry_balance, "open", expiration, timeframe,
            json.dumps(model_votes or {}), json.dumps(indicator_votes or {}), wm_confidence
        ))
        conn.commit()
        rowid = cur.lastrowid
        conn.close()
        return rowid

    def close_trade(self, platform_id: Optional[str] = None, trade_id: Optional[int] = None,
                    result: Optional[str] = None, profit: Optional[float] = None,
                    exit_balance: Optional[float] = None, exit_time: Optional[str] = None,
                    resolved_by: Optional[str] = None) -> int:
        if not platform_id and not trade_id:
            raise ValueError("platform_id or trade_id required")
        exit_time = exit_time or datetime.datetime.utcnow().isoformat()
        conn = self._connect()
        cur = conn.cursor()
        if platform_id:
            cur.execute("""
                UPDATE trades SET result=?, profit=?, exit_balance=?, exit_time=?, status='closed', resolved_by=?
                WHERE platform_id=?
            """, (result, profit, exit_balance, exit_time, resolved_by, platform_id))
        else:
            cur.execute("""
                UPDATE trades SET result=?, profit=?, exit_balance=?, exit_time=?, status='closed', resolved_by=?
                WHERE id=?
            """, (result, profit, exit_balance, exit_time, resolved_by, trade_id))
        conn.commit()
        updated = cur.rowcount
        conn.close()
        return updated

    def get_stats(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT
              COUNT(*) as total,
              SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) as wins,
              SUM(CASE WHEN result='loss' THEN 1 ELSE 0 END) as losses,
              SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) as open_trades,
              COALESCE(SUM(profit),0) as pnl
            FROM trades
        """)
        row = cur.fetchone()
        conn.close()
        if not row:
            return {}
        total, wins, losses, open_trades, pnl = row
        winrate = (wins / total) if total else None
        return {"total": total, "wins": wins, "losses": losses, "open": open_trades, "pnl": pnl, "winrate": winrate}

    def get_winrate(self, asset: str = None, lookback: int = 50, include_draws: bool = True):
        """Devuelve winrate sobre las últimas `lookback` trades (por-asset si se pasa `asset`).
        - include_draws=True -> winrate = wins / total
        - include_draws=False -> winrate = wins / (wins+losses)
        Retorna None si no hay trades suficientes.
        """
        sql = "SELECT result FROM trades"
        params = []
        if asset:
            sql += " WHERE asset = ?"
            params.append(asset)
        sql += " ORDER BY entry_time DESC LIMIT ?"
        params.append(lookback)
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        total = len(rows)
        if total == 0:
            return None
        wins = sum(1 for r in rows if r == 'win')
        losses = sum(1 for r in rows if r == 'loss')
        draws = sum(1 for r in rows if r == 'draw')
        if include_draws:
            return wins / total
        denom = wins + losses
        return (wins / denom) if denom > 0 else None

    def list_trades(self, limit: int = 100):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def get_stats_by_asset(self):
        """Devuelve estadísticas agregadas por asset: trades, wins, losses, winrate, pnl, avg_profit, open_trades."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT
                asset,
                COUNT(*) as trades,
                SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result='loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) as open_trades,
                COALESCE(SUM(profit),0) as pnl,
                CASE WHEN COUNT(*)>0 THEN COALESCE(AVG(profit),0) ELSE 0 END as avg_profit
            FROM trades
            GROUP BY asset
            ORDER BY trades DESC
        """)
        rows = []
        for r in cur.fetchall():
            total = r[1]
            wins = r[2] or 0
            losses = r[3] or 0
            winrate = (wins / total) if total else None
            rows.append({
                'asset': r[0],
                'trades': int(total),
                'wins': int(wins),
                'losses': int(losses),
                'open_trades': int(r[4] or 0),
                'pnl': float(r[5] or 0.0),
                'avg_profit': float(r[6] or 0.0),
                'winrate': winrate
            })
        conn.close()
        return rows

    def query_trades(self, asset: str = None, status: str = None, result: str = None,
                     date_from: str = None, date_to: str = None, limit: int = 1000):
        """Consulta filtrada de trades.
        Parámetros opcionales: asset, status, result, date_from, date_to (ISO strings), limit
        """
        conn = self._connect()
        cur = conn.cursor()
        where = []
        params = []
        if asset:
            where.append("asset = ?"); params.append(asset)
        if status:
            where.append("status = ?"); params.append(status)
        if result:
            where.append("result = ?"); params.append(result)
        if date_from:
            where.append("entry_time >= ?"); params.append(date_from)
        if date_to:
            where.append("entry_time <= ?"); params.append(date_to)
        sql = "SELECT * FROM trades"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def update_trade_result(self, trade_id: int = None, platform_id: str = None,
                            result: str = None, profit: float = None,
                            exit_balance: float = None, exit_time: str = None,
                            resolved_by: str = None) -> int:
        """Actualiza resultado/profit/exit_balance de un trade (independiente de su estado)."""
        if not trade_id and not platform_id:
            raise ValueError('trade_id or platform_id required')
        exit_time = exit_time or datetime.datetime.utcnow().isoformat()
        conn = self._connect()
        cur = conn.cursor()
        if trade_id:
            cur.execute("""
                UPDATE trades SET result=?, profit=?, exit_balance=?, exit_time=?, resolved_by=?
                WHERE id=?
            """, (result, profit, exit_balance, exit_time, resolved_by, trade_id))
        else:
            cur.execute("""
                UPDATE trades SET result=?, profit=?, exit_balance=?, exit_time=?, resolved_by=?
                WHERE platform_id=?
            """, (result, profit, exit_balance, exit_time, resolved_by, platform_id))
        conn.commit()
        updated = cur.rowcount
        conn.close()
        return updated

    def delete_unknown_trades(self, limit: int = None) -> int:
        """Elimina trades cuyo resultado es 'unknown'. Si se pasa `limit`, borra hasta ese número (por fecha)."""
        conn = self._connect()
        cur = conn.cursor()
        if limit and isinstance(limit, int) and limit > 0:
            # eliminar mediante subconsulta para limitar
            cur.execute("DELETE FROM trades WHERE id IN (SELECT id FROM trades WHERE result='unknown' ORDER BY entry_time LIMIT ?)", (limit,))
        else:
            cur.execute("DELETE FROM trades WHERE result='unknown'")
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        return deleted

    def delete_trade(self, trade_id: int) -> int:
        """Elimina un trade por su id. Devuelve el número de filas eliminadas (0/1)."""
        if not trade_id:
            return 0
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM trades WHERE id=?", (trade_id,))
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        return deleted
