import os
import zipfile
from pathlib import Path
from datetime import datetime

from configuraciones.config import DOWNLOADS_DIR, LOG_DB_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _find_db():
    # preferir LOG_DB_PATH si existe
    p = Path(LOG_DB_PATH)
    if p.exists():
        return p
    alt = PROJECT_ROOT / 'ia' / 'trades.db'
    return alt if alt.exists() else None


def _find_latest_weights_json():
    js = [p for p in PROJECT_ROOT.rglob('*.json') if 'weight' in p.name.lower() or 'weights' in p.name.lower()]
    if not js:
        return None
    return max(js, key=lambda p: p.stat().st_mtime)


def export_db_and_weights(dest_dir: str | Path | None = None) -> Path:
    dest = Path(dest_dir or (PROJECT_ROOT / DOWNLOADS_DIR))
    dest.mkdir(parents=True, exist_ok=True)

    db_path = _find_db()
    weights = _find_latest_weights_json()

    if db_path is None and weights is None:
        raise FileNotFoundError('No se encontró ni trades.db ni weights JSON en el proyecto.')

    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out = dest / f'artifacts_{ts}.zip'
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        if db_path:
            z.write(db_path, arcname=db_path.name)
        if weights:
            z.write(weights, arcname=weights.name)
    return out


if __name__ == '__main__':
    print(export_db_and_weights())
