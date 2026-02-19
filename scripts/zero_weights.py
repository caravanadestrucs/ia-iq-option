"""Reset rápido de `ia/weights.json` a valores neutrales.
Uso: python scripts/zero_weights.py
Esto preserva la estructura (keys) pero pone todos los pesos y bias a 0.0.
"""
import json
from pathlib import Path

W = Path(__file__).resolve().parents[1] / 'ia' / 'weights.json'
if not W.exists():
    print('No existe', W)
    raise SystemExit(1)

data = json.loads(W.read_text())
weights = {k: 0.0 for k in data.get('weights', {}).keys()}
regime_weights = {r: {k: 0.0 for k in data.get('weights', {}).keys()} for r in data.get('regime_weights', {}).keys()}
regime_bias = {r: 0.0 for r in data.get('regime_bias', {}).keys()}
new = {'weights': weights, 'bias': 0.0, 'regime_weights': regime_weights, 'regime_bias': regime_bias}
W.write_text(json.dumps(new, indent=2))
print('ia/weights.json reiniciado a valores neutrales')