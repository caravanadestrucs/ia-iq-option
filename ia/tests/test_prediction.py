import numpy as np
import pytest

# monkeypatch helpers
def make_dummy(vote):
    class Dummy:
        def __init__(self, vote):
            self.vote = vote
        def predict_proba(self, X):
            # return probability for class 1
            p = 1.0 if self.vote == 'call' else 0.0 if self.vote == 'put' else 0.5
            return np.array([[1-p, p]])
        def predict(self, X):
            return np.array([1 if self.vote == 'call' else 0])
    return Dummy(vote)


from app import predecir

# helper to generate simple closing array with necessary length

def dummy_closes(length):
    return np.linspace(1, 1 + length * 0.001, length)


def test_predecir_requires_three_votes_and_horizon_alignment(monkeypatch):
    # patch all models to return deterministic votes
    import app as A

    # scenario 1: three calls but horizon vote put -> should block (None)
    monkeypatch.setattr(A, 'rf_model', make_dummy('call'))
    monkeypatch.setattr(A, 'xgb_model', make_dummy('call'))
    monkeypatch.setattr(A, 'lstm_model', make_dummy('call'))
    monkeypatch.setattr(A, 'rf_model_h15', make_dummy('put'))
    monkeypatch.setattr(A, 'xgb_model_h15', make_dummy('put'))
    monkeypatch.setattr(A, 'lstm_model_h15', make_dummy('put'))

    signal, vc, vp, votes = A.predecir(dummy_closes(A.WINDOW_SIZE + 10))
    assert signal is None

    # scenario 2: all four agree call -> should return call
    monkeypatch.setattr(A, 'rf_model_h15', make_dummy('call'))
    monkeypatch.setattr(A, 'xgb_model_h15', make_dummy('call'))
    monkeypatch.setattr(A, 'lstm_model_h15', make_dummy('call'))
    signal2, vc2, vp2, votes2 = A.predecir(dummy_closes(A.WINDOW_SIZE + 10))
    assert signal2 == 'call'
    assert vc2 >= 3

    # scenario 3: two call, two put -> tie -> None
    monkeypatch.setattr(A, 'rf_model', make_dummy('call'))
    monkeypatch.setattr(A, 'xgb_model', make_dummy('call'))
    monkeypatch.setattr(A, 'lstm_model', make_dummy('put'))
    monkeypatch.setattr(A, 'rf_model_h15', make_dummy('put'))
    monkeypatch.setattr(A, 'xgb_model_h15', make_dummy('put'))
    monkeypatch.setattr(A, 'lstm_model_h15', make_dummy('call'))
    signal3, _, _, _ = A.predecir(dummy_closes(A.WINDOW_SIZE + 10))
    assert signal3 is None
