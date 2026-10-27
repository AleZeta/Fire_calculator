"""Test di calibrazione della fiscalità tedesca 2026 (classe 1, senza figli, senza imposta ecclesiastica)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.taxes import calcola_dettaglio_tasse_germania, einkommensteuer_32a  # noqa: E402


def test_calibrazione_stipendio_6250_mensili():
    det = calcola_dettaglio_tasse_germania(
        ral=np.array([6_250.0 * 12]), fattore_inflazione=np.array([1.0]), tipo_sanita="GKV"
    )
    mensile = {k: float(v[0]) / 12.0 for k, v in det.items()}
    assert abs(mensile["lohnsteuer"] - 1_162.50) < 0.01
    assert abs(mensile["rv"] - 581.25) < 0.01
    assert abs(mensile["av"] - 81.25) < 0.01
    assert abs(mensile["quota_kv"] - 502.49) < 0.01
    assert abs(mensile["quota_pv"] - 139.50) < 0.01
    assert mensile["soli"] == 0.0


def test_tetti_contributivi_2026():
    det = calcola_dettaglio_tasse_germania(
        ral=np.array([200_000.0]), fattore_inflazione=np.array([1.0]), tipo_sanita="GKV"
    )
    assert abs(det["rv"][0] - 101_400 * 0.093) < 0.01
    assert abs(det["quota_kv"][0] - 69_750 * 0.08645) < 0.01


def test_pv_con_figli_senza_supplemento():
    det = calcola_dettaglio_tasse_germania(
        ral=np.array([60_000.0]), fattore_inflazione=np.array([1.0]), tipo_sanita="GKV", ha_figli=True
    )
    assert abs(det["quota_pv"][0] - 60_000 * 0.018) < 0.01


def test_tariffa_32a_punti_noti():
    assert einkommensteuer_32a(np.array([12_348.0]))[0] == 0.0
    assert einkommensteuer_32a(np.array([100_000.0]))[0] == 30_864.0
    assert einkommensteuer_32a(np.array([50_000.0]))[0] < einkommensteuer_32a(np.array([60_000.0]))[0]


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print("OK", nome)
