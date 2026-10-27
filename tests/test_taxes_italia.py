"""Test della fiscalità italiana 2026 (IRPEF 23/33/43% e detrazione lavoro dipendente)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.taxes import detrazione_lavoro_dipendente, calcola_dettaglio_tasse_italia  # noqa: E402

F1 = np.array([1.0])


def _det(r):
    return float(detrazione_lavoro_dipendente(np.array([r]), F1)[0])


def test_fasce_detrazione():
    assert _det(10_000) == 1_955.0
    assert _det(15_000) == 1_955.0
    assert abs(_det(28_000) - 1_910.0) < 1e-9
    assert abs(_det(39_000) - 1_910.0 * 11_000 / 22_000) < 1e-9
    assert _det(50_000) == 0.0 and _det(80_000) == 0.0


def test_secondo_scaglione_33():
    ral = np.array([60_000.0])
    det = calcola_dettaglio_tasse_italia(ral, F1)
    imponibile = 60_000 * (1 - 0.0919)
    irpef_lorda = 28_000 * 0.23 + 22_000 * 0.33 + (imponibile - 50_000) * 0.43
    assert abs(det["irpef"][0] - (irpef_lorda + imponibile * 0.02)) < 0.01


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print("OK", nome)
