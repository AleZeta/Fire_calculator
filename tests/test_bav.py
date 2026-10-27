"""Test della Direktversicherung (bAV): limiti di esenzione fiscale e contributiva 2026."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.taxes import calcola_dettaglio_tasse_germania  # noqa: E402

F1 = np.array([1.0])
RAL = np.array([75_000.0])


def _det(bav):
    return calcola_dettaglio_tasse_germania(RAL, F1, tipo_sanita="GKV", bav_annuo=bav)


def test_bav_riduce_tasse_e_contributi():
    base, con = _det(0.0), _det(3_000.0)
    assert con["lohnsteuer"][0] < base["lohnsteuer"][0]
    assert abs((base["rv"][0] - con["rv"][0]) - 3_000 * 0.093) < 0.01  # sotto il 4%: esente anche da SV


def test_bav_limiti_4_e_8_percento_bbg():
    det = _det(8_112.0)
    # contributi: esenti solo i primi 4.056 €
    assert abs(det["rv"][0] - (75_000 - 4_056) * 0.093) < 0.01
    # oltre l'8% della BBG (8.112 €) nessun ulteriore vantaggio fiscale
    assert _det(20_000.0)["lohnsteuer"][0] == det["lohnsteuer"][0]


def test_bav_non_cambia_il_caso_senza_versamento():
    assert abs(_det(0.0)["lohnsteuer"][0] / 12 - 1_162.50) < 0.01


def test_contributo_datore_occupa_parte_del_limite():
    # il datore versa 2.000 €: restano 2.056 € di esenzione contributiva (4% BBG) per il dipendente
    det = calcola_dettaglio_tasse_germania(RAL, F1, tipo_sanita="GKV", bav_annuo=4_056.0, bav_datore_annuo=2_000.0)
    assert abs(det["rv"][0] - (75_000 - 2_056) * 0.093) < 0.01


def test_datore_zero_equivale_a_nessun_contributo():
    a = _det(3_000.0)
    b = calcola_dettaglio_tasse_germania(RAL, F1, tipo_sanita="GKV", bav_annuo=3_000.0, bav_datore_annuo=0.0)
    assert a["lohnsteuer"][0] == b["lohnsteuer"][0] and a["rv"][0] == b["rv"][0]


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print("OK", nome)
