"""Test del periodo di lavoro in Svizzera prima del FIRE (vendite senza imposta, carico azzerato, pensioni)."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SimulationConfig  # noqa: E402
from src.simulation import (  # noqa: E402
    accumula_multi_asset_tax_efficient,
    config_con_anni_svizzeri,
    depositi_con_svizzera,
)

N = 5


def _mercato(anni, rendimento_eq=0.08):
    md_eq = np.full((anni, N), 1.0 + rendimento_eq)
    md_bd = np.full((anni, N), 1.01)
    return md_eq, md_bd


def _cfg(**kw):
    return replace(SimulationConfig(), anni_acc=10, capitale_iniziale=100_000.0, usa_bond_tent=False,
                   target_equity_accumulo=0.8, ter_etf=0.0, bollo=0.0, **kw)


def test_vendite_in_svizzera_senza_imposta():
    md_eq, md_bd = _mercato(10)
    dep = np.zeros((10, N))
    finestra = (5, 8)
    with_tax = _cfg(quota_azioni_fine_svizzera=0.4, aliquota_capital_gain=0.26)
    no_tax = _cfg(quota_azioni_fine_svizzera=0.4, aliquota_capital_gain=0.0)
    a = accumula_multi_asset_tax_efficient(with_tax, dep, md_eq, md_bd, finestra_svizzera=finestra)[0]
    b = accumula_multi_asset_tax_efficient(no_tax, dep, md_eq, md_bd, finestra_svizzera=finestra)[0]
    assert np.allclose(a[:9], b[:9])  # fino a fine periodo svizzero l'aliquota non conta


def test_quota_azionaria_scende_verso_il_target_svizzero():
    md_eq, md_bd = _mercato(10, rendimento_eq=0.0)
    dep = np.zeros((10, N))
    cfg = _cfg(quota_azioni_fine_svizzera=0.4)
    _, eq, bd, _, _ = accumula_multi_asset_tax_efficient(cfg, dep, md_eq, md_bd, finestra_svizzera=(5, 8))
    quota_fine = eq[8] / (eq[8] + bd[8])
    assert np.allclose(quota_fine, 0.4, atol=0.01)


def test_carico_azzerato_a_fine_periodo():
    md_eq, md_bd = _mercato(10)
    dep = np.zeros((10, N))
    cfg = _cfg(quota_azioni_fine_svizzera=0.8, quota_carico_azzerato_svizzera=1.0)
    _, eq, _, carico_eq, _ = accumula_multi_asset_tax_efficient(cfg, dep, md_eq, md_bd, finestra_svizzera=(5, 8))
    assert np.allclose(carico_eq[8], eq[8])
    cfg0 = _cfg(quota_azioni_fine_svizzera=0.8, quota_carico_azzerato_svizzera=0.0)
    _, eq0, _, carico0, _ = accumula_multi_asset_tax_efficient(cfg0, dep, md_eq, md_bd, finestra_svizzera=(5, 8))
    assert np.all(carico0[8] < eq0[8])  # senza azzeramento il carico resta quello originale


def test_depositi_svizzeri_solo_nella_finestra():
    cfg = replace(SimulationConfig(), ral_svizzera_chf=100_000.0, aliquota_effettiva_svizzera=0.2,
                  spese_mese_svizzera_chf=3_000.0, cambio_chf_eur=1.0)
    dep = np.full((6, N), 1_000.0)
    f = np.ones((6, N))
    d = depositi_con_svizzera(cfg, dep, f, 2, 4)
    assert np.allclose(d[2:4], 100_000 * 0.8 - 36_000)
    assert np.allclose(d[:2], 1_000.0) and np.allclose(d[4:], 1_000.0)


def test_anni_svizzeri_si_sommano_ad_ahv_e_secondo_pilastro():
    cfg = replace(SimulationConfig(), anni_pregressi_svizzera=1.0, ral_media_svizzera_chf=60_000.0,
                  capitale_libero_passaggio_chf=4_000.0, ral_svizzera_chf=100_000.0, tasso_altersgutschrift_bvg=0.10)
    c = config_con_anni_svizzeri(cfg, 3)
    assert c.anni_pregressi_svizzera == 4.0
    assert abs(c.ral_media_svizzera_chf - (60_000 + 3 * 100_000) / 4) < 1e-9
    assert abs(c.capitale_libero_passaggio_chf - (4_000 + 3 * (88_200 - 26_460) * 0.10)) < 1e-6  # salario coordinato massimo 88.200
    assert config_con_anni_svizzeri(cfg, 0) is cfg


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print("OK", nome)
