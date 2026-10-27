"""Test del modello di mercato normale (media, volatilità e correlazioni storiche)."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SimulationConfig  # noqa: E402
from src.market_data import carica_dataset_sincrono_eur, genera_matrici_multiasset_sincrone  # noqa: E402


def test_modello_normale_riproduce_media_vol_e_correlazione():
    cfg = SimulationConfig(modello_mercato="normale", num_simulazioni=20_000, nome_dataset_azionario="sp500")
    np.random.seed(0)
    _, md_eq, md_bd, infl = genera_matrici_multiasset_sincrone(cfg)
    df = carica_dataset_sincrono_eur(cfg)
    storico = df[["sp500_eur", "bund_10y_tr_eur", "inflation_germany"]].to_numpy()
    campione = np.stack([(md_eq[1:] - 1).ravel(), (md_bd[1:] - 1).ravel(), infl[1:].ravel()], axis=1)
    assert np.allclose(campione.mean(axis=0)[:2], storico.mean(axis=0)[:2], atol=0.005)
    assert abs(campione[:, 0].std() - storico[:, 0].std()) < 0.01
    corr_attesa = np.corrcoef(storico[:, 0], storico[:, 2])[0, 1]
    assert abs(np.corrcoef(campione[:, 0], campione[:, 2])[0, 1] - corr_attesa) < 0.03


def test_modello_normale_a_forma_corretta():
    cfg = replace(SimulationConfig(), modello_mercato="normale", num_simulazioni=10)
    m_eq, md_eq, md_bd, infl = genera_matrici_multiasset_sincrone(cfg)
    assert m_eq.shape == md_eq.shape == md_bd.shape == infl.shape == (cfg.anni_vita + 1, 10)
    assert np.all(md_eq[0] == 1.0) and np.all(infl[0] == 0.0)


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print("OK", nome)
