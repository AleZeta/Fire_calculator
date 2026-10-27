"""Test del modello di mercato: mercati emergenti (periodo comune, mix annuale) e modello normale."""

import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SimulationConfig, DATA_DIR  # noqa: E402
from src.market_data import carica_dataset_sincrono_eur, genera_matrici_multiasset_sincrone  # noqa: E402


def _cartella_dati(con_em: bool) -> Path:
    cartella = Path(tempfile.mkdtemp())
    shutil.copy(DATA_DIR / "historical_triads_eur.csv", cartella / "historical_triads_eur.csv")
    if con_em:  # dati SINTETICI, solo per verificare la meccanica
        anni = np.arange(1988, 2025)
        pd.DataFrame({"Year": anni, "msci_em_eur": np.linspace(-0.2, 0.3, len(anni))}).to_csv(
            cartella / "msci_em_eur.csv", index=False
        )
    return cartella


def test_senza_file_emergenti_errore_chiaro():
    cfg = SimulationConfig(data_dir=_cartella_dati(False), quota_azioni_emergenti=0.2)
    try:
        carica_dataset_sincrono_eur(cfg)
    except FileNotFoundError as e:
        assert "msci_em_eur" in str(e)
    else:
        raise AssertionError("atteso FileNotFoundError")


def test_periodo_comune_e_mix_annuale():
    cfg = SimulationConfig(data_dir=_cartella_dati(True), quota_azioni_emergenti=0.2, nome_dataset_azionario="sp500")
    df = carica_dataset_sincrono_eur(cfg)
    assert df["Year"].min() == 1988 and df["Year"].max() == 2024
    attesa = 0.8 * df["sp500_eur"] + 0.2 * df["msci_em_eur"]
    assert np.allclose(df["azioni_eur"], attesa)
    completo = carica_dataset_sincrono_eur(cfg, includi_em=False)
    assert completo["Year"].min() == 1970  # lo stress test storico usa la serie completa


def test_quota_zero_non_richiede_il_file():
    cfg = SimulationConfig(data_dir=_cartella_dati(False), quota_azioni_emergenti=0.0, num_simulazioni=50)
    m_eq, md_eq, md_bd, infl = genera_matrici_multiasset_sincrone(cfg)
    assert md_eq.shape == (cfg.anni_vita + 1, 50)


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
