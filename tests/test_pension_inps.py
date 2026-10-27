"""Test del calcolo INPS pro-rata: contributi sulla RAL simulata e rivalutazione per anno di versamento."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pension import (  # noqa: E402
    calcola_pensione_inps_italia_prorata,
    stima_irpef_italia,
    COEFFICIENTI_TRASFORMAZIONE_ITALIA,
)

ANNI = 45


def _inps(anni_futuri, ral, infl=0.0, pil=0.0, anni_pregressi=0):
    inflazione = np.full((ANNI, 1), infl)
    fattore = np.cumprod(1.0 + inflazione, axis=0)
    ral_serie = np.full((ANNI, 1), ral)
    netta, _ = calcola_pensione_inps_italia_prorata(
        anni_italia_totali=anni_pregressi + anni_futuri, anni_ue_totali=anni_pregressi + anni_futuri,
        ral_media_italia=35_000.0, inflazione=inflazione, eta_iniziale=30, pil_reale=pil,
        fattore_infl=fattore, ral_infl_serie=ral_serie, anni_futuri_italia=anni_futuri,
    )
    return float(netta[0]), fattore


def test_contributi_sulla_ral_simulata_senza_rivalutazione():
    netta, fattore = _inps(anni_futuri=20, ral=100_000.0)
    lorda = 20 * 100_000.0 * 0.33 * COEFFICIENTI_TRASFORMAZIONE_ITALIA[67]
    attesa = float(stima_irpef_italia(np.array([lorda]), np.array([fattore[40, 0]]))[0])
    assert abs(netta - attesa) < 0.01


def test_rivalutazione_solo_dal_versamento():
    # un versamento nell'anno t cresce solo dei tassi degli anni successivi: (1+g)^(40 - t); con 10 anni UE la decorrenza è a 71 anni
    g = 0.05
    netta, fattore = _inps(anni_futuri=10, ral=100_000.0, pil=g)
    lorda = 33_000.0 * sum((1.0 + g) ** (40 - t) for t in range(10)) * COEFFICIENTI_TRASFORMAZIONE_ITALIA[71]
    attesa = float(stima_irpef_italia(np.array([lorda]), np.array([fattore[40, 0]]))[0])
    assert abs(netta - attesa) < 0.01


def test_pensione_cresce_con_la_rivalutazione():
    netta_g, _ = _inps(anni_futuri=10, ral=100_000.0, pil=0.03)
    netta_0, _ = _inps(anni_futuri=10, ral=100_000.0, pil=0.0)
    assert netta_g > netta_0


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print("OK", nome)
