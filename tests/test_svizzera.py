"""Test delle prestazioni svizzere (AHV in proporzione agli anni versati, libero passaggio)."""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SimulationConfig  # noqa: E402
from src.pension import calcola_rendita_ahv_annua_chf, rendite_svizzere_nette_eur  # noqa: E402


def test_senza_dati_nessuna_prestazione():
    assert rendite_svizzere_nette_eur(SimulationConfig()) == (0.0, 0.0)


def test_ahv_estremi_e_proporzione_sugli_anni():
    assert abs(calcola_rendita_ahv_annua_chf(44, 15_120) - 1_260 * 12) < 1e-6   # carriera piena, reddito minimo
    assert abs(calcola_rendita_ahv_annua_chf(44, 120_000) - 2_520 * 12) < 1e-6  # reddito sopra il massimo
    assert abs(calcola_rendita_ahv_annua_chf(22, 90_720) - 2_520 * 6) < 1e-6    # metà degli anni, metà rendita
    assert calcola_rendita_ahv_annua_chf(0, 80_000) == 0.0


def test_conversione_cambio_imposta_e_interesse():
    cfg = replace(SimulationConfig(), anni_pregressi_svizzera=44.0, ral_media_svizzera_chf=90_720.0, cambio_chf_eur=1.1,
                  aliquota_imposta_pensione_svizzera=0.2)
    ahv, bvg = rendite_svizzere_nette_eur(cfg)
    assert abs(ahv - 2_520 * 12 * 1.1 * 0.8) < 1e-6 and bvg == 0.0
    cfg = replace(SimulationConfig(), capitale_libero_passaggio_chf=10_000.0, eta_iniziale=30, eta_sblocco_libero_passaggio=60,
                  tasso_libero_passaggio=0.0, coeff_rendita_libero_passaggio=0.04, aliquota_imposta_pensione_svizzera=0.0)
    assert abs(rendite_svizzere_nette_eur(cfg)[1] - 400.0) < 1e-9


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print("OK", nome)
