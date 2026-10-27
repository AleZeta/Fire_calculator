"""Calibrazione del motore di decumulo sulla regola del 4% (30 anni, rendimenti normali 10% / vol 15%, inflazione 3%).

Il motore, con tasse, pensioni, smile, guardrail e buffer disattivati, viene confrontato con un'implementazione di
riferimento indipendente sugli stessi rendimenti. Nota: il motore espone il capitale all'inizio di ogni età,
quindi 30 prelievi richiedono 31 righe (eta_morte = eta_pensione + 31).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SimulationConfig  # noqa: E402
from src.simulation import calcola_curva_spese_dinamica, simula_decumulo_fire_avanzato  # noqa: E402

MU, VOL, INFL, SWR, ANNI, CAPITALE = 0.10, 0.15, 0.03, 0.04, 30, 1_000_000.0


def successo_motore_e_riferimento(quota_azioni: float, n: int = 20_000, seed: int = 1):
    rng = np.random.default_rng(seed)
    cfg = SimulationConfig(
        paese="Italia", num_simulazioni=n, eta_iniziale=65, eta_morte=65 + ANNI + 1, anni_acc=1,
        capitale_iniziale=CAPITALE, spese_mese=CAPITALE * SWR / 12, ter_etf=0.0, bollo=0.0,
        aliquota_capital_gain=0.0, aliquota_capital_gain_bond=0.0, franchigia_capital_gain_annua=0.0,
        usa_curva_spese_smile=False, usa_guardrails_spesa=False, abilita_pensione_pubblica=False,
        anni_buffer=0, usa_bond_tent=False, target_equity_fire=quota_azioni, target_bond_fire=1.0 - quota_azioni,
    )
    righe = ANNI + 2
    md_eq = np.ones((righe, n))
    md_eq[1:] = 1.0 + rng.normal(MU, VOL, (righe - 1, n))
    md_bd = np.ones((righe, n))
    md_bd[1:] = 1.0 + INFL + 0.015  # obbligazioni: 1,5% reale, senza volatilità
    inflazione = np.full((righe, n), INFL)
    inflazione[0] = 0.0
    spese = calcola_curva_spese_dinamica(cfg, inflazione)
    nessuna_pensione = {"pensione_netta_grv": np.zeros(n), "pensione_netta_inps": np.zeros(n),
                        "eta_accesso_grv": ANNI, "eta_accesso_inps": ANNI}
    eq0, bd0 = np.full(n, CAPITALE * quota_azioni), np.full(n, CAPITALE * (1.0 - quota_azioni))
    capitale, _ = simula_decumulo_fire_avanzato(
        0, eq0, bd0, eq0.copy(), bd0.copy(), spese, nessuna_pensione,
        np.cumprod(md_eq, axis=0), md_eq, md_bd, inflazione, cfg,
    )
    successo_motore = float(np.mean(capitale[ANNI] > 0))

    eq, bd = eq0.copy(), bd0.copy()
    for t in range(ANNI):
        tot = eq + bd
        tot = tot - np.minimum(CAPITALE * SWR * (1.0 + INFL) ** t, tot)
        eq, bd = tot * quota_azioni * md_eq[t + 1], tot * (1.0 - quota_azioni) * md_bd[t + 1]
    successo_riferimento = float(np.mean(eq + bd > 1e-6))
    return successo_motore, successo_riferimento


def test_100_azioni_coincide_con_il_riferimento():
    motore, rif = successo_motore_e_riferimento(1.0)
    assert abs(motore - rif) < 0.005
    assert 0.88 < motore < 0.93  # in linea con ~91,5% atteso per 10% / 15% / 30 anni


def test_con_obbligazioni_entro_1_5_punti_dal_riferimento():
    for quota in (0.75, 0.5):
        motore, rif = successo_motore_e_riferimento(quota)
        assert abs(motore - rif) < 0.015, (quota, motore, rif)


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print("OK", nome)
