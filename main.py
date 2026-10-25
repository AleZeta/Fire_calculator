"""Script principale per eseguire la simulazione FIRE (Germania / Italia) da terminale e generare il Report PDF."""

import argparse
from config.settings import SimulationConfig
from src.market_data import (
    carica_serie_rendimenti,
    genera_matrice_rendimenti_vettorizzata,
    genera_rendimenti_obbligazionari,
    genera_matrici_multiasset_sincrone,
)
from src.inflation import genera_inflazione
from src.simulation import esegui_simulazione_completa, simula_stress_test_storico
from src.pdf_report import genera_report_pdf_fire


def run_pipeline(paese: str = "Germania", anni_accumulo_target: int = 15) -> None:
    config = SimulationConfig(paese=paese)

    print("=" * 68)
    print(f"      SIMULATORE MONTE CARLO - FIRE CALCULATOR ({config.paese.upper()})     ")
    print("=" * 68)
    print(f"Paese & Fiscalità:       {config.paese}")
    print(f"Capitale iniziale:       {config.capitale_iniziale:,.0f} €")
    print(f"RAL iniziale:            {config.ral:,.0f} €")
    print(f"Spese mensili:           {config.spese_mese:,.0f} €")
    print(
        f"Asset Allocation:        Accumulo: {int(config.target_equity_accumulo * 100)}% Eq / "
        f"{int(config.target_bond_accumulo * 100)}% Bd -> FIRE: {int(config.target_equity_fire * 100)}% Eq / "
        f"{int(config.target_bond_fire * 100)}% Bd"
    )
    print(f"Simulazioni:             {config.num_simulazioni:,}")
    print(f"Orizzonte di vita:       {config.eta_iniziale} -> {config.eta_morte} anni")
    print("-" * 68)

    print("[Passo 1/3] Generazione matrici Monte Carlo...")
    if config.usa_campionamento_sincrono:
        m_eq, md_eq, md_bond, inflazione = genera_matrici_multiasset_sincrone(config)
    else:
        serie_rendimenti = carica_serie_rendimenti(config)
        m_eq, md_eq = genera_matrice_rendimenti_vettorizzata(serie_rendimenti, config)
        inflazione = genera_inflazione(config)
        md_bond = genera_rendimenti_obbligazionari(config, inflazione)

    print(f"[Passo 2/3] Simulazione FIRE ({config.paese}) con Outflow Rebalancing e Pensione Pubblica...")
    risultati = esegui_simulazione_completa(config, m_eq, md_eq, md_bond, inflazione)

    print("[Passo 3/3] Generazione Report PDF Esecutivo...")
    anno_sel = min(anni_accumulo_target, config.anni_acc - 1)
    stress_dict = {
        "Dot-Com 2000": simula_stress_test_storico(config, 2000, anno_sel),
        "Crisi 2008": simula_stress_test_storico(config, 2008, anno_sel),
        "Stagflazione 1973": simula_stress_test_storico(config, 1973, anno_sel),
        "Crollo 2022": simula_stress_test_storico(config, 2022, anno_sel),
    }
    pdf_bytes = genera_report_pdf_fire(config, risultati, anno_sel, stress_dict)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = config.output_dir / f"report_esecutivo_fire_{config.paese.lower()}.pdf"
    pdf_path.write_bytes(pdf_bytes)

    print("-" * 68)
    print(f"Simulazione completata! Report salvato in: {pdf_path.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIRE Calculator Monte Carlo Simulator")
    parser.add_argument("--paese", type=str, default="Germania", choices=["Germania", "Italia"], help="Paese di riferimento")
    parser.add_argument("--anni-acc", type=int, default=15, help="Anni di accumulo target per il report")
    args = parser.parse_args()

    run_pipeline(paese=args.paese, anni_accumulo_target=args.anni_acc)
