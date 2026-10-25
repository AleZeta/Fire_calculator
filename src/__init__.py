"""Package src per simulatore FIRE avanzato (Italia / Germania)."""

from .inflation import genera_inflazione, INFLAZIONE_STORICA_ITALIANA
from .market_data import (
    carica_dataset_sincrono_eur,
    genera_matrici_multiasset_sincrone,
    carica_serie_rendimenti,
    genera_matrice_rendimenti_vettorizzata,
    genera_rendimenti_obbligazionari,
)
from .pension import calcola_pensione_totale_ue
from .taxes import (
    calcola_stipendio_netto_automatico,
    calcola_dettaglio_tasse_germania,
    calcola_dettaglio_tasse_italia,
)
from .simulation import (
    calcola_curva_spese_dinamica,
    calcola_depositi_e_tfr,
    accumula_multi_asset_tax_efficient,
    simula_decumulo_fire_avanzato,
    esegui_simulazione_completa,
    simula_stress_test_storico,
)
from .visualization import (
    format_euro,
    disegna_heatmap_su_asse,
    render_fig_heatmap_successo,
    render_fig_heatmap_wd,
    plot_fan_chart_portafoglio_fire,
)
from .pdf_report import genera_report_pdf_fire

__all__ = [
    "genera_inflazione",
    "INFLAZIONE_STORICA_ITALIANA",
    "carica_dataset_sincrono_eur",
    "genera_matrici_multiasset_sincrone",
    "carica_serie_rendimenti",
    "genera_matrice_rendimenti_vettorizzata",
    "genera_rendimenti_obbligazionari",
    "calcola_pensione_totale_ue",
    "calcola_stipendio_netto_automatico",
    "calcola_dettaglio_tasse_germania",
    "calcola_dettaglio_tasse_italia",
    "calcola_curva_spese_dinamica",
    "calcola_depositi_e_tfr",
    "accumula_multi_asset_tax_efficient",
    "simula_decumulo_fire_avanzato",
    "esegui_simulazione_completa",
    "simula_stress_test_storico",
    "format_euro",
    "disegna_heatmap_su_asse",
    "render_fig_heatmap_successo",
    "render_fig_heatmap_wd",
    "plot_fan_chart_portafoglio_fire",
    "genera_report_pdf_fire",
]
