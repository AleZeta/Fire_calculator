"""Modulo per il calcolo realistico e progressivo delle imposte e dei contributi sul salario (Germania e Italia)."""

from typing import Union, Dict
import numpy as np


def calcola_dettaglio_tasse_germania(
    ral: np.ndarray,
    fattore_inflazione: np.ndarray,
    tipo_sanita: str = "PKV",
    costo_mensile_pkv_base: float = 400.0,
    anno: Union[int, np.ndarray] = 0,
    inflazione_medica_extra: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Restituisce il dettaglio disaggregato (vettorizzato 1D o 2D) delle singole voci fiscali e sanitarie in Germania:
    1. Contributi sociali dipendente (Sozialabgaben): RV (9.3%) + AV (1.3%), KV/PV o premio PKV.
    2. Imposta progressiva sul reddito (Lohnsteuer) + Solidaritätszuschlag con Milderungszone (§ 4 SolzG).
    """
    tetto_rv_av = 90_600.0 * fattore_inflazione
    tetto_kv_pv = 62_100.0 * fattore_inflazione

    ral_capped_rv = np.minimum(ral, tetto_rv_av)
    quota_rv = ral_capped_rv * 0.093
    quota_av = ral_capped_rv * 0.013
    contributi_previdenziali = quota_rv + quota_av

    if tipo_sanita.upper() == "PKV":
        fattore_pkv = fattore_inflazione * ((1.0 + inflazione_medica_extra) ** anno)
        costo_sanita_lavoratore = costo_mensile_pkv_base * 12.0 * fattore_pkv
        quota_sanitaria_deducibile = costo_sanita_lavoratore * 0.82
        quota_kv = costo_sanita_lavoratore
        quota_pv = np.zeros_like(quota_kv)
    else:
        ral_capped_kv = np.minimum(ral, tetto_kv_pv)
        quota_kv = ral_capped_kv * 0.0815
        quota_pv = ral_capped_kv * 0.022
        costo_sanita_lavoratore = quota_kv + quota_pv
        quota_sanitaria_deducibile = costo_sanita_lavoratore

    imponibile = np.maximum(0.0, ral - contributi_previdenziali - quota_sanitaria_deducibile)
    no_tax_area = 11_784.0 * fattore_inflazione
    reddito_tassabile = np.maximum(0.0, imponibile - no_tax_area)

    soglia_top = 66_760.0 * fattore_inflazione
    parte_media = np.clip(reddito_tassabile, 0.0, soglia_top)
    parte_alta = np.maximum(0.0, reddito_tassabile - soglia_top)
    imposta_base = (parte_media * 0.26) + (parte_alta * 0.42)

    freigrenze_soli = 18_130.0 * fattore_inflazione
    eccedenza_soli = np.maximum(0.0, imposta_base - freigrenze_soli)
    soli = np.minimum(imposta_base * 0.055, eccedenza_soli * 0.119)

    totale_trattenute = contributi_previdenziali + costo_sanita_lavoratore + imposta_base + soli

    return {
        "sanita": costo_sanita_lavoratore,
        "quota_kv": quota_kv,
        "quota_pv": quota_pv,
        "rv": quota_rv,
        "av": quota_av,
        "lohnsteuer": imposta_base,
        "soli": soli,
        "totale": totale_trattenute
    }


def calcola_netto_salario_germania(
    ral: np.ndarray,
    fattore_inflazione: np.ndarray,
    tipo_sanita: str = "PKV",
    costo_mensile_pkv_base: float = 400.0,
    anno: Union[int, np.ndarray] = 0,
    inflazione_medica_extra: float = 0.01
) -> np.ndarray:
    """Calcola lo stipendio netto annuo in Germania (Steuerklasse 1, dipendente) con PKV o GKV."""
    det = calcola_dettaglio_tasse_germania(
        ral=ral,
        fattore_inflazione=fattore_inflazione,
        tipo_sanita=tipo_sanita,
        costo_mensile_pkv_base=costo_mensile_pkv_base,
        anno=anno,
        inflazione_medica_extra=inflazione_medica_extra
    )
    return np.maximum(0.0, ral - det["totale"])


def calcola_dettaglio_tasse_italia(
    ral: np.ndarray,
    fattore_inflazione: np.ndarray
) -> Dict[str, np.ndarray]:
    """Restituisce il dettaglio disaggregato (vettorizzato 1D o 2D) delle singole voci fiscali e contributive in Italia."""
    contributi = ral * 0.0919
    imponibile_irpef = np.maximum(0.0, ral - contributi)

    s1 = 28_000.0 * fattore_inflazione
    s2 = 50_000.0 * fattore_inflazione
    q1 = np.clip(imponibile_irpef, 0.0, s1)
    q2 = np.clip(imponibile_irpef - s1, 0.0, s2 - s1)
    q3 = np.maximum(0.0, imponibile_irpef - s2)
    irpef_nazionale = (q1 * 0.23) + (q2 * 0.35) + (q3 * 0.43)
    addizionali = imponibile_irpef * 0.02

    soglia_det = 15_000.0 * fattore_inflazione
    max_det = 1_910.0 * fattore_inflazione
    detrazione = np.where(
        ral <= soglia_det,
        max_det,
        np.clip(max_det - (ral - soglia_det) * 0.05, 0.0, max_det)
    )
    imposta_netta = np.maximum(0.0, irpef_nazionale + addizionali - detrazione)

    return {
        "ivs": contributi,
        "irpef": imposta_netta,
        "addizionali": addizionali,
        "sanita": np.zeros_like(contributi),
        "totale": contributi + imposta_netta
    }


def calcola_netto_salario_italia(ral: np.ndarray, fattore_inflazione: np.ndarray) -> np.ndarray:
    """Calcola lo stipendio netto annuo in Italia (lavoratore dipendente)."""
    det = calcola_dettaglio_tasse_italia(ral, fattore_inflazione)
    return np.maximum(0.0, ral - det["totale"])


def calcola_stipendio_netto_automatico(
    ral: np.ndarray,
    paese: str,
    fattore_inflazione: np.ndarray,
    tipo_sanita: str = "PKV",
    costo_mensile_pkv: float = 400.0,
    anno: Union[int, np.ndarray] = 0,
    inflazione_medica_extra: float = 0.01
) -> np.ndarray:
    """Selettore unificato per il calcolo progressivo del salario netto (supporta array 1D e 2D)."""
    if paese.lower() in ("germania", "de", "germany"):
        return calcola_netto_salario_germania(
            ral=ral,
            fattore_inflazione=fattore_inflazione,
            tipo_sanita=tipo_sanita,
            costo_mensile_pkv_base=costo_mensile_pkv,
            anno=anno,
            inflazione_medica_extra=inflazione_medica_extra
        )
    return calcola_netto_salario_italia(ral, fattore_inflazione)
