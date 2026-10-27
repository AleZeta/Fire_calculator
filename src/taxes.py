"""Modulo per il calcolo realistico e progressivo delle imposte e dei contributi sul salario (Germania e Italia)."""

from typing import Union, Dict
import numpy as np

from config.settings import (
    TETTO_RV_AV_2026,
    TETTO_KV_PV_2026,
    QUOTA_BAV_ESENTE_TASSE,
    QUOTA_BAV_ESENTE_CONTRIBUTI,
)


# Parametri fiscali Germania 2026 (valori nominali odierni, indicizzati con il fattore di inflazione)
GRUNDFREIBETRAG_2026 = 12_348.0
WERBUNGSKOSTEN_PAUSCHALE_2026 = 1_230.0
SONDERAUSGABEN_PAUSCHALE_2026 = 36.0
SOLI_FREIGRENZE_2026 = 20_350.0
ALIQUOTA_RV = 0.093
ALIQUOTA_AV = 0.013
# Nella Vorsorgepauschale la quota KV esclude lo 0,3% (metà del 0,6% per il Krankengeld)
ESCLUSIONE_KRANKENGELD_KV = 0.003


def einkommensteuer_32a(zve: np.ndarray) -> np.ndarray:
    """Imposta sul reddito (Grundtarif § 32a EStG 2026) su un reddito imponibile annuo già deflazionato; reddito e imposta sono arrotondati per difetto all'euro."""
    x = np.floor(np.maximum(zve, 0.0))
    y = (np.clip(x, 12_348.0, 17_799.0) - 12_348.0) / 10_000.0
    z = (np.clip(x, 17_799.0, 69_878.0) - 17_799.0) / 10_000.0
    zona1 = (914.51 * y + 1_400.0) * y
    zona2 = (173.10 * z + 2_397.0) * z + 1_034.87
    zona3 = 0.42 * x - 11_135.63
    zona4 = 0.45 * x - 19_470.38
    imposta = np.where(
        x <= 12_348.0, 0.0,
        np.where(x <= 17_799.0, zona1,
                 np.where(x <= 69_878.0, zona2,
                          np.where(x <= 277_825.0, zona3, zona4)))
    )
    return np.floor(imposta)  # § 32a: importo arrotondato per difetto all'euro intero


def calcola_dettaglio_tasse_germania(
    ral: np.ndarray,
    fattore_inflazione: np.ndarray,
    tipo_sanita: str = "PKV",
    costo_mensile_pkv_base: float = 400.0,
    anno: Union[int, np.ndarray] = 0,
    inflazione_medica_extra: float = 0.01,
    aliquota_kv: float = 0.08645,
    aliquota_pv: float = 0.018,
    supplemento_pv_senza_figli: float = 0.006,
    ha_figli: bool = False,
    bav_annuo: Union[float, np.ndarray] = 0.0,
    bav_datore_annuo: Union[float, np.ndarray] = 0.0,
    crescita_reale_tetti: float = 0.0
) -> Dict[str, np.ndarray]:
    """
    Restituisce il dettaglio disaggregato (vettorizzato 1D o 2D) delle singole voci fiscali e sanitarie in Germania
    (Steuerklasse 1, senza imposta ecclesiastica), parametri 2026:
    1. Contributi dipendente: RV 9.3% + AV 1.3% (tetto 101.400 €), KV/PV su tetto 69.750 €.
    2. Lohnsteuer con tariffa progressiva § 32a EStG su imponibile = RAL - Pauschbeträge - Vorsorgepauschale.
    3. Solidaritätszuschlag con Milderungszone (§ 4 SolzG).
    Con Direktversicherung (bav_annuo, Entgeltumwandlung) l'imponibile fiscale scende fino all'8% della BBG e
    quello contributivo fino al 4%; il versamento non rientra in "totale" (è risparmio, non trattenuta).
    I limiti valgono sul totale dei versamenti: il contributo del datore (bav_datore_annuo) ne occupa una parte.
    """
    # I tetti contributivi seguono i salari: inflazione + crescita reale
    fattore_tetti = fattore_inflazione * ((1.0 + crescita_reale_tetti) ** anno)
    tetto_rv_av = TETTO_RV_AV_2026 * fattore_tetti
    tetto_kv_pv = TETTO_KV_PV_2026 * fattore_tetti

    bav = np.asarray(bav_annuo, dtype=float) * np.ones_like(ral)
    bav_datore = np.asarray(bav_datore_annuo, dtype=float)
    bav_esente_tasse = np.minimum(bav, np.maximum(0.0, QUOTA_BAV_ESENTE_TASSE * tetto_rv_av - bav_datore))
    bav_esente_contributi = np.minimum(
        bav, np.maximum(0.0, QUOTA_BAV_ESENTE_CONTRIBUTI * tetto_rv_av - bav_datore)
    )
    ral_fiscale = ral - bav_esente_tasse
    ral = ral - bav_esente_contributi  # base contributiva

    ral_capped_rv = np.minimum(ral, tetto_rv_av)
    quota_rv = ral_capped_rv * ALIQUOTA_RV
    quota_av = ral_capped_rv * ALIQUOTA_AV
    contributi_previdenziali = quota_rv + quota_av

    if tipo_sanita.upper() == "PKV":
        fattore_pkv = fattore_inflazione * ((1.0 + inflazione_medica_extra) ** anno)
        costo_sanita_lavoratore = costo_mensile_pkv_base * 12.0 * fattore_pkv
        quota_sanitaria_deducibile = costo_sanita_lavoratore * 0.82
        quota_kv = costo_sanita_lavoratore
        quota_pv = np.zeros_like(quota_kv)
    else:
        ral_capped_kv = np.minimum(ral, tetto_kv_pv)
        aliquota_pv_tot = aliquota_pv + (0.0 if ha_figli else supplemento_pv_senza_figli)
        quota_kv = ral_capped_kv * aliquota_kv
        quota_pv = ral_capped_kv * aliquota_pv_tot
        costo_sanita_lavoratore = quota_kv + quota_pv
        # Vorsorgepauschale: la quota KV deducibile esclude la componente Krankengeld
        quota_sanitaria_deducibile = (
            ral_capped_kv * (aliquota_kv - ESCLUSIONE_KRANKENGELD_KV) + quota_pv
        )

    pauschbetraege = (WERBUNGSKOSTEN_PAUSCHALE_2026 + SONDERAUSGABEN_PAUSCHALE_2026) * fattore_inflazione
    imponibile = np.maximum(0.0, ral_fiscale - pauschbetraege - quota_rv - quota_sanitaria_deducibile)
    imposta_base = einkommensteuer_32a(imponibile / fattore_inflazione) * fattore_inflazione

    freigrenze_soli = SOLI_FREIGRENZE_2026 * fattore_inflazione
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
        "bav": bav,
        "totale": totale_trattenute
    }


def calcola_netto_salario_germania(
    ral: np.ndarray,
    fattore_inflazione: np.ndarray,
    tipo_sanita: str = "PKV",
    costo_mensile_pkv_base: float = 400.0,
    anno: Union[int, np.ndarray] = 0,
    inflazione_medica_extra: float = 0.01,
    **parametri_contributivi
) -> np.ndarray:
    """Calcola lo stipendio netto annuo in Germania (Steuerklasse 1, dipendente) con PKV o GKV."""
    det = calcola_dettaglio_tasse_germania(
        ral=ral,
        fattore_inflazione=fattore_inflazione,
        tipo_sanita=tipo_sanita,
        costo_mensile_pkv_base=costo_mensile_pkv_base,
        anno=anno,
        inflazione_medica_extra=inflazione_medica_extra,
        **parametri_contributivi
    )
    return np.maximum(0.0, ral - det["totale"] - det["bav"])


def detrazione_lavoro_dipendente(reddito: np.ndarray, fattore_inflazione: np.ndarray) -> np.ndarray:
    """Detrazione per lavoro dipendente (art. 13 TUIR) per fasce di reddito complessivo, soglie indicizzate."""
    r = reddito / fattore_inflazione
    detrazione = np.where(
        r <= 15_000.0, 1_955.0,
        np.where(
            r <= 28_000.0, 1_910.0 + 1_190.0 * (28_000.0 - r) / 13_000.0,
            np.where(r <= 50_000.0, 1_910.0 * (50_000.0 - r) / 22_000.0, 0.0)
        )
    )
    return detrazione * fattore_inflazione


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
    irpef_nazionale = (q1 * 0.23) + (q2 * 0.33) + (q3 * 0.43)  # IRPEF 2026: 23% / 33% / 43%
    addizionali = imponibile_irpef * 0.02

    detrazione = detrazione_lavoro_dipendente(imponibile_irpef, fattore_inflazione)
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
    inflazione_medica_extra: float = 0.01,
    **parametri_contributivi
) -> np.ndarray:
    """Selettore unificato per il calcolo progressivo del salario netto (supporta array 1D e 2D)."""
    if paese.lower() in ("germania", "de", "germany"):
        return calcola_netto_salario_germania(
            ral=ral,
            fattore_inflazione=fattore_inflazione,
            tipo_sanita=tipo_sanita,
            costo_mensile_pkv_base=costo_mensile_pkv,
            anno=anno,
            inflazione_medica_extra=inflazione_medica_extra,
            **parametri_contributivi
        )
    return calcola_netto_salario_italia(ral, fattore_inflazione)
