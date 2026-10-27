"""Modulo per la previdenza pubblica con Totalizzazione Europea (Regolamento UE 883/2004) Italia-Germania."""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from config.settings import TETTO_RV_AV_2026


COEFFICIENTI_TRASFORMAZIONE_ITALIA = {
    57: 0.04270, 58: 0.04378, 59: 0.04493, 60: 0.04615, 61: 0.04744,
    62: 0.04882, 63: 0.05028, 64: 0.05184, 65: 0.05352, 66: 0.05531,
    67: 0.05723, 68: 0.05932, 69: 0.06154, 70: 0.06395, 71: 0.06655
}


def stima_irpef_italia(pensione_lorda: np.ndarray, fattore_inflazione_cum: np.ndarray) -> np.ndarray:
    """Calcola la pensione netta italiana deducendo l'IRPEF con scaglioni indicizzati all'inflazione."""
    s1 = 28_000.0 * fattore_inflazione_cum
    s2 = 50_000.0 * fattore_inflazione_cum

    q1 = np.clip(pensione_lorda, 0.0, s1)
    q2 = np.clip(pensione_lorda - s1, 0.0, s2 - s1)
    q3 = np.maximum(0.0, pensione_lorda - s2)

    irpef_lorda = (q1 * 0.23) + (q2 * 0.33) + (q3 * 0.43)
    detrazione_base = 1_955.0 * fattore_inflazione_cum
    detrazione = np.where(
        pensione_lorda <= (8_500.0 * fattore_inflazione_cum),
        irpef_lorda,
        np.clip(detrazione_base, 0.0, irpef_lorda)
    )

    return np.maximum(0.0, pensione_lorda - np.maximum(0.0, irpef_lorda - detrazione))


def stima_imposte_pensione_germania(
    pensione_lorda: np.ndarray,
    fattore_inflazione_cum: np.ndarray,
    tipo_sanita: str = "PKV"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola la pensione netta tedesca considerando il regime sanitario (PKV vs GKV) e l'Einkommensteuer.
    - Se GKV (KVdR): prelievo contributi sanitari e non autosufficienza (~11.5%).
    - Se PKV: nessun prelievo KVdR alla fonte; la DRV eroga il sussidio pubblico dell'8.15% (§ 106 SGB VI).
    """
    grundfreibetrag = 12_348.0 * fattore_inflazione_cum  # Grundfreibetrag 2026
    is_pkv = tipo_sanita.upper() == "PKV"

    if is_pkv:
        zuschuss_drv = pensione_lorda * 0.0815
        contributi_sanitari = np.zeros_like(pensione_lorda)
        base_imponibile = pensione_lorda
    else:
        zuschuss_drv = np.zeros_like(pensione_lorda)
        contributi_sanitari = pensione_lorda * 0.115
        base_imponibile = pensione_lorda - contributi_sanitari

    imponibile_tassabile = np.maximum(0.0, base_imponibile - grundfreibetrag)
    einkommensteuer = imponibile_tassabile * 0.22

    if is_pkv:
        pensione_netta = np.maximum(0.0, pensione_lorda + zuschuss_drv - einkommensteuer)
    else:
        pensione_netta = np.maximum(0.0, pensione_lorda - contributi_sanitari - einkommensteuer)

    return pensione_netta, zuschuss_drv


def calcola_pensione_grv_germania_prorata(
    anni_germania_totali: int,
    anni_ue_totali: int,
    ral_infl_serie: np.ndarray,
    anni_futuri: int,
    inflazione: np.ndarray,
    eta_iniziale: int,
    eta_pensione: int = 67,
    crescita_reale_salari: float = 0.008,
    tipo_sanita: str = "PKV",
    fattore_infl: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola la quota pro-rata tedesca GRV (vettorizzata):
    - La Wartezeit (5 anni) si valuta sulla somma totale degli anni UE (Germania + Italia).
    - La rendita viene pagata in proporzione ai soli anni lavorati in Germania.
    Restituisce: (pensione_netta, pensione_lorda, zuschuss_drv)
    """
    num_sim = ral_infl_serie.shape[1]
    if anni_ue_totali < 5 or anni_germania_totali <= 0:
        zeros = np.zeros(num_sim)
        return zeros, zeros, zeros

    if fattore_infl is None:
        fattore_infl = np.cumprod(1.0 + inflazione, axis=0)

    # Valori 2026: Durchschnittsentgelt provvisorio, BBG RV, Rentenwert dal 1.7.2026 (€/Entgeltpunkt/mese)
    durchschnittsentgelt_base = 51_944.0
    beitragsbemessungsgrenze_base = TETTO_RV_AV_2026
    rentenwert_annuo_base = 42.52 * 12.0

    punti_totali = np.zeros(num_sim)

    # Anni pregressi in Germania (stimati su RAL iniziale)
    anni_pregressi_de = anni_germania_totali - anni_futuri
    if anni_pregressi_de > 0:
        ep_pregresso = np.minimum(ral_infl_serie[0, :], beitragsbemessungsgrenze_base) / durchschnittsentgelt_base
        punti_totali += ep_pregresso * anni_pregressi_de

    # Anni futuri lavorati in Germania (calcolo vettorizzato 2D)
    if anni_futuri > 0:
        f_infl_sub = fattore_infl[:anni_futuri, :]
        anni_col = np.arange(anni_futuri)[:, np.newaxis]
        tetto_sub = beitragsbemessungsgrenze_base * f_infl_sub
        # Il Durchschnittsentgelt cresce con inflazione + crescita reale dei salari
        medio_sub = durchschnittsentgelt_base * f_infl_sub * ((1.0 + crescita_reale_salari) ** anni_col)
        ral_valida = np.minimum(ral_infl_serie[:anni_futuri, :], tetto_sub)
        punti_totali += np.sum(ral_valida / medio_sub, axis=0)

    anni_fino_pensione = max(0, eta_pensione - eta_iniziale)
    idx_pens = min(anni_fino_pensione - 1, inflazione.shape[0] - 1)
    f_infl_pens = fattore_infl[idx_pens, :]

    rentenwert_al_ritiro = rentenwert_annuo_base * f_infl_pens * ((1.0 + crescita_reale_salari) ** anni_fino_pensione)
    pensione_lorda = punti_totali * rentenwert_al_ritiro

    pensione_netta, zuschuss_drv = stima_imposte_pensione_germania(pensione_lorda, f_infl_pens, tipo_sanita=tipo_sanita)
    return pensione_netta, pensione_lorda, zuschuss_drv


def calcola_pensione_inps_italia_prorata(
    anni_italia_totali: int,
    anni_ue_totali: int,
    ral_media_italia: float,
    inflazione: np.ndarray,
    eta_iniziale: int,
    aliquota_ivs: float = 0.33,
    pil_reale: float = 0.008,
    fattore_infl: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, int]:
    """
    Calcola la quota pro-rata italiana INPS (Legge Dini / Riforma Fornero con Totalizzazione UE):
    - Con Totalizzazione UE >= 20 anni: accesso a 67 anni.
    - Con Totalizzazione UE tra 5 e 19 anni: accesso a 71 anni.
    - Sotto i 5 anni UE totali: nessuna pensione.
    """
    num_sim = inflazione.shape[1]
    if anni_ue_totali < 5 or anni_italia_totali <= 0:
        return np.zeros(num_sim), 67

    eta_accesso = 67 if anni_ue_totali >= 20 else 71
    if fattore_infl is None:
        fattore_infl = np.cumprod(1.0 + inflazione, axis=0)

    versamento_base_annuo = min(ral_media_italia, 119_650.0) * aliquota_ivs
    anni_attesa = max(0, eta_accesso - eta_iniziale)

    if anni_attesa > 0:
        fattore_montante = np.prod(1.0 + inflazione[:anni_attesa, :] + pil_reale, axis=0)
    else:
        fattore_montante = np.ones(num_sim)

    montante = (versamento_base_annuo * anni_italia_totali) * fattore_montante
    coeff = COEFFICIENTI_TRASFORMAZIONE_ITALIA.get(eta_accesso, 0.05723)
    pensione_lorda = montante * coeff

    idx_acc = min(anni_attesa - 1, inflazione.shape[0] - 1)
    pensione_netta = stima_irpef_italia(pensione_lorda, fattore_infl[idx_acc, :])

    return pensione_netta, eta_accesso


def calcola_pensione_totale_ue(
    paese_attuale: str,
    anni_lavoro_futuri: int,
    anni_pregressi_italia: int,
    anni_pregressi_germania: int,
    ral_pregressa_italia: float,
    ral_infl_serie: np.ndarray,
    inflazione: np.ndarray,
    eta_iniziale: int,
    eta_pensione_target: int = 67,
    pil_reale: float = 0.008,
    tipo_sanita: str = "PKV",
    fattore_infl: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calcola la pensione combinata europea (Totalizzazione Internazionale UE):
    Restituisce quote pro-rata GRV e INPS con le rispettive età di decorrenza e importi netti.
    """
    if fattore_infl is None:
        fattore_infl = np.cumprod(1.0 + inflazione, axis=0)

    if paese_attuale.lower() in ("germania", "de", "germany"):
        anni_de_totali = anni_pregressi_germania + anni_lavoro_futuri
        anni_it_totali = anni_pregressi_italia
        anni_futuri_de = anni_lavoro_futuri
    else:
        anni_it_totali = anni_pregressi_italia + anni_lavoro_futuri
        anni_de_totali = anni_pregressi_germania
        anni_futuri_de = 0

    anni_ue_totali = anni_de_totali + anni_it_totali

    p_grv_netta, p_grv_lorda, zuschuss_drv = calcola_pensione_grv_germania_prorata(
        anni_germania_totali=anni_de_totali,
        anni_ue_totali=anni_ue_totali,
        ral_infl_serie=ral_infl_serie,
        anni_futuri=anni_futuri_de,
        inflazione=inflazione,
        eta_iniziale=eta_iniziale,
        eta_pensione=eta_pensione_target,
        crescita_reale_salari=pil_reale,
        tipo_sanita=tipo_sanita,
        fattore_infl=fattore_infl
    )

    p_inps, eta_inps = calcola_pensione_inps_italia_prorata(
        anni_italia_totali=anni_it_totali,
        anni_ue_totali=anni_ue_totali,
        ral_media_italia=ral_pregressa_italia,
        inflazione=inflazione,
        eta_iniziale=eta_iniziale,
        pil_reale=pil_reale,
        fattore_infl=fattore_infl
    )

    return {
        "pensione_netta_grv": p_grv_netta,
        "pensione_lorda_grv": p_grv_lorda,
        "zuschuss_drv": zuschuss_drv,
        "eta_accesso_grv": eta_pensione_target,
        "pensione_netta_inps": p_inps,
        "eta_accesso_inps": eta_inps,
        "anni_ue_totali": anni_ue_totali,
        "anni_de_totali": anni_de_totali,
        "anni_it_totali": anni_it_totali,
        "tipo_sanita": tipo_sanita
    }


def calcola_capitale_bav(
    contributi_annui: np.ndarray,
    inflazione: np.ndarray,
    rendimento_reale: float
) -> np.ndarray:
    """
    Capitale accumulato nella Direktversicherung a fine di ciascun anno di versamento (shape: anni_acc + 1, sim).
    Rendimento nominale = inflazione dell'anno + rendimento reale del contratto.
    """
    anni_acc, num_sim = contributi_annui.shape
    capitale = np.zeros((anni_acc + 1, num_sim))
    for t in range(anni_acc):
        capitale[t + 1, :] = (capitale[t, :] + contributi_annui[t, :]) * (1.0 + inflazione[t, :] + rendimento_reale)
    return capitale


def calcola_pensione_bav_netta(
    capitale_a_fire: np.ndarray,
    anni_lavoro: int,
    inflazione: np.ndarray,
    eta_iniziale: int,
    eta_pensione: int,
    rendimento_reale: float,
    coeff_rendita: float,
    aliquota_imposta: float,
    aliquota_kv_pv: float,
    tipo_sanita: str = "PKV"
) -> np.ndarray:
    """
    Rendita netta annua della Direktversicherung alla decorrenza (€ nominali all'età di pensione).
    Dopo l'uscita dal lavoro (FIRE) i versamenti cessano e il capitale continua a rendere fino alla decorrenza.
    La rendita è tassata per intero (imposta sul reddito) e, in GKV, soggetta a KV+PV a quota piena.
    """
    anni_a_pensione = max(0, eta_pensione - eta_iniziale)
    capitale = capitale_a_fire
    if anni_a_pensione > anni_lavoro:
        crescita = np.prod(1.0 + inflazione[anni_lavoro:anni_a_pensione, :] + rendimento_reale, axis=0)
        capitale = capitale_a_fire * crescita
    lorda = capitale * coeff_rendita
    prelievo = aliquota_imposta + (aliquota_kv_pv if tipo_sanita.upper() == "GKV" else 0.0)
    return lorda * (1.0 - prelievo)
