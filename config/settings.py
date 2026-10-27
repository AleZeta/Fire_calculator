"""Modulo di configurazione avanzata dei parametri per la simulazione FIRE (Italia / Germania)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Parametri contributivi Germania 2026 (€/anno, indicizzati all'inflazione nelle simulazioni)
TETTO_RV_AV_2026 = 101_400.0   # Beitragsbemessungsgrenze RV/AV
TETTO_KV_PV_2026 = 69_750.0    # Beitragsbemessungsgrenze KV/PV
QUOTA_BAV_ESENTE_TASSE = 0.08  # § 3 Nr. 63 EStG: 8% della BBG RV (8.112 € nel 2026)
QUOTA_BAV_ESENTE_CONTRIBUTI = 0.04  # § 1 SvEV: 4% della BBG RV (4.056 € nel 2026)


@dataclass
class SimulationConfig:
    """Parametri demografici, finanziari, previdenziali e fiscali per Italia e Germania."""

    # Paese di residenza e lavoro attuale: 'Germania' oppure 'Italia'
    paese: str = "Germania"

    # Storico contributivo pregresso (Totalizzazione Europea)
    anni_pregressi_italia: int = 2      # Anni già lavorati in Italia
    anni_pregressi_germania: int = 3    # Anni già lavorati in Germania
    ral_pregressa_italia: float = 35_000.0  # RAL media nei periodi lavorati in Italia

    # Parametri generali Monte Carlo
    num_simulazioni: int = 5_000
    giorni_anno: int = 253

    # Orizzonte di vita ed età
    eta_iniziale: int = 30
    eta_pensione: int = 67  # Età di pensionamento canonica (67a Regelaltersgrenze / Vecchiaia)
    eta_morte: int = 100    # Orizzonte massimo
    anni_acc: int = 30      # Anni massimi di accumulo testati

    # Profilo lavorativo ed economico
    capitale_iniziale: float = 100_000.0
    ral: float = 85_000.0
    aumento_ral_annuo_reale: float = 0.02  # Crescita reale annua della RAL (oltre inflazione)
    spese_mese: float = 2_000.0

    # Asset Allocation Differenziata: Accumulo vs FIRE (Decumulo)
    target_equity_accumulo: float = 0.80  # 80% Azionario durante il lavoro
    target_bond_accumulo: float = 0.20    # 20% Obbligazionario durante il lavoro
    target_equity_fire: float = 0.60      # 60% Azionario durante il FIRE
    target_bond_fire: float = 0.40        # 40% Obbligazionario durante il FIRE
    usa_bond_tent: bool = True            # Attiva Glidepath Dinamico (Bond Tent anticrisi)

    # Costi e Rendimenti
    ter_etf: float = 0.0018             # Costi di gestione ETF (0.18%)
    bond_real_return: float = 0.015     # Rendimento reale annuo bond (1.5%)
    bond_volatility: float = 0.05       # Volatilità annua bond (5%)

    # Parametri Fiscali (configurati automaticamente dal paese)
    bollo: float = 0.0                  # 0% in Germania; 0.20% (0.002) in Italia
    aliquota_capital_gain: float = 0.26375  # Germania: 25% Abgeltungsteuer + 5.5% Soli; Italia: 26%
    aliquota_capital_gain_bond: float = 0.26375  # Germania: 26.375%; Italia: 12.5% titoli governativi White List
    franchigia_capital_gain_annua: float = 1_000.0  # Germania: Sparer-Pauschbetrag 1.000 €/anno

    # Trattamento Sanitario (Assicurazione Malattia / Pflegeversicherung in Germania)
    tipo_sanita_germania: str = "PKV"   # 'PKV' (Privata) oppure 'GKV' (Pubblica)
    costo_mensile_pkv_iniziale: float = 400.0  # Quota a carico lavoratore oggi (€/mese, oltre al contributo datore)
    # Contributi GKV/PV a carico del dipendente (2026)
    aliquota_kv_dipendente: float = 0.08645  # 7,3% + metà del contributo aggiuntivo medio (2,69% / 2)
    aliquota_pv_base: float = 0.018          # Pflegeversicherung quota dipendente
    supplemento_pv_senza_figli: float = 0.006  # Kinderlosenzuschlag
    ha_figli: bool = False                   # Se False si applica il supplemento PV
    inflazione_medica_extra_annua: float = 0.01  # Crescita reale dei premi sanitari oltre l'inflazione (+1%/anno)
    costo_mensile_pkv_fire_nominale: float = 780.0  # Quota a carico in FIRE prima dei 67 anni senza datore di lavoro (€/mese odierni)
    costo_mensile_gkv_fire_nominale: float = 450.0  # Quota volontaria GKV in FIRE prima dei 67 anni (€/mese odierni)

    # Pensione aziendale (Direktversicherung, solo Germania): Entgeltumwandlung dallo stipendio lordo
    abilita_bav: bool = False
    bav_mensile: float = 200.0              # Versamento lordo mensile (€ odierni, indicizzato all'inflazione)
    bav_rendimento_reale: float = 0.01      # Rendimento reale annuo del contratto (oltre l'inflazione)
    bav_coeff_rendita: float = 0.036        # Rendita annua lorda per € di capitale all'età di pensione
    bav_rivalutazione_rendita: float = 0.01 # Rivalutazione annua nominale della rendita (Überschüsse)
    aliquota_imposta_bav_pensione: float = 0.22  # Imposta sul reddito sulla rendita (tassata per intero)
    aliquota_kv_pv_bav_pensione: float = 0.19    # KV+PV sulle rendite aziendali (solo GKV, quota piena)

    # Campionamento Sincrono Triadi (Joint Vector Block Bootstrap: Stock EUR + Bond EUR + Inflazione Locale)
    usa_campionamento_sincrono: bool = True
    abilita_tfr: bool = False           # False in Germania; True in Italia
    aliquota_accantonamento_tfr: float = 0.0691  # 6.91% della RAL annua
    tassazione_separata_tfr: float = 0.23

    # Previdenza Pubblica
    abilita_pensione_pubblica: bool = True
    pil_reale_medio: float = 0.008

    # Strategia di Prelievo FIRE & Cash Buffer
    anni_buffer: int = 2                # Anni di spese liquide nel cash buffer (ottimale 1-2 anni)
    safe_wd_rate: float = np.pi / 100   # ~3.1416% benchmark di sicurezza

    # Regole di Prelievo Flessibile (Dynamic Spending)
    usa_curva_spese_smile: bool = True  # Retirement Smile: spese modulate per età
    usa_guardrails_spesa: bool = True   # Taglio discrezionale spese in caso di stress
    taglio_guardrail_spese: float = 0.10
    soglia_guardrail_wd: float = 0.05

    # Indice Azionario Scelto: '^SP500TR' (S&P 500 Total Return) oppure 'URTH' (MSCI World ETF)
    nome_dataset_azionario: str = "sp500"
    sp500_ticker: str = "^SP500TR"
    msci_world_ticker: str = "URTH"
    rendimento_medio_fisso: float = 0.08
    anni_blocco_rendimenti: int = 3

    # Inflazione
    scenario_inflazione: str = "lognormale"
    inflazione_media: float = 0.03

    # Percorsi
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)

    def __post_init__(self):
        """Allinea automaticamente le impostazioni fiscali e previdenziali al paese selezionato."""
        if self.is_germania:
            self.bollo = 0.0
            self.aliquota_capital_gain = 0.26375  # 25% + 5.5% Solidaritätszuschlag
            self.aliquota_capital_gain_bond = 0.26375
            self.franchigia_capital_gain_annua = 1_000.0  # Sparer-Pauschbetrag
            self.abilita_tfr = False
        else:  # Italia
            self.bollo = 0.0020  # 0.20% dossier titoli
            self.aliquota_capital_gain = 0.26
            self.aliquota_capital_gain_bond = 0.125  # 12.5% titoli governativi White List
            self.franchigia_capital_gain_annua = 0.0
            self.abilita_tfr = True

    @property
    def is_germania(self) -> bool:
        return self.paese.lower() in ("germania", "de", "germany")

    @property
    def ticker_azionario_attivo(self) -> str:
        if self.nome_dataset_azionario.lower() in ("msci_world", "world", "urth"):
            return self.msci_world_ticker
        return self.sp500_ticker

    @property
    def anni_vita(self) -> int:
        return self.eta_morte - self.eta_iniziale

    @property
    def prelievo_iniziale_annuo(self) -> float:
        return self.spese_mese * 12.0

    def get_allocazione_anno(self, idx_t: int, anno_fire: int) -> Tuple[float, float]:
        """Restituisce le quote target (equity_ratio, bond_ratio) per l'anno idx_t dato l'anno di ritiro anno_fire."""
        if idx_t < anno_fire:
            return self.target_equity_accumulo, self.target_bond_accumulo
        t_fire = idx_t - anno_fire
        if self.usa_bond_tent and t_fire <= 10:
            eq_r = (self.target_equity_fire - 0.15) + (0.15 * (t_fire / 10.0))
        else:
            eq_r = self.target_equity_fire
        return eq_r, 1.0 - eq_r

    def format_allocazione_anno(self, idx_t: int, anno_fire: int) -> str:
        """Formatta la stringa di allocazione 'XX% / YY%' per le tabelle di riepilogo e PDF."""
        eq_r, bd_r = self.get_allocazione_anno(idx_t, anno_fire)
        return f"{eq_r * 100:.1f}% / {bd_r * 100:.1f}%".replace(".0%", "%")
