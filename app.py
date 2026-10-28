"""Dashboard interattiva Web in Streamlit per il simulatore FIRE avanzato (Germania / Italia)."""

import sys
from pathlib import Path

# Assicura che la directory radice del progetto sia sempre in sys.path (per Streamlit Cloud / Linux)
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Invalida cache dei sottomoduli interni per prevenire conflitti di memoria con gli hot-reload di Streamlit
for _mod in list(sys.modules.keys()):
    if _mod.startswith("src.") or _mod == "src":
        del sys.modules[_mod]

import streamlit as st
import numpy as np
import pandas as pd
import hmac

import matplotlib.pyplot as plt

from config.settings import SimulationConfig, DATA_DIR
from src.market_data import (
    carica_dataset_sincrono_eur,
    carica_serie_rendimenti,
    genera_matrice_rendimenti_vettorizzata,
    genera_rendimenti_obbligazionari,
    genera_matrici_multiasset_sincrone,
)
from src.inflation import genera_inflazione
from src.simulation import (
    esegui_simulazione_completa,
    simula_stress_test_storico,
    esegui_analisi_sensitivita_strategie,
    parametri_contributivi_config,
)
from src.taxes import (
    calcola_stipendio_netto_automatico,
    calcola_dettaglio_tasse_germania,
    calcola_dettaglio_tasse_italia,
)
from src.visualization import (
    render_fig_heatmap_successo,
    render_fig_heatmap_wd,
    plot_fan_chart_portafoglio_fire,
    render_fig_heatmap_sensitivita,
)
from src.pdf_report import genera_report_pdf_fire

st.set_page_config(
    page_title="FIRE Calculator (Germania / Italia)",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dati MSCI Emerging Markets: non versionati. Online arrivano dai Secrets di Streamlit Cloud (chiave msci_em_csv)
# e vengono scritti in data/msci_em_eur.csv; in locale basta il file.
_file_em = DATA_DIR / "msci_em_eur.csv"
_motivo_em = ""
if not _file_em.exists():
    try:
        _file_em.write_text(st.secrets["msci_em_csv"].strip() + "\n", encoding="utf-8")
    except KeyError:
        _motivo_em = "Secret 'msci_em_csv' non trovato nei Secrets di questa app."
    except Exception as e:  # nessun secrets.toml (locale) o cartella non scrivibile: l'app funziona senza lo slider
        _motivo_em = f"Secret non leggibile ({type(e).__name__})."

def mostra_guida():
    st.markdown("### 📖 Guida ai Campi e ai Risultati")
    st.caption("Cosa significa ogni campo della barra laterale e come leggere i risultati. Tutti gli importi sono in € di oggi, indicizzati all'inflazione dello scenario simulato.")

    with st.expander("💶 Reddito e Spese", expanded=True):
        st.markdown(
            "- **RAL Iniziale:** stipendio lordo annuo di oggi. Cresce ogni anno con l'inflazione più la *Crescita Reale Annua*.\n"
            "- **Spese Mensili Attuali:** quanto spendi al mese mentre lavori. **Quello che investi ogni anno = stipendio netto − spese attuali** "
            "(non c'è un campo \"risparmio mensile\": è il risultato).\n"
            "- **Spese Mensili Obiettivo in FIRE:** quanto vuoi spendere dopo il ritiro. Determina il capitale necessario.\n"
            "- **Capitale Iniziale:** patrimonio già investito oggi.\n"
            "- **Curva Spese Dinamica (Retirement Smile):** le spese reali non sono piatte. Il modello usa 100% fino a 55 anni, 88% tra 55 e 75 "
            "(si viaggia e si spende meno col passare degli anni) e 105% oltre i 75 (sanità e assistenza). Disattivandola le spese restano costanti."
        )
    with st.expander("🏛️ Carriera, Profilo e Pensioni"):
        st.markdown(
            "- **Anni già lavorati in Italia / Germania:** servono per la pensione. I periodi in paesi UE diversi si sommano per i requisiti minimi.\n"
            "- **Età Pensione Canonica:** da quando arrivano la pensione tedesca (GRV) e italiana (INPS).\n"
            "- **Orizzonte Massimo:** età fino a cui il capitale deve durare. Il successo è la quota di scenari in cui il patrimonio non si esaurisce prima.\n"
            "- **Anni Futuri di Accumulo Testati:** fino a quanti anni di lavoro da oggi vengono provate le età di ritiro FIRE.\n"
            "- **Includi Pensioni Pubbliche:** se disattivato, il piano si regge solo sul portafoglio."
        )
    with st.expander("🇨🇭 Svizzera (AHV e libero passaggio)"):
        st.markdown(
            "- **Anni con contributi e reddito medio:** la rendita AHV è una rendita piena (tra circa 1.260 e 2.520 CHF al mese a seconda del reddito medio) "
            "riproporzionata sugli anni versati su 44. Parte dai 65 anni ed è indicizzata all'inflazione.\n"
            "- **Conto di libero passaggio:** il capitale del 2° pilastro. Si assume uno sblocco a 60 anni, interesse 0,5% l'anno e conversione in una rendita del 4% annuo del capitale.\n"
            "- **Cambio:** euro per 1 franco. Le prestazioni sono tassate nel paese di residenza (stima 15%).\n"
            "- Sono stime semplificate: per importi reali serve l'estratto della cassa di compensazione (AHV) e della fondazione di libero passaggio.\n"
            "- **Trasferimento prima del FIRE:** negli ultimi anni di lavoro stipendio e spese sono svizzeri (in CHF, convertiti col cambio). In Svizzera non si paga capital gain sui privati, "
            "quindi la quota azionaria scende verso il valore indicato vendendo senza imposta. Con *Vendi e ricompra* il prezzo di carico riparte dal valore attuale. "
            "Gli anni svizzeri aumentano anche AHV e 2° pilastro e riducono gli anni tedeschi o italiani per la pensione. Stress test storici e analisi di sensitività non includono questo periodo."
        )
    with st.expander("🏥 Sanità in Germania (PKV / GKV)"):
        st.markdown(
            "- **GKV:** assicurazione pubblica, contributo in percentuale sullo stipendio. **PKV:** assicurazione privata con premio fisso a tuo carico.\n"
            "- **Costo PKV in FIRE:** senza datore di lavoro paghi il premio intero fino ai 67 anni.\n"
            "- **Crescita Reale Premi PKV:** i premi privati salgono più dell'inflazione (storicamente circa +1% l'anno)."
        )
    with st.expander("💼 Pensione Aziendale (Direktversicherung / bAV)"):
        st.markdown(
            "- **Versamento Mensile:** la *tua* parte lorda, tolta dallo stipendio prima di tasse e contributi. Ha vantaggio fiscale fino all'8% della BBG "
            "e vantaggio anche sui contributi sociali fino al 4%.\n"
            "- **Contributo Datore (%):** quota che aggiunge l'azienda, in percentuale del tuo versamento. Il totale nel contratto è tuo versamento + contributo.\n"
            "- **Rendimento Reale:** crescita annua oltre l'inflazione, al netto dei costi. Circa 3–4% per un fondo azionario, 0–1% per una garanzia pura.\n"
            "- Il capitale resta bloccato fino alla pensione e la rendita è tassata (imposta sul reddito più contributi sanitari). Dopo il FIRE i versamenti si fermano."
        )
    with st.expander("📈 Asset Allocation & Mercati"):
        st.markdown(
            "- **Indice Azionario:** MSCI World o S&P 500, in EUR. Le obbligazioni sono il Bund tedesco a 10 anni.\n"
            "- **Modello di Mercato:** *Storico a blocchi* pesca blocchi di anni reali (azioni, bond e inflazione insieme) e conserva shock come il 1973 e il 2022. "
            "*Normale* usa rendimenti casuali con media, volatilità e correlazioni dello storico: più liscio, senza crisi estreme.\n"
            "- **Durata Blocchi:** quanti anni consecutivi reali vengono pescati insieme.\n"
            "- **% Mercati Emergenti:** quota di emergenti dentro la parte azionaria (se i dati sono disponibili). Accorcia il periodo storico: "
            "non confrontare un risultato con emergenti con uno senza.\n"
            "- **% Azioni Accumulo / FIRE:** il resto è obbligazionario.\n"
            "- **Glidepath (Bond Tent):** aumenta le obbligazioni intorno al FIRE per ridurre il rischio di crolli nei primi anni di prelievo.\n"
            "- **TER ETF:** costo annuo del fondo. **Sparer-Pauschbetrag / Bollo / Capital Gain / TFR:** fiscalità sui rendimenti del paese scelto."
        )
    with st.expander("🛡️ Opzioni Decumulo"):
        st.markdown(
            "- **Cash Buffer (anni di spese):** liquidità tenuta da parte per non vendere in un mercato in calo.\n"
            "- **Safe Withdrawal Rate Benchmark:** soglia di prelievo (in % del patrimonio) usata dalla seconda heatmap.\n"
            "- **Guardrails:** se il prelievo supera il 5% del patrimonio, o supera il 4% in un anno di mercato in calo, la spesa di quell'anno viene tagliata del 10%.\n"
            "- **Numero di Scenari:** più scenari = risultati più stabili ma più lenti."
        )
    with st.expander("🎯 Come leggere i risultati"):
        st.markdown(
            "- **Probabilità Capitale > 0 a 100 anni:** quota di scenari in cui il patrimonio non finisce prima dell'orizzonte. 99% = 1 scenario su 100 fallisce.\n"
            "- **Età FIRE minima (≥ 99%):** la prima età di ritiro che raggiunge quel 99%. È il numero migliore per confrontare due scelte: se scende, la scelta migliora.\n"
            "- **Patrimonio al FIRE:** valore mediano accumulato all'età scelta.\n"
            "- Cambia **un solo input alla volta** e ignora differenze sotto circa 1 punto percentuale: sono rumore della simulazione.\n"
            "- Non è un consiglio finanziario: è un modello basato su dati storici."
        )


@st.dialog("📖 Guida ai Campi", width="large")
def guida_in_finestra():
    mostra_guida()


st.title("🔥 FIRE Calculator — Simulatore Monte Carlo Avanzato")
st.markdown(
    "Simulatore di Indipendenza Finanziaria e Ritiro Anticipato (**FIRE**) con modello multi-asset differenziato (Accumulo vs FIRE), "
    "**ribilanciamento tax-efficient**, **Totalizzazione Europea delle Pensioni (Germania GRV + Italia INPS)**, "
    "fiscalità salariale progressiva e gestione dinamica del **Cash Buffer**."
)
if st.button("📖 Guida ai campi e ai risultati"):
    guida_in_finestra()

# ----------------- PROFILO PERSONALE (solo con password, dai Secrets) -----------------
# I dati personali NON stanno nel codice: sono nei Secrets (online) o in .streamlit/secrets.toml (locale, ignorato da git).
# Chiavi: profilo_password (stringa) e tabella [profilo]. Senza password giusta l'app usa i valori generici.
PROFILO: dict = {}
_pwd = st.sidebar.text_input("🔐 Profilo personale (password)", type="password", help="Precompila i campi con i tuoi dati. Vuoto = valori generici.")
if _pwd:
    try:
        if hmac.compare_digest(_pwd.encode(), str(st.secrets["profilo_password"]).encode()):
            PROFILO = dict(st.secrets["profilo"])
            st.sidebar.success("Profilo caricato")
        else:
            st.sidebar.error("Password errata")
    except Exception:
        st.sidebar.error("Profilo non configurato in questa app")


def _p(chiave, predefinito):
    """Valore del profilo personale (se caricato) con lo stesso tipo del predefinito."""
    try:
        return type(predefinito)(PROFILO.get(chiave, predefinito))
    except (TypeError, ValueError):
        return predefinito


# ----------------- SIDEBAR: PARAMETRI CONFIGURAZIONE CON FORM -----------------
st.sidebar.header("🌍 Quadro Territoriale & Fiscale")
paese_scelto = st.sidebar.selectbox(
    "Dove vuoi lavorare e risiedere da qui in poi?",
    options=["Germania 🇩🇪", "Italia 🇮🇹"],
    index=0
)
is_germania = "Germania" in paese_scelto
paese_codice = "Germania" if is_germania else "Italia"

with st.sidebar.form("form_parametri_fire"):
    if is_germania:
        st.info(
            "🇩🇪 **Regime Germania attivo**:\n"
            "- Imposte lavoro progressive (Lohnsteuer + Sozialabgaben)\n"
            "- Soli con Milderungszone (§ 4 SolzG)\n"
            "- Nessuna imposta patrimoniale di bollo (0%)\n"
            "- Abgeltungsteuer 26.375% (Sparer-Pauschbetrag 1.000 €/a)\n"
            "- Pensione GRV calcolata in Entgeltpunkte"
        )
    else:
        st.info(
            "🇮🇹 **Regime Italia attivo**:\n"
            "- Imposte progressive (IRPEF a scaglioni + Contributi IVS)\n"
            "- Bollo dossier titoli: 0.20% annuo\n"
            "- Capital gain: 26% equity / 12.5% bond\n"
            "- TFR dipendenti: 6.91% annuo\n"
            "- Pensione INPS contributiva"
        )

    st.subheader("🏛️ Carriera Pregressa (Totalizzazione UE)")
    st.caption("I periodi lavorati in diversi paesi UE si sommano per i requisiti minimi pensionistici (Regolamento UE 883/2004).")
    anni_preg_it = st.number_input("Anni già lavorati in Italia", min_value=0, max_value=40, value=_p("anni_preg_it", 2), step=1)
    anni_preg_de = st.number_input("Anni già lavorati in Germania", min_value=0, max_value=40, value=_p("anni_preg_de", 3), step=1)
    ral_preg_it = st.number_input("RAL media avuta in Italia (€)", min_value=10_000, max_value=200_000, value=_p("ral_preg_it", 35_000), step=2_500)

    with st.expander("🇨🇭 Carriera pregressa in Svizzera (facoltativo)"):
        st.caption("Rendita AHV (1° pilastro) in proporzione agli anni versati e capitale del 2° pilastro rimasto su un conto di libero passaggio. Stime semplificate.")
        anni_preg_ch = st.number_input("Anni con contributi in Svizzera", min_value=0.0, max_value=40.0, value=_p("anni_preg_ch", 0.0), step=0.25)
        ral_ch_chf = st.number_input("Reddito lordo annuo medio in Svizzera (CHF)", min_value=0, max_value=300_000, value=_p("ral_media_ch_chf", 0), step=2_500)
        cap_lp_chf = st.number_input("Capitale su conto di libero passaggio (CHF)", min_value=0, max_value=2_000_000, value=_p("libero_passaggio_chf", 0), step=500,
                                     help="Saldo del 2° pilastro (BVG/LPP) non ritirato. Da aggiornare con l'ultimo estratto conto.")
        cambio_chf = st.number_input("Cambio (€ per 1 CHF)", min_value=0.5, max_value=1.5, value=_p("cambio_chf_eur", 1.0), step=0.01)

    with st.expander("🇨🇭 Trasferimento in Svizzera prima del FIRE (facoltativo)"):
        abilita_ch = st.checkbox(
            "Lavora in Svizzera negli ultimi anni prima del FIRE",
            value=_p("svizzera", False),
            help="In Svizzera i privati non pagano imposta sul capital gain: durante questo periodo le azioni si vendono senza tasse per scendere verso la quota prudente. "
                 "Stipendio e spese sono quelli svizzeri; dopo il FIRE si torna al paese scelto."
        )
        anni_ch_fire = 3
        ral_ch_fire = 100_000
        aliq_ch_fire = 20.0
        spese_ch_fire = 3_500
        quota_az_ch = 50
        quota_carico_ch = 0
        if abilita_ch:
            anni_ch_fire = st.number_input("Anni in Svizzera prima del FIRE", min_value=1, max_value=5, value=3, step=1)
            ral_ch_fire = st.number_input("RAL lorda in Svizzera (CHF)", min_value=30_000, max_value=500_000, value=_p("ral_svizzera_chf", 100_000), step=5_000)
            aliq_ch_fire = st.number_input("Imposte e contributi totali sul lordo (%)", min_value=5.0, max_value=45.0, value=_p("aliquota_svizzera_pct", 20.0), step=1.0,
                                           help="Imposte sul reddito, AHV/ALV, 2° pilastro e assicurazione malattia: stima complessiva da adattare al cantone.")
            spese_ch_fire = st.number_input("Spese mensili in Svizzera (CHF)", min_value=1_000, max_value=15_000, value=_p("spese_svizzera_chf", 3_500), step=100)
            quota_az_ch = st.slider("Quota azioni a fine periodo (%)", min_value=20, max_value=80, value=50, step=5,
                                    help="La quota azionaria scende verso questo valore vendendo azioni senza imposta sul capital gain.")
            quota_carico_ch = st.slider("Vendi e ricompra: quota del portafoglio (%)", min_value=0, max_value=100, value=0, step=10,
                                        help="Vende e ricompra questa quota a fine periodo: il prezzo di carico riparte dal valore attuale e le plusvalenze passate non vengono tassate al ritorno.")
            st.caption("⚠️ Verifica con un consulente: il trasferimento di residenza ha effetti fiscali (es. perdita del domicilio fiscale, regole antielusive) che il modello non valuta.")

    st.subheader("👤 Profilo Personale & Orizzonte")
    eta_iniziale = st.number_input("Età Attuale", min_value=18, max_value=65, value=_p("eta", 30), step=1)
    eta_pensione = st.number_input("Età Pensione Canonica (Regelaltersgrenze / Vecchiaia)", min_value=60, max_value=75, value=67, step=1)
    eta_morte = st.number_input("Orizzonte Massimo (Fine Vita)", min_value=75, max_value=110, value=100, step=1)
    anni_acc = st.slider("Anni Futuri di Accumulo Testati", min_value=5, max_value=35, value=25, step=1)

    st.subheader("💶 Reddito e Spese")
    capitale_iniziale = st.number_input("Capitale Iniziale Investito (€)", min_value=0, max_value=5_000_000, value=_p("capitale_iniziale", 100_000), step=5_000)
    ral = st.number_input("RAL Iniziale Attuale (€)", min_value=10_000, max_value=500_000, value=_p("ral", 85_000), step=2_500)
    crescita_ral_reale = st.slider("Crescita Reale Annua RAL (% oltre inflazione)", min_value=0.0, max_value=6.0, value=_p("crescita_ral_reale_pct", 2.0), step=0.2) / 100.0
    spese_mese_accumulo = st.number_input(
        "Spese Mensili Attuali, mentre lavori (€)", min_value=500, max_value=20_000, value=_p("spese_attuali", 2_000), step=100,
        help="Quanto spendi oggi al mese (€ di oggi, indicizzati all'inflazione). Quello che avanza dallo stipendio netto viene investito."
    )
    spese_mese = st.number_input(
        "Spese Mensili Obiettivo in FIRE (€)", min_value=500, max_value=20_000, value=_p("spese_fire", 2_000), step=100,
        help="Quanto vuoi spendere al mese dopo il ritiro (€ di oggi). Determina il capitale necessario."
    )

    if is_germania:
        st.subheader("🏥 Sanità in Germania (PKV / GKV)")
        tipo_sanita_input = st.radio(
            "Regime Sanitario in Accumulo",
            options=["Privata (PKV)", "Pubblica (GKV)"],
            index=1 if _p("sanita", "PKV") == "GKV" else 0,
            help="Scegli tra sanità privata (premio a tuo carico con deduzione fiscale) o pubblica (aliquota fissa su busta paga)."
        )
        tipo_sanita = "PKV" if "PKV" in tipo_sanita_input else "GKV"

        if tipo_sanita == "PKV":
            costo_pkv_iniziale = st.number_input(
                "Costo Mensile PKV a tuo carico (€)",
                min_value=100.0, max_value=2_000.0, value=_p("costo_pkv", 400.0), step=25.0,
                help="Quota mensile a tuo carico detratta dalla busta paga (oltre al contributo del datore di lavoro)."
            )
            costo_pkv_fire = st.number_input(
                "Costo Mensile PKV in FIRE prima dei 67a (€)",
                min_value=200.0, max_value=3_000.0, value=_p("costo_pkv_fire", 780.0), step=25.0,
                help="In FIRE senza datore paghi il 100% del premio. Dai 67 anni la DRV versa il sussidio dell'8.15% (§ 106 SGB VI)."
            )
            inflaz_medica = st.slider(
                "Crescita Reale Premi PKV (% oltre inflazione)",
                min_value=0.0, max_value=3.0, value=1.0, step=0.2,
                help="Aumento dei premi dovuto all'invecchiamento e all'inflazione medica (storicamente inflazione + 1%)."
            ) / 100.0
        else:
            costo_pkv_iniziale = 400.0
            costo_pkv_fire = 780.0
            inflaz_medica = 0.01
    else:
        tipo_sanita = "GKV"
        costo_pkv_iniziale = 400.0
        costo_pkv_fire = 780.0
        inflaz_medica = 0.01

    abilita_bav = False
    bav_mensile = 0.0
    bav_contributo_datore = 0.0
    bav_rendimento_reale = SimulationConfig().bav_rendimento_reale
    if is_germania:
        abilita_bav = st.checkbox(
            "Includi Pensione Aziendale (Direktversicherung)",
            value=_p("bav", False),
            help="Versamento mensile lordo (Entgeltumwandlung): esente da imposte fino all'8% della BBG (676 €/mese) e da contributi fino al 4% (338 €/mese); la rendita è tassata per intero alla pensione."
        )
        if abilita_bav:
            bav_mensile = st.number_input(
                "Versamento Mensile Direktversicherung (€ lordi)",
                min_value=0.0, max_value=1_500.0, value=_p("bav_mensile", 200.0), step=25.0
            )
            bav_contributo_datore = st.number_input(
                "Contributo Datore sulla Direktversicherung (% del tuo versamento)",
                min_value=0.0, max_value=100.0, value=_p("bav_datore_pct", 0.0), step=5.0,
                help="Per legge il datore deve aggiungere il 15% se risparmia contributi sociali (§ 1a BetrAVG). Verifica con HR; 0 = prudente."
            ) / 100.0
            bav_rendimento_reale = st.number_input(
                "Rendimento Reale Annuo del Contratto (%)",
                min_value=-2.0, max_value=8.0, value=_p("bav_rendimento_pct", round(bav_rendimento_reale * 100.0, 2)), step=0.5,
                help="Oltre l'inflazione, al netto dei costi del contratto. Fondo azionario: circa 3-4%; garanzia pura: circa 0-1%."
            ) / 100.0

    st.subheader("📈 Asset Allocation & Mercati")
    scelta_azionario = st.selectbox(
        "Indice Azionario di Riferimento",
        options=["MSCI World Total Return (EUR 1970-2024)", "S&P 500 Total Return (EUR 1970-2024)"],
        index=0
    )
    nome_azionario = "msci_world" if "MSCI" in scelta_azionario else "sp500"

    scelta_modello = st.selectbox(
        "Modello di Mercato",
        options=["Storico a blocchi (consigliato)", "Normale (media, volatilità e correlazioni storiche)"],
        index=0,
        help="Storico: estrae blocchi di anni reali (azioni, bond e inflazione insieme: preserva shock come 1973-74 e 2022). "
             "Normale: rendimenti gaussiani con media, volatilità e correlazioni stimate sullo storico; più liscio, senza code spesse né sequenze."
    )
    modello_mercato = "normale" if "Normale" in scelta_modello else "storico"
    usa_sincrono = True

    quota_em = 0.0
    if (DATA_DIR / "msci_em_eur.csv").exists():
        quota_em = st.slider(
            "% Mercati Emergenti nella parte azionaria",
            min_value=0, max_value=50, value=_p("quota_em_pct", 0), step=5,
            help="Ribilanciata ogni anno. Il periodo storico parte dal primo anno in cui esistono tutti gli indici scelti."
        ) / 100.0
        if quota_em > 0:
            anni_storico = carica_dataset_sincrono_eur(SimulationConfig(quota_azioni_emergenti=quota_em))["Year"]
            st.caption(
                f"⚠️ Con gli emergenti il periodo storico è {anni_storico.min()}–{anni_storico.max()} "
                f"({len(anni_storico)} anni invece di 1970–2024): i risultati non sono confrontabili con quelli senza emergenti."
            )
    else:
        st.caption("Per aggiungere i mercati emergenti inserisci data/msci_em_eur.csv (colonne Year, msci_em_eur)."
                   + (f" {_motivo_em}" if _motivo_em else ""))

    durata_blocco = st.select_slider(
        "Durata Blocchi Temporali (Anni)",
        options=[2, 3, 4, 5],
        value=3,
        help="3 anni è la durata media ottimale di un bear market pluriennale (es. 2000-2002, 1973-1975, 2022-2023)."
    )

    col_alloc1, col_alloc2 = st.columns(2)
    with col_alloc1:
        eq_acc = st.slider("% Azioni Accumulo", min_value=0, max_value=100, value=80, step=5) / 100.0
        bd_acc = 1.0 - eq_acc
    with col_alloc2:
        eq_fire = st.slider("% Azioni FIRE", min_value=0, max_value=100, value=60, step=5) / 100.0
        bd_fire = 1.0 - eq_fire

    usa_bond_tent = st.checkbox(
        "Attiva Glidepath Dinamico (Bond Tent anticrisi)",
        value=True,
        help="Aumenta progressivamente la quota bond a ridosso del FIRE e nei primi 10 anni di decumulo per neutralizzare il Sequence of Returns Risk, tornando poi all'esposizione azionaria target."
    )

    ter = st.number_input("TER Annuo ETF (%)", min_value=0.0, max_value=1.5, value=0.18, step=0.02) / 100.0

    if is_germania:
        franchigia_cg = st.number_input("Sparer-Pauschbetrag Franchigia (€/anno)", min_value=0, max_value=4_000, value=1_000, step=100)
        bollo = 0.0
        aliquota_cg = 0.26375
        aliquota_cg_bd = 0.26375
        abilita_tfr = False
    else:
        bollo = st.number_input("Imposta di Bollo Dossier Titoli (%)", min_value=0.0, max_value=1.0, value=0.20, step=0.02) / 100.0
        aliquota_cg = st.number_input("Capital Gain Azionario (%)", min_value=0.0, max_value=35.0, value=26.0, step=1.0) / 100.0
        aliquota_cg_bd = st.number_input("Capital Gain Titoli di Stato (%)", min_value=0.0, max_value=35.0, value=12.5, step=0.5) / 100.0
        franchigia_cg = 0.0
        abilita_tfr = st.checkbox("Includi TFR Dipendenti (6.91% annuo)", value=True)

    st.subheader("🏛️ Opzioni Decumulo")
    abilita_pensione = st.checkbox("Includi Pensioni Pubbliche (GRV + INPS)", value=True)
    anni_buffer = st.slider("Cash Buffer Anticrisi (Anni di Spese)", min_value=0, max_value=7, value=2, step=1)
    safe_wd_rate = st.slider("Safe Withdrawal Rate Benchmark (%)", min_value=2.0, max_value=5.0, value=3.14, step=0.1) / 100.0
    usa_smile = st.checkbox("Curva Spese Dinamica (Retirement Smile)", value=True)
    usa_guardrails = st.checkbox("Guardrails Flessibilità Spese (Guyton-Klinger light)", value=True)

    st.subheader("⚙️ Motore Monte Carlo")
    num_simulazioni = st.select_slider("Numero di Scenari Monte Carlo", options=[1_000, 2_500, 5_000, 10_000], value=5_000)

    st.markdown("---")
    submit_btn = st.form_submit_button(
        "🚀 Esegui / Ricalcola Simulazione",
        type="primary",
        use_container_width=True
    )

cfg = SimulationConfig(
    paese=paese_codice,
    anni_pregressi_italia=int(anni_preg_it),
    anni_pregressi_germania=int(anni_preg_de),
    ral_pregressa_italia=float(ral_preg_it),
    anni_pregressi_svizzera=float(anni_preg_ch),
    ral_media_svizzera_chf=float(ral_ch_chf),
    capitale_libero_passaggio_chf=float(cap_lp_chf),
    cambio_chf_eur=float(cambio_chf),
    abilita_svizzera=bool(abilita_ch),
    anni_svizzera=int(anni_ch_fire),
    ral_svizzera_chf=float(ral_ch_fire),
    aliquota_effettiva_svizzera=float(aliq_ch_fire) / 100.0,
    spese_mese_svizzera_chf=float(spese_ch_fire),
    quota_azioni_fine_svizzera=float(quota_az_ch) / 100.0,
    quota_carico_azzerato_svizzera=float(quota_carico_ch) / 100.0,
    num_simulazioni=num_simulazioni,
    eta_iniziale=eta_iniziale,
    eta_pensione=eta_pensione,
    eta_morte=eta_morte,
    anni_acc=anni_acc,
    capitale_iniziale=capitale_iniziale,
    ral=ral,
    aumento_ral_annuo_reale=crescita_ral_reale,
    spese_mese=spese_mese,
    spese_mese_accumulo=float(spese_mese_accumulo),
    tipo_sanita_germania=tipo_sanita,
    costo_mensile_pkv_iniziale=float(costo_pkv_iniziale),
    costo_mensile_pkv_fire_nominale=float(costo_pkv_fire),
    inflazione_medica_extra_annua=float(inflaz_medica),
    abilita_bav=abilita_bav,
    bav_mensile=float(bav_mensile),
    bav_contributo_datore=float(bav_contributo_datore),
    bav_rendimento_reale=float(bav_rendimento_reale),
    nome_dataset_azionario=nome_azionario,
    target_equity_accumulo=eq_acc,
    target_bond_accumulo=bd_acc,
    target_equity_fire=eq_fire,
    target_bond_fire=bd_fire,
    ter_etf=ter,
    bollo=bollo,
    aliquota_capital_gain=aliquota_cg,
    aliquota_capital_gain_bond=aliquota_cg_bd,
    franchigia_capital_gain_annua=franchigia_cg,
    abilita_tfr=abilita_tfr,
    abilita_pensione_pubblica=abilita_pensione,
    anni_buffer=anni_buffer,
    safe_wd_rate=safe_wd_rate,
    usa_curva_spese_smile=usa_smile,
    usa_guardrails_spesa=usa_guardrails,
    usa_campionamento_sincrono=usa_sincrono,
    modello_mercato=modello_mercato,
    quota_azioni_emergenti=quota_em,
    anni_blocco_rendimenti=int(durata_blocco),
    usa_bond_tent=usa_bond_tent
)

# Calcolo automatico immediato del netto iniziale per trasparenza nella sidebar
netto_stimato_anno0 = float(calcola_stipendio_netto_automatico(
    np.array([ral]),
    paese_codice,
    np.array([1.0]),
    tipo_sanita=tipo_sanita,
    costo_mensile_pkv=costo_pkv_iniziale,
    anno=0,
    inflazione_medica_extra=inflaz_medica,
    **parametri_contributivi_config(cfg)
)[0])
aliquota_effettiva_stimata = (1.0 - netto_stimato_anno0 / ral) * 100.0
st.sidebar.caption(
    f"💡 **Stipendio Netto Attuale Stimato**: ~**{netto_stimato_anno0:,.0f} €/anno** (~**{netto_stimato_anno0 / 12:,.0f} €/mese**)\n"
    f"Trattenute complessive calcolate automaticamente: **{aliquota_effettiva_stimata:.1f}%**"
)


# ----------------- CACHING A DUE LIVELLI & STATO SESSIONE -----------------
@st.cache_data(show_spinner="Caricamento serie storiche e matrici di mercato...")
def get_dati_mercato(num_sim: int, eta_iniziale: int, eta_morte: int, ticker_az: str, paese: str, anni_blocco: int, usa_sincrono_flag: bool, modello: str, quota_em_flag: float):
    dummy_cfg = SimulationConfig(
        num_simulazioni=num_sim,
        eta_iniziale=eta_iniziale,
        eta_morte=eta_morte,
        nome_dataset_azionario=ticker_az,
        paese=paese,
        anni_blocco_rendimenti=anni_blocco,
        usa_campionamento_sincrono=usa_sincrono_flag,
        modello_mercato=modello,
        quota_azioni_emergenti=quota_em_flag
    )
    if usa_sincrono_flag:
        return genera_matrici_multiasset_sincrone(dummy_cfg)
    serie = carica_serie_rendimenti(dummy_cfg)
    m_eq, md_eq = genera_matrice_rendimenti_vettorizzata(serie, dummy_cfg)
    inflazione = genera_inflazione(dummy_cfg)
    md_bond = genera_rendimenti_obbligazionari(dummy_cfg, inflazione)
    return m_eq, md_eq, md_bond, inflazione


@st.cache_data(show_spinner="Calcolo della simulazione FIRE multi-asset...")
def calcola_simulazione_fire(config: SimulationConfig, _m_eq, _md_eq, _md_bd, _infl):
    return esegui_simulazione_completa(config, _m_eq, _md_eq, _md_bd, _infl)


@st.cache_data(show_spinner="Esecuzione analisi di sensitività strategica su allocazioni e worst-case...")
def calcola_sensitivita_cached(config: SimulationConfig, _m_eq, _md_eq, _md_bd, _infl, eta_rif: int):
    return esegui_analisi_sensitivita_strategie(config, _m_eq, _md_eq, _md_bd, _infl, eta_fire_rif=eta_rif)


if "sim_data" not in st.session_state or submit_btn:
    with st.spinner("Calcolo della simulazione FIRE in corso..."):
        m_eq, md_eq, md_bond, inflazione = get_dati_mercato(
            cfg.num_simulazioni,
            cfg.eta_iniziale,
            cfg.eta_morte,
            cfg.nome_dataset_azionario,
            cfg.paese,
            cfg.anni_blocco_rendimenti,
            cfg.usa_campionamento_sincrono,
            cfg.modello_mercato,
            cfg.quota_azioni_emergenti
        )
        risultati = calcola_simulazione_fire(cfg, m_eq, md_eq, md_bond, inflazione)
        f_infl_media = np.mean(np.cumprod(1.0 + inflazione, axis=0), axis=1)
        f_infl_media[0] = 1.0
        st.session_state["sim_data"] = {
            "cfg": cfg,
            "m_eq": m_eq,
            "md_eq": md_eq,
            "md_bond": md_bond,
            "inflazione": inflazione,
            "f_infl_media": f_infl_media,
            "risultati": risultati,
        }

sim_data = st.session_state["sim_data"]
cfg = sim_data["cfg"]
m_eq = sim_data["m_eq"]
md_eq = sim_data["md_eq"]
md_bond = sim_data["md_bond"]
inflazione = sim_data["inflazione"]
f_infl_media = sim_data["f_infl_media"]
risultati = sim_data["risultati"]

# ----------------- CONTROLLO UNIFICATO ETÀ RITIRO FIRE & STATISTICHE IN EVIDENZA -----------------
min_fire_age = int(cfg.eta_iniziale + 1)
max_fire_age = int(cfg.eta_iniziale + cfg.anni_acc - 1)

perc_positivi = risultati["percentuali_positivi"]
idx_fine_vita = cfg.anni_vita - 1
succ_fine_vita = perc_positivi[:, idx_fine_vita]

indici_99 = np.where(succ_fine_vita >= 99.0)[0]
anni_minimi_99 = int(indici_99[0]) if len(indici_99) > 0 else None

default_fire_age = int(cfg.eta_iniziale + (anni_minimi_99 if anni_minimi_99 is not None else min(12, cfg.anni_acc - 1)))
default_fire_age = max(min_fire_age, min(max_fire_age, default_fire_age))

if "eta_fire_target" not in st.session_state:
    st.session_state["eta_fire_target"] = default_fire_age
st.session_state["eta_fire_target"] = max(min_fire_age, min(max_fire_age, st.session_state["eta_fire_target"]))

st.subheader(f"📊 Sintesi Risultati — Quadro Fiscale: {cfg.paese} | Asset: {cfg.ticker_azionario_attivo}")

eta_fire_sel = st.slider(
    "🎯 Seleziona l'Età Programmata per il Ritiro FIRE:",
    min_value=min_fire_age,
    max_value=max_fire_age,
    value=st.session_state["eta_fire_target"],
    key="slider_eta_fire_main",
    help="Slider unificato principale: imposta l'età di pensionamento FIRE per tutte le metriche, i grafici delle tab, le tabelle e il report PDF."
)
st.session_state["eta_fire_target"] = eta_fire_sel
anno_acc_sel = eta_fire_sel - cfg.eta_iniziale

col1, col2, col3, col4, col5 = st.columns(5)

prob_100a = succ_fine_vita[anno_acc_sel]
cap_fire_mediano = float(np.median(risultati["capitale_acc"][anno_acc_sel, :]))

with col1:
    st.metric(
        "Probabilità Capitale > 0 a 100 anni",
        f"{prob_100a:.1f}%",
        f"Ritiro ad età {eta_fire_sel}a ({anno_acc_sel}a lavoro)"
    )

with col2:
    st.metric(
        "Patrimonio Accumulato al FIRE",
        f"{cap_fire_mediano:,.0f} €",
        f"All'età di {eta_fire_sel} anni"
    )

with col3:
    if cfg.abilita_pensione_pubblica:
        p_tot = float(np.median(risultati["pensioni_list"][anno_acc_sel]))
        p_det = risultati["pensioni_dettaglio_list"][anno_acc_sel]
        p_grv = float(np.median(p_det["pensione_netta_grv"]))
        p_inps = float(np.median(p_det["pensione_netta_inps"]))
        st.metric(
            "Pensione Pubblica Totale UE",
            f"{p_tot:,.0f} €/anno",
            f"GRV: {p_grv:,.0f} € + INPS: {p_inps:,.0f} € (dai 67a)"
        )
    else:
        st.metric("Pensione Pubblica", "Disabilitata / N/D")

with col4:
    spese_annue_nom = cfg.spese_mese * 12
    st.metric(
        "Spesa Annua Obiettivo in FIRE",
        f"{spese_annue_nom:,.0f} €/anno",
        f"{cfg.spese_mese:,.0f} €/mese"
    )

with col5:
    if anni_minimi_99 is not None:
        st.metric(
            "Età FIRE minima (≥ 99% successo)",
            f"{cfg.eta_iniziale + anni_minimi_99} anni",
            f"{anni_minimi_99}a di lavoro"
        )
    else:
        st.metric(
            "Età FIRE minima (≥ 99% successo)",
            "Non raggiunta",
            f"entro i {max_fire_age} anni",
            delta_color="off"
        )

# Pre-calcolo condiviso delle 4 traiettorie di Stress Test Storici per l'età selezionata (~2 ms)
MAPPA_ANNI_STRESS = {
    "Dot-Com 2000": 2000,
    "Crisi 2008": 2008,
    "Stagflazione 1973": 1973,
    "Crollo 2022": 2022,
}
tutti_stress_dict = {
    nome: simula_stress_test_storico(cfg, anno_s, anno_acc_sel)
    for nome, anno_s in MAPPA_ANNI_STRESS.items()
}


@st.cache_data(show_spinner=False)
def _cached_heatmap_successo(success_rate_matrix, x_labels, y_labels):
    return render_fig_heatmap_successo(success_rate_matrix, x_labels, y_labels)


@st.cache_data(show_spinner=False)
def _cached_heatmap_wd(wd_data_matrix, x_labels, y_labels):
    return render_fig_heatmap_wd(wd_data_matrix, x_labels, y_labels)


# ----------------- TABS CON GRAFICI (5 Tab Principali) -----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Heatmap (Successo & SWR)",
    "📈 Evoluzione Patrimonio & Stress Test",
    "🏛️ Spese e Pensioni UE (GRV + INPS)",
    "🔬 Analisi di Sensitività Strategie",
    "📥 Riepilogo & Esportazione Piano",
    "📖 Guida ai Campi"
])

with tab1:
    st.markdown(f"### 1. Probabilità che il Portafoglio rimanga Integro (Capitale > 0) — {cfg.paese}")
    st.caption("Percentuale di scenari Monte Carlo in cui il patrimonio non si esaurisce prima dell'età target.")

    eta_min_col = cfg.eta_pensione
    idx_col_start = eta_min_col - cfg.eta_iniziale
    success_rate = np.round(perc_positivi[:, idx_col_start:]).astype(float)

    y_labels = [cfg.eta_iniziale + i for i in range(cfg.anni_acc)]
    x_labels = [i for i in range(eta_min_col, cfg.eta_morte)]

    fig_heat = _cached_heatmap_successo(success_rate, x_labels, y_labels)
    st.pyplot(fig_heat)

    st.markdown("---")
    st.markdown(f"### 2. Sostenibilità Safe Withdrawal Rate (Prelievo rolling 3a < {cfg.safe_wd_rate * 100:.2f}%) — {cfg.paese}")
    st.caption("Percentuale di scenari in cui il tasso di prelievo medio non supera la soglia di sicurezza. Visualizzazione concentrata dai 67 ai 100 anni con valori sempre indicati.")

    wd_sub = risultati["percentuali_wd_rate_roll"][:, idx_col_start:]
    fig_wd = _cached_heatmap_wd(wd_sub, x_labels, y_labels)
    st.pyplot(fig_wd)

with tab2:
    st.markdown("### 📈 Evoluzione del Patrimonio nel Tempo: Fan Chart & Stress Test Storici")
    st.caption(
        f"Asset Allocation dinamica: {int(cfg.target_equity_accumulo * 100)}% Azioni in accumulo -> "
        f"{int(cfg.target_equity_fire * 100)}% Azioni in FIRE"
        + (" (con Glidepath dinamico Bond Tent attivo)" if cfg.usa_bond_tent else "")
    )

    col_t2_1, col_t2_2 = st.columns([1.4, 1])
    with col_t2_1:
        st.info(f"🎯 **Ipotesi di Ritiro Attiva**: Età **{eta_fire_sel} anni** ({anno_acc_sel} anni di lavoro/accumulo futuro da oggi). Modificabile tramite lo slider principale in alto.")
    with col_t2_2:
        scelta_scala = st.radio(
            "Scala Asse Y:",
            options=["Scala Logaritmica (Symlog) 🌟", "Scala Lineare"],
            index=0,
            horizontal=True
        )

    scelta_stress = st.multiselect(
        "⚡ Sovrapponi Crisi Storiche Reali (Stress Test Sequenza di Rendimenti al Ritiro):",
        options=list(MAPPA_ANNI_STRESS.keys()),
        default=list(MAPPA_ANNI_STRESS.keys()),
        help="Simula cosa sarebbe accaduto al tuo portafoglio se fossi andato in FIRE esattamente all'inizio di uno di questi shock storici reali in EUR."
    )

    stress_dict = {s_name: tutti_stress_dict[s_name] for s_name in scelta_stress}
    matrice_scelta = risultati["capitale_vita"][anno_acc_sel]
    scala_modalita = "logaritmica" if "logaritmica" in scelta_scala.lower() else "lineare"
    fig_fan = plot_fan_chart_portafoglio_fire(cfg, matrice_scelta, anno_acc_sel, stress_dict, scala_y=scala_modalita)
    st.pyplot(fig_fan)
    plt.close(fig_fan)

with tab3:
    st.markdown("### 🏛️ Totalizzazione Europea: Pensione Tedesca (GRV) + Pensione Italiana (INPS)")
    st.markdown(
        f"Grazie al **Regolamento UE 883/2004**, i tuoi **{cfg.anni_pregressi_italia} anni in Italia** e **{cfg.anni_pregressi_germania} anni in Germania** "
        f"si sommano per soddisfare la *Wartezeit* minima di 5 anni! All'età pensionabile riceverai due assegni *pro-rata* distinti."
    )
    st.info(f"🎯 **Ipotesi di Ritiro Attiva**: Età **{eta_fire_sel} anni** ({anno_acc_sel} anni di lavoro/accumulo futuro da oggi). Modificabile tramite lo slider principale in alto.")

    anno_sel_pensione = anno_acc_sel
    p_dettaglio = risultati["pensioni_dettaglio_list"][anno_sel_pensione]
    p_grv_med = float(np.median(p_dettaglio["pensione_netta_grv"]))
    p_inps_med = float(np.median(p_dettaglio["pensione_netta_inps"]))
    eta_inps_acc = p_dettaglio["eta_accesso_inps"]
    eta_grv_acc = p_dettaglio["eta_accesso_grv"]

    eta_fire = cfg.eta_iniziale + anno_sel_pensione
    idx_accesso = cfg.eta_pensione - cfg.eta_iniziale

    # Stipendio netto medio anno per anno durante l'accumulo
    stipendio_netto_annuo = np.mean(risultati["depositi"] + risultati["spese"][:cfg.anni_acc, :] * cfg.rapporto_spese_accumulo, axis=1)
    ultimo_stipendio_netto = float(stipendio_netto_annuo[max(0, anno_sel_pensione - 1)]) if anno_sel_pensione > 0 else 0.0
    anni_gap_fire = max(0, cfg.eta_pensione - eta_fire)

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.metric(
            f"Stipendio Netto al Ritiro (Età {eta_fire}a)",
            f"{ultimo_stipendio_netto:,.0f} €/anno" if anno_sel_pensione > 0 else "0 € (FIRE immediato)",
            f"Si azzera a {eta_fire} anni"
        )
    with col_p2:
        st.metric(
            f"Pensione Tedesca GRV (dai {eta_grv_acc}a)",
            f"{p_grv_med:,.0f} €/anno",
            f"Su {p_dettaglio['anni_de_totali']} anni totali in DE"
        )
    with col_p3:
        st.metric(
            f"Pensione Italiana INPS (dai {eta_inps_acc}a)",
            f"{p_inps_med:,.0f} €/anno",
            f"Su {p_dettaglio['anni_it_totali']} anni totali in IT"
        )
    with col_p4:
        p_tot_med = p_grv_med + p_inps_med
        st.metric(
            "Pensione Totale UE Combinata",
            f"{p_tot_med:,.0f} €/anno",
            f"{anni_gap_fire} anni di FIRE prima della pensione"
        )

    anni_x_vita = np.arange(cfg.eta_iniziale, cfg.eta_morte)
    spese_medie = np.mean(risultati["spese"], axis=1)
    spese_medie[:anno_sel_pensione] *= cfg.rapporto_spese_accumulo  # mentre lavori si spende la cifra attuale

    curva_entrate_nette = np.zeros(len(anni_x_vita))
    if anno_sel_pensione > 0:
        anni_lav_effettivi = min(anno_sel_pensione, len(stipendio_netto_annuo))
        curva_entrate_nette[:anni_lav_effettivi] = stipendio_netto_annuo[:anni_lav_effettivi]

    anni_pensionati = np.arange(cfg.eta_pensione, cfg.eta_morte)
    fattore_riv = f_infl_media[idx_accesso:cfg.anni_vita] / max(float(f_infl_media[idx_accesso]), 1e-9)

    if cfg.abilita_pensione_pubblica:
        curva_grv = p_grv_med * fattore_riv
        curva_inps = np.where(anni_pensionati >= eta_inps_acc, p_inps_med * fattore_riv, 0.0)
        curva_tot_pensione = curva_grv + curva_inps
        curva_entrate_nette[idx_accesso:] = curva_tot_pensione

    fig_spese, ax_spese = plt.subplots(figsize=(16, 7))

    ax_spese.plot(anni_x_vita, spese_medie, label="Spese Medie di Vita (con Retirement Smile)", color="#2b5c8f", linewidth=2.8)

    if anno_sel_pensione > 0:
        anni_lav_x = anni_x_vita[:anno_sel_pensione]
        ax_spese.plot(
            anni_lav_x,
            curva_entrate_nette[:anno_sel_pensione],
            label=f"Guadagni Netti da Lavoro (Stipendio fino a {eta_fire}a)",
            color="#27ae60",
            linewidth=2.8
        )
        ax_spese.fill_between(
            anni_lav_x,
            spese_medie[:anno_sel_pensione],
            curva_entrate_nette[:anno_sel_pensione],
            where=(curva_entrate_nette[:anno_sel_pensione] >= spese_medie[:anno_sel_pensione]),
            color="#2ecc71",
            alpha=0.20,
            label="Capacità di Risparmio Investita (Stipendio > Spese)"
        )

    ax_spese.axvline(
        eta_fire,
        color="#c0392b",
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
        label=f"Inizio FIRE (Età {eta_fire}a: stop guadagni da lavoro)"
    )

    if eta_fire < cfg.eta_pensione:
        anni_fire_gap = np.arange(eta_fire, cfg.eta_pensione)
        ax_spese.plot(
            anni_fire_gap,
            np.zeros(len(anni_fire_gap)),
            color="#7f8c8d",
            linestyle=":",
            linewidth=2.2,
            label=f"Fase FIRE senza Pensione (0 € entrate, {anni_gap_fire} anni)"
        )
        ax_spese.fill_between(
            anni_fire_gap,
            0,
            spese_medie[anno_sel_pensione:idx_accesso],
            color="#e74c3c",
            alpha=0.15,
            label="Fabbisogno FIRE a carico del Portafoglio/Buffer"
        )

    if cfg.abilita_pensione_pubblica:
        ax_spese.plot(
            anni_pensionati,
            curva_tot_pensione,
            "-.",
            label=f"Ripresa Entrate: Pensione Pubblica Totale UE (Rivalutata dai {min(eta_grv_acc, eta_inps_acc)}a)",
            color="#8e44ad",
            linewidth=2.8
        )
        ax_spese.plot(anni_pensionati, curva_grv, "--", label=f"• di cui Quota Tedesca GRV (dai {eta_grv_acc}a)", color="#e67e22", linewidth=1.6)
        if p_inps_med > 0:
            ax_spese.plot(anni_pensionati, curva_inps, ":", label=f"• di cui Quota Italiana INPS (dai {eta_inps_acc}a)", color="#16a085", linewidth=1.6)

        ax_spese.fill_between(
            anni_pensionati,
            curva_tot_pensione,
            spese_medie[idx_accesso:],
            where=(spese_medie[idx_accesso:] > curva_tot_pensione),
            color="#e67e22",
            alpha=0.12,
            label="Quota Spese oltre la Pensione (a carico Portafoglio)"
        )

    ax_spese.set_title(
        f"Flusso Entrate Nette (Stipendio -> Stop FIRE a {eta_fire}a -> Ripresa Pensione UE) vs Spese di Vita ({cfg.paese})",
        fontsize=13,
        pad=12
    )
    ax_spese.set_xlabel("Età", fontsize=11)
    ax_spese.set_ylabel("Euro nominali annui (€)", fontsize=11)
    ax_spese.grid(True, linestyle="--", alpha=0.5)
    ax_spese.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, framealpha=0.95, fontsize=9.5)
    fig_spese.tight_layout()
    st.pyplot(fig_spese)
    plt.close(fig_spese)

    # ----------------- SECONDO GRAFICO: TASSE E SANITÀ DISAGGREGATE (VETTORIZZATO) -----------------
    st.markdown("---")
    st.markdown(f"### 🧾 Dettaglio Trattenute, Fisco e Sanità nel Tempo ({cfg.paese})")
    st.caption(
        "Ripartizione delle uscite fiscali e contributive per singola voce durante la carriera lavorativa, "
        "la fase FIRE (dove restano a tuo carico gli oneri sanitari) e la fase di pensionamento pubblico."
    )

    n_anni_vita = len(anni_x_vita)
    serie_sanita = np.zeros(n_anni_vita)
    serie_rv_ivs = np.zeros(n_anni_vita)
    serie_av = np.zeros(n_anni_vita)
    serie_irpef_lohn = np.zeros(n_anni_vita)
    serie_soli_addiz = np.zeros(n_anni_vita)

    ral_medie = np.mean(risultati["ral_infl"], axis=1)
    f_infl_vita = f_infl_media[:n_anni_vita]

    # Fase 1: Carriera lavorativa dipendente (vettorizzata)
    if anno_sel_pensione > 0:
        ral_lav = ral_medie[:anno_sel_pensione]
        f_infl_lav = f_infl_vita[:anno_sel_pensione]
        if is_germania:
            det = calcola_dettaglio_tasse_germania(
                ral=ral_lav,
                fattore_inflazione=f_infl_lav,
                tipo_sanita=cfg.tipo_sanita_germania,
                costo_mensile_pkv_base=cfg.costo_mensile_pkv_iniziale,
                anno=np.arange(anno_sel_pensione),
                inflazione_medica_extra=cfg.inflazione_medica_extra_annua,
                **parametri_contributivi_config(cfg)
            )
            serie_sanita[:anno_sel_pensione] = det["sanita"]
            serie_rv_ivs[:anno_sel_pensione] = det["rv"]
            serie_av[:anno_sel_pensione] = det["av"]
            serie_irpef_lohn[:anno_sel_pensione] = det["lohnsteuer"]
            serie_soli_addiz[:anno_sel_pensione] = det["soli"]
        else:
            det_it = calcola_dettaglio_tasse_italia(ral_lav, f_infl_lav)
            serie_rv_ivs[:anno_sel_pensione] = det_it["ivs"]
            serie_irpef_lohn[:anno_sel_pensione] = det_it["irpef"]
            serie_soli_addiz[:anno_sel_pensione] = det_it["addizionali"]

    # Fase 2: Decumulo FIRE prima dei 67 anni (vettorizzata)
    if is_germania and anno_sel_pensione < idx_accesso:
        t_fire_slice = np.arange(anno_sel_pensione, idx_accesso)
        f_infl_fire = f_infl_vita[anno_sel_pensione:idx_accesso]
        if cfg.tipo_sanita_germania.upper() == "PKV":
            serie_sanita[anno_sel_pensione:idx_accesso] = (
                cfg.costo_mensile_pkv_fire_nominale * 12.0 * f_infl_fire
                * ((1.0 + cfg.inflazione_medica_extra_annua) ** t_fire_slice)
            )
        else:
            serie_sanita[anno_sel_pensione:idx_accesso] = cfg.costo_mensile_gkv_fire_nominale * 12.0 * f_infl_fire

    # Fase 3: Pensione Pubblica Canonica dai 67 anni (vettorizzata)
    t_pens_slice = np.arange(idx_accesso, n_anni_vita)
    f_infl_pens = f_infl_vita[idx_accesso:]
    fatt_riv_pens = fattore_riv if cfg.abilita_pensione_pubblica else f_infl_pens

    if is_germania:
        p_lorda_raw = p_dettaglio.get("pensione_lorda_grv", p_grv_med * 1.15)
        p_lorda_base = float(np.mean(p_lorda_raw)) if isinstance(p_lorda_raw, np.ndarray) else float(p_lorda_raw)
        p_lorda_vec = p_lorda_base * fatt_riv_pens

        if cfg.tipo_sanita_germania.upper() == "PKV":
            premio_pkv_vec = (
                cfg.costo_mensile_pkv_fire_nominale * 12.0 * f_infl_pens
                * ((1.0 + cfg.inflazione_medica_extra_annua) ** t_pens_slice)
            )
            zuschuss_vec = p_lorda_vec * 0.0815
            serie_sanita[idx_accesso:] = np.maximum(0.0, premio_pkv_vec - zuschuss_vec)
            serie_irpef_lohn[idx_accesso:] = np.maximum(0.0, (p_lorda_vec - 11_784.0 * f_infl_pens) * 0.22)
        else:
            quota_kvdr_vec = p_lorda_vec * 0.115
            serie_sanita[idx_accesso:] = quota_kvdr_vec
            serie_irpef_lohn[idx_accesso:] = np.maximum(0.0, (p_lorda_vec - quota_kvdr_vec - 11_784.0 * f_infl_pens) * 0.22)
    else:
        serie_irpef_lohn[idx_accesso:] = (p_inps_med * 1.23 * fatt_riv_pens) * 0.23

    fig_tax, ax_tax = plt.subplots(figsize=(16, 7))
    width = 0.72

    if is_germania:
        lbl_san = f"Assicurazione Sanitaria ({'PKV Privata' if cfg.tipo_sanita_germania.upper() == 'PKV' else 'GKV Pubblica + Pflege'})"
        ax_tax.bar(anni_x_vita, serie_sanita, width, label=lbl_san, color="#27ae60", alpha=0.92)
        bottom = serie_sanita.copy()

        ax_tax.bar(anni_x_vita, serie_rv_ivs, width, bottom=bottom, label="Pensione Pubblica (Rentenversicherung RV 9.3%)", color="#e67e22", alpha=0.90)
        bottom += serie_rv_ivs

        ax_tax.bar(anni_x_vita, serie_av, width, bottom=bottom, label="Disoccupazione (Arbeitslosenversicherung AV 1.3%)", color="#f39c12", alpha=0.90)
        bottom += serie_av

        ax_tax.bar(anni_x_vita, serie_irpef_lohn, width, bottom=bottom, label="Imposta sul Reddito (Lohnsteuer / Einkommensteuer)", color="#2980b9", alpha=0.90)
        bottom += serie_irpef_lohn

        ax_tax.bar(anni_x_vita, serie_soli_addiz, width, bottom=bottom, label="Solidaritätszuschlag (Soli 5.5%)", color="#8e44ad", alpha=0.90)
    else:
        ax_tax.bar(anni_x_vita, serie_rv_ivs, width, label="Contributi Previdenziali IVS (9.19%)", color="#e67e22", alpha=0.92)
        bottom = serie_rv_ivs.copy()

        ax_tax.bar(anni_x_vita, serie_irpef_lohn, width, bottom=bottom, label="IRPEF Progressiva", color="#2980b9", alpha=0.90)
        bottom += serie_irpef_lohn

        ax_tax.bar(anni_x_vita, serie_soli_addiz, width, bottom=bottom, label="Addizionali Regionali e Comunali (~2%)", color="#8e44ad", alpha=0.90)

    ax_tax.axvline(
        eta_fire,
        color="#c0392b",
        linestyle="--",
        linewidth=2.0,
        alpha=0.9,
        label=f"Inizio FIRE (Età {eta_fire}a: stop contributi lavoro)"
    )

    if eta_fire < cfg.eta_pensione:
        ax_tax.axvline(
            cfg.eta_pensione,
            color="#2c3e50",
            linestyle=":",
            linewidth=1.8,
            alpha=0.8,
            label=f"Età Pensione Canonica ({cfg.eta_pensione}a)"
        )

    ax_tax.set_title(
        f"Composizione Uscite: Tasse, Contributi e Spesa Sanitaria ({'Germania' if is_germania else 'Italia'}) — Ritiro a {eta_fire}a",
        fontsize=13,
        pad=12
    )
    ax_tax.set_xlabel("Età", fontsize=11)
    ax_tax.set_ylabel("Euro nominali annui (€)", fontsize=11)
    ax_tax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_tax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, framealpha=0.95, fontsize=9.5)
    fig_tax.tight_layout()
    st.pyplot(fig_tax)
    plt.close(fig_tax)

with tab4:
    st.markdown("### 🔬 Analisi di Sensitività Strategica: Asset Allocation Pre/Post FIRE, Sanità e Bond Tent")
    st.markdown(
        "Questa analisi effettua una scansione sistematica incrociata per identificare la strategia ottimale che ti consente "
        "di raggiungere il **FIRE il prima possibile** garantendo **almeno il 99% di probabilità di successo fino a 100 anni** "
        "e la massima resilienza contro il peggior scenario (1° percentile e crolli storici reali).\n\n"
        "Tutti i tuoi parametri personali (RAL, spese, capitale di partenza, carriera pregressa IT/DE) restano fissati, "
        "mentre vengono variati simultaneamente:\n"
        "- **Allocazione Azionaria Pre-FIRE** (Fase di Accumulo: da 50% a 90%)\n"
        "- **Allocazione Azionaria Post-FIRE** (Fase di Decumulo: da 30% a 70%, con vincolo Post ≤ Pre)\n"
        "- **Glidepath Dinamico (Bond Tent)**: Attivo vs Disattivato\n"
        + ("- **Regime Sanitario**: Pubblica (GKV) vs Privata (PKV)" if is_germania else "")
    )

    df_sens = calcola_sensitivita_cached(cfg, m_eq, md_eq, md_bond, inflazione, eta_fire_sel)

    # Controlli interattivi per la vista della Tabella e della Heatmap
    col_s1, col_s2, col_s3 = st.columns([1, 1, 1.2])
    with col_s1:
        if is_germania:
            filtro_sanita = st.selectbox(
                "Regime Sanitario da Esaminare:",
                options=["GKV (Pubblica)", "PKV (Privata)"],
                index=0 if cfg.tipo_sanita_germania.upper() == "GKV" else 1
            )
            sanita_val = "GKV" if "GKV" in filtro_sanita else "PKV"
        else:
            sanita_val = cfg.tipo_sanita_germania
            st.info("Sanità: SSN Italia")

    with col_s2:
        filtro_bt = st.selectbox(
            "Glidepath Dinamico (Bond Tent):",
            options=["Sì (Attivo)", "No (Costante)"],
            index=0 if cfg.usa_bond_tent else 1
        )
        bt_val = "Sì" if "Sì" in filtro_bt else "No"

    with col_s3:
        st.caption(
            "🎯 **Criteri di Decisione Ottimale**:\n"
            "- **Età Min 99%**: Età minima di pensionamento FIRE con successo ≥ 99% a 100 anni.\n"
            f"- **Worst-Case 100a**: Capitale residuo al 1° percentile all'età di 100 anni con ritiro a {eta_fire_sel}a."
        )

    # 1. Heatmap 2D
    st.markdown("#### 🗺️ Matrice 2D: Età Minima di Ritiro con Successo ≥ 99% fino a 100 Anni")
    st.caption("Ogni cella indica l'età più giovane in cui puoi ritirarti raggiungendo il 99% di successo a fine vita (100 anni).")
    fig_sens = render_fig_heatmap_sensitivita(df_sens, sanita_sel=sanita_val, bond_tent_sel=bt_val)
    st.pyplot(fig_sens)
    plt.close(fig_sens)

    # 2. Ranking delle Strategie
    st.markdown("---")
    st.markdown(f"#### 🏆 Classifica delle Migliori Strategie per Worst-Case Survival (Ritiro a {eta_fire_sel}a)")
    st.caption("Ordinamento: 1° Età Minima di FIRE al 99%, 2° Tasso di Successo, 3° Capitale al 1° Percentile peggiore.")

    sub_df = df_sens[df_sens["Bond Tent"] == bt_val]
    if is_germania:
        sub_df = sub_df[sub_df["Sanità"] == sanita_val]

    cols_display = [
        "Azioni Pre-FIRE", "Azioni Post-FIRE", "Bond Tent",
        "Età Min 99%", "Età Min 95%", f"Succ a {eta_fire_sel}a (%)",
        f"1° Perc 100a (Ritiro {eta_fire_sel}a)", f"Mediana 100a (Ritiro {eta_fire_sel}a)",
        "Crisi Storiche OK / 4", "Peggiore Shock Storico (€)"
    ]
    if is_germania:
        cols_display.insert(0, "Sanità")

    st.dataframe(
        sub_df[cols_display].reset_index(drop=True),
        use_container_width=True,
        height=380
    )

    if len(sub_df) > 0:
        top_row = sub_df.iloc[0]
        st.success(
            f"💡 **Strategia Consigliata per la Massima Copertura dai Worst-Case**: "
            f"Accumulo con **{top_row['Azioni Pre-FIRE']} Azioni** ➔ Decumulo FIRE con **{top_row['Azioni Post-FIRE']} Azioni** "
            f"(Bond Tent: **{top_row['Bond Tent']}** | Sanità: **{top_row.get('Sanità', 'N/D')}**).\n\n"
            f"- **Età Minima FIRE al 99%**: **{top_row['Età Min 99%']} anni**\n"
            f"- **Capitale Residuo nel Caso Peggiore (1° Perc. a 100a)**: **{top_row[f'1° Perc 100a (Ritiro {eta_fire_sel}a)']:,.0f} €**\n"
            f"- **Resistenza a Crisi Storiche (1973, 2000, 2008, 2022)**: **{top_row['Crisi Storiche OK / 4']} su 4** superate senza esaurimento del capitale."
        )

with tab5:
    st.markdown("### 📥 Riepilogo Esecutivo Anno per Anno & Esportazione Piano FIRE")
    st.markdown(
        "Esamina la tabella completa della tua traiettoria di vita con il piano selezionato ed esporta i dati in formato **CSV compatibile con Excel** o scarica il **Dossier PDF Esecutivo** professionale."
    )
    st.info(f"🎯 **Ipotesi di Ritiro Attiva**: Età **{eta_fire_sel} anni** ({anno_acc_sel} anni di lavoro/accumulo futuro da oggi). Modificabile tramite lo slider principale in alto.")

    anno_export_sel = anno_acc_sel
    eta_fire_exp = eta_fire_sel
    n_anni = cfg.anni_vita
    righe_summary = []

    p_dettaglio_exp = risultati["pensioni_dettaglio_list"][anno_export_sel]
    p_grv_base_exp = float(np.median(p_dettaglio_exp["pensione_netta_grv"]))
    p_inps_base_exp = float(np.median(p_dettaglio_exp["pensione_netta_inps"]))
    eta_inps_exp = p_dettaglio_exp["eta_accesso_inps"]
    idx_acc_exp = cfg.eta_pensione - cfg.eta_iniziale

    cap_p01_exp, cap_mediano_exp = np.percentile(
        risultati["capitale_vita"][anno_export_sel], [1, 50], axis=1
    )
    dep_medie_exp = np.mean(risultati["depositi"], axis=1)
    costo_sanita_fire_base = (
        cfg.costo_mensile_pkv_fire_nominale * 12.0
        if cfg.tipo_sanita_germania.upper() == "PKV"
        else cfg.costo_mensile_gkv_fire_nominale * 12.0
    )

    for idx_t in range(n_anni):
        eta_curr = cfg.eta_iniziale + idx_t
        f_infl_t = float(f_infl_media[idx_t])
        alloc_str = cfg.format_allocazione_anno(idx_t, anno_export_sel)

        if idx_t < anno_export_sel:
            fase = "Lavoro & Accumulo"
            ral_val = float(ral_medie[idx_t])
            dep_val = float(dep_medie_exp[idx_t])
            spesa_val = float(spese_medie[idx_t])
            netto_val = dep_val + spesa_val
            tasse_val = max(0.0, ral_val - netto_val)
            p_grv_val = 0.0
            p_inps_val = 0.0
            prelievo_val = 0.0
        elif idx_t < idx_acc_exp:
            fase = "FIRE Anticipato"
            ral_val = dep_val = netto_val = p_grv_val = p_inps_val = 0.0
            spesa_val = float(spese_medie[idx_t])
            tasse_val = float(costo_sanita_fire_base * f_infl_t) if is_germania else 0.0
            prelievo_val = spesa_val + tasse_val
        else:
            fase = "Pensione Canonica"
            ral_val = dep_val = netto_val = 0.0
            spesa_val = float(spese_medie[idx_t])
            fatt_riv = float(fattore_riv[idx_t - idx_acc_exp])
            p_grv_val = p_grv_base_exp * fatt_riv
            p_inps_val = (p_inps_base_exp * fatt_riv) if eta_curr >= eta_inps_exp else 0.0
            tasse_val = p_grv_val * 0.15
            prelievo_val = max(0.0, spesa_val - (p_grv_val + p_inps_val))

        righe_summary.append({
            "Età": eta_curr,
            "Fase": fase,
            "Allocazione (Azioni / Bond)": alloc_str,
            "RAL Lorda (€)": round(ral_val),
            "Stipendio Netto (€)": round(netto_val),
            "Tasse & Sanità (€)": round(tasse_val),
            "Risparmio Investito (€)": round(dep_val),
            "Spese Vita (€)": round(spesa_val),
            "Pensione Tedesca GRV (€)": round(p_grv_val),
            "Pensione Italiana INPS (€)": round(p_inps_val),
            "Prelievo Portafoglio (€)": round(prelievo_val),
            "Patrimonio Peggior Scenario (1° Perc.) (€)": round(float(cap_p01_exp[idx_t])),
            "Patrimonio Mediano (50° Perc.) (€)": round(float(cap_mediano_exp[idx_t]))
        })

    df_summary = pd.DataFrame(righe_summary)

    # ----------------- TASTI EXPORT IN PRIMO PIANO (SOPRA ALLA TABELLA) -----------------
    st.markdown("---")
    st.subheader("📑 Esportazione Documentale del Piano")
    st.caption("Scarica immediatamente i risultati dell'analisi con un solo clic.")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("##### 📊 Proiezione Completa Tabellare (CSV)")
        st.caption("Formato standard compatibile con Microsoft Excel e software di finanza personale (separatore ';' e codifica UTF-8 BOM).")
        csv_data = df_summary.to_csv(index=False, sep=";").encode("utf-8-sig")
        st.download_button(
            label="📥 Scarica Tabella CSV (Excel)",
            data=csv_data,
            file_name=f"piano_fire_dettagliato_ritiro_{eta_fire_exp}a.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_d2:
        st.markdown("##### 📄 Dossier Strategico FIRE (PDF Esecutivo)")
        st.caption("Dossier completo ad alta risoluzione: KPI cards, Fan Chart logaritmico, doppie Heatmap, flussi di cassa, tasse, stress test e analisi di sensitività (5 pagine).")
        df_sens_export = calcola_sensitivita_cached(cfg, m_eq, md_eq, md_bond, inflazione, eta_fire_sel)
        st.download_button(
            label="📥 Scarica Report PDF Esecutivo (5 Pagine)",
            data=lambda: genera_report_pdf_fire(cfg, risultati, anno_export_sel, tutti_stress_dict, df_sens=df_sens_export),
            file_name=f"report_esecutivo_fire_ritiro_{eta_fire_exp}a.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    # ----------------- TABELLA DETTAGLIATA (POSIZIONATA SOTTO AI TASTI DI EXPORT) -----------------
    st.markdown("---")
    st.subheader("📋 Proiezione Numerica Dettagliata Anno per Anno")
    st.caption(
        f"Ipotesi di piano attiva: Ritiro programmato ad età **{eta_fire_exp} anni**. "
        "Tutti i valori monetari sono espressi in euro nominali dell'anno di riferimento (inclusivi di inflazione cumulata)."
    )
    st.dataframe(df_summary, use_container_width=True, height=520)

with tab6:
    mostra_guida()
