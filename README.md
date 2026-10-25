# 🔥 FIRE Calculator — Simulatore Monte Carlo Avanzato (Germania 🇩🇪 & Italia 🇮🇹)

Simulatore Monte Carlo ad alta precisione in Python per la pianificazione dell'**Indipendenza Finanziaria e Ritiro Anticipato (*FIRE - Financial Independence, Retire Early*)**, progettato specificamente per la realtà fiscale, previdenziale e sanitaria di **Germania** e **Italia**.

Disponibile come **Dashboard Web interattiva (Streamlit)** sia in locale che via cloud, e come **pipeline da terminale (CLI)** con esportazione automatica di report PDF esecutivi.

---

## 🚀 Funzionalità Chiave della Modellazione

### 1. Macroeconomia Sincronizzata in EUR (Joint Vector Block Bootstrap)
- **Dataset Reale Unificato (1970–2024)** con triplette storiche reali:
  - Azionario: **MSCI World Net Total Return** o **S&P 500 Total Return** in EUR.
  - Obbligazionario: **German Bund 10-Year Total Return** (prezzo + cedole reinvestite in EUR).
  - Inflazione locale: **Destatis CPI** (Germania) o **Istat FOI** (Italia).
- **Block Bootstrap a blocchi di 3 anni**: preserva le correlazioni incrociate reali (es. crollo congiunto di stock e bond con inflazione elevata nel 1973-74 e nel 2022) e l'autocorrelazione ciclica dei bear market pluriennali.

### 2. Fiscalità Progressiva del Lavoro
- **Germania 🇩🇪**:
  - Contributi previdenziali e disoccupazione (RV $9.3\%$, AV $1.3\%$ con tetto *Beitragsbemessungsgrenze* indicizzato).
  - Imposta sul reddito (*Lohnsteuer*) con *Grundfreibetrag* indicizzato e aliquote progressive fino al $42\%$.
  - *Solidaritätszuschlag* con zona di raccordo (*Milderungszone* § 4 SolzG).
- **Italia 🇮🇹**:
  - Contributi IVS dipendente ($9.19\%$).
  - IRPEF a scaglioni ($23\%$, $35\%$, $43\%$), detrazioni da lavoro dipendente e addizionali regionali/comunali.
  - TFR dipendenti ($6.91\%$ annuo rivalutato), liquidato al FIRE con tassazione separata ($23\%$).

### 3. Totalizzazione Europea delle Pensioni (Regolamento UE 883/2004)
- I periodi lavorati in diversi paesi UE si sommano per soddisfare i requisiti minimi di accesso (*Wartezeit* 5 anni).
- **Pensione Tedesca (GRV)**: calcolata in *Entgeltpunkte* (EP) sul reddito relativo medio (*Durchschnittsentgelt*) rivalutata alla crescita reale dei salari e all'inflazione.
- **Pensione Italiana (INPS)**: montante contributivo rivalutato al tasso di capitalizzazione (PIL + inflazione) e convertito in rendita a 67 o 71 anni con i coefficienti di trasformazione ufficiali.
- Entrambe le pensioni godono di **perequazione automatica integrale all'inflazione**.

### 4. Gestione Sanitaria in Germania (PKV Privata vs GKV Pubblica)
- **Fase di lavoro**: premio a carico del dipendente con deduzione fiscale per la copertura di base (§ 10 EStG).
- **Fase FIRE pre-67 anni**: assenza del datore di lavoro e pagamento del $100\%$ del premio PKV (indicizzato con inflazione medica reale $+1\%$/anno per invecchiamento).
- **Dai 67 anni**: la DRV versa il sussidio pubblico del **$8.15\%$ della pensione lorda** (*Beitragszuschuss* § 106 SGB VI).

### 5. Glidepath della Bond Tent con Tassazione del Ribilanciamento
- **Fase pre-FIRE (ultimi 5 anni di lavoro)**: la quota azionaria scende gradualmente dall'$80\%$ al $45\%$ per neutralizzare il *Sequence of Returns Risk*.
- **Inflow Rebalancing iniziale**: i nuovi risparmi vanno prima nei bond per ridurre le imposte.
- **Tassazione del Ribilanciamento Attivo**: se i risparmi non bastano per raggiungere il target della tenda, le azioni in eccesso vengono vendute, viene calcolata la plusvalenza sul prezzo medio di carico (`carico_eq`) e liquidata l'imposta ($26.375\%$ DE con franchigia *Sparer-Pauschbetrag*, $26\%$ IT). Solo il ricavato netto acquista obbligazioni.
- **Fase post-FIRE (primi 10 anni di decumulo)**: risalita graduale verso l'equity target ($60\%$). La vendita di obbligazioni o azioni per mantenere il glidepath è soggetta a calcolo analitico delle imposte con compensazione delle minusvalenze pregresse nello **zainetto fiscale quadriennale**.

### 6. Prelievo Flessibile, Cash Buffer e Stress Test Storici
- **Retirement Smile**: curva delle spese per età ($100\%$ prima dei 55a, $88\%$ tra 55 e 75a, $105\%$ oltre i 75a).
- **Cash Buffer Dinamico**: 1–2 anni di spese liquide per non vendere asset durante mercati orso.
- **Stress Test Storici Sovrapposti**: simulazione deterministica dell'impatto sul piano di 4 shock storici reali:
  - *Bolla Dot-Com (2000)*
  - *Crisi Subprime (2008)*
  - *Stagflazione (1973)*
  - *Doppio Shock Azioni-Bond (2022)*

---

## 📁 Struttura del Progetto

```text
Fire_calculator/
├── config/
│   ├── __init__.py
│   └── settings.py          # Parametri finanziari, fiscali, demografici e previdenziali
├── src/
│   ├── __init__.py
│   ├── market_data.py       # Dati sincroni EUR, bootstrap vettoriale a blocchi, cache LRU
│   ├── inflation.py         # Modelli inflazione (reale, lognormale, fissa)
│   ├── taxes.py             # Calcolo progressivo Lohnsteuer/Soli (DE) e IRPEF/IVS (IT)
│   ├── pension.py           # Totalizzazione UE: GRV (DE) e INPS contributivo (IT)
│   ├── simulation.py        # Motore Monte Carlo: accumulo, decumulo, bond tent e minusvalenze
│   ├── visualization.py     # Fan chart in scala Symlog ed heatmap ad alto contrasto
│   └── pdf_report.py        # Generatore dossier esecutivo PDF in 4 pagine A4 Landscape
├── data/
│   └── historical_triads_eur.csv # Dataset sincronizzato 1970-2024 (Azioni, Bund 10Y TR, Inflazione)
├── output/                  # Cartella per report PDF esportati
├── app.py                   # Dashboard Web interattiva (Streamlit)
├── main.py                  # Script CLI per simulazioni e generazione PDF da terminale
├── requirements.txt         # Dipendenze Python ottimizzate
├── DOCUMENTAZIONE_MODELLAZIONE.md # Trattato teorico della modellazione e manuale utente
└── README.md
```

---

## 💻 Installazione ed Esecuzione Locale

### Prerequisiti
- Python 3.10 o superiore.

### Installazione Dipendenze
```powershell
# Creazione ambiente virtuale (se non già presente)
python -m venv .venv

# Attivazione ambiente virtuale (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Installazione librerie richieste
pip install -r requirements.txt
```

### 1. Avvio della Dashboard Web (Streamlit)
```powershell
streamlit run app.py
```
L'applicazione si aprirà automaticamente nel browser all'indirizzo `http://localhost:8501`.

### 2. Esecuzione da Riga di Comando (CLI)
Per eseguire la simulazione e generare direttamente il Report PDF esecutivo nella cartella `output/`:
```powershell
# Simulazione per la Germania:
python main.py --paese Germania --anni-acc 15

# Simulazione per l'Italia:
python main.py --paese Italia --anni-acc 15
```

---

## ☁️ Pubblicazione su Streamlit Community Cloud

Il progetto è pre-configurato per essere distribuito gratuitamente su **Streamlit Community Cloud**:

1. Carica il repository su [GitHub](https://github.com) (assicurati di includere `app.py`, `requirements.txt`, `config/`, `src/` e `data/historical_triads_eur.csv`).
2. Accedi a [share.streamlit.io](https://share.streamlit.io) con il tuo account GitHub.
3. Clicca su **"Create app"**:
   - Seleziona il tuo repository.
   - Inserisci `app.py` nel campo *Main file path*.
   - Clicca su **Deploy!**.
4. In circa 60 secondi l'applicazione sarà attiva e condivisibile tramite link pubblico o privato protetto.

---

## 📑 Struttura dell'Interfaccia Web

- **Barra Superiore Unificata**:
  - Slider unificato dell'età di ritiro FIRE (aggiorna simultaneamente tutte le tab e i report).
  - 4 KPI cards in evidenza: *Probabilità Capitale > 0 a 100 anni*, *Patrimonio Accumulato al FIRE*, *Pensione Pubblica Totale UE* e *Spesa Annua Obiettivo*.
- **Tab 1: 🎯 Heatmap (Successo & SWR)**:
  - Matrice 1: Tasso di successo (patrimonio non esaurito) fino a 100 anni per ogni età di ritiro (soglia rigorosa al $99\%$).
  - Matrice 2: Sostenibilità del Safe Withdrawal Rate (rolling 3 anni sotto il $3.14\%$).
- **Tab 2: 📈 Evoluzione Patrimonio & Stress Test**:
  - Fan Chart con fasce percentili $10^\circ–90^\circ$, $25^\circ–75^\circ$, mediana ($50^\circ$) e **1° percentile peggiore**.
  - Selettore scala: **Logaritmica (Symlog - default)** o Lineare.
  - Sovrapposizione in tempo reale dei 4 stress test storici reali.
- **Tab 3: 🏛️ Spese e Pensioni UE (GRV + INPS)**:
  - Grafico stipendio da lavoro $\to$ stop al FIRE $\to$ ripresa pensioni UE.
  - Grafico a barre impilate di trattenute, fisco, previdenza e sanità (PKV/GKV).
- **Tab 4: 📥 Riepilogo & Esportazione Piano**:
  - Download in 1 clic di **Tabella CSV (Excel)** e **Dossier PDF Esecutivo di 4 pagine**.
  - Tabella dettagliata anno per anno con colonna **Allocazione (Azioni / Bond)**.

---

## 📖 Approfondimento della Modellazione
Per un'analisi teorica completa delle formule, delle scelte metodologiche, dei limiti e delle FAQ concettuali, consulta il documento:
👉 **[DOCUMENTAZIONE_MODELLAZIONE.md](DOCUMENTAZIONE_MODELLAZIONE.md)**
