# FIRE Calculator — Simulatore Monte Carlo Avanzato per l'Indipendenza Finanziaria

Simulatore Monte Carlo in Python per la pianificazione del **FIRE (*Financial Independence, Retire Early*)** adattato alla realtà finanziaria e previdenziale italiana.

Funziona interamente in locale sia da riga di comando (CLI), sia tramite **dashboard interattiva Web (Streamlit)**, sia con **Jupyter Notebook**.

---

## 🚀 Nuove Funzionalità Avanzate Implementate

1. **Ribilanciamento Intelligente Tax-Efficient (senza vendita deliberata)**:
   - In Italia ogni vendita genera un evento tassabile con addebito immediato del **26% di capital gain**.
   - **In fase di accumulo (*Inflow Rebalancing*)**: i nuovi risparmi mensili/annuali vengono convogliati nell'asset sotto-allocato fino al target percentuale (es. 80/20), senza vendere mai quote e generando **zero imposte**.
   - **In fase di decumulo (*Outflow Rebalancing*)**: i prelievi necessari per coprire spese o ricostituire il cash buffer vengono estratti prioritariamente dall'asset che è sovrappesato. Se l'azionario subisce un crollo (*mercato orso*), i prelievi vengono effettuati dal cash buffer o dalle obbligazioni, evitando di vendere azioni ai minimi (*Sequence of Returns Risk*) e riducendo l'impatto fiscale.

2. **Previdenza Pubblica INPS (Sistema Contributivo Reale)**:
   - Modella l'accumulo del montante contributivo durante gli anni di effettivo lavoro prima del FIRE (aliquota IVS 33% su RAL, rivalutata alla crescita del PIL reale e all'inflazione).
   - All'età della pensione (default 67 anni), applica i **coefficienti di trasformazione ministeriali ufficiali** (legge Dini/Fornero) e stima l'IRPEF netta a scaglioni.
   - La pensione pubblica netta va a coprire il fabbisogno di vita, riducendo corrispondentemente il prelievo necessario dal portafoglio privato negli ultimi decenni di vita.

3. **Costi Reali e Fisco Italiano**:
   - **Imposta di Bollo**: applicata al controvalore del dossier titoli (0,20% annuo).
   - **TER degli ETF**: dedotto annualmente (default 0,18% annuo).
   - **Zainetto Fiscale**: compensazione delle plusvalenze con le minusvalenze realizzate negli ultimi 4 anni solari.

4. **Strategie Flessibili di Prelievo (Dynamic Spending)**:
   - **Retirement Smile**: curva delle spese per età (fase attiva 30-55 anni al 100%, fase tranquilla 56-74 anni al 88%, fase terza età >75 anni al 105% per salute/assistenza).
   - **Guardrails (Guyton-Klinger light)**: riduzione discrezionale del 10% sulle spese voluttuarie se il withdrawal rate supera la soglia critica del 5% o in caso di forte shock di mercato.

5. **Ottimizzazione delle Performance**:
   - Vettorializzazione del Block Bootstrap dei rendimenti (ridotto da 700.000 iterazioni Python a pochi passaggi).
   - Filtro rolling su array con `scipy.ndimage.uniform_filter1d` (calcolo istantaneo del Safe Withdrawal Rate su 10.000 scenari).

---

## 📁 Struttura del Progetto

```text
Fire_calculator/
├── config/
│   ├── __init__.py
│   └── settings.py          # Parametri finanziari, INPS, fiscalità e decumulo
├── src/
│   ├── __init__.py
│   ├── market_data.py       # Dati storici S&P500, rendimenti obbligazionari e bootstrap vettorizzato
│   ├── inflation.py         # Modelli inflazione (storica italiana, lognormale, fissa)
│   ├── pension.py           # Modello previdenziale INPS contributivo e IRPEF
│   ├── simulation.py        # Accumulo multi-asset, decumulo, ribilanciamento e minusvalenze
│   └── visualization.py     # Boxplot ed heatmap ad alta risoluzione
├── data/                    # Cache locale per matrici (.npy) e serie storiche (.csv)
├── output/                  # Grafici generati in PNG
├── notebooks/
│   ├── Fire_calculator.ipynb# Notebook interattivo aggiornato
│   └── legacy_colab_Fire_calculator.ipynb # Notebook originale Colab d'archivio
├── app.py                   # Dashboard interattiva Web (Streamlit)
├── main.py                  # Script eseguibile da riga di comando
├── requirements.txt         # Dipendenze Python
└── README.md
```

---

## 💻 Modalità d'Uso

### 1. Dashboard Web Interattiva (Streamlit)
Il modo più intuitivo e visivo per esplorare le simulazioni, modificare parametri e vedere le heatmap in tempo reale:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

### 2. Script da Riga di Comando (CLI)
Per eseguire rapidamente la simulazione ed esportare tutti i 9 grafici in `output/`:
```powershell
# Esecuzione standard con caricamento da cache:
.\.venv\Scripts\python.exe main.py

# Ricalcola da zero nuove matrici Monte Carlo:
.\.venv\Scripts\python.exe main.py --regenerate

# Salva solo i grafici in output/ senza aprire finestre a schermo:
.\.venv\Scripts\python.exe main.py --save-only
```

### 3. Jupyter Notebook
```powershell
.\.venv\Scripts\jupyter-lab.exe
```
Apri il file [`notebooks/Fire_calculator.ipynb`](notebooks/Fire_calculator.ipynb).
