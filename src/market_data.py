"""Gestione dati di mercato, block bootstrap vettorizzato e rendimenti multi-asset sincroni (MSCI World / S&P 500 + Bund + Inflazione Locale)."""

from functools import lru_cache
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from config.settings import SimulationConfig


@lru_cache(maxsize=4)
def _leggi_csv_sincrono_cached(percorso_csv: str) -> pd.DataFrame:
    """Lettura memorizzata in cache del dataset storico unificato in EUR."""
    csv_file = Path(percorso_csv)
    if not csv_file.exists():
        raise FileNotFoundError(f"File dati storici sincroni non trovato: {csv_file}")
    return pd.read_csv(csv_file)


def carica_serie_emergenti_eur(config: SimulationConfig) -> pd.DataFrame:
    """Rendimenti annuali MSCI Emerging Markets Net TR in EUR (colonne: Year, msci_em_eur), forniti dall'utente."""
    csv_file = config.data_dir / "msci_em_eur.csv"
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Dati dei mercati emergenti non trovati: {csv_file}. "
            "Serve un CSV con colonne Year, msci_em_eur (rendimento annuale decimale, es. 0.12 = +12%)."
        )
    df_em = pd.read_csv(csv_file)
    if not {"Year", "msci_em_eur"}.issubset(df_em.columns):
        raise ValueError(f"{csv_file} deve contenere le colonne Year e msci_em_eur")
    return df_em[["Year", "msci_em_eur"]]


def carica_dataset_sincrono_eur(config: SimulationConfig, includi_em: bool = True) -> pd.DataFrame:
    """
    Carica il dataset storico unificato (1970 - 2024) in EUR contenente:
    - MSCI World Net Total Return in EUR
    - S&P 500 Total Return in EUR
    - 10-Year German Bund Total Return in EUR (prezzo + cedole reinvestite)
    - Inflazione Destatis (Germania)
    - Inflazione Istat FOI (Italia)
    """
    csv_file = config.data_dir / "historical_triads_eur.csv"
    df = _leggi_csv_sincrono_cached(str(csv_file.resolve()))
    if not includi_em or config.quota_azioni_emergenti <= 0.0:
        return df

    # Con i mercati emergenti il periodo utilizzato parte dal primo anno in cui esistono tutti gli indici scelti
    df = df.merge(carica_serie_emergenti_eur(config), on="Year", how="inner").reset_index(drop=True)
    col_stock = "msci_world_eur" if _usa_msci_world(config) else "sp500_eur"
    q = config.quota_azioni_emergenti
    df["azioni_eur"] = (1.0 - q) * df[col_stock] + q * df["msci_em_eur"]
    return df


def _usa_msci_world(config: SimulationConfig) -> bool:
    return config.nome_dataset_azionario.lower() in ("msci_world", "world", "urth")


def _genera_normale(
    serie_stock: np.ndarray,
    serie_bond: np.ndarray,
    serie_infl: np.ndarray,
    durata: int,
    num_sim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rendimenti annuali normali multivariati (azioni, obbligazioni, inflazione) con media, volatilità e
    correlazioni stimate sul campione storico. Niente autocorrelazione né code spesse.
    """
    dati = np.column_stack((serie_stock, serie_bond, serie_infl))
    estrazioni = np.random.multivariate_normal(dati.mean(axis=0), np.cov(dati, rowvar=False), size=(durata - 1, num_sim))
    azioni = np.maximum(estrazioni[:, :, 0], -0.95)
    obbligazioni = np.maximum(estrazioni[:, :, 1], -0.95)
    inflazione = np.maximum(estrazioni[:, :, 2], -0.02)
    return azioni, obbligazioni, inflazione


def genera_matrici_multiasset_sincrone(
    config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Esegue il Joint Vector Block Bootstrap (campionamento a blocchi sincrono) su triplette storiche reali in EUR:
    Ogni anno t estratto all'interno del blocco preleva simultaneamente:
    1. Rendimento Azionario Total Return in EUR (MSCI World Net TR o S&P 500 EUR)
    2. Rendimento Obbligazionario Governativo Total Return in EUR (Bund 10Y TR)
    3. Inflazione Locale (Destatis CPI per la Germania, Istat FOI per l'Italia)

    Preserva le vere correlazioni incrociate (es. shock 1973-74 o 2022) e l'autocorrelazione ciclica.
    """
    df = carica_dataset_sincrono_eur(config)
    durata = config.anni_vita + 1
    num_sim = config.num_simulazioni
    num_anni_blocco = config.anni_blocco_rendimenti

    if config.quota_azioni_emergenti > 0.0:
        col_stock = "azioni_eur"  # mix indice scelto + emergenti, ribilanciato ogni anno
    else:
        col_stock = "msci_world_eur" if _usa_msci_world(config) else "sp500_eur"
    serie_stock = df[col_stock].to_numpy()
    serie_bond = df["bund_10y_tr_eur"].to_numpy()

    col_infl = "inflation_germany" if config.is_germania else "inflation_italy"
    serie_infl = df[col_infl].to_numpy()

    if config.modello_mercato.lower() == "normale":
        r_eq, r_bd, r_infl = _genera_normale(serie_stock, serie_bond, serie_infl, durata, num_sim)
        md_eq = np.ones((durata, num_sim))
        md_bd = np.ones((durata, num_sim))
        infl = np.zeros((durata, num_sim))
        md_eq[1:, :], md_bd[1:, :], infl[1:, :] = 1.0 + r_eq, 1.0 + r_bd, r_infl
        return np.cumprod(md_eq, axis=0), md_eq, md_bd, infl

    n_osservazioni = len(df)
    upper_bound = max(1, n_osservazioni - num_anni_blocco)

    num_blocchi = int(np.ceil((durata - 1) / num_anni_blocco))
    indici_inizio = np.random.randint(0, upper_bound + 1, size=(num_blocchi, num_sim))
    offsets = np.arange(num_anni_blocco)[np.newaxis, :, np.newaxis]
    idx_storico = (indici_inizio[:, np.newaxis, :] + offsets).reshape(-1, num_sim)[:durata - 1, :]

    md_eq = np.ones((durata, num_sim))
    md_bd = np.ones((durata, num_sim))
    infl = np.zeros((durata, num_sim))

    md_eq[1:, :] = 1.0 + serie_stock[idx_storico]
    md_bd[1:, :] = 1.0 + serie_bond[idx_storico]
    infl[1:, :] = serie_infl[idx_storico]

    m_eq = np.cumprod(md_eq, axis=0)
    return m_eq, md_eq, md_bd, infl


def carica_serie_rendimenti(config: SimulationConfig) -> pd.Series:
    """Recupera i rendimenti storici azionari da cache locale o download on-demand."""
    ticker = config.ticker_azionario_attivo
    safe_name = ticker.replace("^", "").lower()
    cache_file = config.data_dir / f"{safe_name}_historical.csv"

    if cache_file.exists():
        try:
            df_cache = pd.read_csv(cache_file, index_col=0)
            col = pd.to_numeric(df_cache.iloc[:, 0], errors="coerce").dropna()
            rendimenti = col.pct_change(config.giorni_anno).dropna()
            if len(rendimenti) > config.giorni_anno:
                return rendimenti
        except Exception:
            pass

    try:
        import yfinance as yf  # Lazy import solo se manca il CSV locale
        start_date = "1988-01-01" if "^" in ticker else "2012-01-01"
        raw_data = yf.download(ticker, start=start_date, progress=False)
        if raw_data is not None and not raw_data.empty:
            serie_prezzi = raw_data["Close"].iloc[:, 0] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data["Close"]
            serie_prezzi = pd.to_numeric(serie_prezzi, errors="coerce").dropna()
            config.data_dir.mkdir(parents=True, exist_ok=True)
            serie_prezzi.to_frame(name="Price").to_csv(cache_file)
            rendimenti = serie_prezzi.pct_change(config.giorni_anno).dropna()
            if len(rendimenti) > 0:
                return rendimenti
    except Exception:
        pass

    return pd.Series(np.full(1000, config.rendimento_medio_fisso))


def genera_matrice_rendimenti_vettorizzata(
    df_rendimenti: pd.Series,
    config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Generatore storico disaccoppiato vettorizzato."""
    durata = config.anni_vita + 1
    num_sim = config.num_simulazioni
    num_anni_blocco = config.anni_blocco_rendimenti
    giorni_anno = config.giorni_anno
    valori_serie = df_rendimenti.to_numpy()

    upper_bound = len(valori_serie) - (num_anni_blocco * giorni_anno) - 1
    if upper_bound <= 0:
        md = 1.0 + np.random.choice(valori_serie, size=(durata, num_sim))
        md[0, :] = 1.0
        return np.cumprod(md, axis=0), md

    md = np.ones((durata, num_sim))
    num_blocchi = int(np.ceil((durata - 1) / num_anni_blocco))
    indici_inizio = np.random.randint(0, upper_bound, size=(num_blocchi, num_sim))
    offsets = (np.arange(num_anni_blocco) * giorni_anno)[np.newaxis, :, np.newaxis]
    idx_offset = (indici_inizio[:, np.newaxis, :] + offsets).reshape(-1, num_sim)[:durata - 1, :]
    md[1:, :] = 1.0 + valori_serie[idx_offset]

    return np.cumprod(md, axis=0), md


def genera_rendimenti_obbligazionari(
    config: SimulationConfig,
    inflazione: np.ndarray
) -> np.ndarray:
    """Rendimenti obbligazionari sintetici."""
    durata = config.anni_vita + 1
    num_sim = config.num_simulazioni
    shock_volatilita = np.random.normal(0.0, config.bond_volatility, size=(durata, num_sim))
    md_bond = 1.0 + inflazione + config.bond_real_return + shock_volatilita
    md_bond[0, :] = 1.0
    return md_bond
