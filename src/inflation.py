"""Modellazione dell'inflazione per la simulazione Monte Carlo."""

import numpy as np
from config.settings import SimulationConfig

# Serie storica dell'inflazione italiana (1954 - 2022) in percentuale
INFLAZIONE_STORICA_ITALIANA = np.array([
    2.3, 3.4, 1.3, 2.8, -0.4, 2.3, 2.1, 4.7, 7.5, 5.9, 4.6, 2.3, 3.7, 1.4, 2.6,
    5.0, 4.8, 5.7, 10.8, 19.1, 17.0, 16.8, 17.0, 12.1, 14.8, 21.2, 17.8, 16.5,
    14.7, 10.8, 9.2, 5.8, 4.8, 5.0, 6.3, 6.5, 6.2, 5.3, 4.7, 4.1, 5.3, 4.0, 2.0,
    2.0, 1.7, 2.5, 2.7, 2.5, 2.7, 2.2, 1.9, 2.1, 1.8, 3.3, 0.8, 1.5, 2.7, 3.0,
    1.2, 0.2, 0.1, -0.1, 1.2, 1.2, 0.6, -0.2, 1.9, 8.1, 8.7
]) / 100.0


def genera_inflazione(config: SimulationConfig) -> np.ndarray:
    """
    Genera la matrice di inflazione per la durata della vita simulata.

    Parametri:
        config: istanza di SimulationConfig con orizzonte temporale e numero di scenari.

    Ritorna:
        np.ndarray di forma (anni_vita + 1, num_simulazioni).
    """
    durata = config.anni_vita + 1
    num_sim = config.num_simulazioni
    scenario = config.scenario_inflazione.lower()
    media_target = config.inflazione_media

    if scenario == "fissa":
        return np.full((durata, num_sim), media_target)

    if scenario == "reale":
        indices = np.random.randint(0, len(INFLAZIONE_STORICA_ITALIANA), (durata, num_sim))
        return INFLAZIONE_STORICA_ITALIANA[indices]

    if scenario == "reale riscalata":
        rapporto_scala = media_target / INFLAZIONE_STORICA_ITALIANA.mean()
        serie_riscalata = INFLAZIONE_STORICA_ITALIANA * rapporto_scala
        indices = np.random.randint(0, len(serie_riscalata), (durata, num_sim))
        return serie_riscalata[indices]

    if scenario == "lognormale":
        var_storica = INFLAZIONE_STORICA_ITALIANA.var()
        sigma = np.sqrt(np.log(1.0 + var_storica / (media_target ** 2)))
        mu = np.log(media_target) - (sigma ** 2) / 2.0
        return np.random.lognormal(mean=mu, sigma=sigma, size=(durata, num_sim))

    return np.full((durata, num_sim), media_target)
