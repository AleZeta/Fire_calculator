"""Modulo di visualizzazione grafica per il simulatore FIRE (Heatmap e Fan Chart)."""

from typing import Optional, Dict, Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

from config.settings import SimulationConfig

# Stile grafico pulito senza dipendenza da seaborn
plt.rcParams.update({
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "axes.edgecolor": "#cbd5e1",
    "axes.grid": True,
    "grid.color": "#cbd5e1",
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
})

CMAP_RED_GREEN = mcolors.LinearSegmentedColormap.from_list(
    "RedGreen", ["#dc2626", "#fde047", "#16a34a"]
)


def format_euro(x: float, pos=None) -> str:
    """Formatta i valori monetari sugli assi in €, k€ o M€."""
    if abs(x) >= 1e6:
        return f"{x * 1e-6:.1f} M€"
    if abs(x) >= 1e3:
        return f"{x * 1e-3:.0f} k€"
    if x == 0:
        return "0 €"
    return f"{x:.0f} €"


def disegna_heatmap_su_asse(
    ax: plt.Axes,
    matrix: np.ndarray,
    x_labels: Sequence[int],
    y_labels: Sequence[int],
    cbar_label: str,
    annot_size: float = 7.8
) -> None:
    """
    Disegna una heatmap annotata ad alto contrasto direttamente su un asse Matplotlib
    senza il sovraccarico di seaborn.
    """
    n_rows, n_cols = matrix.shape
    masked_data = np.ma.masked_invalid(matrix)

    im = ax.imshow(masked_data, cmap=CMAP_RED_GREEN, aspect="auto", vmin=0.0, vmax=100.0)
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Griglia bianca tra le celle
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#ffffff66", linestyle="-", linewidth=0.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8.0)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(y_labels, rotation=0, fontsize=8.0)

    rounded = np.rint(matrix)
    for i in range(n_rows):
        row_vals = matrix[i]
        row_rnd = rounded[i]
        for j in range(n_cols):
            val = row_vals[j]
            if not np.isnan(val):
                txt_color = "#ffffff" if (val < 32.0 or val > 85.0) else "#0f172a"
                ax.text(
                    j, i, f"{int(row_rnd[j])}",
                    ha="center", va="center",
                    color=txt_color, fontsize=annot_size
                )


def render_fig_heatmap_successo(
    success_rate_matrix: np.ndarray,
    x_labels: Sequence[int],
    y_labels: Sequence[int]
) -> plt.Figure:
    """Genera la figura Heatmap per il Tasso di Successo."""
    fig_h = max(6.8, len(y_labels) * 0.30)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    disegna_heatmap_su_asse(
        ax, success_rate_matrix, x_labels, y_labels,
        cbar_label="Tasso di Successo (%)", annot_size=7.8
    )
    ax.set_ylabel("Età Ritiro FIRE", fontsize=10.5, labelpad=8)
    ax.set_xlabel("Età Raggiunta (Orizzonte)", fontsize=10.5, labelpad=8)
    fig.tight_layout()
    return fig


def render_fig_heatmap_wd(
    wd_data_matrix: np.ndarray,
    x_labels: Sequence[int],
    y_labels: Sequence[int]
) -> plt.Figure:
    """Genera la figura Heatmap per la Sostenibilità del Safe Withdrawal Rate."""
    fig_h = max(6.8, len(y_labels) * 0.30)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    disegna_heatmap_su_asse(
        ax, wd_data_matrix, x_labels, y_labels,
        cbar_label="% Scenari Sostenibili", annot_size=7.5
    )
    ax.set_ylabel("Età Ritiro FIRE", fontsize=10.5, labelpad=8)
    ax.set_xlabel("Età", fontsize=10.5, labelpad=8)
    fig.tight_layout()
    return fig


def plot_fan_chart_portafoglio_fire(
    config: SimulationConfig,
    capitale_scelto: np.ndarray,
    anni_accumulo: int,
    stress_lines: Optional[Dict[str, np.ndarray]] = None,
    scala_y: str = "logaritmica"
) -> plt.Figure:
    """
    Genera un Fan Chart (fasce percentili 10-90, 25-75, mediana, 1° percentile)
    con supporto per scala logaritmica (symlog) o lineare e traiettorie di stress test storici.
    """
    anni_x = np.arange(config.eta_iniziale, config.eta_morte)
    eta_fire = config.eta_iniziale + anni_accumulo

    p01, p10, p25, p50, p75, p90 = np.percentile(
        capitale_scelto, [1, 10, 25, 50, 75, 90], axis=1
    )

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.fill_between(anni_x, p10, p90, color="#38bdf8", alpha=0.18, label="Intervallo 10°–90° Percentile (80% degli scenari)")
    ax.fill_between(anni_x, p25, p75, color="#0284c7", alpha=0.30, label="Intervallo Interquartile 25°–75° Percentile")
    ax.plot(anni_x, p50, color="#0369a1", linewidth=3.0, label=f"Mediana Monte Carlo (50°) — Valore fine vita: {p50[-1]:,.0f} €")
    ax.plot(anni_x, p01, color="#ef4444", linestyle="--", linewidth=2.0, label=f"Peggior Scenario 1° Percentile (Stress estremo) — Fine vita: {p01[-1]:,.0f} €")

    colori_stress = {
        "Dot-Com 2000": "#ea580c",
        "Crisi 2008": "#9333ea",
        "Stagflazione 1973": "#059669",
        "Crollo 2022": "#ca8a04"
    }
    if stress_lines:
        for nome_scen, traiettoria in stress_lines.items():
            c = colori_stress.get(nome_scen, "#d97706")
            ax.plot(anni_x, traiettoria, linestyle=":", linewidth=2.2, color=c, label=f"Storico Reale: {nome_scen}")

    ax.axvline(eta_fire, color="#dc2626", linestyle="--", linewidth=1.8, alpha=0.9, label=f"Inizio FIRE (Età {eta_fire}a)")
    ax.axvline(config.eta_pensione, color="#334155", linestyle=":", linewidth=1.8, alpha=0.8, label=f"Pensione Canonica ({config.eta_pensione}a)")
    ax.axhline(0, color="#64748b", linestyle="-", linewidth=1.0, alpha=0.7)

    val_max_display = float(np.percentile(p90, 95))
    if stress_lines:
        for t in stress_lines.values():
            val_max_display = max(val_max_display, float(np.max(t)))

    top_y = max(1.5e6, val_max_display * 1.15)
    if "logaritmica" in scala_y.lower() or "symlog" in scala_y.lower():
        ax.set_yscale("symlog", linthresh=50_000, linscale=0.5)
        ax.set_ylim(bottom=0, top=top_y)
        ticks_symlog = [0, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000]
        ax.set_yticks([t for t in ticks_symlog if t <= top_y])
        subtitle_scala = "(Scala Logaritmica Symlog — Ideale per apprezzare sia i primi anni che il decumulo)"
    else:
        ax.set_ylim(bottom=0, top=top_y)
        subtitle_scala = "(Scala Lineare Estesa fino al 90° Percentile)"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_euro))

    glide_info = " | Bond Tent Attivo" if config.usa_bond_tent else ""
    ax.set_title(
        f"Fan Chart Evoluzione Patrimonio FIRE — Ritiro a {eta_fire}a ({anni_accumulo}a lavoro) | "
        f"Allocazione: {int(config.target_equity_fire * 100)}% Azioni / {int(config.target_bond_fire * 100)}% Bond{glide_info}\n{subtitle_scala}",
        fontsize=12,
        pad=10
    )
    ax.set_xlabel("Età (anni)", fontsize=11)
    ax.set_ylabel("Patrimonio Netto Residuo (€)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, framealpha=0.95, fontsize=9.0)
    fig.tight_layout()

    return fig


def render_fig_heatmap_sensitivita(
    df_sens: "pd.DataFrame",
    sanita_sel: str = "GKV",
    bond_tent_sel: str = "No"
) -> plt.Figure:
    """
    Genera una heatmap 2D che mostra l'età minima di FIRE per raggiungere il 99% di successo
    incrociando l'allocazione azionaria Pre-FIRE (righe) e Post-FIRE (colonne).
    """
    # Filtra per sanità e bond tent selezionati
    sub = df_sens[df_sens["Bond Tent"] == bond_tent_sel]
    if "Sanità" in df_sens.columns and sanita_sel in df_sens["Sanità"].values:
        sub = sub[sub["Sanità"] == sanita_sel]

    y_allocs = ["50%", "60%", "70%", "80%", "90%"]
    x_allocs = ["30%", "40%", "50%", "60%", "70%"]

    matrix_eta = np.full((len(y_allocs), len(x_allocs)), np.nan)

    for i, acc in enumerate(y_allocs):
        for j, fire in enumerate(x_allocs):
            match = sub[(sub["Azioni Pre-FIRE"] == acc) & (sub["Azioni Post-FIRE"] == fire)]
            if len(match) > 0:
                val = match["Età Min 99%"].values[0]
                if val < 900:
                    matrix_eta[i, j] = float(val)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap_eta = mcolors.LinearSegmentedColormap.from_list(
        "AgeGreenRed", ["#16a34a", "#84cc16", "#eab308", "#f97316", "#dc2626"]
    )

    valid_vals = matrix_eta[np.isfinite(matrix_eta)]
    min_v = float(np.min(valid_vals)) if len(valid_vals) > 0 else 45.0
    max_v = float(np.max(valid_vals)) if len(valid_vals) > 0 else 52.0
    if max_v == min_v:
        max_v += 1.0

    im = ax.imshow(matrix_eta, cmap=cmap_eta, aspect="auto", vmin=min_v, vmax=max_v)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cbar.set_label("Età Minima Ritiro FIRE (Anni)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)

    n_rows, n_cols = matrix_eta.shape
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#ffffff77", linestyle="-", linewidth=1.0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(x_allocs, fontsize=9.5)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(y_allocs, fontsize=9.5)

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix_eta[i, j]
            if np.isnan(val):
                ax.text(j, i, "-", ha="center", va="center", color="#94a3b8", fontsize=10.0)
            else:
                rel = (val - min_v) / (max_v - min_v)
                txt_col = "#ffffff" if (rel < 0.25 or rel > 0.75) else "#0f172a"
                ax.text(j, i, f"{int(val)} anni", ha="center", va="center", color=txt_col, fontsize=10.0, weight="bold")

    ax.set_title(
        f"Età Minima FIRE al 99% di Successo a 100 Anni (Sanità: {sanita_sel} | Bond Tent: {bond_tent_sel})",
        fontsize=11.5, weight="bold", pad=10, color="#0f172a"
    )
    ax.set_ylabel("Allocazione Azioni PRE-FIRE (Accumulo)", fontsize=10.0, labelpad=8)
    ax.set_xlabel("Allocazione Azioni POST-FIRE (Decumulo)", fontsize=10.0, labelpad=8)
    fig.tight_layout()

    return fig

