"""Modulo per la generazione di report PDF esecutivi completi per il simulatore FIRE."""

import io
import textwrap
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

from config.settings import SimulationConfig
from src.visualization import disegna_heatmap_su_asse, format_euro


def _aggiungi_banner_intestazione(
    fig: plt.Figure,
    rect: List[float],
    titolo: str,
    sottotitolo: Optional[str] = None
) -> None:
    """Crea il banner di intestazione scuro per le pagine del report PDF."""
    ax_head = fig.add_axes(rect)
    ax_head.set_facecolor("#0f172a")
    ax_head.set_xticks([])
    ax_head.set_yticks([])
    for spine in ax_head.spines.values():
        spine.set_visible(False)

    if sottotitolo:
        ax_head.text(0.02, 0.60, titolo, color="#ffffff", fontsize=15, weight="bold", va="center")
        ax_head.text(0.02, 0.22, sottotitolo, color="#94a3b8", fontsize=9.2, va="center")
    else:
        ax_head.text(0.02, 0.50, titolo, color="#ffffff", fontsize=13, weight="bold", va="center")


def genera_report_pdf_fire(
    config: SimulationConfig,
    risultati: Dict[str, Any],
    anno_acc_sel: int,
    stress_dict: Optional[Dict[str, np.ndarray]] = None
) -> bytes:
    """
    Genera un report esecutivo PDF multi-pagina (formato A4 Landscape) con grafica
    professionale ad alta risoluzione, KPI cards, Fan Chart logaritmico, Matrici Heatmap,
    Flussi di cassa, disaggregazione fiscale/sanitaria e tabella dettagliata.
    """
    buffer = io.BytesIO()

    eta_fire = config.eta_iniziale + anno_acc_sel
    anni_x = np.arange(config.eta_iniziale, config.eta_morte)
    cap_matrice = risultati["capitale_vita"][anno_acc_sel]

    p01, p10, p25, p50, p75, p90 = np.percentile(
        cap_matrice, [1, 10, 25, 50, 75, 90], axis=1
    )

    perc_pos = risultati["percentuali_positivi"]
    p_det = risultati["pensioni_dettaglio_list"][anno_acc_sel]
    p_grv_med = float(np.median(p_det["pensione_netta_grv"]))
    p_inps_med = float(np.median(p_det["pensione_netta_inps"]))
    p_tot_med = p_grv_med + p_inps_med

    with PdfPages(buffer) as pdf:
        # =========================================================================
        # PAGINA 1: DASHBOARD ESECUTIVA, KPI CARDS & FAN CHART LOGARITMICO
        # =========================================================================
        fig1 = plt.figure(figsize=(11.69, 8.27), dpi=150)
        fig1.patch.set_facecolor("#f8fafc")

        sub_info = (
            f"Quadro: {config.paese} | Asset Allocation: {int(config.target_equity_accumulo * 100)}% Eq lavoro -> "
            f"{int(config.target_equity_fire * 100)}% Eq FIRE | Glidepath Bond Tent: {'Attivo (default)' if config.usa_bond_tent else 'Disattivo'} | "
            f"Ipotesi Ritiro FIRE: Età {eta_fire} anni ({anno_acc_sel}a accumulo futuro)"
        )
        _aggiungi_banner_intestazione(
            fig1, [0.04, 0.88, 0.92, 0.09],
            "REPORT ESECUTIVO — PIANO DI INDIPENDENZA FINANZIARIA (FIRE)",
            sub_info
        )

        prob_100_val = perc_pos[anno_acc_sel, config.anni_vita - 1]
        kpi_data = [
            ("Probabilità Successo a 100a", f"{prob_100_val:.1f}%", f"Capitale > 0 con ritiro a {eta_fire}a", "#0284c7"),
            ("Patrimonio FIRE Mediano", f"{p50[anno_acc_sel]:,.0f} €", f"Cap. iniziale: {config.capitale_iniziale:,.0f} €", "#0d9488"),
            ("Pensione Pubblica UE a 67a", f"{p_tot_med:,.0f} €/anno", f"GRV: {p_grv_med:,.0f} € + INPS: {p_inps_med:,.0f} €", "#7c3aed"),
            ("Spesa Annua Obiettivo", f"{config.spese_mese * 12:,.0f} €/anno", f"{config.spese_mese:,.0f} €/mese (SWR: {config.safe_wd_rate * 100:.2f}%)", "#b45309")
        ]

        card_w = 0.215
        gap = 0.02
        start_x = 0.04
        for idx, (title, val, note, col_accent) in enumerate(kpi_data):
            c_x = start_x + idx * (card_w + gap)
            ax_c = fig1.add_axes([c_x, 0.73, card_w, 0.12])
            ax_c.set_facecolor("#ffffff")
            for sp in ax_c.spines.values():
                sp.set_color("#cbd5e1")
                sp.set_linewidth(1.0)
            ax_c.set_xticks([])
            ax_c.set_yticks([])

            ax_c.axhline(0.98, color=col_accent, linewidth=3.5)
            ax_c.text(0.06, 0.72, title.upper(), fontsize=7.5, weight="bold", color="#64748b", va="center")
            ax_c.text(0.06, 0.40, val, fontsize=12.5, weight="bold", color="#0f172a", va="center")
            ax_c.text(0.06, 0.16, note, fontsize=7.5, color="#64748b", va="center")

        # Main Chart: Fan Chart in Scala Symlog
        ax_main = fig1.add_axes([0.08, 0.10, 0.68, 0.58])
        ax_main.set_facecolor("#ffffff")
        for sp in ax_main.spines.values():
            sp.set_color("#cbd5e1")

        ax_main.fill_between(anni_x, p10, p90, color="#38bdf8", alpha=0.18, label="Intervallo 10°–90° Perc. (80% scenari)")
        ax_main.fill_between(anni_x, p25, p75, color="#0284c7", alpha=0.30, label="Intervallo Interquartile 25°–75°")
        ax_main.plot(anni_x, p50, color="#0369a1", linewidth=3.0, label=f"Mediana (50°) | 100a: {p50[-1]:,.0f} €")
        ax_main.plot(anni_x, p01, color="#ef4444", linestyle="--", linewidth=2.0, label=f"1° Perc. Peggiore | 100a: {p01[-1]:,.0f} €")

        if stress_dict:
            col_stress = {"Dot-Com 2000": "#ea580c", "Crisi 2008": "#9333ea", "Stagflazione 1973": "#059669", "Crollo 2022": "#ca8a04"}
            for s_name, s_traj in stress_dict.items():
                c = col_stress.get(s_name, "#d97706")
                ax_main.plot(anni_x, s_traj, linestyle=":", linewidth=2.2, color=c, label=f"Storico: {s_name}")

        ax_main.axvline(eta_fire, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Inizio FIRE (Età {eta_fire}a)")
        ax_main.axvline(config.eta_pensione, color="#334155", linestyle=":", linewidth=1.8, label=f"Pensione ({config.eta_pensione}a)")

        max_val_90 = float(np.percentile(p90, 95))
        top_y = max(1.5e6, max_val_90 * 1.15)
        ax_main.set_yscale("symlog", linthresh=50_000, linscale=0.5)
        ax_main.set_ylim(bottom=0, top=top_y)

        ticks_symlog = [0, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 25_000_000]
        ax_main.set_yticks([t for t in ticks_symlog if t <= top_y])
        ax_main.yaxis.set_major_formatter(ticker.FuncFormatter(format_euro))

        ax_main.set_title(
            f"Fan Chart Evoluzione Patrimonio FIRE (Scala Logaritmica Symlog) — Ritiro programmato a {eta_fire} anni",
            fontsize=12, weight="bold", color="#1e293b", pad=10
        )
        ax_main.set_xlabel("Età (anni)", fontsize=10, color="#334155")
        ax_main.set_ylabel("Patrimonio Netto Residuo (€)", fontsize=10, color="#334155")
        ax_main.grid(True, which="both", linestyle="--", alpha=0.45)
        ax_main.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.95, fontsize=7.8)

        pdf.savefig(fig1)
        plt.close(fig1)

        # =========================================================================
        # PAGINA 2: MATRICI HEATMAP (TASSO DI SUCCESSO & SOSTENIBILITÀ SWR)
        # =========================================================================
        fig2 = plt.figure(figsize=(11.69, 8.27), dpi=150)
        fig2.patch.set_facecolor("#f8fafc")

        _aggiungi_banner_intestazione(
            fig2, [0.04, 0.91, 0.92, 0.06],
            "MATRICI DI RISCHIO E SOSTENIBILITÀ MONTE CARLO (HEATMAP)"
        )

        eta_min_col = config.eta_pensione
        idx_col_start = eta_min_col - config.eta_iniziale
        y_lbls = [config.eta_iniziale + i for i in range(config.anni_acc)]
        x_lbls = [i for i in range(eta_min_col, config.eta_morte)]

        # Heatmap 1: Tasso di Successo
        ax_h1 = fig2.add_axes([0.08, 0.52, 0.87, 0.34])
        succ_matrix = np.round(perc_pos[:, idx_col_start:]).astype(float)
        disegna_heatmap_su_asse(
            ax_h1, succ_matrix, x_lbls, y_lbls,
            cbar_label="Tasso di Successo (%)", annot_size=6.8
        )
        ax_h1.set_title("1. Tasso di Successo Portafoglio (Capitale > 0 fino all'età target)", fontsize=10.5, weight="bold", color="#1e293b", pad=6)
        ax_h1.set_ylabel("Età Ritiro FIRE", fontsize=8.5)
        ax_h1.set_xlabel("Età Raggiunta", fontsize=8.5)

        # Heatmap 2: Sostenibilità SWR
        ax_h2 = fig2.add_axes([0.08, 0.08, 0.87, 0.34])
        wd_sub = risultati["percentuali_wd_rate_roll"][:, idx_col_start:]
        disegna_heatmap_su_asse(
            ax_h2, wd_sub, x_lbls, y_lbls,
            cbar_label="% Scenari Sostenibili", annot_size=6.8
        )
        ax_h2.set_title(f"2. Sostenibilità Safe Withdrawal Rate (Prelievo rolling 3a < {config.safe_wd_rate * 100:.2f}%)", fontsize=10.5, weight="bold", color="#1e293b", pad=6)
        ax_h2.set_ylabel("Età Ritiro FIRE", fontsize=8.5)
        ax_h2.set_xlabel("Età", fontsize=8.5)

        pdf.savefig(fig2)
        plt.close(fig2)

        # =========================================================================
        # PAGINA 3: FLUSSI DI CASSA & FISCO / PREVIDENZA / STRESS TEST COMPARISON
        # =========================================================================
        fig3 = plt.figure(figsize=(11.69, 8.27), dpi=150)
        fig3.patch.set_facecolor("#f8fafc")

        _aggiungi_banner_intestazione(
            fig3, [0.04, 0.91, 0.92, 0.06],
            "FLUSSI DI CASSA, PREVIDENZA PUBBLICA & DETTAGLIO FISCALE"
        )

        # Grafico Sinistra: Entrate Nette vs Spese
        ax_f1 = fig3.add_axes([0.08, 0.49, 0.41, 0.33])
        ax_f1.set_facecolor("#ffffff")
        for sp in ax_f1.spines.values():
            sp.set_color("#cbd5e1")

        spese_medie = np.mean(risultati["spese"], axis=1)
        stipendio_netto = np.mean(risultati["depositi"] + risultati["spese"][:config.anni_acc, :], axis=1)

        ax_f1.plot(anni_x, spese_medie, label="Spese di Vita (Retirement Smile)", color="#0284c7", linewidth=2.5)
        if anno_acc_sel > 0:
            anni_lav = min(anno_acc_sel, len(stipendio_netto))
            ax_f1.plot(anni_x[:anni_lav], stipendio_netto[:anni_lav], label=f"Stipendio Netto (fino a {eta_fire}a)", color="#16a34a", linewidth=2.5)

        if config.abilita_pensione_pubblica:
            anni_pens = np.arange(config.eta_pensione, config.eta_morte)
            f_riv = (1.0 + config.inflazione_media) ** np.arange(len(anni_pens))
            ax_f1.plot(anni_pens, p_tot_med * f_riv, "-.", label="Pensione Pubblica Totale UE", color="#9333ea", linewidth=2.5)

        ax_f1.axvline(eta_fire, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Stop Lavoro ({eta_fire}a)")
        ax_f1.set_title("Flusso Entrate Nette vs Spese di Vita", fontsize=10.5, weight="bold", color="#1e293b", pad=8)
        ax_f1.set_xlabel("Età", fontsize=8.5)
        ax_f1.set_ylabel("Euro nominali annui (€)", fontsize=8.5)
        ax_f1.yaxis.set_major_formatter(ticker.FuncFormatter(format_euro))
        ax_f1.grid(True, linestyle="--", alpha=0.45)
        ax_f1.legend(loc="upper left", fontsize=7.2, framealpha=0.92)

        # Grafico Destra: Composizione Tasse e Sanità
        ax_f2 = fig3.add_axes([0.56, 0.49, 0.40, 0.33])
        ax_f2.set_facecolor("#ffffff")
        for sp in ax_f2.spines.values():
            sp.set_color("#cbd5e1")

        n_anni_v = len(anni_x)
        sanita_v = np.zeros(n_anni_v)
        fisco_v = np.zeros(n_anni_v)
        prev_v = np.zeros(n_anni_v)

        ral_medie = np.mean(risultati["ral_infl"], axis=1)
        idx_pens_canonica = config.eta_pensione - config.eta_iniziale
        is_pkv = config.tipo_sanita_germania.upper() == "PKV"

        if anno_acc_sel > 0:
            ral_sub = ral_medie[:anno_acc_sel]
            prev_v[:anno_acc_sel] = ral_sub * (0.093 if config.is_germania else 0.0919)
            sanita_v[:anno_acc_sel] = (config.costo_mensile_pkv_iniziale * 12.0) if is_pkv else (ral_sub * 0.08)
            fisco_v[:anno_acc_sel] = np.maximum(0.0, ral_sub * 0.22)

        if anno_acc_sel < idx_pens_canonica:
            sanita_v[anno_acc_sel:idx_pens_canonica] = config.costo_mensile_pkv_fire_nominale * 12.0

        sanita_v[idx_pens_canonica:] = max(0.0, config.costo_mensile_pkv_fire_nominale * 12.0 - p_grv_med * 0.0815)
        fisco_v[idx_pens_canonica:] = p_tot_med * 0.15

        width = 0.7
        ax_f2.bar(anni_x, sanita_v, width, label=f"Sanità ({config.tipo_sanita_germania})", color="#16a34a", alpha=0.88)
        ax_f2.bar(anni_x, prev_v, width, bottom=sanita_v, label="Previdenza (RV / IVS)", color="#ea580c", alpha=0.88)
        ax_f2.bar(anni_x, fisco_v, width, bottom=sanita_v + prev_v, label="Imposte Dirette (Lohnsteuer / IRPEF)", color="#0284c7", alpha=0.88)

        ax_f2.axvline(eta_fire, color="#dc2626", linestyle="--", linewidth=1.5, label="Inizio FIRE")
        ax_f2.set_title(f"Composizione Uscite: Fisco, Previdenza e Sanità ({config.paese})", fontsize=10.5, weight="bold", color="#1e293b", pad=8)
        ax_f2.set_xlabel("Età", fontsize=8.5)
        ax_f2.set_ylabel("Euro annui (€)", fontsize=8.5)
        ax_f2.yaxis.set_major_formatter(ticker.FuncFormatter(format_euro))
        ax_f2.grid(True, linestyle="--", alpha=0.45)
        ax_f2.legend(loc="upper right", fontsize=7.2, framealpha=0.92)

        # Tabella Sintesi Stress Test
        ax_tbl_stress = fig3.add_axes([0.08, 0.06, 0.88, 0.35])
        ax_tbl_stress.set_facecolor("#ffffff")
        for sp in ax_tbl_stress.spines.values():
            sp.set_color("#cbd5e1")
        ax_tbl_stress.set_xticks([])
        ax_tbl_stress.set_yticks([])

        ax_tbl_stress.text(0.01, 0.92, f"RESILIENZA DEL PORTAFOGLIO AGLI SHOCK STORICI (RITIRO AD ETÀ {eta_fire} ANNI)", fontsize=10, weight="bold", color="#0f172a")

        def _riga_stress(titolo_s: str, chiave_s: str, desc_s: str, val_rischio: str) -> List[str]:
            fine_v = float(stress_dict[chiave_s][-1]) if (stress_dict and chiave_s in stress_dict) else None
            return [
                titolo_s,
                textwrap.fill(desc_s, 32),
                f"{fine_v:,.0f} €" if fine_v is not None else "N/D",
                "Capitale Integro" if (fine_v is not None and fine_v > 0) else "Esaurito",
                textwrap.fill(val_rischio, 26)
            ]

        stress_rows = [
            _riga_stress("Bolla Dot-Com (2000)", "Dot-Com 2000", "Bear market azionario prolungato di 3 anni consecutivi", "Resistente grazie a bond tent & buffer"),
            _riga_stress("Crisi Subprime (2008)", "Crisi 2008", "Crollo azionario violento (-37%) con rapido rimbalzo", "Assorbito da ribilanciamento tax-free"),
            _riga_stress("Stagflazione (1973)", "Stagflazione 1973", "Inflazione a due cifre con crollo simultaneo di stock e bond", "Massimo Sequence Risk reale"),
            _riga_stress("Doppio Shock (2022)", "Crollo 2022", "Rialzo tassi ECB: Bund -20% e Azioni -13% in EUR", "Contenuto dal cash buffer"),
        ]
        stress_cols = ["Scenario di Stress", "Natura dello Shock", "Capitale a 100a", "Esito Portfolio", "Valutazione Rischio"]

        t_elem = ax_tbl_stress.table(
            cellText=stress_rows,
            colLabels=stress_cols,
            colWidths=[0.16, 0.34, 0.13, 0.13, 0.24],
            cellLoc="left",
            loc="center",
            bbox=[0.0, 0.04, 1.0, 0.82]
        )
        t_elem.auto_set_font_size(False)
        t_elem.set_fontsize(7.8)
        for (r, c), cell in t_elem.get_celld().items():
            if r == 0:
                cell.set_facecolor("#1e293b")
                cell.set_text_props(color="#ffffff", weight="bold")
            else:
                cell.set_facecolor("#f8fafc" if r % 2 == 0 else "#ffffff")
            if c in (0, 2, 3):
                cell.set_text_props(ha="center")
            cell.set_edgecolor("#cbd5e1")

        pdf.savefig(fig3)
        plt.close(fig3)

        # =========================================================================
        # PAGINA 4: PROIEZIONE FINANZIARIA DETTAGLIATA ANNO PER ANNO
        # =========================================================================
        fig4 = plt.figure(figsize=(11.69, 8.27), dpi=150)
        fig4.patch.set_facecolor("#f8fafc")

        _aggiungi_banner_intestazione(
            fig4, [0.04, 0.91, 0.92, 0.06],
            f"PROIEZIONE DETTAGLIATA ANNO PER ANNO — RITIRO PROGRAMMATO A {eta_fire} ANNI"
        )

        anni_stampati = [
            eta for eta in range(config.eta_iniziale, config.eta_morte)
            if eta in (config.eta_iniziale, eta_fire, config.eta_pensione, config.eta_morte - 1) or (eta % 3 == 0)
        ]

        headers_tab = [
            "Età", "Fase", "Alloc. (Eq/Bd)", "RAL (€)", "Netto (€)", "Tasse/Sanità",
            "Risparmio", "Spese (€)", "Pensione UE", "Prelievo", "Peggior 1% (€)", "Mediana 50% (€)"
        ]

        table_rows = []
        f_infl_cum = (1.0 + config.inflazione_media) ** np.arange(len(anni_x))
        dep_medie = np.mean(risultati["depositi"], axis=1)

        for eta_t in anni_stampati:
            idx_t = eta_t - config.eta_iniziale
            alloc_str = config.format_allocazione_anno(idx_t, anno_acc_sel)

            if idx_t < anno_acc_sel:
                fase = "Accumulo"
                ral_v = float(ral_medie[idx_t])
                dep_v = float(dep_medie[idx_t])
                spesa_v = float(spese_medie[idx_t])
                netto_v = dep_v + spesa_v
                tasse_v = max(0.0, ral_v - netto_v)
                pens_v = 0.0
                prel_v = 0.0
            elif idx_t < idx_pens_canonica:
                fase = "FIRE"
                ral_v = netto_v = dep_v = pens_v = 0.0
                spesa_v = float(spese_medie[idx_t])
                tasse_v = float(config.costo_mensile_pkv_fire_nominale * 12.0 * f_infl_cum[idx_t])
                prel_v = spesa_v + tasse_v
            else:
                fase = "Pensione"
                ral_v = netto_v = dep_v = 0.0
                spesa_v = float(spese_medie[idx_t])
                pens_v = float(p_tot_med * (f_infl_cum[idx_t] / f_infl_cum[idx_pens_canonica]))
                tasse_v = float(pens_v * 0.15)
                prel_v = max(0.0, spesa_v - pens_v)

            table_rows.append([
                str(eta_t),
                fase,
                alloc_str,
                f"{ral_v:,.0f}" if ral_v > 0 else "-",
                f"{netto_v:,.0f}" if netto_v > 0 else "-",
                f"{tasse_v:,.0f}",
                f"{dep_v:,.0f}" if dep_v > 0 else "-",
                f"{spesa_v:,.0f}",
                f"{pens_v:,.0f}" if pens_v > 0 else "-",
                f"{prel_v:,.0f}" if prel_v > 0 else "-",
                f"{float(p01[idx_t]):,.0f}",
                f"{float(p50[idx_t]):,.0f}"
            ])

        ax_tbl_full = fig4.add_axes([0.04, 0.08, 0.92, 0.78])
        ax_tbl_full.set_facecolor("#ffffff")
        for sp in ax_tbl_full.spines.values():
            sp.set_color("#cbd5e1")
        ax_tbl_full.set_xticks([])
        ax_tbl_full.set_yticks([])

        col_w_page4 = [0.045, 0.08, 0.095, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.10, 0.10]
        t_elem2 = ax_tbl_full.table(
            cellText=table_rows,
            colLabels=headers_tab,
            colWidths=col_w_page4,
            cellLoc="center",
            loc="center",
            bbox=[0.01, 0.02, 0.98, 0.96]
        )
        t_elem2.auto_set_font_size(False)
        t_elem2.set_fontsize(7.2)

        for (r, c), cell in t_elem2.get_celld().items():
            if r == 0:
                cell.set_facecolor("#0f172a")
                cell.set_text_props(color="#ffffff", weight="bold")
            else:
                eta_val = int(table_rows[r - 1][0])
                if eta_val == eta_fire:
                    cell.set_facecolor("#fee2e2")
                    cell.set_text_props(weight="bold", color="#991b1b")
                elif eta_val == config.eta_pensione:
                    cell.set_facecolor("#ede9fe")
                    cell.set_text_props(weight="bold", color="#5b21b6")
                else:
                    cell.set_facecolor("#f8fafc" if r % 2 == 0 else "#ffffff")
            cell.set_edgecolor("#e2e8f0")

        pdf.savefig(fig4)
        plt.close(fig4)

    buffer.seek(0)
    return buffer.getvalue()
