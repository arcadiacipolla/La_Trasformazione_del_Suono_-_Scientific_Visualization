"""
src/style.py — The Shape of Music
──────────────────────────────────────────────────────────────────────────────
Impostazioni grafiche condivise da tutti i moduli del notebook.

Principi applicati:
  · Edward Tufte   — data-ink ratio, no chartjunk, spine ridotte
  · Colin Ware     — attributi pre-attentivi: un solo colore ad alta saturazione
                     (HIGHLIGHT) per guidare lo sguardo, grigio per il resto
  · ColorBrewer    — palette colorblind-safe (Set2) per cluster e generi
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Colori ────────────────────────────────────────────────────────────────────

# Pre-attentivo (Ware): colore ad alta saturazione riservato alla feature chiave.
# Tutto il resto resta in MUTED per non competere visivamente.
HIGHLIGHT = "#BF5200"   # arancione bruciato

# Grigio desaturato per serie secondarie, sfondi, linee di riferimento
MUTED     = "#B4B2A9"
MUTED_ALT = "#D3D1C7"   # variante più chiara per griglie e bande

# Quasi-nero per testo e spine — meno aggressivo del nero puro
INK       = "#2C2C2A"

# Bianco leggermente caldo per sfondi (evita il bianco freddo puro)
PAPER     = "#FAFAF8"

# Palette ColorBrewer Set2 — 8 colori, testata per daltonismo (Ware, cap. 4)
# Usata per cluster (M4) e generi (M3). Non superare 6-7 categorie simultanee.
CLUSTER_PALETTE = [
    "#66C2A5",  # teal          → cluster 0
    "#FC8D62",  # arancio       → cluster 1
    "#8DA0CB",  # azzurro       → cluster 2
    "#E78AC3",  # rosa          → cluster 3
    "#A6D854",  # verde         → cluster 4
    "#FFD92F",  # giallo        → cluster 5
    "#E5C494",  # sabbia        → cluster 6
    "#B3B3B3",  # grigio neutro → cluster 7
]

# Alias per usare la stessa palette con i generi (M3)
GENRE_PALETTE = CLUSTER_PALETTE


# ── rcParams Tufte-inspired ───────────────────────────────────────────────────
# Applicare con mpl.rcParams.update(TUFTE_RC) o chiamando apply_style().

TUFTE_RC: dict = {
    # Spine: solo bottom e left — le altre sono chartjunk (Tufte, VDQI p. 96)
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,

    # Griglia tenue come guida, non come struttura dominante
    "axes.grid":            True,
    "grid.color":           MUTED_ALT,
    "grid.linestyle":       "--",
    "grid.linewidth":       0.5,
    "grid.alpha":           0.6,
    "axes.axisbelow":       True,   # griglia sotto i dati, non sopra

    # Tick leggeri, verso l'esterno
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.size":     4,
    "ytick.major.size":     4,
    "xtick.minor.size":     0,
    "ytick.minor.size":     0,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,

    # Colori del testo e degli assi
    "text.color":           INK,
    "axes.labelcolor":      INK,
    "xtick.color":          INK,
    "ytick.color":          INK,
    "axes.edgecolor":       INK,

    # Font: sans-serif leggibile, gerarchia chiara
    "font.family":          "sans-serif",
    "font.size":            11,
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "axes.titlepad":        10,
    "axes.labelsize":       11,
    "axes.labelpad":        6,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "legend.fontsize":      10,
    "legend.frameon":       False,  # niente bordo legenda (chartjunk)
    "legend.handlelength":  1.5,

    # Figure
    "figure.dpi":           150,
    "figure.facecolor":     PAPER,
    "axes.facecolor":       PAPER,
    "savefig.facecolor":    PAPER,
    "savefig.dpi":          150,
    "figure.autolayout":    False,
}


# ── Funzioni ──────────────────────────────────────────────────────────────────

def apply_style(ax: plt.Axes | None = None) -> plt.Axes | None:
    """
    Aggiorna i rcParams globali con TUFTE_RC.
    Se viene passato un Axes esistente, rimuove anche le spine top/right
    e imposta i tick verso l'esterno — utile quando rcParams non ha ancora
    effetto sull'asse già creato.

    Uso tipico nel notebook (inizio di ogni sezione):
        from src.style import apply_style
        apply_style()
        fig, ax = plt.subplots(...)
    """
    mpl.rcParams.update(TUFTE_RC)

    if ax is not None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    return ax


def label_line(ax: plt.Axes, line, label: str, x_offset: float = 0.01,
               color: str | None = None) -> None:
    """
    Etichettatura diretta di una linea (Tufte: evitare la legenda esterna).
    Posiziona il testo all'estremità destra della linea.

    Parametri
    ---------
    ax       : Axes su cui disegnare
    line     : oggetto Line2D restituito da ax.plot(...)
    label    : testo da mostrare
    x_offset : offset orizzontale relativo all'asse x (in unità dei dati)
    color    : colore del testo; se None usa il colore della linea
    """
    xdata, ydata = line.get_xdata(), line.get_ydata()
    x_end = xdata[-1] + (xdata[-1] - xdata[0]) * x_offset
    y_end = ydata[-1]
    col = color or line.get_color()
    ax.text(
        x_end, y_end, label,
        color=col, fontsize=10, va="center",
        fontweight="bold" if col == HIGHLIGHT else "normal",
    )


def save_fig(fig: plt.Figure, name: str,
             figures_dir: str = "figures") -> str:
    """
    Salva la figura in <figures_dir>/<name>.png con tight_layout.
    Crea la cartella se non esiste. Restituisce il path completo.

    Uso tipico:
        save_fig(fig, "m2_temporal_evolution")
    """
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    return path
