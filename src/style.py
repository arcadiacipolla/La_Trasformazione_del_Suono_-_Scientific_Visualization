"""
src/style.py: impostazioni grafiche condivise da tutti i moduli del notebook

Principi applicati:
 - Edward Tufte:  data-ink ratio, no chartjunk, spine ridotte
 - Colin Ware:    attributi pre-attentivi: un solo colore ad alta saturazione
                  (HIGHLIGHT) per guidare lo sguardo, grigio per il resto
 - ColorBrewer:   palette colorblind-safe (Set2) per cluster e generi
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt


# Colori

HIGHLIGHT = "#BF5200" # Pre-attentivo (Ware): colore ad alta saturazione riservato alla feature chiave
MUTED     = "#B4B2A9" # Grigio desaturato per serie secondarie, sfondi, linee di riferimento
MUTED_ALT = "#D3D1C7" # variante più chiara per griglie e bande
INK       = "#2C2C2A" # Quasi-nero per testo e spine (meno aggressivo del nero puro)
PAPER     = "#FAFAF8" # Bianco leggermente caldo per sfondi (evita il bianco freddo puro)

# Palette ColorBrewer Set2 — 8 colori, testata per daltonismo (Ware, cap. 4)
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
GENRE_PALETTE = CLUSTER_PALETTE



# rcParams basato su Tufte

TUFTE_RC: dict = {
    # Solo spine bottom e left
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,
    # Griglia tenue come guida, non come struttura dominante
    "axes.grid":            True,
    "grid.color":           MUTED_ALT,
    "grid.linestyle":       "--",
    "grid.linewidth":       0.5,
    "grid.alpha":           0.2,
    "axes.axisbelow":       True,
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
    # Font sans-serif leggibile, gerarchia chiara
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
    "legend.frameon":       False,
    "legend.handlelength":  1.5,
    # Figure
    "figure.dpi":           150,
    "figure.facecolor":     PAPER,
    "axes.facecolor":       PAPER,
    "savefig.facecolor":    PAPER,
    "savefig.dpi":          150,
    "figure.autolayout":    False,
}



# Funzioni

def apply_style(ax: plt.Axes | None = None) -> plt.Axes | None:
    """
    Aggiorna i rcParams globali con TUFTE_RC
    """
    mpl.rcParams.update(TUFTE_RC)

    if ax is not None:
    # rimuovi le spine top/right e imposta i tick verso l'esterno
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    return ax


def label_line(ax: plt.Axes, line, label: str, x_offset: float = 0.01,
               color: str | None = None) -> None:
    """
    Posiziona l'etichetta all'estremità destra della linea
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
    Salva la figura (con tight_layout)
    """
    os.makedirs(figures_dir, exist_ok=True) # Crea la cartella se non esiste
    path = os.path.join(figures_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Salvataggio → {path}")
    return path
