"""
src/preprocessing.py — The Shape of Music
──────────────────────────────────────────────────────────────────────────────
Pipeline di pulizia, normalizzazione e arricchimento del dataset Spotify.

Dataset di input (da Kaggle "Spotify Dataset 1921-2020, 600k+ Tracks"):
  · data/tracks.csv   — 586 k brani con feature audio
  · data/artists.csv  — artisti con generi associati

Output:
  · data/spotify_clean.csv — dataset pulito, pronto per il notebook

Principi applicati:
  · Stephen Few  — table design: report qualità dati leggibile, niente griglie
  · Lie Factor   — normalizzazione fedele al range reale del dataset,
                   senza distorcere le distribuzioni con range teorici arbitrari
"""

import os
import ast
import pandas as pd
import numpy as np

# ── Percorsi di default ───────────────────────────────────────────────────────

DATA_DIR      = "data"
RAW_TRACKS    = os.path.join(DATA_DIR, "tracks.csv")
RAW_ARTISTS   = os.path.join(DATA_DIR, "artists.csv")
CLEAN_PATH    = os.path.join(DATA_DIR, "spotify_clean.csv")

# ── Costanti ──────────────────────────────────────────────────────────────────

# Feature audio da normalizzare: loudness (dB, circa −60..0) e tempo (BPM)
# Tutte le altre sono già in [0, 1] per definizione dell'API Spotify.
FEATURES_TO_NORMALIZE = ["loudness", "tempo"]

# Feature audio usate nell'analisi (M2, M3, M4, M5)
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "loudness",   # normalizzata da noi → loudness_norm dopo il rename
    "tempo",      # normalizzata da noi → tempo_norm dopo il rename
]

# Range temporale dichiarato nel titolo del progetto
YEAR_MIN = 1921
YEAR_MAX = 2020

# Soglia di popolarità per la classificazione "hit" (M5)
# Top ~35 % dei brani nel dataset pulito
POPULARITY_THRESHOLD = 60

# Mappatura Spotify micro-generi → 5 macro-categorie per M3 / M4
# La logica: cerchiamo la prima keyword presente nel micro-genere.
# Ordine importante: le parole più specifiche vengono prima di quelle generiche.
GENRE_KEYWORDS: dict[str, str] = {
    # Pop
    "k-pop":        "Pop",
    "indie pop":    "Pop",
    "synth-pop":    "Pop",
    "electropop":   "Pop",
    "dance pop":    "Pop",
    "art pop":      "Pop",
    "pop":          "Pop",
    # Classical
    "baroque":      "Classical",
    "romantic era": "Classical",
    "opera":        "Classical",
    "orchestra":    "Classical",
    "chamber":      "Classical",
    "classical":    "Classical",
    "piano":        "Classical",
    # Electronic
    "dubstep":      "Electronic",
    "techno":       "Electronic",
    "trance":       "Electronic",
    "ambient":      "Electronic",
    "house":        "Electronic",
    "edm":          "Electronic",
    "electronic":   "Electronic",
    "electronica":  "Electronic",
    "electro":      "Electronic",
    # Hip-Hop / R&B
    "trap":         "Hip-Hop",
    "gangsta":      "Hip-Hop",
    "hip hop":      "Hip-Hop",
    "hip-hop":      "Hip-Hop",
    "r&b":          "Hip-Hop",
    "soul":         "Hip-Hop",
    "rap":          "Hip-Hop",
    # Rock
    "heavy metal":  "Rock",
    "hard rock":    "Rock",
    "punk":         "Rock",
    "grunge":       "Rock",
    "metal":        "Rock",
    "alt-rock":     "Rock",
    "indie rock":   "Rock",
    "rock":         "Rock",
}


# ── Funzioni ──────────────────────────────────────────────────────────────────

def load_data(
    tracks_path: str = RAW_TRACKS,
    artists_path: str = RAW_ARTISTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carica tracks.csv e artists.csv dal dataset Kaggle.

    tracks.csv — colonne principali:
        id, name, popularity, duration_ms, explicit, artists, id_artists,
        release_date, danceability, energy, key, loudness, mode,
        speechiness, acousticness, instrumentalness, liveness, valence,
        tempo, time_signature

    artists.csv — colonne principali:
        id, followers, genres, name, popularity

    Nota: release_date può avere formato YYYY, YYYY-MM o YYYY-MM-DD.
    """
    tracks  = pd.read_csv(tracks_path,  low_memory=False)
    artists = pd.read_csv(artists_path, low_memory=False)

    # Estrae l'anno da release_date (qualunque formato)
    tracks["year"] = (
        pd.to_datetime(tracks["release_date"], errors="coerce")
        .dt.year
        .astype("Int64")
    )

    print(f"  tracks  loaded: {len(tracks):>7,} rows × {tracks.shape[1]} cols")
    print(f"  artists loaded: {len(artists):>7,} rows × {artists.shape[1]} cols")
    return tracks, artists


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove brani duplicati per (name, artists).
    Mantiene il record con popularity più alta — scelta conservativa
    che massimizza l'informazione sul riconoscimento del brano.
    """
    before = len(df)
    df = (
        df.sort_values("popularity", ascending=False)
          .drop_duplicates(subset=["name", "artists"], keep="first")
          .reset_index(drop=True)
    )
    print(f"  Duplicates removed : {before - len(df):>6,}  →  {len(df):,} rows")
    return df


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove le righe con valori mancanti nelle colonne critiche
    (feature audio + year + popularity).
    Le colonne non critiche (key, mode, time_signature) vengono mantenute.
    """
    critical = [c for c in AUDIO_FEATURES + ["year", "popularity"]
                if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical).reset_index(drop=True)
    print(f"  Null rows removed  : {before - len(df):>6,}  →  {len(df):,} rows")
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza loudness e tempo in [0, 1] tramite min-max scaling.

    Tufte / Lie Factor: usiamo il range REALE del dataset (non teorico)
    così da non distorcere la distribuzione dei valori.
    Le colonne normalizzate vengono rinominate aggiungendo il suffisso '_norm'
    per tenere traccia della trasformazione.
    """
    for col in FEATURES_TO_NORMALIZE:
        if col not in df.columns:
            continue
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
            print(f"  Normalized '{col}': [{col_min:.2f}, {col_max:.2f}] → [0, 1]"
                  f"  → '{col}_norm'")
        else:
            df[f"{col}_norm"] = 0.0
    return df


def filter_years(
    df: pd.DataFrame,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
) -> pd.DataFrame:
    """
    Filtra i brani nel range temporale 1921-2020.
    Rimuove outlier di data (valori impossibili o fuori range dichiarato).
    """
    before = len(df)
    df = df[df["year"].between(year_min, year_max)].reset_index(drop=True)
    print(f"  Year [{year_min}–{year_max}]   : {before - len(df):>6,} removed  →  {len(df):,} rows")
    return df


def _parse_id_list(value: str) -> list[str]:
    """
    Converte la stringa "['id1', 'id2']" in una lista Python.
    Gestisce sia apici singoli che doppi, e valori malformati.
    """
    if not isinstance(value, str):
        return []
    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else [str(result)]
    except (ValueError, SyntaxError):
        return [value.strip("[]'\" ")]


def _map_genre(genres_list: list[str]) -> str:
    """
    Mappa una lista di micro-generi Spotify al macro-genere più vicino.
    Strategia: scansiona GENRE_KEYWORDS in ordine (più specifico → più generico).
    Se nessuna keyword corrisponde, restituisce 'Other'.
    """
    genres_str = " ".join(genres_list).lower()
    for keyword, macro in GENRE_KEYWORDS.items():
        if keyword in genres_str:
            return macro
    return "Other"


def add_primary_genre(
    tracks: pd.DataFrame,
    artists: pd.DataFrame,
) -> pd.DataFrame:
    """
    Arricchisce il dataset tracce con una colonna 'genre' (macro-categoria).

    Pipeline:
    1. Espande id_artists (lista di ID) → un ID per riga
    2. Join con artists.csv su artist_id
    3. Parsea la colonna genres dell'artista (lista di micro-generi)
    4. Mappa al macro-genere tramite GENRE_KEYWORDS
    5. Per ogni traccia mantiene il genere del primo artista trovato

    Le tracce senza genere identificabile ricevono il valore 'Other'.
    """
    # Prepara il lookup artista → genere
    artists_clean = artists[["id", "genres"]].copy()
    artists_clean["genres_list"] = artists_clean["genres"].apply(_parse_id_list)
    artists_clean["genre"] = artists_clean["genres_list"].apply(_map_genre)
    artist_genre = artists_clean.set_index("id")["genre"].to_dict()

    # Per ogni traccia, prende il genere del primo artista con genere noto
    def _get_track_genre(id_artists_str: str) -> str:
        ids = _parse_id_list(id_artists_str)
        for aid in ids:
            g = artist_genre.get(aid, "Other")
            if g != "Other":
                return g
        return "Other"

    tracks["genre"] = tracks["id_artists"].apply(_get_track_genre)

    dist = tracks["genre"].value_counts()
    print("  Genre distribution (top 7):")
    for g, n in dist.head(7).items():
        pct = n / len(tracks) * 100
        print(f"    {g:<12} {n:>6,}  ({pct:.1f}%)")

    return tracks


def add_hit_label(
    df: pd.DataFrame,
    threshold: int = POPULARITY_THRESHOLD,
) -> pd.DataFrame:
    """
    Aggiunge la colonna binaria 'is_hit' usata come target in M5.
    is_hit = 1 se popularity >= threshold, 0 altrimenti.

    Il valore di default (60) corrisponde circa al top 35 % dei brani
    nel dataset pulito — abbastanza selettivo da essere significativo,
    abbastanza ampio da avere esempi positivi in ogni decade.
    """
    df["is_hit"] = (df["popularity"] >= threshold).astype(int)
    pct = df["is_hit"].mean() * 100
    print(f"  'is_hit' (popularity ≥ {threshold}): {pct:.1f}% brani positivi")
    return df


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un DataFrame riassuntivo della qualità dei dati.

    Principi Stephen Few (table design):
    · Solo colonne necessarie — niente misure ridondanti
    · Numeri arrotondati a 4 decimali — leggibilità senza falsa precisione
    · Nessuna formattazione decorativa — la struttura emerge dall'allineamento

    Il DataFrame restituito può essere stampato nel notebook con:
        display(data_quality_report(df))
    oppure formattato con:
        display(data_quality_report(df).style.format("{:.4f}", subset=[...]))
    """
    cols = [c for c in AUDIO_FEATURES + ["year", "popularity"]
            if c in df.columns]
    print(cols)

    report = pd.DataFrame({
        "feature":  cols,
        # "n":        [int(df[c].notna().sum()) for c in cols],
        # "null_%":   [round(df[c].isna().mean() * 100, 2) for c in cols],
        "min":      [round(df[c].min(), 4) for c in cols],
        "max":      [round(df[c].max(), 4) for c in cols],
        "mean":     [round(df[c].mean(), 4) for c in cols],
        "std":      [round(df[c].std(), 4) for c in cols],
        "range":    [round(df[c].max() - df[c].min(), 4) for c in cols],
    })
    return report


def save_clean(df: pd.DataFrame, out_path: str = CLEAN_PATH) -> None:
    """Salva il DataFrame pulito in data/spotify_clean.csv."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  ({len(df):,} rows, {df.shape[1]} cols)")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_pipeline(
    tracks_path:  str = RAW_TRACKS,
    artists_path: str = RAW_ARTISTS,
    out_path:     str = CLEAN_PATH,
) -> pd.DataFrame:
    """
    Esegue l'intera pipeline in sequenza e restituisce il DataFrame pulito.

    Uso nel notebook (inizio di M1):
        from src.preprocessing import run_pipeline
        df = run_pipeline()
    """
    print("[*] Fase di pipeline")
    tracks, artists = load_data(tracks_path, artists_path)
    df = drop_duplicates(tracks)
    df = handle_nulls(df)
    df = normalize_features(df)
    df = filter_years(df)
    df = add_primary_genre(df, artists)
    df = add_hit_label(df)
    save_clean(df, out_path)
    print(f"[+] Dataset elaborato con successo --> {len(df):,} brani")
    return df


if __name__ == "__main__":
    run_pipeline()
