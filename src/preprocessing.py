"""
src/preprocessing.py: pipeline di pulizia, normalizzazione e arricchimento del dataset Spotify

Dataset di input (da Kaggle "Spotify Dataset 1921-2020, 600k+ Tracks"):
 - data/tracks.csv:  586k brani con feature audio
 - data/artists.csv: artisti con generi associati

Output:
 - data/spotify_clean.csv: dataset pulito, pronto per il notebook

Principi applicati:
 - Stephen Few (table design): report qualità dati leggibile, niente griglie
 - Lie Factor: normalizzazione fedele al range reale del dataset,
               senza distorcere le distribuzioni con range teorici arbitrari
"""

import os
import ast
import pandas as pd
import numpy as np


# Percorsi file

DATA_DIR    = "data"
RAW_TRACKS  = os.path.join(DATA_DIR, "tracks.csv")
RAW_ARTISTS = os.path.join(DATA_DIR, "artists.csv")
CLEAN_PATH  = os.path.join(DATA_DIR, "spotify_clean.csv")



# Costanti

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "loudness",
    "tempo",
]

# Feature audio da normalizzare: loudness (dB, circa −60..0) e tempo (BPM)
# (le altre feature sono già in [0, 1] per definizione dell'API Spotify)
FEATURES_TO_NORMALIZE = ["loudness", "tempo"]

# Range temporale del dataset
YEAR_MIN = 1921
YEAR_MAX = 2020

# Soglia di popolarità per la classificazione "hit"
POPULARITY_THRESHOLD = 60

# Mappatura dei generi in 5 macro-categorie
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



# Funzioni

def load_data(
    tracks_path: str = RAW_TRACKS,
    artists_path: str = RAW_ARTISTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carica tracks.csv e artists.csv dal dataset Kaggle
    """
    tracks  = pd.read_csv(tracks_path,  low_memory=False)
    artists = pd.read_csv(artists_path, low_memory=False)

    # Estrae l'anno da release_date
    tracks["year"] = (
        pd.to_datetime(tracks["release_date"], errors="coerce")
        .dt.year
        .astype("Int64")
    )

    print(f"    Dataset {tracks_path} caricato: {len(tracks):>7,} righe × {tracks.shape[1]} colonne")
    print(f"    Dataset {artists_path} caricato: {len(artists):>7,} righe × {artists.shape[1]} colonne")
    return tracks, artists


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove brani duplicati per (name, artists) mantenendo il record con popularity più alta
    """
    before = len(df)
    df = (
        df.sort_values("popularity", ascending=False)
          .drop_duplicates(subset=["name", "artists"], keep="first")
          .reset_index(drop=True)
    )
    print(f"    Duplicati rimossi: {before - len(df):>6,}  ->  {len(df):,} righe")
    return df


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove le righe con valori mancanti nelle colonne critiche (feature audio + year + popularity)
    """
    critical = [c for c in AUDIO_FEATURES + ["year", "popularity"]
                if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical).reset_index(drop=True)
    print(f"    Valori nulli rimossi: {before - len(df):>6,}  ->  {len(df):,} righe")
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza loudness e tempo in [0, 1] tramite min-max scaling
    """
    for col in FEATURES_TO_NORMALIZE:
        if col not in df.columns:
            continue
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
            print(f"    Normalizzazione '{col}' in '{col}_norm': "
                  f"[{col_min:.2f}, {col_max:.2f}] → [0, 1]")
        else:
            df[f"{col}_norm"] = 0.0
    return df


def filter_years(
    df: pd.DataFrame,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
) -> pd.DataFrame:
    """
    Filtra i brani nel range temporale 1921-2020
    """
    before = len(df)
    df = df[df["year"].between(year_min, year_max)].reset_index(drop=True)
    print(f"    Range temporale [{year_min}–{year_max}]: {before - len(df):>6,} righe rimosse  ->  {len(df):,} righe")
    return df


def _parse_id_list(value: str) -> list[str]:
    """
    Converte la stringa "['id1', 'id2']" in una lista Python
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
    Mappa una lista di generi di Spotify alla macro-categoria di appartenenza
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
    """
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
    print("    Distribuzione generi (top 7):")
    for g, n in dist.items():
        pct = n / len(tracks) * 100
        print(f"      {g:<12} {n:>6,}  ({pct:.1f}%)")

    return tracks


def add_hit_label(
    df: pd.DataFrame,
    threshold: int = POPULARITY_THRESHOLD,
) -> pd.DataFrame:
    """
    Aggiunge la colonna binaria 'is_hit' (is_hit = 1 se popularity >= threshold, 0 altrimenti)
    """
    df["is_hit"] = (df["popularity"] >= threshold).astype(int)
    pct = df["is_hit"].mean() * 100
    print(f"    Aggiunta colonna 'is_hit' (popularity ≥ {threshold}): {pct:.1f}% brani positivi")
    return df


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un DataFrame riassuntivo della qualità dei dati applicando i principi di Stephen Few
    """
    colonne = [c for c in AUDIO_FEATURES + ["year", "popularity"]
            if c in df.columns]

    report = pd.DataFrame({
        "feature":  colonne,
        # "n":        [int(df[c].notna().sum()) for c in colonne],
        # "null_%":   [round(df[c].isna().mean() * 100, 2) for c in colonne],
        "min":      [round(df[c].min(), 4) for c in colonne],
        "max":      [round(df[c].max(), 4) for c in colonne],
        "mean":     [round(df[c].mean(), 4) for c in colonne],
        "std":      [round(df[c].std(), 4) for c in colonne],
        "range":    [round(df[c].max() - df[c].min(), 4) for c in colonne],
    })
    return report


def save_clean(df: pd.DataFrame, out_path: str = CLEAN_PATH) -> None:
    """Salva il DataFrame pulito in data/spotify_clean.csv."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"    Salvataggio → {out_path}  ({len(df):,} righe, {df.shape[1]} colonne)")


def run_preprocessing_pipeline(
    tracks_path:  str = RAW_TRACKS,
    artists_path: str = RAW_ARTISTS,
    out_path:     str = CLEAN_PATH,
) -> pd.DataFrame:
    """
    Esegue l'intera pipeline in sequenza e restituisce il DataFrame pulito.
    """
    print("[*] Avvio pipeline di preprocessing del dataset")
    tracks, artists = load_data(tracks_path, artists_path)
    df = drop_duplicates(tracks)
    df = handle_nulls(df)
    df = normalize_features(df)
    df = filter_years(df)
    df = add_primary_genre(df, artists)
    df = add_hit_label(df)
    save_clean(df, out_path)
    print(f"[+] Dataset preprocessato con successo --> {len(df):,} brani")
    return df


if __name__ == "__main__":
    run_preprocessing_pipeline()
