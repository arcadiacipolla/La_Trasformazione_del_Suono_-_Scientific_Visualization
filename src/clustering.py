"""
src/clustering.py : K-Means clustering sulle feature audio del dataset Spotify

Pipeline:
 1. Selezione delle 8 feature audio
 2. StandardScaler
 3. Elbow Method + Silhouette Score per trovare k ottimale
 4. K-Means finale con il k scelto
 5. Profilazione cluster come Z-score (input del Parallel Coordinates)
 6. Naming automatico dei cluster da profilo dominante
 7. PCA 2D opzionale per scatter di controllo

Output:
 - data/spotify_clustered.csv: dataset con colonna 'cluster' aggiunta
 - dict results:               informazioni necessarie per M4: Archetipi
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*n_init.*")


# Costanti

DATA_DIR       = "data"
CLUSTERED_PATH = os.path.join(DATA_DIR, "spotify_clustered.csv")

# Le 8 feature audio per il clustering.
CLUSTER_FEATURES: list[str] = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "loudness_norm",
]

K_RANGE_DEFAULT = range(2, 11)   # k = 2..10

# Soglie Z-score per il naming automatico dei cluster
# Nota: un cluster è "dominato" da una feature se il suo Z-score medio supera
# DOMINANT_THR; è "basso" su quella feature se scende sotto LOW_THR
DOMINANT_THR =  0.55
LOW_THR      = -0.55

# Etichette semantiche per ogni feature (alta, bassa)
_FEATURE_LABELS: dict[str, tuple[str, str | None]] = {
    "acousticness":     ("acustico",    "elettronico"),
    "energy":           ("energico",    "calmo"),
    "danceability":     ("danzabile",   None),
    "valence":          ("positivo",    "malinconico"),
    "instrumentalness": ("strumentale", "vocale"),
    "speechiness":      ("parlato",     None),
    "liveness":         ("live",        None),
    "loudness_norm":    ("forte",       "silenzioso"),
    "loudness":         ("forte",       "silenzioso"),
}



# Funzioni

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estrae le feature audio dal DataFrame pulito
    """
    selected = []
    for col in CLUSTER_FEATURES:
        if col in df.columns:
            selected.append(col)
        elif col == "loudness_norm" and "loudness" in df.columns:
            selected.append("loudness")
            print("    Warning: 'loudness_norm' non trovata -> uso 'loudness' (non normalizzata)")
        else:
            print(f"    Warning: feature '{col}' assente nel DataFrame (saltata)")

    print(f"    Feature selezionate ({len(selected)}): {selected}")
    return df[selected].copy()


def scale(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Applica StandardScaler alle feature
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"    StandardScaler → shape: {X_scaled.shape}")
    return X_scaled, scaler


def elbow_curve(
    X_scaled:     np.ndarray,
    k_range:      range = K_RANGE_DEFAULT,
    random_state: int   = 42,
) -> dict[str, list]:
    """
    Calcola l'inertia (WCSS) per ogni k nel range
    """
    ks, inertias = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        ks.append(k)
        inertias.append(round(km.inertia_, 1))
        print(f"      k={k:2d}  inertia={km.inertia_:>12,.0f}")

    return {"k": ks, "inertia": inertias}


def silhouette_scores(
    X_scaled:     np.ndarray,
    k_range:      range = K_RANGE_DEFAULT,
    random_state: int   = 42,
    sample_size:  int   = 20_000,
) -> dict:
    """
    Calcola il Silhouette Score medio per ogni k
    """
    n = X_scaled.shape[0]
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    X_sample = X_scaled[idx]

    ks, scores = [], []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_sample)
        score  = silhouette_score(X_sample, labels, random_state=random_state)
        ks.append(k)
        scores.append(round(float(score), 4))
        print(f"      k={k:2d}  silhouette={score:.4f}")

    best_k = ks[int(np.argmax(scores))]
    print(f"    -> k ottimale (silhouette max): {best_k}")
    return {"k": ks, "scores": scores, "best_k": best_k}


def find_optimal_k(
    X_scaled:     np.ndarray,
    k_range:      range = K_RANGE_DEFAULT,
    random_state: int   = 42,
) -> tuple[int, dict, dict]:
    """
    Combina Elbow Method e Silhouette Score
    La scelta finale segue il Silhouette (più oggettivo)
    """
    print("    Elbow Method")
    elbow_data = elbow_curve(X_scaled, k_range, random_state)

    print("    Silhouette Score")
    sil_data   = silhouette_scores(X_scaled, k_range, random_state)

    return sil_data["best_k"], elbow_data, sil_data


def run_kmeans(
    df:           pd.DataFrame,
    X_scaled:     np.ndarray,
    k:            int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans]:
    """
    Addestra K-Means con il k scelto e aggiunge la colonna 'cluster'
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_scaled)

    print(f"    K-Means (k={k}) completato")
    dist = df["cluster"].value_counts().sort_index()
    for c, n in dist.items():
        bar = "█" * int(n / len(df) * 40)
        print(f"      cluster {c}: {n:>6,}  ({n/len(df)*100:.1f}%)  {bar}")

    return df, km


def cluster_profiles(
    df:       pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calcola il profilo Z-score di ogni cluster sulle feature audio
    """
    if features is None:
        features = [c for c in CLUSTER_FEATURES if c in df.columns]
        if "loudness_norm" not in df.columns and "loudness" in df.columns:
            features = [
                "loudness" if c == "loudness_norm" else c for c in features
            ]

    # Z-score per ogni feature sull'intero dataset
    df_z = df[features].copy()
    for col in features:
        mu, sigma = df_z[col].mean(), df_z[col].std()
        if sigma > 0:
            df_z[col] = (df_z[col] - mu) / sigma
        else:
            df_z[col] = 0.0

    df_z["cluster"] = df["cluster"].values
    profiles = df_z.groupby("cluster")[features].mean().round(3)

    print(f"    Profili cluster calcolati: shape {profiles.shape}")
    return profiles


def name_clusters(profiles: pd.DataFrame) -> dict[int, str]:
    """
    Assegna un nome descrittivo a ogni cluster basandosi sul profilo Z-score
    """
    names: dict[int, str] = {}

    for cluster_id, row in profiles.iterrows():
        high_parts, low_parts = [], []

        for feat, score in row.items():
            if feat not in _FEATURE_LABELS:
                continue
            high_label, low_label = _FEATURE_LABELS[feat]
            if score > DOMINANT_THR and high_label:
                high_parts.append((score, high_label))
            elif score < LOW_THR and low_label:
                low_parts.append((abs(score), low_label))

        # Ordina per intensità del segnale, prendi i top descrittori
        high_parts.sort(key=lambda x: -x[0])
        low_parts.sort(key=lambda x: -x[0])

        parts = [l for _, l in high_parts[:2]] + [l for _, l in low_parts[:1]]
        label = "-".join(parts) if parts else f"cluster {cluster_id}"
        names[int(cluster_id)] = label

    print("    Nomi cluster automatici:")
    for cid, name in names.items():
        print(f"      {cid} → '{name}'")

    return names


def pca_2d(
    X_scaled: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Riduce le feature a 2 componenti principali per visualizzazione
    """
    pca        = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    var        = pca.explained_variance_ratio_

    print(f"    PCA 2D: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%"
          f"    (spiegato: {sum(var)*100:.1f}%)")

    df_pca = pd.DataFrame({"pc1": components[:, 0], "pc2": components[:, 1]})
    return df_pca, var


def save_clustered(df: pd.DataFrame, out_path: str = CLUSTERED_PATH) -> None:
    """Salva il DataFrame arricchito con la colonna 'cluster'."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"    Salvataggio → {out_path}  ({len(df):,} righe, {df.shape[1]} colonne)")


def run_cluster_pipeline(
    df:           pd.DataFrame,
    k:            int | None = None,
    k_range:      range      = K_RANGE_DEFAULT,
    out_path:     str        = CLUSTERED_PATH,
    random_state: int        = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Esegue l'intera pipeline di clustering.

    Se k=None, determina automaticamente k ottimale via Silhouette Score.
    Se k è fornito, salta la ricerca e usa quel valore direttamente
    (utile per iterazioni veloci dopo la prima esecuzione).

    Restituisce (df_clustered, results) dove results contiene:
     - 'k'              : k usato
     - 'kmeans'         : modello KMeans addestrato
     - 'scaler'         : StandardScaler per inversione
     - 'feature_names'  : lista feature usate
     - 'profiles'       : DataFrame Z-score (k × features) per M4
     - 'cluster_names'  : dict {int → str} per le etichette
     - 'pca_df'         : DataFrame con pc1, pc2 per scatter opzionale
     - 'pca_explained'  : varianza spiegata da PC1 e PC2
     - 'elbow_data'     : {'k', 'inertia'} per grafico Elbow in M4
     - 'silhouette_data': {'k', 'scores', 'best_k'} per grafico Sil. in M4
    """
    print("[*] Avvio pipeline di clustering")

    X             = select_features(df)
    feature_names = list(X.columns)
    X_scaled, scaler = scale(X)

    if k is None:
        print("    Ricerca k ottimale (Elbow + Silhouette)...")
        k, elbow_data, sil_data = find_optimal_k(X_scaled, k_range, random_state)
    else:
        print(f"    k fornito manualmente: {k}")
        print("    Calcolo diagnostici comunque (per i grafici M4)...")
        elbow_data = elbow_curve(X_scaled, k_range, random_state)
        sil_data   = silhouette_scores(X_scaled, k_range, random_state)

    df_c, km   = run_kmeans(df, X_scaled, k, random_state)
    profiles   = cluster_profiles(df_c, feature_names)
    names      = name_clusters(profiles)
    pca_df, pca_var = pca_2d(X_scaled)

    save_clustered(df_c, out_path)

    results = {
        "k":               k,
        "kmeans":          km,
        "scaler":          scaler,
        "feature_names":   feature_names,
        "profiles":        profiles,
        "cluster_names":   names,
        "pca_df":          pca_df,
        "pca_explained":   pca_var,
        "elbow_data":      elbow_data,
        "silhouette_data": sil_data,
    }

    print(f"[*] Cluster effettuato con successo --> {len(df_c):,} brani")
    return df_c, results


if __name__ == "__main__":
    clean_path = os.path.join(DATA_DIR, "spotify_clean.csv")
    if os.path.exists(clean_path):
        df = pd.read_csv(clean_path)
        run_cluster_pipeline(df)
    else:
        print(f"[-] Errore: '{clean_path}' non trovato. Esegui prima src/preprocessing.py.")
