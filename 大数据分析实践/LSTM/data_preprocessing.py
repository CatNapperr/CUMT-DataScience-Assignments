import os
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SEQ_LEN = 3
TRAIN_END = 2012
VAL_END = 2015
TEST_END = 2017

FEATURE_NAMES = [
    "Age", "MPG",
    "FG36", "FGA36", "3P36", "3PA36", "FT36", "FTA36",
    "AST36", "TRB36", "STL36", "BLK36", "TOV36",
    "TS%", "AST_TOV", "3P_Rate",
]

LABEL_BINS = [0, 12, 18, 25, float("inf")]
LABEL_NAMES = ["Role Player", "Rotation Player", "Star", "Superstar"]


def _deduplicate(df):
    """Keep TOT row per player-season, or the only row if player stayed on one team."""
    df = df.copy()
    df["is_tot"] = df["Tm"] == "TOT"
    idx_tot = df[df["is_tot"]].groupby(["Player", "Year"]).size()
    idx_single = (
        df.groupby(["Player", "Year"])
        .size()
        .pipe(lambda s: s[s == 1])
    )
    mask = df["is_tot"] | df.set_index(["Player", "Year"]).index.isin(idx_single.index)
    return df[mask].drop(columns=["is_tot"]).reset_index(drop=True)


def _fill_missing(df):
    for col in ["FG%", "3P%", "FT%", "2P%", "eFG%"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def _engineer_features(df):
    df = df.copy()

    # Drop rows with G=0 or MP=0 — cannot compute any per-minute stats
    df = df[(df["G"] > 0) & (df["MP"] > 0)].copy()

    df["MPG"] = df["MP"] / df["G"]

    per36_cols = ["FG", "FGA", "3P", "3PA", "FT", "FTA",
                  "AST", "TRB", "STL", "BLK", "TOV", "PTS"]
    for c in per36_cols:
        df[f"{c}36"] = df[c] / df["MP"] * 36

    # TS%
    denom = 2 * (df["FGA"] + 0.44 * df["FTA"])
    df["TS%"] = np.where(denom > 0, df["PTS"] / denom, 0)

    # AST_TOV — clamp extreme values (e.g. pure passers with 0 TOV)
    ast_tov_raw = np.where(df["TOV36"] > 0, df["AST36"] / df["TOV36"], df["AST36"])
    df["AST_TOV"] = np.clip(ast_tov_raw, 0, 20)

    # 3P_Rate
    df["3P_Rate"] = np.where(df["FGA36"] > 0, df["3PA36"] / df["FGA36"], 0)

    return df


def _make_labels(df):
    df["label"] = pd.cut(df["PTS36"], bins=LABEL_BINS, right=False, labels=[0, 1, 2, 3])
    return df


def _drop_redundant(df):
    drop_cols = [
        "PTS", "AST", "TRB", "STL", "BLK", "TOV",
        "FG", "FGA", "3P", "3PA", "FT", "FTA",
        "FG%", "3P%", "FT%", "2P%", "eFG%",
        "MP", "G", "ORB", "DRB", "PF",
        "PTS36",  # leaked target
        "Tm",      # not needed for modeling
    ]
    existing = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=existing)


def _build_sequences(df):
    """Sliding window SEQ_LEN=3 over each player's career timeline.

    Returns X(N,3,16), y(N,), target_years(N,), players(N,), positions(N,).
    """
    X_list, y_list, years_list, players_list, pos_list = [], [], [], [], []
    grouped = df.groupby("Player")
    for player_name, group in grouped:
        group = group.sort_values("Year").reset_index(drop=True)
        if len(group) < SEQ_LEN + 1:
            continue
        vals = group[FEATURE_NAMES].values.astype(np.float32)
        labels = group["label"].values
        years = group["Year"].values
        pos = group["Pos"].iloc[0] if "Pos" in group.columns else ""
        for i in range(len(group) - SEQ_LEN):
            X_list.append(vals[i:i + SEQ_LEN])
            y_list.append(labels[i + SEQ_LEN])
            years_list.append(years[i + SEQ_LEN])
            players_list.append(player_name)
            pos_list.append(pos)
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    target_years = np.array(years_list, dtype=np.int64)
    players = np.array(players_list)
    positions = np.array(pos_list)
    return X, y, target_years, players, positions


PROCESSED_DIR = "data/processed"


def load_processed_data():
    """Load preprocessed data from disk. Must run load_and_preprocess() first."""
    data = {}
    for name in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
                 "players_test", "positions_test"]:
        path = f"{PROCESSED_DIR}/{name}.npy"
        if os.path.exists(path):
            data[name] = np.load(path)
    with open(f"{PROCESSED_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return (data["X_train"], data["y_train"]), \
           (data["X_val"], data["y_val"]), \
           (data["X_test"], data["y_test"]), scaler, \
           data.get("players_test"), data.get("positions_test")


def load_and_preprocess(csv_path: str, force: bool = False):
    if not force and os.path.exists(f"{PROCESSED_DIR}/X_train.npy"):
        print("Found cached preprocessed data. Loading from disk...")
        return load_processed_data()
    df = pd.read_csv(csv_path)

    df = _deduplicate(df)
    df = _fill_missing(df)
    df = _engineer_features(df)
    df = _make_labels(df)
    df = _drop_redundant(df)

    # Build sequences first (full timeline), then split by target year
    X, y, target_years, players, positions = _build_sequences(df)

    train_mask = target_years <= TRAIN_END
    val_mask   = (target_years > TRAIN_END) & (target_years <= VAL_END)
    test_mask  = (target_years > VAL_END) & (target_years <= TEST_END)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    players_test = players[test_mask]
    positions_test = positions[test_mask]

    # StandardScaler fit on training set only, then transform all
    scaler = StandardScaler()
    N_train, S, D = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, D)).reshape(N_train, S, D)

    N_val, S, D = X_val.shape
    X_val = scaler.transform(X_val.reshape(-1, D)).reshape(N_val, S, D)

    N_test, S, D = X_test.shape
    X_test = scaler.transform(X_test.reshape(-1, D)).reshape(N_test, S, D)

    # Persist to disk
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, arr in [("X_train", X_train), ("y_train", y_train),
                      ("X_val", X_val), ("y_val", y_val),
                      ("X_test", X_test), ("y_test", y_test),
                      ("players_test", players_test),
                      ("positions_test", positions_test)]:
        np.save(f"{PROCESSED_DIR}/{name}.npy", arr)
    with open(f"{PROCESSED_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved processed data to {PROCESSED_DIR}/")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler


if __name__ == "__main__":
    csv_path = "data/NBA_Season_Stats.csv"
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _, _, _ = load_and_preprocess(csv_path)

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    unique, counts = np.unique(y_train, return_counts=True)
    print("\nLabel distribution (train):")
    for u, c in zip(unique, counts):
        print(f"  {u} ({LABEL_NAMES[u]}): {c} ({c/len(y_train)*100:.1f}%)")

    unique_t, counts_t = np.unique(y_test, return_counts=True)
    print("\nLabel distribution (test):")
    for u, c in zip(unique_t, counts_t):
        print(f"  {u} ({LABEL_NAMES[u]}): {c} ({c/len(y_test)*100:.1f}%)")
