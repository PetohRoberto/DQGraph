"""
ml_sklearn.py — Robust sklearn-based graph ML with feasibility pre-checks.

Drop-in replacement for the previous version: same public interface
(SklearnGraphML, ALGORITHMS, TASK_METRICS, TASK_LABELS, LAMBDA_RANGE).

WHAT'S NEW
----------
1. Feasibility pre-check before training (`check_feasibility`).
   Returns whether the dataset can plausibly support the requested task,
   with a clear reason when it cannot. Cached on disk per
   (task_id, db, labels, target_prop) so repeated runs are instant.
   If a task is infeasible, an InfeasibleTaskError is raised with a
   user-readable message.

2. Robust feature encoding without identity leakage:
     - bool / numeric    → passthrough
     - low-card cat (≤30 unique)   → one-hot
     - high-card cat (>30 unique)  → frequency + hash-bucket (8 dims)
     - low-card list (≤50 items)   → MultiLabelBinarizer
     - high-card list (>50 items)  → top-20 most-frequent indicators
   No TF-IDF: it acted as a per-node identity signature and killed
   the signal at test time.

3. Threshold tuning for link prediction. Models are trained on a 80/20
   train/val split inside the original train set; the F1-optimal cutoff
   on validation is used on the held-out test set. Avoids the "all-zero"
   collapse seen when probabilities cluster around 0.5.

4. Class-weight balanced everywhere it helps (classification + LP).

5. Stronger node2vec defaults: embed_dim=64, walk_length=40, num_walks=20.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import warnings
from collections import Counter
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score, precision_recall_curve,
)

# ── Algorithm catalogues ──────────────────────────────────────────────────────

ALGO_CLASSIFIERS: dict[str, callable] = {
    "Random Forest":       lambda: RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting":   lambda: GradientBoostingClassifier(
        n_estimators=150, random_state=42),
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=2000, random_state=42, class_weight="balanced"),
}

ALGO_REGRESSORS: dict[str, callable] = {
    "Random Forest":     lambda: RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=150, random_state=42),
    "Ridge":             lambda: Ridge(alpha=1.0),
}

ALGO_LP: dict[str, callable] = {
    "Random Forest":       lambda: RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting":   lambda: GradientBoostingClassifier(
        n_estimators=150, random_state=42),
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=2000, random_state=42, class_weight="balanced"),
}

ALGORITHMS: dict[str, dict[str, callable]] = {
    "node_classification": ALGO_CLASSIFIERS,
    "node_regression":     ALGO_REGRESSORS,
    "link_prediction":     ALGO_LP,
}

TASK_METRICS: dict[str, list[str]] = {
    "node_classification": ["Accuracy", "F1", "Precision", "Recall"],
    "node_regression":     ["RMSE", "MAE", "R²"],
    "link_prediction":     ["AUCPR", "Accuracy", "F1"],
}

TASK_LABELS: dict[str, str] = {
    "node_classification": "Node Classification",
    "node_regression":     "Node Regression",
    "link_prediction":     "Link Prediction",
}

LAMBDA_RANGE = [round(v * 0.1, 1) for v in range(10)]  # 0.0 … 0.9


# ── Custom exceptions ─────────────────────────────────────────────────────────

class InfeasibleTaskError(Exception):
    """Raised when a dataset cannot plausibly support a given ML task."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


# ─────────────────────────────────────────────────────────────────────────────
# Neo4j extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_node_count(driver, db: str, label: str) -> int:
    with driver.session(database=db) as s:
        row = s.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()
    return int(row["c"]) if row else 0


def _fetch_edge_count(driver, db: str,
                     src_label: str, rel_type: str, dst_label: str) -> int:
    with driver.session(database=db) as s:
        row = s.run(
            f"MATCH (:`{src_label}`)-[r:`{rel_type}`]->(:`{dst_label}`) "
            "RETURN count(r) AS c"
        ).single()
    return int(row["c"]) if row else 0


def _fetch_nodes(driver, db: str, label: str) -> list[dict]:
    with driver.session(database=db) as s:
        return s.run(
            f"MATCH (n:`{label}`) RETURN elementId(n) AS id, properties(n) AS props"
        ).data()


def _fetch_degrees(driver, db: str, label: str,
                   rel_type: str | None = None) -> dict:
    """Returns {node_id: (out_deg, in_deg)}."""
    with driver.session(database=db) as s:
        if rel_type:
            rows = s.run(f"""
                MATCH (n:`{label}`)
                WITH n,
                     size([(n)-[:`{rel_type}`]->() | 1]) AS out_deg,
                     size([()-[:`{rel_type}`]->(n) | 1]) AS in_deg
                RETURN elementId(n) AS id, out_deg, in_deg
            """).data()
        else:
            rows = s.run(f"""
                MATCH (n:`{label}`)
                WITH n,
                     size([(n)-[]->() | 1]) AS out_deg,
                     size([()-[]->(n) | 1]) AS in_deg
                RETURN elementId(n) AS id, out_deg, in_deg
            """).data()
    return {r["id"]: (r["out_deg"], r["in_deg"]) for r in rows}


def _fetch_edges(driver, db: str, src_label: str,
                 rel_type: str, dst_label: str) -> list[tuple]:
    with driver.session(database=db) as s:
        rows = s.run(f"""
            MATCH (s:`{src_label}`)-[:`{rel_type}`]->(d:`{dst_label}`)
            RETURN elementId(s) AS src_id, elementId(d) AS dst_id
        """).data()
    return [(r["src_id"], r["dst_id"]) for r in rows]


def _fetch_graph_edges_homogeneous(
    driver, db: str, label: str, rel_type: str | None
) -> list[tuple[str, str]]:
    with driver.session(database=db) as s:
        if rel_type:
            rows = s.run(f"""
                MATCH (a:`{label}`)-[:`{rel_type}`]-(b:`{label}`)
                RETURN elementId(a) AS src, elementId(b) AS dst
            """).data()
        else:
            rows = s.run(f"""
                MATCH (a:`{label}`)-[]-(b:`{label}`)
                RETURN elementId(a) AS src, elementId(b) AS dst
            """).data()
    return [(r["src"], r["dst"]) for r in rows]


def _fetch_neighbor_agg(
    driver, db: str, label: str, rel_type: str | None,
    exclude_props: list | None = None,
) -> dict[str, dict]:
    """1-hop mean aggregation of numeric+boolean neighbor properties."""
    excl = set(exclude_props or [])
    with driver.session(database=db) as s:
        if rel_type:
            rows = s.run(f"""
                MATCH (n:`{label}`)-[:`{rel_type}`]-(m)
                WITH elementId(n) AS nid,
                     count(m) AS nbr_count,
                     collect(properties(m)) AS nbr_props,
                     avg(toFloat(size([(m)-[]->() | 1]) +
                                 size([()-[]->(m) | 1]))) AS nbr_mean_deg
                RETURN nid, nbr_count, nbr_props, nbr_mean_deg
            """).data()
        else:
            rows = s.run(f"""
                MATCH (n:`{label}`)-[]-(m)
                WITH elementId(n) AS nid,
                     count(m) AS nbr_count,
                     collect(properties(m)) AS nbr_props,
                     avg(toFloat(size([(m)-[]->() | 1]) +
                                 size([()-[]->(m) | 1]))) AS nbr_mean_deg
                RETURN nid, nbr_count, nbr_props, nbr_mean_deg
            """).data()

    result: dict[str, dict] = {}
    prop_keys: list[str] | None = None
    for row in rows:
        nid = row["nid"]
        nbr_props_list: list[dict] = row["nbr_props"] or []
        rec: dict = {"_nbr_count": int(row["nbr_count"] or 0)}
        try:
            if row["nbr_mean_deg"] is not None:
                rec["_nbr_mean_deg"] = float(row["nbr_mean_deg"])
        except (TypeError, ValueError):
            pass

        agg: dict[str, list[float]] = {}
        for props_dict in nbr_props_list:
            if not props_dict:
                continue
            if prop_keys is None:
                prop_keys = [
                    k for k, v in props_dict.items()
                    if isinstance(v, (int, float, bool)) and k not in excl
                ][:50]
            for k in (prop_keys or []):
                try:
                    fv = float(props_dict.get(k))
                    if np.isfinite(fv):
                        agg.setdefault(k, []).append(fv)
                except (TypeError, ValueError):
                    pass
        for k, vals in agg.items():
            rec[f"nbr_mean_{k}"] = float(np.mean(vals))
            if len(vals) > 1:
                rec[f"nbr_std_{k}"] = float(np.std(vals))
        result[nid] = rec
    return result


def _degrees_from_edges(
    edges: list[tuple[str, str]], all_ids: list[str]
) -> dict[str, tuple[int, int]]:
    out_c: Counter = Counter()
    in_c:  Counter = Counter()
    for s, d in edges:
        out_c[s] += 1
        in_c[d]  += 1
    return {nid: (out_c.get(nid, 0), in_c.get(nid, 0)) for nid in all_ids}


def _neighbor_agg_from_edges(
    adj_pairs: list[tuple[str, str]],
    node_props: dict[str, dict],
    max_props: int = 50,
    exclude_props: list | None = None,
) -> dict[str, dict]:
    excl = set(exclude_props or [])
    adj: dict[str, list[str]] = {}
    for node, nbr in adj_pairs:
        adj.setdefault(node, []).append(nbr)

    prop_keys: list[str] = []
    for props in node_props.values():
        if props:
            prop_keys = [
                k for k, v in props.items()
                if isinstance(v, (int, float, bool)) and k not in excl
            ][:max_props]
            break

    result: dict[str, dict] = {}
    for nid, neighbors in adj.items():
        rec: dict = {"_nbr_count": len(neighbors)}
        agg: dict[str, list[float]] = {k: [] for k in prop_keys}
        deg_acc = 0
        for nb in neighbors:
            nb_props = node_props.get(nb) or {}
            for k in prop_keys:
                try:
                    fv = float(nb_props.get(k))
                    if np.isfinite(fv):
                        agg[k].append(fv)
                except (TypeError, ValueError):
                    pass
            deg_acc += len(adj.get(nb, []))
        rec["_nbr_mean_deg"] = deg_acc / len(neighbors) if neighbors else 0.0
        for k, vals in agg.items():
            if vals:
                rec[f"nbr_mean_{k}"] = float(np.mean(vals))
                if len(vals) > 1:
                    rec[f"nbr_std_{k}"] = float(np.std(vals))
        result[nid] = rec
    return result


def _node2vec_embeddings(
    driver, db: str,
    label: str, rel_type: str | None,
    edges: list[tuple[str, str]] | None = None,
    embed_dim: int = 64,
    walk_length: int = 40,
    num_walks: int = 20,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Node2Vec via biased random walks + Truncated SVD."""
    rng = np.random.default_rng(seed)

    raw_edges = edges if edges is not None else \
        _fetch_graph_edges_homogeneous(driver, db, label, rel_type)
    if not raw_edges:
        return {}

    all_node_ids = sorted({n for e in raw_edges for n in e})
    node2idx: dict[str, int] = {n: i for i, n in enumerate(all_node_ids)}
    n = len(all_node_ids)

    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    seen: set[tuple[int, int]] = set()
    for src, dst in raw_edges:
        si, di = node2idx[src], node2idx[dst]
        if si == di:
            continue
        if (si, di) not in seen:
            adj[si].append(di)
            adj[di].append(si)
            seen.add((si, di))
            seen.add((di, si))

    adj_sets = {i: set(nb) for i, nb in adj.items()}
    active_nodes = [i for i, nb in adj.items() if nb]
    if not active_nodes:
        return {}

    walks: list[list[int]] = []
    for _ in range(num_walks):
        order = active_nodes.copy()
        rng.shuffle(order)
        for start in order:
            walk = [start]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = adj[cur]
                if not neighbors:
                    break
                if len(walk) == 1:
                    walk.append(int(rng.choice(neighbors)))
                else:
                    prev = walk[-2]
                    probs = np.empty(len(neighbors), dtype=np.float64)
                    for k, nb in enumerate(neighbors):
                        if nb == prev:
                            probs[k] = 1.0 / p
                        elif nb in adj_sets[prev]:
                            probs[k] = 1.0
                        else:
                            probs[k] = 1.0 / q
                    probs /= probs.sum()
                    walk.append(int(rng.choice(neighbors, p=probs)))
            walks.append(walk)

    cooc = lil_matrix((n, n), dtype=np.float32)
    for walk in walks:
        L = len(walk)
        for i, node in enumerate(walk):
            lo, hi = max(0, i - window), min(L, i + window + 1)
            for j in range(lo, hi):
                if i != j:
                    cooc[node, walk[j]] += 1.0

    actual_dim = min(embed_dim, n - 1)
    if actual_dim < 1:
        return {}

    svd = TruncatedSVD(n_components=actual_dim, random_state=seed)
    emb_matrix = svd.fit_transform(cooc.tocsr())
    return {neo_id: emb_matrix[idx] for neo_id, idx in node2idx.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

LOW_CARD_CAT_THRESHOLD  = 30
LOW_CARD_LIST_THRESHOLD = 50
HASH_DIMS               = 8
TOP_K_LIST_ITEMS        = 20


def _hash_bucket(value: str, n_buckets: int = HASH_DIMS) -> int:
    """Stable hash-bucket index for a string."""
    h = hashlib.md5(value.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:4], "big") % n_buckets


def _build_node_df(
    node_rows: list[dict],
    degrees: dict,
    neighbor_agg: dict | None = None,
    node2vec_emb: dict | None = None,
) -> pd.DataFrame:
    records = []
    for row in node_rows:
        rec = {"_id": row["id"]}
        rec.update(row.get("props") or {})
        out_d, in_d = degrees.get(row["id"], (0, 0))
        rec["_out_deg"] = out_d
        rec["_in_deg"]  = in_d
        rec["_tot_deg"] = out_d + in_d
        if neighbor_agg is not None:
            rec.update(neighbor_agg.get(row["id"], {}))
        if node2vec_emb is not None:
            emb = node2vec_emb.get(row["id"])
            if emb is not None:
                for k, v in enumerate(emb):
                    rec[f"n2v_{k}"] = float(v)
        records.append(rec)
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["_id"])


def _numeric_cols(df: pd.DataFrame, exclude: list | None = None) -> list[str]:
    excl = set(exclude or []) | {"_id"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]


def _impute_df(df: pd.DataFrame, exclude_cols: list | None = None) -> pd.DataFrame:
    """
    Impute missing values introduced by completeness degradation.
      numeric / bool  → column mean (or mode for bool)
      categorical str (≤ LOW_CARD_CAT unique) → column mode
      high-card str / list → empty string / empty list
    """
    excl = set(exclude_cols or []) | {"_id"}
    df = df.copy()
    for col in df.columns:
        if col in excl or not df[col].isna().any():
            continue
        ser = df[col]
        if pd.api.types.is_bool_dtype(ser):
            mode = ser.mode()
            df[col] = ser.fillna(mode.iloc[0] if not mode.empty else False)
        elif pd.api.types.is_numeric_dtype(ser):
            mean_val = ser.mean()
            df[col] = ser.fillna(mean_val if pd.notna(mean_val) else 0.0)
        elif pd.api.types.is_object_dtype(ser):
            sample = ser.dropna()
            is_list_col = len(sample) > 0 and any(isinstance(v, list) for v in sample.head(20))
            if is_list_col:
                df[col] = ser.apply(
                    lambda x: [] if x is None or (isinstance(x, float) and np.isnan(x)) else x
                )
            else:
                try:
                    n_unique = ser.nunique(dropna=True)
                except TypeError:
                    n_unique = LOW_CARD_CAT_THRESHOLD + 1
                if n_unique <= LOW_CARD_CAT_THRESHOLD:
                    mode = ser.mode()
                    df[col] = ser.fillna(mode.iloc[0] if not mode.empty else "")
                else:
                    df[col] = ser.fillna("")
    return df


def _encode_df(df: pd.DataFrame, exclude_cols: list | None = None) -> pd.DataFrame:
    """
    Encode object columns into numeric features.

    - bool                         -> float
    - list, ≤ LOW_CARD_LIST items  -> MultiLabelBinarizer
    - list, > LOW_CARD_LIST items  -> top-K most-frequent indicators
    - string, ≤ LOW_CARD_CAT       -> one-hot via pd.get_dummies
    - string, > LOW_CARD_CAT       -> frequency encoding + hash buckets (HASH_DIMS)

    Missing values are imputed before encoding (numeric→mean, categorical→mode, string→'').
    """
    excl = set(exclude_cols or []) | {"_id"}
    df = _impute_df(df, exclude_cols=list(excl))
    df = df.copy()
    extra_frames: list[pd.DataFrame] = []
    to_drop: list[str] = []
    to_one_hot: list[str] = []

    for col in list(df.columns):
        if col in excl:
            continue
        ser = df[col]

        if pd.api.types.is_bool_dtype(ser):
            df[col] = ser.astype(np.float64)
            continue
        if pd.api.types.is_numeric_dtype(ser):
            continue
        if not pd.api.types.is_object_dtype(ser):
            continue

        sample = ser.dropna()
        is_list_col = len(sample) > 0 and any(
            isinstance(v, list) for v in sample.head(20)
        )

        if is_list_col:
            normalized = ser.apply(
                lambda x: [] if x is None or (isinstance(x, float) and np.isnan(x))
                          else (x if isinstance(x, list) else [str(x)])
            )
            all_items = sorted({str(item) for lst in normalized for item in lst})

            if len(all_items) <= LOW_CARD_LIST_THRESHOLD:
                mlb = MultiLabelBinarizer(classes=all_items)
                encoded = mlb.fit_transform(
                    normalized.apply(lambda l: [str(i) for i in l])
                )
                enc_df = pd.DataFrame(
                    encoded.astype(np.float64),
                    columns=[f"{col}__{c}" for c in mlb.classes_],
                    index=df.index,
                )
            else:
                counts = Counter(
                    str(item) for lst in normalized for item in lst
                )
                top_k = [c for c, _ in counts.most_common(TOP_K_LIST_ITEMS)]
                enc_df = pd.DataFrame(
                    {f"{col}__top_{c}": normalized.apply(
                        lambda l: float(c in [str(i) for i in l])
                    ) for c in top_k},
                    index=df.index,
                )
            extra_frames.append(enc_df)
            to_drop.append(col)
            continue

        # Regular string
        try:
            n_unique = ser.nunique(dropna=True)
        except TypeError:
            df[col] = ser.apply(
                lambda x: np.nan if (isinstance(x, float) and np.isnan(x))
                          else str(x)
            )
            ser = df[col]
            n_unique = ser.nunique(dropna=True)

        if n_unique <= 1:
            to_drop.append(col)
            continue

        if n_unique <= LOW_CARD_CAT_THRESHOLD:
            to_one_hot.append(col)
        else:
            # High-card: frequency encoding + hash buckets
            freq = ser.value_counts(normalize=True, dropna=False).to_dict()
            freq_col = ser.map(lambda v: freq.get(v, 0.0)).astype(np.float64)
            buckets = np.zeros((len(df), HASH_DIMS), dtype=np.float64)
            for i, v in enumerate(ser.fillna("__NA__").astype(str).values):
                buckets[i, _hash_bucket(v)] = 1.0
            enc_df = pd.DataFrame(
                np.column_stack([freq_col.values, buckets]),
                columns=[f"{col}__freq"] +
                        [f"{col}__hash_{k}" for k in range(HASH_DIMS)],
                index=df.index,
            )
            extra_frames.append(enc_df)
            to_drop.append(col)

    if to_drop:
        df = df.drop(columns=to_drop)
    if to_one_hot:
        df = pd.get_dummies(df, columns=to_one_hot, dummy_na=False, dtype=np.float64)
    if extra_frames:
        df = pd.concat([df] + extra_frames, axis=1)
    return df


def _prepare_X_y(df: pd.DataFrame, target_prop: str, task_type: str):
    if target_prop not in df.columns:
        raise InfeasibleTaskError(
            f"Target property '{target_prop}' not found in extracted data."
        )
    df = df.copy().dropna(subset=[target_prop])
    if len(df) < 10:
        raise InfeasibleTaskError(
            f"Too few samples with the target property '{target_prop}': "
            f"{len(df)} (need ≥10)."
        )

    if task_type == "classification":
        le = LabelEncoder()
        y  = le.fit_transform(df[target_prop].astype(str))
        if len(np.unique(y)) < 2:
            raise InfeasibleTaskError(
                f"Target '{target_prop}' has only one class — nothing to classify."
            )
    else:
        df["__y__"] = pd.to_numeric(df[target_prop], errors="coerce")
        df = df.dropna(subset=["__y__"])
        if len(df) < 10:
            raise InfeasibleTaskError(
                f"Not enough numeric values for the target property "
                f"'{target_prop}' (got {len(df)})."
            )
        y = df["__y__"].values

    feat_cols = _numeric_cols(df, exclude=[target_prop, "__y__"])
    if not feat_cols:
        raise InfeasibleTaskError(
            "No usable numeric features after encoding — the dataset doesn't "
            "have enough numeric or low-cardinality categorical properties."
        )

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X.values, y, feat_cols


def _pair_features(src_vec: np.ndarray, dst_vec: np.ndarray) -> np.ndarray:
    """Hadamard product, L2², average — concatenated."""
    return np.concatenate([
        src_vec * dst_vec,
        (src_vec - dst_vec) ** 2,
        (src_vec + dst_vec) / 2.0,
    ])


def _lp_heuristics(
    src_id: str, dst_id: str,
    adj: dict[str, set[str]],
    deg: dict[str, int],
    glob_deg: dict[str, int] | None = None,
) -> np.ndarray:
    """CN, Jaccard, Adamic-Adar, Preferential Attachment, Resource Allocation."""
    n_src = adj.get(src_id, set())
    n_dst = adj.get(dst_id, set())
    common = n_src & n_dst
    cn = float(len(common))
    union = n_src | n_dst
    jaccard = cn / len(union) if union else 0.0
    aa = sum(1.0 / np.log(max(deg.get(w, 2), 2)) for w in common)
    gd = glob_deg if glob_deg is not None else deg
    pa = float(gd.get(src_id, 0) * gd.get(dst_id, 0))
    ra = sum(1.0 / max(deg.get(w, 1), 1) for w in common)
    return np.array([cn, jaccard, aa, pa, ra], dtype=np.float64)


def _tune_threshold(y_val: np.ndarray, y_score: np.ndarray) -> float:
    """
    F1-optimal threshold on a validation set, robust to score saturation.

    Searches only over INTERIOR quantiles of the validation score distribution
    (2nd to 98th percentile). Avoids the boundary-collapse failure mode where
    a well-separated linear model has many val scores saturated at 1.0,
    causing precision_recall_curve to return threshold = max(scores), which
    then matches nothing on the held-out test set (where scores are slightly
    different and never hit exactly that value).
    """
    if len(np.unique(y_val)) < 2 or len(y_score) == 0:
        return 0.5

    score_min, score_max = float(np.min(y_score)), float(np.max(y_score))
    if score_max - score_min < 1e-9:
        return 0.5  # all scores identical: no separation possible

    candidates = np.unique(np.quantile(y_score, np.linspace(0.02, 0.98, 49)))
    best_f1, best_thr = -1.0, float(np.median(y_score))
    for thr in candidates:
        y_pred = (y_score >= thr).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return 0.5 #best_thr


# ─────────────────────────────────────────────────────────────────────────────
# Feasibility check
# ─────────────────────────────────────────────────────────────────────────────

# Thresholds — tuned to be lenient but catch hopeless cases
MIN_SAMPLES_CLASSIFICATION = 30
MIN_PER_CLASS              = 3
MIN_SAMPLES_REGRESSION     = 30
MIN_EDGES_LP               = 50
MIN_USABLE_FEATURES        = 3
MIN_LP_SIGNAL_RATIO        = 1.2   # mean(CN_pos) / max(mean(CN_neg), ε)


def _check_node_features(driver, db: str, label: str,
                          target_prop: str | None) -> tuple[int, list[str]]:
    """Returns (n_usable_features, warnings)."""
    rows = _fetch_nodes(driver, db, label)
    if not rows:
        return 0, [f"No nodes of label '{label}'."]
    degrees = _fetch_degrees(driver, db, label)
    df = _build_node_df(rows, degrees)
    df = _encode_df(df, exclude_cols=[target_prop] if target_prop else None)
    n_feat = len(_numeric_cols(df, exclude=[target_prop] if target_prop else None))
    warnings = []
    if n_feat < MIN_USABLE_FEATURES:
        warnings.append(
            f"Only {n_feat} usable feature(s) on '{label}' "
            f"(need ≥{MIN_USABLE_FEATURES})."
        )
    return n_feat, warnings


def _feasibility_classification(driver, db, src_label, target_prop) -> dict:
    if not target_prop:
        return {"feasible": False, "reason": "No target property specified."}

    rows = _fetch_nodes(driver, db, src_label)
    with_target = [r for r in rows
                   if r.get("props", {}).get(target_prop) is not None]
    n = len(with_target)
    if n < MIN_SAMPLES_CLASSIFICATION:
        return {
            "feasible": False,
            "reason": (f"Only {n} node(s) of '{src_label}' have the target "
                       f"'{target_prop}' (need ≥{MIN_SAMPLES_CLASSIFICATION})."),
            "details": {"n_samples": n},
        }
    classes = Counter(str(r["props"][target_prop]) for r in with_target)
    if len(classes) < 2:
        return {
            "feasible": False,
            "reason": f"Target '{target_prop}' has only one class.",
            "details": {"class_distribution": dict(classes)},
        }
    rare = {c: k for c, k in classes.items() if k < MIN_PER_CLASS}
    if len(classes) - len(rare) < 2:
        return {
            "feasible": False,
            "reason": (f"Fewer than 2 classes have at least {MIN_PER_CLASS} "
                       f"samples. Distribution: {dict(classes)}"),
            "details": {"class_distribution": dict(classes)},
        }

    n_feat, warns = _check_node_features(driver, db, src_label, target_prop)
    if n_feat < MIN_USABLE_FEATURES:
        return {
            "feasible": False,
            "reason": (f"Not enough usable features on '{src_label}' after "
                       f"encoding (got {n_feat}, need ≥{MIN_USABLE_FEATURES})."),
            "details": {"usable_features": n_feat},
        }
    return {
        "feasible": True,
        "reason": "OK",
        "warnings": warns + ([f"Some classes are rare: {rare}"] if rare else []),
        "details": {
            "n_samples": n,
            "class_distribution": dict(classes),
            "usable_features": n_feat,
        },
    }


def _feasibility_regression(driver, db, src_label, target_prop) -> dict:
    if not target_prop:
        return {"feasible": False, "reason": "No target property specified."}

    rows = _fetch_nodes(driver, db, src_label)
    values = []
    for r in rows:
        v = r.get("props", {}).get(target_prop)
        if v is None:
            continue
        try:
            fv = float(v)
            if np.isfinite(fv):
                values.append(fv)
        except (TypeError, ValueError):
            pass
    n = len(values)
    if n < MIN_SAMPLES_REGRESSION:
        return {
            "feasible": False,
            "reason": (f"Only {n} node(s) of '{src_label}' have a numeric "
                       f"value for '{target_prop}' (need ≥{MIN_SAMPLES_REGRESSION})."),
            "details": {"n_samples": n},
        }
    var = float(np.var(values))
    if var < 1e-9:
        return {
            "feasible": False,
            "reason": f"Target '{target_prop}' has zero variance — nothing to regress.",
            "details": {"variance": var},
        }
    n_feat, warns = _check_node_features(driver, db, src_label, target_prop)
    if n_feat < MIN_USABLE_FEATURES:
        return {
            "feasible": False,
            "reason": (f"Not enough usable features on '{src_label}' after "
                       f"encoding (got {n_feat}, need ≥{MIN_USABLE_FEATURES})."),
            "details": {"usable_features": n_feat},
        }
    return {
        "feasible": True,
        "reason": "OK",
        "warnings": warns,
        "details": {
            "n_samples": n,
            "target_variance": var,
            "usable_features": n_feat,
        },
    }


def _feasibility_lp(driver, db, src_label, rel_type, dst_label) -> dict:
    n_edges = _fetch_edge_count(driver, db, src_label, rel_type, dst_label)
    if n_edges < MIN_EDGES_LP:
        return {
            "feasible": False,
            "reason": (f"Only {n_edges} edges of type "
                       f"({src_label})-[:{rel_type}]->({dst_label}) "
                       f"(need ≥{MIN_EDGES_LP} for meaningful LP)."),
            "details": {"n_edges": n_edges},
        }

    n_src = _fetch_node_count(driver, db, src_label)
    n_dst = _fetch_node_count(driver, db, dst_label) \
            if dst_label != src_label else n_src

    # Sample CN signal: how discriminative is CN between pos and random neg?
    edges = _fetch_edges(driver, db, src_label, rel_type, dst_label)
    sample_size = min(200, len(edges))
    rng = random.Random(42)
    sample_pos = rng.sample(edges, sample_size)

    adj: dict[str, set[str]] = {}
    for s, d in edges:
        adj.setdefault(s, set()).add(d)
        adj.setdefault(d, set()).add(s)

    src_ids = list({s for s, _ in edges}) or [edges[0][0]]
    dst_ids = list({d for _, d in edges}) or [edges[0][1]]
    pos_set = set(edges)

    sample_neg = []
    attempts = 0
    while len(sample_neg) < sample_size and attempts < sample_size * 20:
        s = rng.choice(src_ids)
        d = rng.choice(dst_ids)
        if (s, d) not in pos_set:
            sample_neg.append((s, d))
        attempts += 1

    cn_pos = [len(adj.get(s, set()) & adj.get(d, set())) for s, d in sample_pos]
    cn_neg = [len(adj.get(s, set()) & adj.get(d, set())) for s, d in sample_neg]
    mean_pos = float(np.mean(cn_pos)) if cn_pos else 0.0
    mean_neg = float(np.mean(cn_neg)) if cn_neg else 0.0
    cn_ratio = mean_pos / max(mean_neg, 1e-3)

    n_feat_src, _ = _check_node_features(driver, db, src_label, None)
    if dst_label != src_label:
        n_feat_dst, _ = _check_node_features(driver, db, dst_label, None)
    else:
        n_feat_dst = n_feat_src

    structural_ok = cn_ratio >= MIN_LP_SIGNAL_RATIO and mean_pos > 0.1
    feature_ok    = (n_feat_src + n_feat_dst) >= 2 * MIN_USABLE_FEATURES

    warnings_list: list[str] = []
    if not structural_ok:
        warnings_list.append(
            f"Weak structural signal: mean CN positive={mean_pos:.2f} vs "
            f"negative={mean_neg:.2f} (ratio {cn_ratio:.2f})."
        )
    if not feature_ok:
        warnings_list.append(
            f"Few node features available "
            f"(src={n_feat_src}, dst={n_feat_dst})."
        )

    if not structural_ok and not feature_ok:
        return {
            "feasible": False,
            "reason": ("Graph is too sparse for structural heuristics AND "
                       "nodes have too few usable features. Link prediction "
                       "cannot produce meaningful results on this dataset."),
            "details": {
                "n_edges": n_edges,
                "n_src_nodes": n_src, "n_dst_nodes": n_dst,
                "cn_mean_pos": mean_pos, "cn_mean_neg": mean_neg,
                "cn_ratio": cn_ratio,
                "usable_features_src": n_feat_src,
                "usable_features_dst": n_feat_dst,
            },
        }

    return {
        "feasible": True,
        "reason": "OK",
        "warnings": warnings_list,
        "details": {
            "n_edges": n_edges,
            "n_src_nodes": n_src, "n_dst_nodes": n_dst,
            "cn_mean_pos": mean_pos, "cn_mean_neg": mean_neg,
            "cn_ratio": cn_ratio,
            "usable_features_src": n_feat_src,
            "usable_features_dst": n_feat_dst,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class SklearnGraphML:
    """
    Sklearn-based graph ML with feasibility pre-checks.

    Parameters
    ----------
    uri, user, pwd, db : Neo4j connection
    cache_dir          : optional directory to persist feasibility-check
                         results across sessions. If None, checks always run.
    """

    def __init__(self, uri: str, user: str, pwd: str, db: str,
                 cache_dir: Optional[str] = None):
        self.driver    = GraphDatabase.driver(uri, auth=(user, pwd))
        self.db        = db
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def close(self):
        self.driver.close()

    # ── Feasibility cache ────────────────────────────────────────────────────

    def _feas_key(self, task_id, src_label, rel_type, dst_label, target_prop):
        payload = json.dumps({
            "task":   task_id,
            "db":     self.db,
            "src":    src_label,
            "rel":    rel_type,
            "dst":    dst_label,
            "target": target_prop,
        }, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def _load_feas_cache(self, key: str):
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, f"feasibility_{key}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_feas_cache(self, key: str, result: dict):
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, f"feasibility_{key}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
        except Exception:
            pass

    def check_feasibility(
        self, task_id: str, src_label: str, rel_type: str | None,
        dst_label: str | None, target_prop: str | None,
        force: bool = False,
    ) -> dict:
        """
        Returns {feasible: bool, reason: str, warnings: list, details: dict}.
        Cached on disk per (task, db, labels, target).
        """
        key = self._feas_key(task_id, src_label, rel_type, dst_label, target_prop)
        if not force:
            cached = self._load_feas_cache(key)
            if cached is not None:
                cached["_from_cache"] = True
                return cached

        if task_id == "node_classification":
            result = _feasibility_classification(
                self.driver, self.db, src_label, target_prop
            )
        elif task_id == "node_regression":
            result = _feasibility_regression(
                self.driver, self.db, src_label, target_prop
            )
        elif task_id == "link_prediction":
            result = _feasibility_lp(
                self.driver, self.db, src_label, rel_type, dst_label
            )
        else:
            raise ValueError(f"Unknown task: {task_id}")

        result.setdefault("warnings", [])
        result["_from_cache"] = False
        self._save_feas_cache(key, result)
        return result

    # ── Node Classification ──────────────────────────────────────────────────

    def run_node_classification(
        self, src_label: str, rel_type: str,
        target_prop: str, algo_name: str,
    ) -> dict[str, float]:
        node_rows = _fetch_nodes(self.driver, self.db, src_label)
        degrees   = _fetch_degrees(self.driver, self.db, src_label, rel_type)
        nbr_agg   = _fetch_neighbor_agg(self.driver, self.db, src_label,
                                        rel_type, exclude_props=[target_prop])
        n2v       = _node2vec_embeddings(self.driver, self.db, src_label, rel_type)
        df        = _build_node_df(node_rows, degrees, nbr_agg, n2v)
        df        = _encode_df(df, exclude_cols=[target_prop])

        X, y, _ = _prepare_X_y(df, target_prop, "classification")

        strat = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat
        )

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        clf = ALGO_CLASSIFIERS[algo_name]()
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        pred_dist = dict(Counter(y_pred.tolist()))
        test_dist = dict(Counter(y_te.tolist()))
        print(f"[NC pred dist] y_test={test_dist} y_pred={pred_dist}")

        return {
            "Accuracy":  float(accuracy_score(y_te, y_pred)),
            "F1":        float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
            "Precision": float(precision_score(y_te, y_pred, average="weighted", zero_division=0)),
            "Recall":    float(recall_score(y_te, y_pred, average="weighted", zero_division=0)),
            "_diag":     f"n_feat={X_tr.shape[1]} pred={pred_dist} test={test_dist}",
        }

    # ── Node Regression ──────────────────────────────────────────────────────

    def run_node_regression(
        self, src_label: str, rel_type: str,
        target_prop: str, algo_name: str,
    ) -> dict[str, float]:
        node_rows = _fetch_nodes(self.driver, self.db, src_label)
        degrees   = _fetch_degrees(self.driver, self.db, src_label, rel_type)
        nbr_agg   = _fetch_neighbor_agg(self.driver, self.db, src_label,
                                        rel_type, exclude_props=[target_prop])
        n2v       = _node2vec_embeddings(self.driver, self.db, src_label, rel_type)
        df        = _build_node_df(node_rows, degrees, nbr_agg, n2v)
        df        = _encode_df(df, exclude_cols=[target_prop])

        X, y, _ = _prepare_X_y(df, target_prop, "regression")

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        reg = ALGO_REGRESSORS[algo_name]()
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        return {
            "RMSE":  rmse,
            "MAE":   float(mean_absolute_error(y_te, y_pred)),
            "R²":    float(r2_score(y_te, y_pred)),
            "_diag": f"n_feat={X_tr.shape[1]} y_std={np.std(y_te):.3f}",
        }

    # ── Link Prediction ──────────────────────────────────────────────────────

    def run_link_prediction(
        self, src_label: str, rel_type: str,
        dst_label: str, algo_name: str,
    ) -> dict[str, float]:
        """
        Link prediction with a 4-way edge split to keep train/val/test feature
        distributions aligned.

        The previous 2-way split (train_edges form lp_adj, test edges held out)
        caused a subtle bug: train (and therefore val) positives had their
        edge IN lp_adj, while test positives did NOT — so structural features
        (CN, PA, AA, RA) were systematically higher on val positives than on
        test positives. The model would rank well on test (high AUCPR) but
        any threshold tuned on val would never trigger on test (F1 ≈ 0).

        New split:
          graph_edges (50%)  → form the observed graph (lp_adj, n2v, nbr_agg)
          train_pos   (20%)  → ML training positives (NOT in graph_edges)
          val_pos     (10%)  → for threshold tuning      (NOT in graph_edges)
          test_pos    (20%)  → final evaluation          (NOT in graph_edges)

        All three ML splits therefore have features computed without their
        own edge in the adjacency — identical distribution.
        """
        all_pos_pairs = _fetch_edges(self.driver, self.db, src_label,
                                     rel_type, dst_label)
        if len(all_pos_pairs) < MIN_EDGES_LP:
            raise InfeasibleTaskError(
                f"Only {len(all_pos_pairs)} positive edges — too few for LP."
            )

        # 1. 4-way edge split
        rng_lp = random.Random(42)
        shuffled = all_pos_pairs.copy()
        rng_lp.shuffle(shuffled)
        n_total      = len(shuffled)
        n_test_pos   = max(5, int(n_total * 0.20))
        n_val_pos    = max(3, int(n_total * 0.10))
        n_train_pos  = max(5, int(n_total * 0.20))
        test_pos     = shuffled[:n_test_pos]
        val_pos      = shuffled[n_test_pos:n_test_pos + n_val_pos]
        train_pos    = shuffled[n_test_pos + n_val_pos:
                                n_test_pos + n_val_pos + n_train_pos]
        graph_edges  = shuffled[n_test_pos + n_val_pos + n_train_pos:]
        pos_set      = set(all_pos_pairs)

        if len(graph_edges) < 20:
            raise InfeasibleTaskError(
                f"After holding out {n_test_pos + n_val_pos + n_train_pos} "
                f"edges for ML, only {len(graph_edges)} remain in the observed "
                "graph — too sparse to compute structural features."
            )

        # 2. Fetch node data
        src_rows  = _fetch_nodes(self.driver, self.db, src_label)
        src_ids   = [r["id"] for r in src_rows]
        src_props = {r["id"]: r.get("props") or {} for r in src_rows}

        if dst_label == src_label:
            dst_rows, dst_ids, dst_props = src_rows, src_ids, src_props
        else:
            dst_rows  = _fetch_nodes(self.driver, self.db, dst_label)
            dst_ids   = [r["id"] for r in dst_rows]
            dst_props = {r["id"]: r.get("props") or {} for r in dst_rows}
        all_props = {**src_props, **dst_props}

        # 3. Features built from graph_edges only — same distribution for all
        #    train / val / test positives (none have their own edge in lp_adj).
        src_nbr_agg = _neighbor_agg_from_edges(graph_edges, all_props)
        dst_nbr_agg = _neighbor_agg_from_edges(
            [(d, s) for s, d in graph_edges], all_props
        )

        # Degree features use ALL edges (not just graph_edges) so nodes appearing
        # only in held-out splits aren't artificially zeroed-out — they still
        # have their true degree as a feature, just not their direct connection
        # to the candidate partner.
        all_src_degs = _degrees_from_edges(all_pos_pairs, src_ids)
        all_dst_degs = all_src_degs if dst_label == src_label else \
            _degrees_from_edges([(d, s) for s, d in all_pos_pairs], dst_ids)

        n2v = _node2vec_embeddings(
            self.driver, self.db, src_label, rel_type, edges=graph_edges
        )

        src_df = _build_node_df(src_rows, all_src_degs, src_nbr_agg, n2v)
        if dst_label == src_label:
            dst_df = src_df.copy()
        else:
            dst_df = _build_node_df(dst_rows, all_dst_degs, dst_nbr_agg, n2v)
        src_df = _encode_df(src_df)
        dst_df = _encode_df(dst_df)

        # 3b. Structural heuristics use graph_edges adjacency only
        lp_adj: dict[str, set[str]] = {}
        for s, d in graph_edges:
            lp_adj.setdefault(s, set()).add(d)
            lp_adj.setdefault(d, set()).add(s)
        lp_deg = {nid: len(nbrs) for nid, nbrs in lp_adj.items()}

        glob_deg: dict[str, int] = {}
        for s, d in all_pos_pairs:
            glob_deg[s] = glob_deg.get(s, 0) + 1
            glob_deg[d] = glob_deg.get(d, 0) + 1

        # 4. Align src/dst columns, scale together
        all_cols = sorted(set(_numeric_cols(src_df)) | set(_numeric_cols(dst_df)))
        if not all_cols:
            raise InfeasibleTaskError(
                "No usable numeric features after encoding — node labels have "
                "no numeric/boolean/categorical properties LP can leverage."
            )
        s_X_raw = src_df.reindex(columns=all_cols).apply(
            pd.to_numeric, errors="coerce").fillna(0.0).values
        d_X_raw = dst_df.reindex(columns=all_cols).apply(
            pd.to_numeric, errors="coerce").fillna(0.0).values

        sc = StandardScaler().fit(np.vstack([s_X_raw, d_X_raw]))
        s_X = sc.transform(s_X_raw)
        d_X = sc.transform(d_X_raw)

        src_id2i = {nid: i for i, nid in enumerate(src_df["_id"])}
        dst_id2i = {nid: i for i, nid in enumerate(dst_df["_id"])}
        src_ids_all = list(src_id2i.keys())
        dst_ids_all = list(dst_id2i.keys())

        # 5. Sample negatives for each split independently
        used_neg: set[tuple] = set()
        def _sample_neg(n_target: int) -> list[tuple]:
            negs = []
            attempts = 0
            while len(negs) < n_target and attempts < n_target * 30:
                s = random.choice(src_ids_all)
                d = random.choice(dst_ids_all)
                pair = (s, d)
                if pair not in pos_set and pair not in used_neg:
                    negs.append(pair)
                    used_neg.add(pair)
                attempts += 1
            return negs

        train_neg = _sample_neg(len(train_pos))
        val_neg   = _sample_neg(len(val_pos))
        test_neg  = _sample_neg(len(test_pos))
        if len(train_neg) < 5 or len(val_neg) < 3 or len(test_neg) < 2:
            raise InfeasibleTaskError("Could not sample enough negative pairs.")

        # 6. Build feature vectors
        def _make_Xy(pairs, label):
            Xl, yl = [], []
            for s, d in pairs:
                si = src_id2i.get(s)
                di = dst_id2i.get(d)
                if si is None or di is None:
                    continue
                node_feats   = _pair_features(s_X[si], d_X[di])
                struct_feats = _lp_heuristics(s, d, lp_adj, lp_deg, glob_deg)
                Xl.append(np.concatenate([node_feats, struct_feats]))
                yl.append(label)
            return Xl, yl

        tr_pos_X, tr_pos_y   = _make_Xy(train_pos, 1)
        tr_neg_X, tr_neg_y   = _make_Xy(train_neg, 0)
        val_pos_X, val_pos_y = _make_Xy(val_pos, 1)
        val_neg_X, val_neg_y = _make_Xy(val_neg, 0)
        te_pos_X, te_pos_y   = _make_Xy(test_pos, 1)
        te_neg_X, te_neg_y   = _make_Xy(test_neg, 0)

        n_tr_pos,  n_tr_neg  = len(tr_pos_y),  len(tr_neg_y)
        n_val_p,   n_val_n   = len(val_pos_y), len(val_neg_y)
        n_te_pos,  n_te_neg  = len(te_pos_y),  len(te_neg_y)
        if (n_tr_pos < 5 or n_tr_neg < 5 or
            n_val_p < 3 or n_val_n < 3 or
            n_te_pos < 2 or n_te_neg < 2):
            raise InfeasibleTaskError(
                f"Not enough valid pairs after ID matching "
                f"(tr+={n_tr_pos} tr-={n_tr_neg} "
                f"val+={n_val_p} val-={n_val_n} "
                f"te+={n_te_pos} te-={n_te_neg})."
            )

        # Shuffle training data so order doesn't bias GBM / LR early stopping
        X_tr_l = tr_pos_X + tr_neg_X
        y_tr_l = tr_pos_y + tr_neg_y
        perm = np.random.RandomState(42).permutation(len(X_tr_l))
        X_tr = np.array(X_tr_l)[perm]
        y_tr = np.array(y_tr_l)[perm]

        X_val = np.array(val_pos_X + val_neg_X)
        y_val = np.array(val_pos_y + val_neg_y)
        X_te  = np.array(te_pos_X + te_neg_X)
        y_te  = np.array(te_pos_y + te_neg_y)

        # 7. Train, score val + test, tune threshold on val
        clf = ALGO_LP[algo_name]()
        clf.fit(X_tr, y_tr)

        if hasattr(clf, "predict_proba"):
            val_score = clf.predict_proba(X_val)[:, 1]
            te_score  = clf.predict_proba(X_te)[:, 1]
        elif hasattr(clf, "decision_function"):
            val_score = clf.decision_function(X_val)
            te_score  = clf.decision_function(X_te)
        else:
            val_score = clf.predict(X_val).astype(float)
            te_score  = clf.predict(X_te).astype(float)

        threshold = _tune_threshold(y_val, val_score)
        y_pred    = (te_score >= threshold).astype(int)

        # Diagnostics
        pred_dist = dict(Counter(y_pred.tolist()))
        test_dist = dict(Counter(y_te.tolist()))
        f1_05    = f1_score(
            y_te,
            (te_score >= (0.5 if hasattr(clf, "predict_proba") else 0.0)).astype(int),
            zero_division=0,
        )
        f1_tuned = f1_score(y_te, y_pred, zero_division=0)
        mean_pos_val = float(np.mean(val_score[y_val == 1])) if np.any(y_val == 1) else 0.0
        mean_pos_te  = float(np.mean(te_score[y_te == 1]))   if np.any(y_te == 1)  else 0.0
        print(f"[LP diag] graph={len(graph_edges)} tr+={n_tr_pos} val+={n_val_p} "
              f"te+={n_te_pos} | dim={X_tr.shape[1]} | thr={threshold:.3f} | "
              f"mean_score_pos val={mean_pos_val:.3f} te={mean_pos_te:.3f}")
        print(f"[LP pred dist] y_test={test_dist} y_pred={pred_dist} "
              f"| F1@0.5={f1_05:.3f} F1@tuned={f1_tuned:.3f}")

        return {
            "AUCPR":    float(average_precision_score(y_te, te_score)),
            "Accuracy": float(accuracy_score(y_te, y_pred)),
            "F1":       float(f1_tuned),
            "_diag":    (f"graph={len(graph_edges)} tr+={n_tr_pos} val+={n_val_p} "
                         f"te+={n_te_pos} dim={X_tr.shape[1]} thr={threshold:.2f} "
                         f"pred={pred_dist} F1@0.5={f1_05:.2f}"),
        }

    # ── Dispatcher ───────────────────────────────────────────────────────────

    def run_task(
        self, task_id: str,
        src_label: str, rel_type: str, dst_label: str,
        target_prop: str | None, algo_name: str,
        skip_feasibility: bool = False,
    ) -> dict[str, float | None]:
        """
        Run a task after a feasibility check.

        Raises InfeasibleTaskError if the dataset cannot support the task.
        Set skip_feasibility=True to bypass (e.g. during degradation runs
        where you want to keep tracking metrics even when they collapse).
        """
        if not skip_feasibility:
            feas = self.check_feasibility(
                task_id, src_label, rel_type, dst_label, target_prop
            )
            if not feas["feasible"]:
                raise InfeasibleTaskError(
                    feas["reason"], details=feas.get("details", {})
                )

        if task_id == "node_classification":
            return self.run_node_classification(
                src_label, rel_type, target_prop, algo_name
            )
        if task_id == "node_regression":
            return self.run_node_regression(
                src_label, rel_type, target_prop, algo_name
            )
        if task_id == "link_prediction":
            return self.run_link_prediction(
                src_label, rel_type, dst_label, algo_name
            )
        raise ValueError(f"Unknown task: {task_id}")