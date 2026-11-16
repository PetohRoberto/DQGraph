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

3. Link prediction without a validation set: edges are split into the
   observed graph (50 %), training (30 %) and test (20 %) positives, and the
   decision threshold is 0.5 (balanced positives / negatives). The former
   validation set only fed a threshold search whose result was not used.

6. Fixed splits (NodeSplit, EdgeSplit), drawn once on the original graph
   before any degradation, so that every λ uses the same split and a
   degradation can target the training data, the test data, or both.

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
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from neo4j_pool import get_driver
from value_types import to_number, value_kind, CONVERTIBLE_SHARE
from scipy.sparse import coo_matrix, lil_matrix
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

# Seed of every random choice: splits, negative sampling, walks, models.
SEED = 42

# Which data a degradation may alter (the three pollution scenarios of
# Mohammed et al.): training only, test only, or both.
DEGRADE_SCOPES = ("both", "train", "test")

# Node tasks: share of the nodes held out for testing.
NODE_TEST_SHARE = 0.20
# Link prediction: shares of the positive edges used as training and test
# examples; the rest (50 %) is the observed graph the features come from.
LP_TRAIN_SHARE = 0.30
LP_TEST_SHARE  = 0.20


@dataclass
class NodeSplit:
    """
    Train / test split of the nodes, drawn ONCE on the original graph so that
    every λ uses the same split and a degradation can target one side only.

    Copies made by the duplicate degradation (origin: copy id -> original id)
    take the side of their original. Under a one-sided scope, a copy whose
    original is on the clean side is used neither for training nor for
    testing, so that the clean side really stays clean.
    """
    train_ids: set
    test_ids: set
    scope: str = "both"
    origin: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)   # original (src, dst) pairs

    def side(self, nid: str) -> str | None:
        base = self.origin.get(nid)
        if base is None:
            return "test" if nid in self.test_ids else "train"
        side = "test" if base in self.test_ids else "train"
        return side if self.scope in ("both", side) else None

    def scope_nodes(self) -> set | None:
        """Nodes the degradation may alter (None = all of them)."""
        return {"both": None, "train": self.train_ids, "test": self.test_ids}[self.scope]

    def scope_pairs(self) -> set | None:
        """Edges the duplicate degradation may copy: those leaving a node of
        the degraded side (None = all of them)."""
        nodes = self.scope_nodes()
        return None if nodes is None else {(s, d) for s, d in self.edges if s in nodes}

    def describe(self) -> str:
        return f"{len(self.train_ids)} training / {len(self.test_ids)} test nodes"


@dataclass
class EdgeSplit:
    """
    Link-prediction split, drawn ONCE on the original graph: observed edges
    (features), training and test positives, and negatives sampled for each
    set (twice as many as positives, so that duplicated positives can keep a
    1:1 ratio; the first ones are used).

    After a degradation, a set keeps its edges that still exist, plus the
    copies of its edges (origin: copy pair -> original pair).
    """
    observed: list
    train_pos: list
    test_pos: list
    train_neg: list
    test_neg: list
    scope: str = "both"
    origin: dict = field(default_factory=dict)
    src_ids: set = field(default_factory=set)   # nodes of the source label

    def _pairs(self, side: str) -> list:
        return (self.train_pos + self.train_neg if side == "train"
                else self.test_pos + self.test_neg)

    def scope_nodes(self) -> set | None:
        """Source-label nodes the property / edge-coverage degradations may
        alter: the endpoints of the degraded side's pairs that the other side
        does not use (None = all nodes)."""
        if self.scope == "both":
            return None
        other = "test" if self.scope == "train" else "train"
        mine  = {n for pair in self._pairs(self.scope) for n in pair}
        theirs = {n for pair in self._pairs(other) for n in pair}
        return (mine - theirs) & self.src_ids

    def scope_pairs(self) -> set | None:
        """Edges the duplicate degradation may copy: the degraded side's
        positives (None = every edge, observed ones included)."""
        return {"both": None, "train": set(self.train_pos),
                "test": set(self.test_pos)}[self.scope]

    def describe(self) -> str:
        return (f"{len(self.observed)} observed edges, {len(self.train_pos)} training "
                f"/ {len(self.test_pos)} test positives")

# ── Algorithm catalogues ──────────────────────────────────────────────────────

ALGO_CLASSIFIERS: dict[str, callable] = {
    "Random Forest":       lambda: RandomForestClassifier(
        n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting":   lambda: GradientBoostingClassifier(
        n_estimators=150, random_state=SEED),
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=2000, random_state=SEED, class_weight="balanced"),
}

ALGO_REGRESSORS: dict[str, callable] = {
    "Random Forest":     lambda: RandomForestRegressor(
        n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=150, random_state=SEED),
    "Ridge":             lambda: Ridge(alpha=1.0),
}

ALGO_LP: dict[str, callable] = {
    "Random Forest":       lambda: RandomForestClassifier(
        n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting":   lambda: GradientBoostingClassifier(
        n_estimators=150, random_state=SEED),
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=2000, random_state=SEED, class_weight="balanced"),
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


# The fetchers sort their rows: Neo4j's return order is not guaranteed (it
# changes e.g. when edges are restored), and every seeded draw depends on it.

def _fetch_nodes(driver, db: str, label: str) -> list[dict]:
    with driver.session(database=db) as s:
        rows = s.run(
            f"MATCH (n:`{label}`) RETURN elementId(n) AS id, properties(n) AS props"
        ).data()
    return sorted(rows, key=lambda r: r["id"])


def _fetch_degrees(driver, db: str, label: str,
                   rel_type: str | None = None) -> dict:
    """Returns {node_id: (out_deg, in_deg)}."""
    with driver.session(database=db) as s:
        if rel_type:
            rows = s.run(f"""
                MATCH (n:`{label}`)
                RETURN elementId(n) AS id,
                       COUNT {{ (n)-[:`{rel_type}`]->() }} AS out_deg,
                       COUNT {{ (n)<-[:`{rel_type}`]-() }} AS in_deg
            """).data()
        else:
            rows = s.run(f"""
                MATCH (n:`{label}`)
                RETURN elementId(n) AS id,
                       COUNT {{ (n)-->() }} AS out_deg,
                       COUNT {{ (n)<--() }} AS in_deg
            """).data()
    return {r["id"]: (r["out_deg"], r["in_deg"]) for r in rows}


def _fetch_edges(driver, db: str, src_label: str,
                 rel_type: str, dst_label: str) -> list[tuple]:
    with driver.session(database=db) as s:
        rows = s.run(f"""
            MATCH (s:`{src_label}`)-[:`{rel_type}`]->(d:`{dst_label}`)
            RETURN elementId(s) AS src_id, elementId(d) AS dst_id
        """).data()
    return sorted((r["src_id"], r["dst_id"]) for r in rows)


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
    return sorted((r["src"], r["dst"]) for r in rows)


def _num(v):
    """Numeric value of a property (bools, numeric strings, dates) or None."""
    return float(v) if isinstance(v, bool) else to_number(v)


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
                     avg(toFloat(COUNT {{ (m)-->() }} +
                                 COUNT {{ (m)<--() }})) AS nbr_mean_deg
                RETURN nid, nbr_count, nbr_props, nbr_mean_deg
            """).data()
        else:
            rows = s.run(f"""
                MATCH (n:`{label}`)-[]-(m)
                WITH elementId(n) AS nid,
                     count(m) AS nbr_count,
                     collect(properties(m)) AS nbr_props,
                     avg(toFloat(COUNT {{ (m)-->() }} +
                                 COUNT {{ (m)<--() }})) AS nbr_mean_deg
                RETURN nid, nbr_count, nbr_props, nbr_mean_deg
            """).data()

    result: dict[str, dict] = {}
    prop_keys: list[str] | None = None
    for row in sorted(rows, key=lambda r: r["nid"]):
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
                # Union over the first node's neighbours, sorted: independent
                # of the order in which Neo4j collected them.
                prop_keys = sorted({
                    k for p in nbr_props_list if p for k, v in p.items()
                    if _num(v) is not None and k not in excl
                })[:50]
            for k in (prop_keys or []):
                fv = _num(props_dict.get(k))
                if fv is not None:
                    agg.setdefault(k, []).append(fv)
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
                if _num(v) is not None and k not in excl
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
                fv = _num(nb_props.get(k))
                if fv is not None:
                    agg[k].append(fv)
            deg_acc += len(adj.get(nb, []))
        rec["_nbr_mean_deg"] = deg_acc / len(neighbors) if neighbors else 0.0
        for k, vals in agg.items():
            if vals:
                rec[f"nbr_mean_{k}"] = float(np.mean(vals))
                if len(vals) > 1:
                    rec[f"nbr_std_{k}"] = float(np.std(vals))
        result[nid] = rec
    return result


def _uniform_walk_cooc(
    adj: dict[int, list[int]], active_nodes: list[int], n: int,
    num_walks: int, walk_length: int, window: int, rng,
):
    """
    Co-occurrence matrix of unbiased random walks (node2vec with p = q = 1).
    Every neighbour has the same probability, so all walks advance together
    as numpy arrays instead of one Python step at a time.
    """
    deg = np.array([len(adj[i]) for i in range(n)], dtype=np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(deg)
    indices = np.fromiter((nb for i in range(n) for nb in adj[i]),
                          dtype=np.int64, count=int(indptr[-1]))

    # Every node reached has ≥1 neighbour (undirected), so walks never stop early.
    starts = np.tile(np.asarray(active_nodes, dtype=np.int64), num_walks)
    walks = np.empty((len(starts), walk_length), dtype=np.int64)
    walks[:, 0] = starts
    for t in range(1, walk_length):
        cur = walks[:, t - 1]
        offset = (rng.random(len(cur)) * deg[cur]).astype(np.int64)
        walks[:, t] = indices[indptr[cur] + offset]

    # Pairs at distance 1..window, both directions (same counts as the loop).
    rows, cols = [], []
    for o in range(1, min(window, walk_length - 1) + 1):
        a, b = walks[:, :-o].ravel(), walks[:, o:].ravel()
        rows += [a, b]
        cols += [b, a]
    if not rows:
        return coo_matrix((n, n), dtype=np.float32).tocsr()
    r, c = np.concatenate(rows), np.concatenate(cols)
    return coo_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                      shape=(n, n)).tocsr()   # duplicates are summed


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
    seed: int = SEED,
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

    if p == 1.0 and q == 1.0:
        cooc = _uniform_walk_cooc(adj, active_nodes, n,
                                  num_walks, walk_length, window, rng)
    else:
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
        cooc = cooc.tocsr()

    actual_dim = min(embed_dim, n - 1)
    if actual_dim < 1:
        return {}

    svd = TruncatedSVD(n_components=actual_dim, random_state=seed)
    emb_matrix = svd.fit_transform(cooc)
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
    if not records:
        return pd.DataFrame(columns=["_id"])
    return _coerce_numeric_like(pd.DataFrame(records))


def _is_text(ser: pd.Series) -> bool:
    """Object or string column (pandas ≥ 3 gives strings their own 'str' dtype)."""
    return (pd.api.types.is_object_dtype(ser)
            or pd.api.types.is_string_dtype(ser))


def _coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    """
    Columns of numbers stored as strings ("1995", "7.5") or of dates become
    floats (dates → decimal year), so they are used as numbers and not hashed
    as text. Same rule as the schema explorer (value_types).
    """
    for col in df.columns:
        if col == "_id" or not _is_text(df[col]):
            continue
        vals = df[col].dropna()
        if vals.empty:
            continue
        kinds = vals.head(200).map(value_kind)
        if not any((kinds == k).mean() >= CONVERTIBLE_SHARE
                   for k in ("numeric", "temporal")):
            continue
        df[col] = df[col].map(
            lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v))
                      else to_number(v)
        ).astype(np.float64)
    return df


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
        elif _is_text(ser):
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
        if not _is_text(ser):
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
    return X.values, y, feat_cols, df["_id"].tolist()


def _node_split(node_rows: list[dict], target_prop: str, task_type: str,
                scope: str = "both") -> NodeSplit:
    """Train / test split of the nodes that carry the target, stratified for
    classification when every class has at least two nodes."""
    ids, labels = [], []
    for r in node_rows:
        v = (r.get("props") or {}).get(target_prop)
        if v is not None:
            ids.append(r["id"])
            labels.append(str(v))
    if len(ids) < 10:
        raise InfeasibleTaskError(
            f"Too few nodes with the target property '{target_prop}': {len(ids)}.")
    counts = Counter(labels)
    strat = (labels if task_type == "classification" and len(counts) > 1
             and min(counts.values()) >= 2 else None)
    train, test = train_test_split(ids, test_size=NODE_TEST_SHARE,
                                   random_state=SEED, stratify=strat)
    return NodeSplit(set(train), set(test), scope)


def _edge_split(pos_pairs: list[tuple], src_ids: list[str], dst_ids: list[str],
                scope: str = "both") -> EdgeSplit:
    """Observed (50 %) / training (30 %) / test (20 %) split of the positive
    edges, and the negatives of each set."""
    if len(pos_pairs) < MIN_EDGES_LP:
        raise InfeasibleTaskError(
            f"Only {len(pos_pairs)} positive edges — too few for LP.")
    rng = random.Random(SEED)
    shuffled = list(pos_pairs)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test  = max(5, int(n * LP_TEST_SHARE))
    n_train = max(5, int(n * LP_TRAIN_SHARE))
    test_pos  = shuffled[:n_test]
    train_pos = shuffled[n_test:n_test + n_train]
    observed  = shuffled[n_test + n_train:]
    if len(observed) < 20:
        raise InfeasibleTaskError(
            f"After holding out {n_test + n_train} edges for ML, only "
            f"{len(observed)} remain in the observed graph — too sparse to "
            "compute structural features.")

    pos_set, used = set(pos_pairs), set()

    def _sample_neg(n_target: int) -> list[tuple]:
        negs, attempts = [], 0
        while len(negs) < n_target and attempts < n_target * 30:
            pair = (rng.choice(src_ids), rng.choice(dst_ids))
            if pair not in pos_set and pair not in used:
                negs.append(pair)
                used.add(pair)
            attempts += 1
        return negs

    train_neg = _sample_neg(2 * len(train_pos))
    test_neg  = _sample_neg(2 * len(test_pos))
    if len(train_neg) < 5 or len(test_neg) < 2:
        raise InfeasibleTaskError("Could not sample enough negative pairs.")
    return EdgeSplit(observed, train_pos, test_pos, train_neg, test_neg, scope)


def _apply_node_split(X: np.ndarray, y: np.ndarray, ids: list, split: NodeSplit):
    side = np.array([split.side(i) or "" for i in ids])
    tr, te = side == "train", side == "test"
    if tr.sum() < 5 or te.sum() < 2:
        raise InfeasibleTaskError(
            f"Not enough nodes on each side of the split "
            f"(training: {int(tr.sum())}, test: {int(te.sum())}).")
    return X[tr], X[te], y[tr], y[te]


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
) -> np.ndarray:
    """CN, Jaccard, Adamic-Adar, Preferential Attachment, Resource Allocation."""
    n_src = adj.get(src_id, set())
    n_dst = adj.get(dst_id, set())
    common = n_src & n_dst
    cn = float(len(common))
    union = n_src | n_dst
    jaccard = cn / len(union) if union else 0.0
    aa = sum(1.0 / np.log(max(deg.get(w, 2), 2)) for w in common)
    pa = float(deg.get(src_id, 0) * deg.get(dst_id, 0))
    ra = sum(1.0 / max(deg.get(w, 1), 1) for w in common)
    return np.array([cn, jaccard, aa, pa, ra], dtype=np.float64)


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


def _check_node_features(fetch, label: str,
                          target_prop: str | None) -> tuple[int, list[str]]:
    """Returns (n_usable_features, warnings)."""
    rows = fetch(_fetch_nodes, label)
    if not rows:
        return 0, [f"No nodes of label '{label}'."]
    degrees = fetch(_fetch_degrees, label)
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


def _feasibility_classification(fetch, src_label, target_prop) -> dict:
    if not target_prop:
        return {"feasible": False, "reason": "No target property specified."}

    rows = fetch(_fetch_nodes, src_label)
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

    n_feat, warns = _check_node_features(fetch, src_label, target_prop)
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


def _feasibility_regression(fetch, src_label, target_prop) -> dict:
    if not target_prop:
        return {"feasible": False, "reason": "No target property specified."}

    rows = fetch(_fetch_nodes, src_label)
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
    n_feat, warns = _check_node_features(fetch, src_label, target_prop)
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


def _feasibility_lp(fetch, src_label, rel_type, dst_label) -> dict:
    n_edges = fetch(_fetch_edge_count, src_label, rel_type, dst_label)
    if n_edges < MIN_EDGES_LP:
        return {
            "feasible": False,
            "reason": (f"Only {n_edges} edges of type "
                       f"({src_label})-[:{rel_type}]->({dst_label}) "
                       f"(need ≥{MIN_EDGES_LP} for meaningful LP)."),
            "details": {"n_edges": n_edges},
        }

    n_src = fetch(_fetch_node_count, src_label)
    n_dst = fetch(_fetch_node_count, dst_label) \
            if dst_label != src_label else n_src

    # Sample CN signal: how discriminative is CN between pos and random neg?
    edges = fetch(_fetch_edges, src_label, rel_type, dst_label)
    sample_size = min(200, len(edges))
    rng = random.Random(SEED)
    sample_pos = rng.sample(edges, sample_size)

    adj: dict[str, set[str]] = {}
    for s, d in edges:
        adj.setdefault(s, set()).add(d)
        adj.setdefault(d, set()).add(s)

    # sorted: a set's iteration order changes between Python processes.
    src_ids = sorted({s for s, _ in edges}) or [edges[0][0]]
    dst_ids = sorted({d for _, d in edges}) or [edges[0][1]]
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

    n_feat_src, _ = _check_node_features(fetch, src_label, None)
    if dst_label != src_label:
        n_feat_dst, _ = _check_node_features(fetch, dst_label, None)
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
        self.driver    = get_driver(uri, user, pwd)
        self.db        = db
        self.cache_dir = cache_dir
        self._memo_store: dict = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def close(self):
        pass  # shared driver, see neo4j_pool

    def _memo(self, fn, *args, **kwargs):
        """Fetch once per instance: algorithms run on the same graph state
        (same λ) share the extracted data instead of re-querying Neo4j.
        Results are shared, callers must treat them as read-only."""
        def _h(v):
            return tuple(v) if isinstance(v, list) else v
        key = (fn, tuple(_h(a) for a in args),
               tuple(sorted((k, _h(v)) for k, v in kwargs.items())))
        if key not in self._memo_store:
            self._memo_store[key] = fn(self.driver, self.db, *args, **kwargs)
        return self._memo_store[key]

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
        if not force and ("feas", key) in self._memo_store:
            return self._memo_store[("feas", key)]
        if not force:
            cached = self._load_feas_cache(key)
            if cached is not None:
                cached["_from_cache"] = True
                return cached

        if task_id == "node_classification":
            result = _feasibility_classification(
                self._memo, src_label, target_prop
            )
        elif task_id == "node_regression":
            result = _feasibility_regression(
                self._memo, src_label, target_prop
            )
        elif task_id == "link_prediction":
            result = _feasibility_lp(
                self._memo, src_label, rel_type, dst_label
            )
        else:
            raise ValueError(f"Unknown task: {task_id}")

        result.setdefault("warnings", [])
        result["_from_cache"] = False
        self._save_feas_cache(key, result)
        self._memo_store[("feas", key)] = result
        return result

    def _node_Xy(self, src_label: str, rel_type: str,
                 target_prop: str, task_type: str):
        """Node features + target, built once and shared by all algorithms."""
        key = ("node_Xy", src_label, rel_type, target_prop, task_type)
        if key not in self._memo_store:
            node_rows = self._memo(_fetch_nodes, src_label)
            degrees   = self._memo(_fetch_degrees, src_label, rel_type)
            nbr_agg   = self._memo(_fetch_neighbor_agg, src_label,
                                   rel_type, exclude_props=[target_prop])
            n2v       = self._memo(_node2vec_embeddings, src_label, rel_type)
            df        = _build_node_df(node_rows, degrees, nbr_agg, n2v)
            df        = _encode_df(df, exclude_cols=[target_prop])
            X, y, _, ids = _prepare_X_y(df, target_prop, task_type)
            self._memo_store[key] = (X, y, ids)
        return self._memo_store[key]

    # ── Fixed split ──────────────────────────────────────────────────────────

    def make_split(self, task_id: str, src_label: str, rel_type: str | None,
                   dst_label: str | None, target_prop: str | None,
                   scope: str = "both") -> NodeSplit | EdgeSplit:
        """Split drawn on the CURRENT graph: call it before degrading, then
        pass it to run_task at every λ."""
        if scope not in DEGRADE_SCOPES:
            raise ValueError(f"Unknown scope '{scope}'.")
        if task_id == "link_prediction":
            pos = self._memo(_fetch_edges, src_label, rel_type, dst_label)
            src_ids = [r["id"] for r in self._memo(_fetch_nodes, src_label)]
            dst_ids = (src_ids if dst_label == src_label else
                       [r["id"] for r in self._memo(_fetch_nodes, dst_label)])
            split = _edge_split(pos, src_ids, dst_ids, scope)
            split.src_ids = set(src_ids)
            return split
        task_type = "classification" if task_id == "node_classification" else "regression"
        split = _node_split(self._memo(_fetch_nodes, src_label), target_prop,
                            task_type, scope)
        if rel_type and dst_label:
            split.edges = self._memo(_fetch_edges, src_label, rel_type, dst_label)
        return split

    # ── Node Classification ──────────────────────────────────────────────────

    def run_node_classification(
        self, src_label: str, rel_type: str,
        target_prop: str, algo_name: str, split: NodeSplit | None = None,
    ) -> dict[str, float]:
        X, y, ids = self._node_Xy(src_label, rel_type, target_prop, "classification")

        if split is not None:
            X_tr, X_te, y_tr, y_te = _apply_node_split(X, y, ids, split)
        else:
            strat = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=NODE_TEST_SHARE, random_state=SEED, stratify=strat
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
        target_prop: str, algo_name: str, split: NodeSplit | None = None,
    ) -> dict[str, float]:
        X, y, ids = self._node_Xy(src_label, rel_type, target_prop, "regression")

        if split is not None:
            X_tr, X_te, y_tr, y_te = _apply_node_split(X, y, ids, split)
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=NODE_TEST_SHARE, random_state=SEED)

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

    def _build_lp_dataset(self, src_label: str, rel_type: str, dst_label: str,
                          split: EdgeSplit | None = None) -> dict:
        """Pair features of the training and test sets, built once per graph
        state and shared by all algorithms (same pairs for each one).

        With a fixed split (drawn on the original graph), each set keeps its
        edges that still exist in the current graph plus the copies of its
        edges; its negatives are the first ones of the split, as many as
        positives. Without one, the split is drawn on the current graph."""
        if split is None:
            split = self.make_split("link_prediction", src_label, rel_type,
                                    dst_label, None)
        current = self._memo(_fetch_edges, src_label, rel_type, dst_label)
        cur_set = set(current)

        def _side(pairs: list) -> list:
            mine = set(pairs)
            return ([p for p in pairs if p in cur_set] +
                    [p for p in current if split.origin.get(p) in mine])

        graph_edges = _side(split.observed)
        train_pos   = _side(split.train_pos)
        test_pos    = _side(split.test_pos)
        train_neg   = split.train_neg[:len(train_pos)]
        test_neg    = split.test_neg[:len(test_pos)]
        if len(graph_edges) < 20:
            raise InfeasibleTaskError(
                f"Only {len(graph_edges)} edges left in the observed graph — too "
                "sparse to compute structural features.")

        # 2. Fetch node data (copies made by a duplicate degradation included)
        src_rows  = self._memo(_fetch_nodes, src_label)
        src_ids   = [r["id"] for r in src_rows]
        src_props = {r["id"]: r.get("props") or {} for r in src_rows}

        if dst_label == src_label:
            dst_rows, dst_ids, dst_props = src_rows, src_ids, src_props
        else:
            dst_rows  = self._memo(_fetch_nodes, dst_label)
            dst_ids   = [r["id"] for r in dst_rows]
            dst_props = {r["id"]: r.get("props") or {} for r in dst_rows}
        all_props = {**src_props, **dst_props}

        # 3. Features built from graph_edges only — same distribution for the
        #    training and test positives (none have their own edge in lp_adj).
        src_nbr_agg = _neighbor_agg_from_edges(graph_edges, all_props)
        dst_nbr_agg = _neighbor_agg_from_edges(
            [(d, s) for s, d in graph_edges], all_props
        )

        # Degrees also come from graph_edges only: counted on all edges, they
        # would include each held-out positive edge itself (+1 on both of its
        # endpoints, never on a negative pair), which leaks the label.
        src_degs = _degrees_from_edges(graph_edges, src_ids)
        dst_degs = src_degs if dst_label == src_label else \
            _degrees_from_edges([(d, s) for s, d in graph_edges], dst_ids)

        n2v = self._memo(_node2vec_embeddings, src_label, rel_type, edges=graph_edges)

        src_df = _build_node_df(src_rows, src_degs, src_nbr_agg, n2v)
        if dst_label == src_label:
            dst_df = src_df.copy()
        else:
            dst_df = _build_node_df(dst_rows, dst_degs, dst_nbr_agg, n2v)
        src_df = _encode_df(src_df)
        dst_df = _encode_df(dst_df)

        # 3b. Structural heuristics use graph_edges adjacency only
        lp_adj: dict[str, set[str]] = {}
        for s, d in graph_edges:
            lp_adj.setdefault(s, set()).add(d)
            lp_adj.setdefault(d, set()).add(s)
        lp_deg = {nid: len(nbrs) for nid, nbrs in lp_adj.items()}

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

        # 5. Build feature vectors
        def _make_Xy(pairs, label):
            Xl, yl = [], []
            for s, d in pairs:
                si = src_id2i.get(s)
                di = dst_id2i.get(d)
                if si is None or di is None:
                    continue
                node_feats   = _pair_features(s_X[si], d_X[di])
                struct_feats = _lp_heuristics(s, d, lp_adj, lp_deg)
                Xl.append(np.concatenate([node_feats, struct_feats]))
                yl.append(label)
            return Xl, yl

        tr_pos_X, tr_pos_y = _make_Xy(train_pos, 1)
        tr_neg_X, tr_neg_y = _make_Xy(train_neg, 0)
        te_pos_X, te_pos_y = _make_Xy(test_pos, 1)
        te_neg_X, te_neg_y = _make_Xy(test_neg, 0)

        n_tr_pos, n_tr_neg = len(tr_pos_y), len(tr_neg_y)
        n_te_pos, n_te_neg = len(te_pos_y), len(te_neg_y)
        if n_tr_pos < 5 or n_tr_neg < 5 or n_te_pos < 2 or n_te_neg < 2:
            raise InfeasibleTaskError(
                f"Not enough valid pairs after ID matching "
                f"(tr+={n_tr_pos} tr-={n_tr_neg} te+={n_te_pos} te-={n_te_neg})."
            )

        # Shuffle training data so order doesn't bias GBM / LR early stopping
        X_tr_l = tr_pos_X + tr_neg_X
        y_tr_l = tr_pos_y + tr_neg_y
        perm = np.random.RandomState(SEED).permutation(len(X_tr_l))
        X_tr = np.array(X_tr_l)[perm]
        y_tr = np.array(y_tr_l)[perm]

        X_te = np.array(te_pos_X + te_neg_X)
        y_te = np.array(te_pos_y + te_neg_y)

        return {
            "X_tr": X_tr, "y_tr": y_tr, "X_te": X_te, "y_te": y_te,
            "n_graph": len(graph_edges), "n_tr_pos": n_tr_pos, "n_te_pos": n_te_pos,
        }

    def run_link_prediction(
        self, src_label: str, rel_type: str,
        dst_label: str, algo_name: str, split: EdgeSplit | None = None,
    ) -> dict[str, float]:
        """
        Link prediction with a 3-way edge split:
          graph_edges (50%) → form the observed graph (lp_adj, n2v, nbr_agg)
          train_pos   (30%) → ML training positives (NOT in graph_edges)
          test_pos    (20%) → final evaluation      (NOT in graph_edges)

        Training and test positives both have features computed without their
        own edge in the adjacency — identical distribution. Negatives are as
        many as positives, so the decision threshold is 0.5 (0 for a
        decision function). There is no validation set: it only fed a
        threshold search whose result was not used.
        """
        key = ("lp_dataset", src_label, rel_type, dst_label)
        if key not in self._memo_store:
            self._memo_store[key] = self._build_lp_dataset(
                src_label, rel_type, dst_label, split)
        ds = self._memo_store[key]
        X_tr, y_tr = ds["X_tr"], ds["y_tr"]
        X_te, y_te = ds["X_te"], ds["y_te"]
        n_graph, n_tr_pos, n_te_pos = ds["n_graph"], ds["n_tr_pos"], ds["n_te_pos"]

        # 6. Train and score the test pairs
        clf = ALGO_LP[algo_name]()
        clf.fit(X_tr, y_tr)

        if hasattr(clf, "predict_proba"):
            te_score, threshold = clf.predict_proba(X_te)[:, 1], 0.5
        elif hasattr(clf, "decision_function"):
            te_score, threshold = clf.decision_function(X_te), 0.0
        else:
            te_score, threshold = clf.predict(X_te).astype(float), 0.5
        y_pred = (te_score >= threshold).astype(int)

        # Diagnostics
        pred_dist = dict(Counter(y_pred.tolist()))
        test_dist = dict(Counter(y_te.tolist()))
        f1 = f1_score(y_te, y_pred, zero_division=0)
        mean_pos_te = float(np.mean(te_score[y_te == 1])) if np.any(y_te == 1) else 0.0
        print(f"[LP diag] graph={n_graph} tr+={n_tr_pos} te+={n_te_pos} "
              f"| dim={X_tr.shape[1]} | mean_score_pos te={mean_pos_te:.3f}")
        print(f"[LP pred dist] y_test={test_dist} y_pred={pred_dist} | F1={f1:.3f}")

        return {
            "AUCPR":    float(average_precision_score(y_te, te_score)),
            "Accuracy": float(accuracy_score(y_te, y_pred)),
            "F1":       float(f1),
            "_diag":    (f"graph={n_graph} tr+={n_tr_pos} te+={n_te_pos} "
                         f"dim={X_tr.shape[1]} pred={pred_dist}"),
        }

    # ── Dispatcher ───────────────────────────────────────────────────────────

    def run_task(
        self, task_id: str,
        src_label: str, rel_type: str, dst_label: str,
        target_prop: str | None, algo_name: str,
        skip_feasibility: bool = False,
        split: NodeSplit | EdgeSplit | None = None,
    ) -> dict[str, float | None]:
        """
        Run a task after a feasibility check.

        Raises InfeasibleTaskError if the dataset cannot support the task.
        Set skip_feasibility=True to bypass (e.g. during degradation runs
        where you want to keep tracking metrics even when they collapse).
        split: fixed split from make_split() (drawn on the original graph);
        None draws one on the current graph.
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
                src_label, rel_type, target_prop, algo_name, split=split
            )
        if task_id == "node_regression":
            return self.run_node_regression(
                src_label, rel_type, target_prop, algo_name, split=split
            )
        if task_id == "link_prediction":
            return self.run_link_prediction(
                src_label, rel_type, dst_label, algo_name, split=split
            )
        raise ValueError(f"Unknown task: {task_id}")