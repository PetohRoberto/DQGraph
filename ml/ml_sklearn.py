"""
ml_sklearn.py — sklearn-based graph ML.

Extracts data from Neo4j into Python/pandas, then trains and evaluates
sklearn models locally — no GDS required.

Tasks supported:
  node_classification   → RandomForest / GradientBoosting / LogisticRegression
  node_regression       → RandomForest / GradientBoosting / Ridge
  link_prediction       → RandomForest / GradientBoosting / LogisticRegression
"""
from __future__ import annotations

import random
import warnings
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score,
)

# ── Algorithm catalogues ──────────────────────────────────────────────────────

ALGO_CLASSIFIERS: dict[str, callable] = {
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":   lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
}

ALGO_REGRESSORS: dict[str, callable] = {
    "Random Forest":     lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Ridge":             lambda: Ridge(),
}

ALGO_LP: dict[str, callable] = {
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":   lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
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

# ── Neo4j extraction helpers ──────────────────────────────────────────────────

def _fetch_nodes(driver, db: str, label: str) -> list[dict]:
    with driver.session(database=db) as s:
        return s.run(
            f"MATCH (n:`{label}`) RETURN elementId(n) AS id, properties(n) AS props"
        ).data()


def _fetch_degrees(driver, db: str, label: str, rel_type: str | None = None) -> dict:
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


def _fetch_neighbor_agg(
    driver, db: str, label: str, rel_type: str | None
) -> dict[str, dict]:
    """
    1-hop mean aggregation of neighbor numeric properties (GNN mean-aggregator in Cypher).
    Returns {node_id: {'nbr_mean_<prop>': float, '_nbr_mean_deg': float, '_nbr_count': int}}.
    Limits to first 12 numeric properties to avoid very wide feature vectors.
    """
    with driver.session(database=db) as s:
        if rel_type:
            rows = s.run(f"""
                MATCH (n:`{label}`)-[:`{rel_type}`]-(m)
                WITH elementId(n) AS nid,
                     count(m) AS nbr_count,
                     collect(properties(m)) AS nbr_props,
                     avg(toFloat(size([(m)-[]->() | 1]) + size([()-[]->(m) | 1]))) AS nbr_mean_deg
                RETURN nid, nbr_count, nbr_props, nbr_mean_deg
            """).data()
        else:
            rows = s.run(f"""
                MATCH (n:`{label}`)-[]-(m)
                WITH elementId(n) AS nid,
                     count(m) AS nbr_count,
                     collect(properties(m)) AS nbr_props,
                     avg(toFloat(size([(m)-[]->() | 1]) + size([()-[]->(m) | 1]))) AS nbr_mean_deg
                RETURN nid, nbr_count, nbr_props, nbr_mean_deg
            """).data()

    result: dict[str, dict] = {}
    prop_keys: list[str] | None = None  # Determined from first row with data

    for row in rows:
        nid = row["nid"]
        nbr_props_list: list[dict] = row["nbr_props"] or []
        nbr_count = int(row["nbr_count"] or 0)
        nbr_mean_deg = row.get("nbr_mean_deg")

        agg: dict[str, list[float]] = {}
        for props_dict in nbr_props_list:
            if not props_dict:
                continue
            if prop_keys is None:
                # Pick up to 12 numeric-valued property keys from the first neighbor
                prop_keys = [
                    k for k, v in props_dict.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ][:12]
            for k in (prop_keys or []):
                v = props_dict.get(k)
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        agg.setdefault(k, []).append(fv)
                except (TypeError, ValueError):
                    pass

        rec: dict = {"_nbr_count": nbr_count}
        if nbr_mean_deg is not None:
            try:
                rec["_nbr_mean_deg"] = float(nbr_mean_deg)
            except (TypeError, ValueError):
                pass
        for k, vals in agg.items():
            rec[f"nbr_mean_{k}"] = float(np.mean(vals))
            if len(vals) > 1:
                rec[f"nbr_std_{k}"] = float(np.std(vals))
        result[nid] = rec

    return result


def _fetch_graph_edges_homogeneous(
    driver, db: str, label: str, rel_type: str | None
) -> list[tuple[str, str]]:
    """Fetch undirected edges between nodes of the same label for graph construction."""
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


def _degrees_from_edges(
    edges: list[tuple[str, str]], all_ids: list[str]
) -> dict[str, tuple[int, int]]:
    """Compute (out_degree, in_degree) per node from a given edge list (no Neo4j query)."""
    from collections import Counter
    out_c: Counter = Counter()
    in_c:  Counter = Counter()
    for s, d in edges:
        out_c[s] += 1
        in_c[d]  += 1
    return {nid: (out_c.get(nid, 0), in_c.get(nid, 0)) for nid in all_ids}


def _neighbor_agg_from_edges(
    adj_pairs: list[tuple[str, str]],
    node_props: dict[str, dict],
    max_props: int = 12,
) -> dict[str, dict]:
    """
    Compute neighbor property aggregation from a given (node, neighbor) pair list.
    adj_pairs defines who counts as a neighbor of whom (pass undirected pairs as both
    (s,d) and (d,s), or directed as needed).
    node_props: {node_id: {prop_name: value}}
    """
    adj: dict[str, list[str]] = {}
    for node, nbr in adj_pairs:
        adj.setdefault(node, []).append(nbr)

    prop_keys: list[str] = []
    for props in node_props.values():
        if props:
            prop_keys = [
                k for k, v in props.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
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
                v = nb_props.get(k)
                try:
                    fv = float(v)
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
    driver,
    db: str,
    label: str,
    rel_type: str | None,
    edges: list[tuple[str, str]] | None = None,
    embed_dim: int = 32,
    walk_length: int = 20,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Node2Vec via biased random walks (p, q) + Truncated SVD on co-occurrence matrix.
    Returns {neo4j_element_id: np.ndarray of shape (embed_dim,)}.

    edges  — precomputed edge list; if None, fetched from Neo4j (full graph).
             Pass train edges only for link prediction to avoid leakage.
    p      — return parameter (high = less backtracking)
    q      — in-out parameter (low = DFS-like, high = BFS-like)
    """
    rng = np.random.default_rng(seed)

    # ── 1. Build adjacency ────────────────────────────────────────────────────
    raw_edges = edges if edges is not None else _fetch_graph_edges_homogeneous(driver, db, label, rel_type)
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

    adj_sets: dict[int, set[int]] = {i: set(nb) for i, nb in adj.items()}
    active_nodes = [i for i, nb in adj.items() if nb]
    if not active_nodes:
        return {}

    # ── 2. Biased random walks ────────────────────────────────────────────────
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

    # ── 3. Co-occurrence matrix (sparse) ──────────────────────────────────────
    cooc = lil_matrix((n, n), dtype=np.float32)
    for walk in walks:
        L = len(walk)
        for i, node in enumerate(walk):
            lo = max(0, i - window)
            hi = min(L, i + window + 1)
            for j in range(lo, hi):
                if i != j:
                    cooc[node, walk[j]] += 1.0

    # ── 4. Truncated SVD ──────────────────────────────────────────────────────
    actual_dim = min(embed_dim, n - 1)
    if actual_dim < 1:
        return {}

    svd = TruncatedSVD(n_components=actual_dim, random_state=seed)
    emb_matrix = svd.fit_transform(cooc.tocsr())  # (n, actual_dim)

    # ── 5. Map back to Neo4j IDs ──────────────────────────────────────────────
    return {neo_id: emb_matrix[idx] for neo_id, idx in node2idx.items()}


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


def _prepare_X_y(df: pd.DataFrame, target_prop: str, task_type: str):
    """
    Returns (X_array, y_array, feature_names) or raises ValueError.
    task_type: 'classification' | 'regression'
    """
    if target_prop not in df.columns:
        raise ValueError(f"Target property '{target_prop}' not found in data.")

    df = df.copy().dropna(subset=[target_prop])
    if len(df) < 10:
        raise ValueError(f"Too few samples with the target property: {len(df)}.")

    if task_type == "classification":
        le = LabelEncoder()
        y  = le.fit_transform(df[target_prop].astype(str))
        if len(np.unique(y)) < 2:
            raise ValueError("Target has only one class.")
    else:
        df = df.copy()
        df["__y__"] = pd.to_numeric(df[target_prop], errors="coerce")
        df = df.dropna(subset=["__y__"])
        if len(df) < 10:
            raise ValueError("Not enough numeric values for the target property.")
        y = df["__y__"].values

    feat_cols = _numeric_cols(df, exclude=[target_prop, "__y__"])
    if not feat_cols:
        raise ValueError("No numeric features available.")

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(df[feat_cols].median())
    return X.values, y, feat_cols


def _pair_features(src_vec: np.ndarray, dst_vec: np.ndarray) -> np.ndarray:
    """Hadamard product, L2 squared, average — concatenated."""
    h = src_vec * dst_vec
    l = (src_vec - dst_vec) ** 2
    a = (src_vec + dst_vec) / 2.0
    return np.concatenate([h, l, a])


# ── Main ML class ─────────────────────────────────────────────────────────────

class SklearnGraphML:
    def __init__(self, uri: str, user: str, pwd: str, db: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self.db     = db

    def close(self):
        self.driver.close()

    # ── Node Classification ───────────────────────────────────────────────────

    def run_node_classification(
        self, src_label: str, rel_type: str,
        target_prop: str, algo_name: str,
    ) -> dict[str, float]:
        node_rows   = _fetch_nodes(self.driver, self.db, src_label)
        degrees     = _fetch_degrees(self.driver, self.db, src_label, rel_type)
        nbr_agg     = _fetch_neighbor_agg(self.driver, self.db, src_label, rel_type)
        n2v         = _node2vec_embeddings(self.driver, self.db, src_label, rel_type)
        df          = _build_node_df(node_rows, degrees, nbr_agg, n2v)

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

        return {
            "Accuracy":  float(accuracy_score(y_te, y_pred)),
            "F1":        float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
            "Precision": float(precision_score(y_te, y_pred, average="weighted", zero_division=0)),
            "Recall":    float(recall_score(y_te, y_pred, average="weighted", zero_division=0)),
        }

    # ── Node Regression ───────────────────────────────────────────────────────

    def run_node_regression(
        self, src_label: str, rel_type: str,
        target_prop: str, algo_name: str,
    ) -> dict[str, float]:
        node_rows   = _fetch_nodes(self.driver, self.db, src_label)
        degrees     = _fetch_degrees(self.driver, self.db, src_label, rel_type)
        nbr_agg     = _fetch_neighbor_agg(self.driver, self.db, src_label, rel_type)
        n2v         = _node2vec_embeddings(self.driver, self.db, src_label, rel_type)
        df          = _build_node_df(node_rows, degrees, nbr_agg, n2v)

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
            "RMSE": rmse,
            "MAE":  float(mean_absolute_error(y_te, y_pred)),
            "R²":   float(r2_score(y_te, y_pred)),
        }

    # ── Link Prediction ───────────────────────────────────────────────────────

    def run_link_prediction(
        self, src_label: str, rel_type: str,
        dst_label: str, algo_name: str,
    ) -> dict[str, float]:
        all_pos_pairs = _fetch_edges(self.driver, self.db, src_label, rel_type, dst_label)
        if len(all_pos_pairs) < 10:
            raise ValueError(f"Not enough positive edges: {len(all_pos_pairs)}.")

        # ── 1. Split edges FIRST — prevents leakage into features ─────────────
        # node2vec, degrees and neighbor stats are computed on train edges only,
        # so the model never sees test edge topology during feature construction.
        rng_lp = random.Random(42)
        shuffled = all_pos_pairs.copy()
        rng_lp.shuffle(shuffled)
        n_test_pos = max(5, int(len(shuffled) * 0.2))
        test_pos  = shuffled[:n_test_pos]
        train_pos = shuffled[n_test_pos:]

        pos_set = set(all_pos_pairs)

        # ── 2. Fetch node data ─────────────────────────────────────────────────
        src_rows  = _fetch_nodes(self.driver, self.db, src_label)
        src_ids   = [r["id"] for r in src_rows]
        src_props = {r["id"]: r.get("props") or {} for r in src_rows}

        if dst_label == src_label:
            dst_rows  = src_rows
            dst_ids   = src_ids
            dst_props = src_props
        else:
            dst_rows  = _fetch_nodes(self.driver, self.db, dst_label)
            dst_ids   = [r["id"] for r in dst_rows]
            dst_props = {r["id"]: r.get("props") or {} for r in dst_rows}

        all_props = {**src_props, **dst_props}

        # ── 3. Structural features from TRAIN edges only ───────────────────────
        src_degs    = _degrees_from_edges(train_pos, src_ids)
        src_nbr_agg = _neighbor_agg_from_edges(train_pos, all_props)
        dst_degs    = _degrees_from_edges([(d, s) for s, d in train_pos], dst_ids)
        dst_nbr_agg = _neighbor_agg_from_edges([(d, s) for s, d in train_pos], all_props)

        if dst_label == src_label:
            n2v    = _node2vec_embeddings(
                self.driver, self.db, src_label, rel_type, edges=train_pos
            )
            src_df = _build_node_df(src_rows, src_degs, src_nbr_agg, n2v)
            dst_df = src_df.copy()
        else:
            # Bipartite: one shared embedding space built from train edges
            shared_n2v = _node2vec_embeddings(
                self.driver, self.db, src_label, rel_type, edges=train_pos
            )
            src_df = _build_node_df(src_rows, src_degs, src_nbr_agg, shared_n2v)
            dst_df = _build_node_df(dst_rows, dst_degs, dst_nbr_agg, shared_n2v)

        # ── 4. Scale features (fit on train nodes only) ────────────────────────
        s_cols  = _numeric_cols(src_df)
        d_cols  = _numeric_cols(dst_df)
        s_X_raw = src_df[s_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        d_X_raw = dst_df[d_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

        s_sc = StandardScaler().fit(s_X_raw) if s_X_raw.shape[1] > 0 else None
        d_sc = StandardScaler().fit(d_X_raw) if d_X_raw.shape[1] > 0 else None
        s_X = s_sc.transform(s_X_raw) if s_sc else np.zeros((len(src_df), 1))
        d_X = d_sc.transform(d_X_raw) if d_sc else np.zeros((len(dst_df), 1))

        src_id2i    = {nid: i for i, nid in enumerate(src_df["_id"])}
        dst_id2i    = {nid: i for i, nid in enumerate(dst_df["_id"])}
        src_ids_all = list(src_id2i.keys())
        dst_ids_all = list(dst_id2i.keys())

        # ── 5. Sample negatives for train and test independently ───────────────
        used_neg: set[tuple] = set()

        def _sample_neg(n_target: int) -> list[tuple]:
            negs: list[tuple] = []
            attempts = 0
            while len(negs) < n_target and attempts < n_target * 20:
                s = random.choice(src_ids_all)
                d = random.choice(dst_ids_all)
                pair = (s, d)
                if pair not in pos_set and pair not in used_neg:
                    negs.append(pair)
                    used_neg.add(pair)
                attempts += 1
            return negs

        train_neg = _sample_neg(min(len(train_pos), 10_000))
        test_neg  = _sample_neg(len(test_pos))

        if len(train_neg) < 5 or len(test_neg) < 2:
            raise ValueError("Could not sample enough negative pairs.")

        # ── 6. Build feature vectors ───────────────────────────────────────────
        def _make_Xy(pairs: list[tuple], label: int) -> tuple[list, list]:
            Xl, yl = [], []
            for s, d in pairs:
                si = src_id2i.get(s)
                di = dst_id2i.get(d)
                if si is None or di is None:
                    continue
                Xl.append(_pair_features(s_X[si], d_X[di]))
                yl.append(label)
            return Xl, yl

        X_tr_l, y_tr_l = [], []
        for xb, yb in [_make_Xy(train_pos, 1), _make_Xy(train_neg, 0)]:
            X_tr_l.extend(xb); y_tr_l.extend(yb)

        X_te_l, y_te_l = [], []
        for xb, yb in [_make_Xy(test_pos, 1), _make_Xy(test_neg, 0)]:
            X_te_l.extend(xb); y_te_l.extend(yb)

        if len(X_tr_l) < 10 or len(X_te_l) < 4:
            raise ValueError("Not enough valid pairs for training.")

        X_tr, y_tr = np.array(X_tr_l), np.array(y_tr_l)
        X_te, y_te = np.array(X_te_l), np.array(y_te_l)

        # ── 7. Train and evaluate ──────────────────────────────────────────────
        clf = ALGO_LP[algo_name]()
        clf.fit(X_tr, y_tr)
        y_pred  = clf.predict(X_te)
        y_score = (clf.predict_proba(X_te)[:, 1]
                   if hasattr(clf, "predict_proba") else y_pred.astype(float))

        return {
            "AUCPR":    float(average_precision_score(y_te, y_score)),
            "Accuracy": float(accuracy_score(y_te, y_pred)),
            "F1":       float(f1_score(y_te, y_pred, zero_division=0)),
        }

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def run_task(
        self, task_id: str,
        src_label: str, rel_type: str, dst_label: str,
        target_prop: str | None, algo_name: str,
    ) -> dict[str, float | None]:
        if task_id == "node_classification":
            return self.run_node_classification(src_label, rel_type, target_prop, algo_name)
        if task_id == "node_regression":
            return self.run_node_regression(src_label, rel_type, target_prop, algo_name)
        if task_id == "link_prediction":
            return self.run_link_prediction(src_label, rel_type, dst_label, algo_name)
        raise ValueError(f"Unknown task: {task_id}")
