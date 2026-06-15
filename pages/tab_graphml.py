"""
pages/tab_graphml.py — Graph ML Impact tab.

- Extraction des données de Neo4j en Python/pandas
- Training avec sklearn (pas de GDS)
- Choix de jusqu'à 3 algorithmes par tâche
- λ fixe : 0.0 → 0.9, pas de 0.1
- Cache par config dans le dossier projet
- Graphique : X = dégradation λ, Y = métrique, une courbe par algo
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from neo4j import GraphDatabase

from ui.schema import SchemaExplorer
from degraders import DataCompletenessDegrader
from ml.ml_sklearn import (
    SklearnGraphML, ALGORITHMS, TASK_METRICS, TASK_LABELS, LAMBDA_RANGE,
    ALGO_CLASSIFIERS, ALGO_REGRESSORS, ALGO_LP,
)
from project_manager import Project

# ── cached schema fetchers ────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def _labels(uri, user, pwd, db):
    s = SchemaExplorer(uri, user, pwd, db); r = s.get_node_labels(); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _rels(uri, user, pwd, db):
    s = SchemaExplorer(uri, user, pwd, db); r = s.get_relationship_types(); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _rels_for_src(uri, user, pwd, db, src):
    s = SchemaExplorer(uri, user, pwd, db); r = s.get_rel_types_for_source(src); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _dst_for(uri, user, pwd, db, src, rel):
    s = SchemaExplorer(uri, user, pwd, db); r = s.get_dst_labels_for(src, rel); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _node_props(uri, user, pwd, db, lbl):
    s = SchemaExplorer(uri, user, pwd, db); r = s.get_node_properties(lbl); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _numeric_props(uri, user, pwd, db, lbl):
    s = SchemaExplorer(uri, user, pwd, db); r = s.get_numeric_properties(lbl); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _categorical_props(uri, user, pwd, db, lbl):
    s = SchemaExplorer(uri, user, pwd, db); r = s.get_categorical_properties(lbl); s.close(); return r


# ── degradation config ────────────────────────────────────────────────────────
DEGRAD_DIMS = {
    "Property Completeness": "property_completeness",
    "Edge Coverage":         "edge_coverage",
    "Property Accuracy":     "property_accuracy",
}

COLORS = ["#4f8cff", "#00e5c0", "#ffd166", "#ff6b6b", "#b57bee", "#ff9f43"]


def _normalize_results(data: dict) -> dict:
    """JSON always serializes dict keys as strings; convert lambda keys back to float."""
    out = {}
    for algo, metrics in data.items():
        out[algo] = {}
        for metric, lam_vals in metrics.items():
            out[algo][metric] = {
                (float(k) if isinstance(k, str) else k): v
                for k, v in lam_vals.items()
            }
    return out


# ── Neo4j backup / restore ────────────────────────────────────────────────────

def _backup_nodes(uri, user, pwd, db, label):
    d = GraphDatabase.driver(uri, auth=(user, pwd))
    with d.session(database=db) as s:
        rows = s.run(f"MATCH (n:`{label}`) "
                     "RETURN elementId(n) AS eid, properties(n) AS props").data()
    d.close()
    return rows


def _restore_nodes(uri, user, pwd, db, label, backup):
    d = GraphDatabase.driver(uri, auth=(user, pwd))
    with d.session(database=db) as s:
        for row in backup:
            s.run("MATCH (n) WHERE elementId(n) = $eid SET n = $props",
                  {"eid": row["eid"], "props": row["props"]})
    d.close()


def _backup_edges(uri, user, pwd, db, src, rel, dst):
    d = GraphDatabase.driver(uri, auth=(user, pwd))
    with d.session(database=db) as s:
        rows = s.run(f"MATCH (s:`{src}`)-[r:`{rel}`]->(d:`{dst}`) "
                     "RETURN elementId(s) AS sid, elementId(d) AS did, "
                     "properties(r) AS props").data()
    d.close()
    return rows


def _restore_edges(uri, user, pwd, db, src, rel, dst, backup):
    d = GraphDatabase.driver(uri, auth=(user, pwd))
    with d.session(database=db) as s:
        s.run(f"MATCH (:`{src}`)-[r:`{rel}`]->(:`{dst}`) DELETE r")
        for row in backup:
            s.run(f"MATCH (s) WHERE elementId(s) = $sid "
                  f"MATCH (d) WHERE elementId(d) = $did "
                  f"CREATE (s)-[r:`{rel}`]->(d) SET r = $props",
                  {"sid": row["sid"], "did": row["did"],
                   "props": row.get("props", {})})
    d.close()


# ── chart ─────────────────────────────────────────────────────────────────────

def _make_multi_algo_chart(
    lambdas: list, algo_data: dict[str, list],
    metric_name: str, task_name: str, degrad_dim: str,
) -> go.Figure:
    """
    algo_data: {algo_name: [value_at_lambda_0, value_at_lambda_1, ...]}
    None values are replaced by NaN in the trace.
    """
    fig = go.Figure()

    for i, (algo_name, vals) in enumerate(algo_data.items()):
        color = COLORS[i % len(COLORS)]
        rgb   = ",".join(str(int(color.lstrip("#")[j:j+2], 16)) for j in (0, 2, 4))

        y_plot = [v if v is not None else float("nan") for v in vals]
        fig.add_trace(go.Scatter(
            x=lambdas,
            y=y_plot,
            mode="lines+markers",
            name=algo_name,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color, line=dict(color="#080b12", width=2)),
            fill="none",
            hovertemplate=f"<b>{algo_name}</b><br>λ = %{{x:.1f}}<br>{metric_name} = %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title={
            "text": f"{task_name} — {metric_name} vs Degradation ({degrad_dim})",
            "font": {"family": "Syne", "size": 13, "color": "#e8eaf0"},
        },
        xaxis={
            "title": {"text": "Degradation λ",
                      "font": {"family": "DM Mono", "size": 11, "color": "#5a6480"}},
            "tickfont": {"family": "DM Mono", "size": 10, "color": "#5a6480"},
            "gridcolor": "#1c2333", "zeroline": False, "dtick": 0.1,
        },
        yaxis={
            "title": {"text": metric_name,
                      "font": {"family": "DM Mono", "size": 11, "color": "#5a6480"}},
            "tickfont": {"size": 10, "color": "#5a6480"},
            "gridcolor": "#1c2333", "zeroline": False,
        },
        legend={
            "font": {"family": "DM Mono", "size": 11, "color": "#e8eaf0"},
            "bgcolor": "rgba(0,0,0,0)",
        },
        height=360,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── apply degradation ─────────────────────────────────────────────────────────

def _degrade(uri, user, pwd, db, src_label, rel_type, dst_label,
             degrad_method, degrad_prop, lam, log):
    try:
        deg = DataCompletenessDegrader(uri, user, pwd, db)
        if degrad_method == "property_completeness" and degrad_prop:
            info = deg.degrade_property_completeness("node", src_label, degrad_prop, lam)
            log(f"  → {info['degraded']} degraded ({info['final_missing']}/{info['total']} missing)")
        elif degrad_method == "edge_coverage":
            info = deg.degrade_edge_coverage(src_label, rel_type, dst_label, lam)
            log(f"  → {info['edges_deleted']} edges deleted")
        elif degrad_method == "property_accuracy" and degrad_prop:
            info = deg.degrade_property_accuracy("node", src_label, degrad_prop, lam)
            log(f"  → {info['corrupted']}/{info['total']} values corrupted (σ={lam})")
        deg.close()
    except Exception as e:
        log(f"  ✗ Degradation failed: {e}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def render_graphml_tab():
    uri  = st.session_state.uri
    user = st.session_state.neo_user
    pwd  = st.session_state.neo_pass
    dbs  = st.session_state.databases
    proj: Project = st.session_state.project

    st.markdown("""
    <div style="font-family:Syne,sans-serif;font-weight:700;font-size:1.3rem;
                color:#e8eaf0;margin:6px 0 4px">Graph ML Impact</div>
    <div style="font-family:DM Mono,monospace;font-size:.7rem;color:#5a6480;
                letter-spacing:.12em;text-transform:uppercase;margin-bottom:20px">
      Degradation impact on ML performance — sklearn</div>
    """, unsafe_allow_html=True)

    ratio = st.slider("⇔ Layout (config ↔ results)", 30, 70, 38, 5, key="ml_ratio")
    left, right = st.columns([ratio, 100 - ratio], gap="medium")

    # ─────────────────────────────────────────────────────────────────────────
    with left:
        # ① Dataset
        st.markdown("**① Dataset**")
        dataset_db = st.selectbox("Database", dbs, key="ml_db")
        labels     = _labels(uri, user, pwd, dataset_db)

        # ② Graph structure
        st.markdown("**② Graph structure**")
        src_label = st.selectbox("Source node label", labels, key="ml_src")

        avail_rels = _rels_for_src(uri, user, pwd, dataset_db, src_label) or []
        if not avail_rels:
            avail_rels = _rels(uri, user, pwd, dataset_db)
        rel_type = st.selectbox("Relationship type", avail_rels, key="ml_rel")

        avail_dsts = _dst_for(uri, user, pwd, dataset_db, src_label, rel_type) or labels
        dst_label  = st.selectbox("Destination node label", avail_dsts, key="ml_dst")

        # ③ ML Task
        st.markdown("**③ ML Task**")
        task_id = st.selectbox(
            "Task",
            options=list(TASK_LABELS.keys()),
            format_func=lambda k: TASK_LABELS[k],
            key="ml_task",
        )

        # Propriété cible (classif / régression)
        target_property = None
        if task_id == "node_classification":
            cat_props = _categorical_props(uri, user, pwd, dataset_db, src_label)
            if cat_props:
                target_property = st.selectbox(
                    "Target property (categorical)", cat_props, key="ml_target")
            else:
                all_props = _node_props(uri, user, pwd, dataset_db, src_label)
                target_property = st.selectbox(
                    "Target property", all_props, key="ml_target") if all_props else None
                st.caption("⚠ No categorical property detected.")

        elif task_id == "node_regression":
            num_props = _numeric_props(uri, user, pwd, dataset_db, src_label)
            if num_props:
                target_property = st.selectbox(
                    "Target property (numeric)", num_props, key="ml_target")
            else:
                all_props = _node_props(uri, user, pwd, dataset_db, src_label)
                target_property = st.selectbox(
                    "Target property", all_props, key="ml_target") if all_props else None
                st.caption("⚠ No numeric property detected.")

        # ④ Algorithms (max 3)
        st.markdown("**④ Algorithms (max 3)**")
        algo_pool = list(ALGORITHMS[task_id].keys())
        selected_algos = st.multiselect(
            "Algorithms to compare",
            options=algo_pool,
            default=algo_pool[:3],
            max_selections=3,
            key="ml_algos",
        )

        # ⑤ Performance metrics
        st.markdown("**⑤ Metrics**")
        perf_metrics = st.multiselect(
            "Metrics to display",
            options=TASK_METRICS[task_id],
            default=TASK_METRICS[task_id][:2],
            key="ml_perf",
        )

        # ⑥ Degradation
        st.markdown("**⑥ Degradation**")
        degrad_dim    = st.selectbox("Degradation dimension",
                                     list(DEGRAD_DIMS.keys()), key="ml_ddim")
        degrad_method = DEGRAD_DIMS[degrad_dim]
        degrad_prop   = None

        if degrad_method == "property_completeness":
            props = _node_props(uri, user, pwd, dataset_db, src_label)
            degrad_prop = st.selectbox("Property to degrade", props,
                                       key="ml_dprop") if props else None

        elif degrad_method == "property_accuracy":
            num_props = _numeric_props(uri, user, pwd, dataset_db, src_label)
            degrad_prop = st.selectbox("Numeric property to corrupt", num_props,
                                       key="ml_dprop_acc") if num_props else None
            if not num_props:
                st.warning("No numeric property found.")

        st.caption(f"Fixed λ: {LAMBDA_RANGE}  (step = 0.1)")

        run_btn = st.button("▶  Run experiment", key="ml_run",
                            width="stretch", type="primary")

    # ─────────────────────────────────────────────────────────────────────────
    with right:
        if not run_btn and "ml_results" not in st.session_state:
            st.markdown("""
            <div style="padding:48px 24px;text-align:center;color:#5a6480;
                        font-family:DM Mono,monospace;font-size:.82rem">
              ◎ Configure the experiment on the left<br>and click
              <strong style="color:#e8eaf0">▶ Run experiment</strong>.
            </div>""", unsafe_allow_html=True)

        # ── Run experiment ──────────────────────────────────────────────────
        if run_btn:
            if not selected_algos:
                st.error("Select at least one algorithm.")
                st.stop()
            if not perf_metrics:
                st.error("Select at least one metric.")
                st.stop()

            # Build cache config key
            cache_config = {
                "task_id":       task_id,
                "db":            dataset_db,
                "src_label":     src_label,
                "rel_type":      rel_type,
                "dst_label":     dst_label,
                "target_prop":   target_property,
                "degrad_method": degrad_method,
                "degrad_prop":   degrad_prop,
                "algorithms":    sorted(selected_algos),
                "perf_metrics":  sorted(perf_metrics),
                "lambdas":       LAMBDA_RANGE,
            }

            cached = proj.load_ml_cache(cache_config)
            if cached:
                st.info("Cache found — loading previous results.")
                st.session_state.ml_results = {
                    "task":       TASK_LABELS[task_id],
                    "dim":        degrad_dim,
                    "algorithms": selected_algos,
                    "metrics":    perf_metrics,
                    "lambdas":    LAMBDA_RANGE,
                    "data":       _normalize_results(cached["results"]),
                    "computed_at": cached.get("computed_at", ""),
                }
                st.rerun()

            # ── Full experiment ─────────────────────────────────────────────
            log_lines: list[str] = []
            log_ph = st.empty()

            def log(msg: str):
                log_lines.append(msg)
                log_ph.code("\n".join(log_lines[-30:]), language="text")

            # data[algo][metric][lambda] = value
            results: dict = {a: {m: {} for m in perf_metrics} for a in selected_algos}

            progress = st.progress(0)
            total    = len(LAMBDA_RANGE)

            log("📦 Backing up nodes...")
            node_backup = _backup_nodes(uri, user, pwd, dataset_db, src_label)
            log(f"   {len(node_backup)} nodes backed up.")

            edge_backup = None
            if degrad_method == "edge_coverage":
                log("📦 Backing up edges...")
                edge_backup = _backup_edges(uri, user, pwd, dataset_db,
                                            src_label, rel_type, dst_label)
                log(f"   {len(edge_backup)} edges backed up.")

            for step_i, lam in enumerate(LAMBDA_RANGE):
                log(f"\n══ λ = {lam:.1f} ═══════════════════")

                # Degrade
                log(f"  Degradation λ={lam:.1f}...")
                _degrade(uri, user, pwd, dataset_db, src_label, rel_type, dst_label,
                         degrad_method, degrad_prop, lam, log)

                # Train each algorithm
                for algo_name in selected_algos:
                    log(f"  [{algo_name}] training...")
                    try:
                        ml = SklearnGraphML(uri, user, pwd, dataset_db)
                        scores = ml.run_task(
                            task_id, src_label, rel_type, dst_label,
                            target_property, algo_name,
                        )
                        ml.close()
                        for m in perf_metrics:
                            val = scores.get(m)
                            results[algo_name][m][lam] = float(val) if val is not None else None
                            log(f"    {m} = {val:.4f}" if val is not None else f"    {m} = None")
                    except Exception as e:
                        log(f"  ✗ {algo_name} : {e}")
                        for m in perf_metrics:
                            results[algo_name][m][lam] = None

                # Restore
                log("  Restoring...")
                try:
                    _restore_nodes(uri, user, pwd, dataset_db, src_label, node_backup)
                    if degrad_method == "edge_coverage" and edge_backup is not None:
                        _restore_edges(uri, user, pwd, dataset_db,
                                       src_label, rel_type, dst_label, edge_backup)
                    log("  ✓ Restored")
                except Exception as e:
                    log(f"  ⚠ Restore failed: {e}")

                progress.progress((step_i + 1) / total)

            log("\n✅ Experiment complete — saving to project...")
            proj.save_ml_cache(cache_config, results)
            log("✅ Results cached.")

            st.session_state.ml_results = {
                "task":       TASK_LABELS[task_id],
                "dim":        degrad_dim,
                "algorithms": selected_algos,
                "metrics":    perf_metrics,
                "lambdas":    LAMBDA_RANGE,
                "data":       _normalize_results(results),
                "computed_at": "",
            }
            log_ph.empty()
            st.rerun()

        # ── Display results ─────────────────────────────────────────────────
        res = st.session_state.get("ml_results")
        if not res:
            return

        task_lbl  = res["task"]
        degrad_lbl = res["dim"]
        algorithms = res["algorithms"]
        metrics    = res["metrics"]
        lambdas    = res["lambdas"]
        data       = res["data"]

        header_parts = [
            f"<span style='font-family:Syne;font-weight:700;font-size:1.1rem'>{task_lbl}</span>",
            f"<span style='font-family:DM Mono;font-size:.68rem;color:#5a6480;"
            f"margin-left:12px'>degradation: {degrad_lbl}</span>",
        ]
        if res.get("computed_at"):
            header_parts.append(
                f"<span style='font-family:DM Mono;font-size:.65rem;color:#5a6480;"
                f"margin-left:12px'>cache: {res['computed_at'][:16].replace('T',' ')}</span>"
            )
        st.markdown(f"<div style='margin-bottom:16px'>{''.join(header_parts)}</div>",
                    unsafe_allow_html=True)

        # One chart per metric
        for metric_name in metrics:
            algo_data: dict[str, list] = {}
            for algo_name in algorithms:
                algo_data[algo_name] = [
                    data.get(algo_name, {}).get(metric_name, {}).get(lam)
                    for lam in lambdas
                ]

            fig = _make_multi_algo_chart(lambdas, algo_data, metric_name, task_lbl, degrad_lbl)
            st.plotly_chart(fig, width="stretch", key=f"ml_fig_{metric_name}")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Clear button
        if st.button("↺  Reset results", key="ml_clear", width="stretch"):
            st.session_state.pop("ml_results", None)
            st.rerun()
