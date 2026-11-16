"""
pages/tab_graphml.py — Graph ML Impact tab.

- Extraction des données de Neo4j en Python/pandas
- Training avec sklearn (pas de GDS)
- Choix de jusqu'à 3 algorithmes par tâche
- λ : plage dans [0.0, 0.9] et pas variable (choisi par l'utilisateur)
- Cache par config dans le dossier projet
- Disposition : configuration à l'horizontale, résultats en dessous
- Graphique : X = dégradation λ, Y = métrique, une courbe par algo
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from neo4j_pool import get_driver
from degraders import DataCompletenessDegrader
from ml.ml_sklearn import (
    SklearnGraphML, ALGORITHMS, TASK_METRICS, TASK_LABELS,
    ALGO_CLASSIFIERS, ALGO_REGRESSORS, ALGO_LP,
)
from project_manager import Project
import experiment_jobs

# ── cached schema fetchers ────────────────────────────────────────────────────
from ui.schema_cache import (
    labels as _labels, rels as _rels, rels_for_src as _rels_for_src,
    dst_for as _dst_for, node_props as _node_props,
    numeric_props as _numeric_props, categorical_props as _categorical_props,
)


# ── degradation config ────────────────────────────────────────────────────────
DEGRAD_DIMS = {
    "Property Completeness": "property_completeness",
    "Edge Coverage":         "edge_coverage",
    "Property Conformity":   "property_conformity",
    "Node-Edge Duplicate":   "node_edge_duplicate",
    "Property Accuracy":     "property_accuracy",
}

# Which data the degradation alters (see ml_sklearn.NodeSplit / EdgeSplit).
DEGRAD_SCOPES = {
    "both":  "Training + test",
    "train": "Training only",
    "test":  "Test only",
}

# Part of the cache key: bump it when the degradation or the ML pipeline
# changes, so that results computed by the previous code are not reused.
# 3: split fixed before degrading, degradation scope, no LP validation set.
PIPELINE_VERSION = 3

COLORS = ["#3968d1", "#00a884", "#d99a2b", "#e15a5a", "#9b5de0", "#e08636"]

LAMBDA_MAX = 0.9


def _lambda_grid(lmin: float, lmax: float, step: float) -> list[float]:
    """λ levels from lmin to lmax every `step`. lmax is always included, so the
    last interval may be shorter. Computed in hundredths to avoid float drift."""
    lo, hi, s = round(lmin * 100), round(lmax * 100), max(1, round(step * 100))
    pts = list(range(lo, hi + 1, s))
    if pts[-1] != hi:
        pts.append(hi)
    return [p / 100 for p in pts]


def _fmt_lam(lam: float) -> str:
    return f"{lam:g}"


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

RESTORE_BATCH = 5000


def _batches(rows, size=RESTORE_BATCH):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def _backup_nodes(uri, user, pwd, db, label):
    with get_driver(uri, user, pwd).session(database=db) as s:
        return s.run(f"MATCH (n:`{label}`) "
                     "RETURN elementId(n) AS eid, properties(n) AS props").data()


def _restore_nodes(uri, user, pwd, db, label, backup):
    with get_driver(uri, user, pwd).session(database=db) as s:
        for chunk in _batches(backup):
            s.run("UNWIND $rows AS row "
                  "MATCH (n) WHERE elementId(n) = row.eid SET n = row.props",
                  rows=chunk)


def _backup_edges(uri, user, pwd, db, src, rel, dst):
    with get_driver(uri, user, pwd).session(database=db) as s:
        return s.run(f"MATCH (s:`{src}`)-[r:`{rel}`]->(d:`{dst}`) "
                     "RETURN elementId(s) AS sid, elementId(d) AS did, "
                     "properties(r) AS props").data()


def _restore_edges(uri, user, pwd, db, src, rel, dst, backup):
    with get_driver(uri, user, pwd).session(database=db) as s:
        s.run(f"MATCH (:`{src}`)-[r:`{rel}`]->(:`{dst}`) DELETE r")
        for chunk in _batches(backup):
            s.run(f"UNWIND $rows AS row "
                  f"MATCH (s) WHERE elementId(s) = row.sid "
                  f"MATCH (d) WHERE elementId(d) = row.did "
                  f"CREATE (s)-[r:`{rel}`]->(d) SET r = coalesce(row.props, {{}})",
                  rows=chunk)


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
            marker=dict(size=8, color=color, line=dict(color="#ffffff", width=2)),
            fill="none",
            hovertemplate=f"<b>{algo_name}</b><br>λ = %{{x}}<br>{metric_name} = %{{y:.4f}}<extra></extra>",
        ))

    # Ticks on the tested λ levels while they stay readable, otherwise auto.
    xticks = {"tickvals": lambdas} if len(lambdas) <= 15 else {}

    fig.update_layout(
        title={
            "text": f"{task_name} — {metric_name} vs Degradation ({degrad_dim})",
            "font": {"family": "Syne", "size": 13, "color": "#191c26"},
        },
        xaxis={
            "title": {"text": "Degradation λ",
                      "font": {"family": "DM Mono", "size": 11, "color": "#6b7085"}},
            "tickfont": {"family": "DM Mono", "size": 10, "color": "#6b7085"},
            "gridcolor": "#e1e4ea", "zeroline": False, **xticks,
        },
        yaxis={
            "title": {"text": metric_name,
                      "font": {"family": "DM Mono", "size": 11, "color": "#6b7085"}},
            "tickfont": {"size": 10, "color": "#6b7085"},
            "gridcolor": "#e1e4ea", "zeroline": False,
        },
        legend={
            "font": {"family": "DM Mono", "size": 11, "color": "#191c26"},
            "bgcolor": "rgba(0,0,0,0)",
        },
        height=360,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── apply degradation ─────────────────────────────────────────────────────────

def _degrade(deg, src_label, rel_type, dst_label, degrad_method, lam, log,
             target_property=None, scope_nodes=None, scope_pairs=None):
    # Every method takes the cumulative λ and only alters the instances still
    # needed to reach it (deg remembers the instances already noised).
    # The ML target is never degraded: only the features are.
    # scope_nodes / scope_pairs: the nodes / edges of the degraded side
    # (training or test data); None = the whole graph.
    keep = [target_property] if target_property else []
    try:
        if degrad_method == "property_completeness":
            info = deg.degrade_all_completeness("node", src_label, lam, exclude_props=keep,
                                                scope_ids=scope_nodes)
            log(f"  → {info['total_properties']} properties targeted at {lam:.0%} missing "
                f"({info['total_removed']} values removed this step)")
        elif degrad_method == "edge_coverage":
            info = deg.degrade_edge_coverage(src_label, rel_type, dst_label, lam,
                                             scope_ids=scope_nodes)
            log(f"  → {info['edges_deleted']} edges deleted (cumul. λ={_fmt_lam(lam)})")
        elif degrad_method == "property_conformity":
            info = deg.degrade_all_conformity("node", src_label, lam, exclude_props=keep,
                                              scope_ids=scope_nodes)
            log(f"  → {info['total_properties']} properties: {info['total_altered']} "
                f"values made non-conforming this step ({lam:.0%} of instances)")
        elif degrad_method == "node_edge_duplicate":
            info = deg.degrade_node_edge_duplicates(src_label, rel_type, dst_label, lam,
                                                    scope_pairs=scope_pairs)
            log(f"  → {info['duplicated']} edges duplicated with their endpoints this step "
                f"({info['total_duplicated']}/{info['total_edges']} edges, "
                f"{info['nodes_created']} nodes created)")
        elif degrad_method == "property_accuracy":
            info = deg.degrade_all_accuracy("node", src_label, lam, exclude_props=keep,
                                            scope_ids=scope_nodes)
            log(f"  → {info['total_numeric_properties']} numeric properties: "
                f"{info['total_corrupted']} values corrupted this step "
                f"({lam:.0%} of instances noisy)")
    except Exception as e:
        log(f"  ✗ Degradation failed: {e}")


# ── background experiment ─────────────────────────────────────────────────────

def _run_experiment(job, uri, user, pwd, db, proj, task_id, src_label, rel_type,
                    dst_label, target_property, degrad_method, degrad_prop,
                    algorithms, perf_metrics, lambdas, cache_config,
                    degrad_scope="both"):
    """Runs in a thread (see experiment_jobs): no Streamlit calls in here."""
    log = job.log

    # data[algo][metric][lambda] = value
    results: dict = {a: {m: {} for m in perf_metrics} for a in algorithms}

    # The train / test split is drawn ONCE, on the original graph, before any
    # degradation: every λ uses the same split, and the degradation can be
    # restricted to the training or the test data.
    try:
        split = SklearnGraphML(uri, user, pwd, db).make_split(
            task_id, src_label, rel_type, dst_label, target_property, degrad_scope)
    except Exception as e:
        log(f"✗ Cannot split the data: {e}")
        raise
    scope_nodes, scope_pairs = split.scope_nodes(), split.scope_pairs()
    log(f"✂ Split drawn on the original graph: {split.describe()}.")
    if degrad_scope == "both":
        log("   Degrading the training and the test data.")
    else:
        n_scope = len(scope_pairs) if degrad_method == "node_edge_duplicate" else len(scope_nodes)
        unit = "edges" if degrad_method == "node_edge_duplicate" else "source nodes"
        log(f"   Degrading the {DEGRAD_SCOPES[degrad_scope].split()[0].lower()} data "
            f"only: {n_scope} {unit} can be altered; λ is a proportion of them.")

    log("📦 Backing up nodes...")
    node_backup = _backup_nodes(uri, user, pwd, db, src_label)
    log(f"   {len(node_backup)} nodes backed up.")

    edge_backup = None
    if degrad_method == "edge_coverage":
        log("📦 Backing up edges...")
        edge_backup = _backup_edges(uri, user, pwd, db, src_label, rel_type, dst_label)
        log(f"   {len(edge_backup)} edges backed up.")

    # One degrader (seeded) for the whole experiment: the levels are cumulative.
    deg = DataCompletenessDegrader(uri, user, pwd, db)

    # Restore in a finally so the graph is never left degraded (stop, error).
    try:
        for step_i, lam in enumerate(lambdas):
            if job.cancel.is_set():
                break
            log(f"\n══ λ = {_fmt_lam(lam)} ═══════════════════")

            log(f"  Degradation up to λ={_fmt_lam(lam)}...")
            _degrade(deg, src_label, rel_type, dst_label,
                     degrad_method, lam, log, target_property=target_property,
                     scope_nodes=scope_nodes, scope_pairs=scope_pairs)

            # Copies made by the duplicate degradation join the side of their
            # original (node copies for node tasks, edge copies for LP).
            split.origin = (deg.dup_origin_edges if task_id == "link_prediction"
                            else deg.dup_origin_nodes)

            # Train each algorithm — one instance per λ so the algorithms
            # share the data extracted from the (degraded) graph.
            ml = SklearnGraphML(uri, user, pwd, db)
            for algo_name in algorithms:
                if job.cancel.is_set():
                    break
                log(f"  [{algo_name}] training...")
                try:
                    scores = ml.run_task(
                        task_id, src_label, rel_type, dst_label,
                        target_property, algo_name, split=split,
                    )
                    if scores.get("_diag"):
                        log(f"    ▸ {scores['_diag']}")
                    for m in perf_metrics:
                        val = scores.get(m)
                        results[algo_name][m][lam] = float(val) if val is not None else None
                        log(f"    {m} = {val:.4f}" if val is not None else f"    {m} = None")
                except Exception as e:
                    log(f"  ✗ {algo_name} : {e}")
                    for m in perf_metrics:
                        results[algo_name][m][lam] = None

            job.progress = (step_i + 1) / len(lambdas)
    finally:
        log("\n  Restoring original data...")
        if degrad_method == "node_edge_duplicate":
            try:
                log(f"  {deg.remove_duplicates()} duplicate nodes deleted")
            except Exception as e:
                log(f"  ⚠ Deleting duplicates failed: {e}")
        try:
            _restore_nodes(uri, user, pwd, db, src_label, node_backup)
            if degrad_method == "edge_coverage" and edge_backup is not None:
                _restore_edges(uri, user, pwd, db, src_label, rel_type, dst_label,
                               edge_backup)
            log("  ✓ Restored")
        except Exception as e:
            log(f"  ⚠ Restore failed: {e}")

    if job.cancel.is_set():
        log("\n■ Experiment stopped — partial results not saved.")
        return None

    log("\n✅ Experiment complete — saving to project...")
    proj.save_ml_cache(cache_config, results)
    log("✅ Results cached.")
    return results


@st.fragment(run_every=2)
def _job_panel(key):
    job = experiment_jobs.get(key)
    if job is None or job.status != "running":
        st.rerun()  # whole page: show the results
    st.info(f"⏳ Experiment running in the background on **{job.db}** — you can "
            "use the Data Quality page meanwhile (evaluations on this database "
            "are paused until it finishes).")
    st.progress(job.progress)
    st.code("\n".join(job.log_lines[-30:]), language="text")
    if not job.cancel.is_set() and st.button("■  Stop experiment", key="ml_stop",
                                             width="stretch"):
        job.cancel.set()
    if job.cancel.is_set():
        st.caption("Stopping after the current training, then restoring the data...")


def _collect_job(job):
    """A finished job: move its results into the session and forget it."""
    experiment_jobs.pop(job.key)
    if job.status == "done":
        st.session_state.ml_results = {
            **job.meta,
            "data":        _normalize_results(job.results),
            "computed_at": "",
            "log":         list(job.log_lines),
        }
        st.session_state.ml_job_msg = ("done", "Experiment complete — results saved.")
    else:
        last = next((l for l in reversed(job.log_lines) if l.strip()), "")
        st.session_state.ml_job_msg = (job.status, f"Experiment {job.status}. {last}",
                                       list(job.log_lines))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def render_graphml_tab():
    uri  = st.session_state.uri
    user = st.session_state.neo_user
    pwd  = st.session_state.neo_pass
    dbs  = st.session_state.databases
    proj: Project = st.session_state.project

    st.markdown("""
    <div style="font-family:Syne,sans-serif;font-weight:700;font-size:1.3rem;
                color:#191c26;margin:6px 0 4px">Graph ML Impact</div>
    <div style="font-family:DM Mono,monospace;font-size:.7rem;color:#6b7085;
                letter-spacing:.12em;text-transform:uppercase;margin-bottom:20px">
      Degradation impact on ML performance — sklearn</div>
    """, unsafe_allow_html=True)

    current = experiment_jobs.get(proj.dir)
    running = current is not None and current.status == "running"

    # ── Configuration (horizontal) ───────────────────────────────────────────
    with st.container(border=True):
        c_data, c_task, c_deg = st.columns(3, gap="large")

        with c_data:
            # ① Dataset
            st.markdown("**① Dataset**")
            dataset_db = st.selectbox("Database", dbs, key="ml_db")
            labels     = _labels(uri, user, pwd, dataset_db)
            if not labels:
                st.warning(f"No node label found in **{dataset_db}**: the database "
                           "is empty, or its schema was read before the data were "
                           "imported (↻ Refresh schema in the Data Quality tab).")

            # ② Graph structure
            st.markdown("**② Graph structure**")
            src_label = st.selectbox("Source node label", labels, key="ml_src")

            avail_rels = _rels_for_src(uri, user, pwd, dataset_db, src_label) or []
            if not avail_rels:
                avail_rels = _rels(uri, user, pwd, dataset_db)
            rel_type = st.selectbox("Relationship type", avail_rels, key="ml_rel")

            avail_dsts = _dst_for(uri, user, pwd, dataset_db, src_label, rel_type) or labels
            dst_label  = st.selectbox("Destination node label", avail_dsts, key="ml_dst")

        with c_task:
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
            # One key per task: a keyed multiselect keeps its selection when its
            # options change and silently drops the values that became invalid,
            # so a shared key turned [Logistic Regression, Random Forest]
            # (classification) into [Random Forest] for regression.
            st.markdown("**④ Algorithms (max 3)**")
            algo_pool = list(ALGORITHMS[task_id].keys())
            selected_algos = st.multiselect(
                "Algorithms to compare",
                options=algo_pool,
                default=algo_pool[:3],
                max_selections=3,
                key=f"ml_algos_{task_id}",
            )

            # ⑤ Performance metrics
            st.markdown("**⑤ Metrics**")
            perf_metrics = st.multiselect(
                "Metrics to display",
                options=TASK_METRICS[task_id],
                default=TASK_METRICS[task_id][:2],
                key=f"ml_perf_{task_id}",
            )

        with c_deg:
            # ⑥ Degradation
            st.markdown("**⑥ Degradation**")
            degrad_dim    = st.selectbox("Degradation dimension",
                                         list(DEGRAD_DIMS.keys()), key="ml_ddim")
            degrad_method = DEGRAD_DIMS[degrad_dim]
            degrad_scope  = st.radio(
                "Degraded data", list(DEGRAD_SCOPES), horizontal=True,
                format_func=DEGRAD_SCOPES.get, key="ml_scope",
                help="The training / test split is drawn once on the original "
                     "graph. Training only: the model learns on degraded data "
                     "and is tested on clean data; test only: the reverse. "
                     "λ is then a proportion of the degraded side.",
            )
            degrad_prop   = None

            lam_range = st.slider(
                "λ range",
                min_value=0.0, max_value=LAMBDA_MAX,
                value=(0.0, LAMBDA_MAX),
                step=0.01,
                key="ml_lambda_range",
            )
            lam_step = st.number_input(
                "λ step",
                min_value=0.01, max_value=LAMBDA_MAX,
                value=0.1, step=0.01, format="%.2f",
                key="ml_lambda_step",
                help="Gap between two tested λ levels. The upper bound of the "
                     "range is always tested, so the last gap may be shorter.",
            )
            selected_lambdas = _lambda_grid(*lam_range, lam_step)
            st.caption(f"{len(selected_lambdas)} levels — λ = "
                       f"[{', '.join(_fmt_lam(l) for l in selected_lambdas)}]")

        run_btn = st.button("▶  Run experiment", key="ml_run",
                            width="stretch", type="primary", disabled=running)

    # ── Results (below) ──────────────────────────────────────────────────────
    job = experiment_jobs.get(proj.dir)
    if job is not None and job.status != "running":
        _collect_job(job)
        job = None
    if job is not None:
        _job_panel(proj.dir)

    msg = st.session_state.pop("ml_job_msg", None)
    if msg:
        (st.success if msg[0] == "done" else st.warning)(msg[1])
        if len(msg) > 2 and msg[2]:   # failed / stopped: the full log
            with st.expander("Experiment log", expanded=True):
                st.code("\n".join(msg[2]), language="text")

    if job is None and not run_btn and "ml_results" not in st.session_state:
        st.markdown("""
        <div style="padding:48px 24px;text-align:center;color:#6b7085;
                    font-family:DM Mono,monospace;font-size:.82rem">
          ◎ Configure the experiment above and click
          <strong style="color:#191c26">▶ Run experiment</strong>.
        </div>""", unsafe_allow_html=True)

    # ── Run experiment ──────────────────────────────────────────────────────
    if run_btn:
        if not selected_algos:
            st.error("Select at least one algorithm.")
            st.stop()
        if not perf_metrics:
            st.error("Select at least one metric.")
            st.stop()

        # Build cache config key
        cache_config = {
            "pipeline":      PIPELINE_VERSION,
            "task_id":       task_id,
            "db":            dataset_db,
            "src_label":     src_label,
            "rel_type":      rel_type,
            "dst_label":     dst_label,
            "target_prop":   target_property,
            "degrad_method": degrad_method,
            "degrad_prop":   degrad_prop,
            "degrad_scope":  degrad_scope,
            "algorithms":    sorted(selected_algos),
            "perf_metrics":  sorted(perf_metrics),
            "lambdas":       selected_lambdas,
        }
        meta = {
            "task":         TASK_LABELS[task_id],
            "dim":          degrad_dim,
            "scope":        DEGRAD_SCOPES[degrad_scope],
            "algorithms":   selected_algos,
            "metrics":      perf_metrics,
            "lambdas":      selected_lambdas,
            "cache_config": cache_config,
        }

        cached = proj.load_ml_cache(cache_config)
        if cached:
            st.info("Cache found — loading previous results.")
            st.session_state.ml_results = {
                **meta,
                "data":        _normalize_results(cached["results"]),
                "computed_at": cached.get("computed_at", ""),
            }
            st.rerun()

        if experiment_jobs.busy_db(uri, dataset_db) is not None:
            st.error(f"An experiment is already degrading '{dataset_db}'.")
            st.stop()

        # ── Full experiment, in the background ──────────────────────────────
        run = dict(uri=uri, user=user, pwd=pwd, db=dataset_db, proj=proj,
                   task_id=task_id, src_label=src_label, rel_type=rel_type,
                   dst_label=dst_label, target_property=target_property,
                   degrad_method=degrad_method, degrad_prop=degrad_prop,
                   algorithms=selected_algos, perf_metrics=perf_metrics,
                   lambdas=selected_lambdas, cache_config=cache_config,
                   degrad_scope=degrad_scope)
        experiment_jobs.start(proj.dir, uri, dataset_db, meta,
                              lambda job: _run_experiment(job, **run))
        st.session_state.pop("ml_results", None)
        st.rerun()

    # ── Display results ─────────────────────────────────────────────────────
    res = st.session_state.get("ml_results")
    if not res:
        return

    task_lbl   = res["task"]
    degrad_lbl = res["dim"]
    algorithms = res["algorithms"]
    metrics    = res["metrics"]
    lambdas    = res["lambdas"]
    data       = res["data"]

    header_parts = [
        f"<span style='font-family:Syne;font-weight:700;font-size:1.1rem'>{task_lbl}</span>",
        f"<span style='font-family:DM Mono;font-size:.68rem;color:#6b7085;"
        f"margin-left:12px'>degradation: {degrad_lbl}"
        f" ({res.get('scope', DEGRAD_SCOPES['both']).lower()})</span>",
        f"<span style='font-family:DM Mono;font-size:.68rem;color:#6b7085;"
        f"margin-left:12px'>{len(lambdas)} λ levels "
        f"({_fmt_lam(lambdas[0])} → {_fmt_lam(lambdas[-1])})</span>",
    ]
    if res.get("computed_at"):
        header_parts.append(
            f"<span style='font-family:DM Mono;font-size:.65rem;color:#6b7085;"
            f"margin-left:12px'>cache: {res['computed_at'][:16].replace('T',' ')}</span>"
        )
    st.markdown(f"<div style='margin:20px 0 4px'>{''.join(header_parts)}</div>",
                unsafe_allow_html=True)
    st.caption(f"Algorithms: {', '.join(algorithms)}  ·  Metrics: {', '.join(metrics)}  ·  "
               f"λ = [{', '.join(_fmt_lam(l) for l in lambdas)}]")

    # An algorithm without a score is an empty curve: say so instead of
    # leaving the user wonder where it went.
    for algo_name in algorithms:
        missing = [lam for lam in lambdas
                   if all(data.get(algo_name, {}).get(m, {}).get(lam) is None
                          for m in metrics)]
        if len(missing) == len(lambdas):
            st.warning(f"**{algo_name}**: no score at any λ, training failed "
                       "(see the experiment log below).")
        elif missing:
            st.info(f"**{algo_name}**: no score at λ = "
                    f"{', '.join(_fmt_lam(l) for l in missing)} (see the experiment log).")

    # One chart per metric, two per row (results use the full width)
    n_cols = min(2, len(metrics)) or 1
    chart_cols = st.columns(n_cols, gap="medium")
    for i, metric_name in enumerate(metrics):
        algo_data: dict[str, list] = {}
        for algo_name in algorithms:
            algo_data[algo_name] = [
                data.get(algo_name, {}).get(metric_name, {}).get(lam)
                for lam in lambdas
            ]

        fig = _make_multi_algo_chart(lambdas, algo_data, metric_name, task_lbl, degrad_lbl)
        with chart_cols[i % n_cols]:
            st.plotly_chart(fig, width="stretch", key=f"ml_fig_{metric_name}")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Exact values behind the curves: one row per λ, one column per metric × algo
    with st.expander("Scores table"):
        table = pd.DataFrame([
            {"λ": _fmt_lam(lam), **{
                f"{m} · {a}": data.get(a, {}).get(m, {}).get(lam)
                for m in metrics for a in algorithms}}
            for lam in lambdas
        ])
        st.dataframe(table, hide_index=True, width="stretch")

    if res.get("log"):
        with st.expander("Experiment log"):
            st.code("\n".join(res["log"]), language="text")
    else:
        st.caption("Results loaded from the cache: the log of the original run "
                   "is not kept.")

    # Clear button — removes display AND disk cache so the next run re-trains
    if st.button("↺  Reset results", key="ml_clear", width="stretch"):
        cached_cfg = res.get("cache_config")
        if cached_cfg:
            proj.delete_ml_cache(cached_cfg)
        st.session_state.pop("ml_results", None)
        st.rerun()
