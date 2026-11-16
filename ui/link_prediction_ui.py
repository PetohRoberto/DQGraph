"""
ui/link_prediction_ui.py
------------------------
Onglet « Link Prediction » dans le Data Quality Dashboard.
S'appuie sur link_prediction.LinkPredictionPipeline (GDS).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ml.link_prediction import LinkPredictionPipeline
from ui.schema import SchemaExplorer


# ── cache helpers ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _labels(uri, user, password, database):
    s = SchemaExplorer(uri, user, password, database)
    r = s.get_node_labels(); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _rel_types(uri, user, password, database):
    s = SchemaExplorer(uri, user, password, database)
    r = s.get_relationship_types(); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _rel_node_labels(uri, user, password, rel_type, database):
    s = SchemaExplorer(uri, user, password, database)
    r = s.get_node_labels_for_rel(rel_type); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _node_props(uri, user, password, label, database):
    s = SchemaExplorer(uri, user, password, database)
    r = s.get_node_properties(label); s.close(); return r


# ── constantes ───────────────────────────────────────────────────────────────

FEATURES = {
    "Common Neighbors":        "COMMON_NEIGHBORS",
    "Adamic-Adar":             "ADAMIC_ADAR",
    "Resource Allocation":     "RESOURCE_ALLOCATION",
    "Preferential Attachment": "PREFERENTIAL_ATTACHMENT",
    "Total Neighbors":         "TOTAL_NEIGHBORS",
}

EMBEDDING_FEATURES = {
    "Node2Vec": "NODE2VEC",
    "GraphSAGE": "GRAPHSAGE",
    "Hadamard": "HADAMARD"
}
ALGORITHMS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest":       "random_forest",
}


# ── mini progress log ─────────────────────────────────────────────────────────

class ProgressLog:
    def __init__(self, container):
        self._container = container
        self._lines: list[str] = []

    def __call__(self, msg: str):
        self._lines.append(msg)
        self._container.markdown(
            "\n\n".join(
                f'<div class="info-box" style="margin:4px 0;padding:8px 14px">{l}</div>'
                for l in self._lines
            ),
            unsafe_allow_html=True,
        )


# ── résultats ─────────────────────────────────────────────────────────────────

def _render_model_metrics(model_info: dict, model_selection: dict, elapsed: float):
    st.markdown('<div class="section-title">📊 Model Performance</div>',
                unsafe_allow_html=True)

    best = model_info.get("bestParameters", {})
    metrics = model_info.get("metrics", {})

    cols = st.columns(3)
    with cols[0]:
        aucpr = metrics.get("AUCPR", {})
        test_val = aucpr.get("test", None) if isinstance(aucpr, dict) else aucpr
        color = "#00a884" if (test_val or 0) >= 0.7 else (
                "#d99a2b" if (test_val or 0) >= 0.5 else "#e15a5a")
        val_str = f"{test_val:.4f}" if test_val is not None else "N/A"
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:{color};font-size:2.2rem">{val_str}</div>
            <div class="score-label">AUCPR (test)</div>
        </div>""", unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:#3968d1;font-size:2.2rem">{elapsed}s</div>
            <div class="score-label">Training time</div>
        </div>""", unsafe_allow_html=True)

    with cols[2]:
        n_candidates = len(model_selection.get("modelCandidatesWithStats", [])) \
                       if model_selection else 0
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:#00a884;font-size:2.2rem">{n_candidates}</div>
            <div class="score-label">Candidates evaluated</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("Model info details (JSON)", expanded=False):
        st.json(model_info)


def _render_predictions(predictions: list[dict], rel_type: str):
    if not predictions:
        st.warning("No predicted links above the threshold.")
        return

    st.markdown(
        f'<div class="section-title">🔮 Predicted links — {rel_type} '
        f'({len(predictions)} results)</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for p in predictions:
        def _repr(node: dict) -> str:
            if not node:
                return "?"
            for key in ("name", "title", "id", "label"):
                if key in node:
                    return str(node[key])
            vals = list(node.values())
            return str(vals[0]) if vals else "?"

        rows.append({
            "Node 1":      _repr(p["node1"]),
            "Node 2":      _repr(p["node2"]),
            "Probability": p["probability"],
            "Confidence":  "🟢 High" if p["probability"] >= 0.7
                           else ("🟡 Medium" if p["probability"] >= 0.5 else "🔴 Low"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig = go.Figure(go.Bar(
        x=[f"{r['Node 1']} → {r['Node 2']}" for r in rows],
        y=[r["Probability"] for r in rows],
        marker_color=[
            "#00a884" if r["Probability"] >= 0.7
            else ("#d99a2b" if r["Probability"] >= 0.5 else "#e15a5a")
            for r in rows
        ],
        text=[f"{r['Probability']:.3f}" for r in rows],
        textposition="outside",
        textfont={"family": "Space Mono", "size": 10, "color": "#191c26"},
    ))
    fig.update_layout(
        title={"text": "Probability score per pair",
               "font": {"family": "Syne", "size": 13, "color": "#6b7085"}},
        height=340,
        margin=dict(l=10, r=10, t=50, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "tickfont": {"family": "Space Mono", "size": 9, "color": "#6b7085"},
            "tickangle": -35,
            "gridcolor": "#e1e4ea",
        },
        yaxis={"range": [0, 1.15], "gridcolor": "#e1e4ea",
               "tickfont": {"size": 10, "color": "#6b7085"}},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="lp_bar")


# ── formulaire principal ──────────────────────────────────────────────────────

def render_link_prediction_tab(uri: str, user: str, password: str, database: str = "neo4j"):
    """Rendu complet de l'onglet Link Prediction."""

    st.markdown("""
    <div class="info-box">
        🔗 <strong>Link Prediction</strong> — Predicts missing links in the graph
        via the GDS pipeline. Select the target relationship type, heuristic features,
        and the ML algorithm.
    </div>
    """, unsafe_allow_html=True)

    lp = LinkPredictionPipeline(uri, user, password, database)
    if not lp.check_gds_available():
        lp.close()
        st.error(
            "❌ **Neo4j Graph Data Science (GDS)** is not available on this instance.\n\n"
            "Install the GDS plugin from [neo4j.com/docs/graph-data-science](https://neo4j.com/docs/graph-data-science/)."
        )
        return
    lp.close()

    col_form, col_result = st.columns([1, 2], gap="large")

    with col_form:
        st.markdown('<div class="conn-panel">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Target relationship")

        rel_types = _rel_types(uri, user, password, database)
        rel_type = st.selectbox("Relationship type to predict", rel_types, key="lp_rel") \
                   if rel_types else st.text_input("Relationship type", key="lp_rel")

        src_labels, dst_labels = _rel_node_labels(uri, user, password, rel_type, database) \
                                  if rel_type else ([], [])

        src_label = st.selectbox("Source node", src_labels, key="lp_src") \
                    if src_labels else st.text_input("Source node", key="lp_src")
        dst_label = st.selectbox("Destination node", dst_labels, key="lp_dst") \
                    if dst_labels else st.text_input("Destination node", key="lp_dst")

        st.markdown("#### 🔬 Heuristic Features")
        selected_feats = st.multiselect(
            "Heuristic features",
            list(FEATURES.keys()),
            default=["Common Neighbors", "Adamic-Adar", "Preferential Attachment"],
            key="lp_features",
        )
        
        st.markdown("#### 🔬 Embedded Features")
        selected_embedded_feats = st.multiselect(
            "Embedding features",
            list(EMBEDDING_FEATURES.keys()),
            default=["Hadamard"],
            key="lp_embedded_features",
        )

        st.markdown("#### 🤖 Algorithm")
        algo_label = st.selectbox("ML model", list(ALGORITHMS.keys()), key="lp_algo")

        st.markdown("#### ⚙️ Paramètres")
        c1, c2 = st.columns(2)
        with c1:
            test_frac = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05, key="lp_test")
        with c2:
            train_frac = st.slider("Train fraction", 0.4, 0.8, 0.6, 0.05, key="lp_train")

        top_n = st.number_input("Top N predictions", 5, 100, 20, key="lp_topn")
        threshold = st.slider("Probability threshold", 0.0, 1.0, 0.3, 0.05, key="lp_thresh")

        st.markdown('</div>', unsafe_allow_html=True)

        run_btn = st.button("▶ Run pipeline", use_container_width=True, key="lp_run")

    with col_result:
        if run_btn:
            if not (selected_feats or selected_embedded_feats):
                st.warning("Select at least one feature.")
            else:
                log_container = st.empty()
                progress = ProgressLog(log_container)

                feat_ids = [FEATURES[f] for f in selected_feats]
                embedded_feat_ids = [EMBEDDING_FEATURES[f] for f in selected_embedded_feats]
                algo = ALGORITHMS[algo_label]

                lp = LinkPredictionPipeline(uri, user, password, database)
                result = lp.run_full_pipeline(
                    src_label=src_label,
                    dst_label=dst_label,
                    rel_type=rel_type,
                    features=embedded_feat_ids,
                    algorithm=algo,
                    test_fraction=test_frac,
                    train_fraction=train_frac,
                    top_n=int(top_n),
                    threshold=threshold,
                    progress_callback=progress,
                )
                lp.close()
                log_container.empty()

                if not result["success"]:
                    st.error(f"❌ GDS error: {result['error']}")
                else:
                    gs = result.get("graph_stats", {})
                    st.markdown(f"""
                    <div class="info-box">
                        Projected graph: <strong>{gs.get('nodeCount', '?')}</strong> nodes ·
                        <strong>{gs.get('relationshipCount', '?')}</strong> relationships ·
                        Pipeline completed in <strong>{result['elapsed']}s</strong>
                    </div>""", unsafe_allow_html=True)

                    _render_model_metrics(
                        result["model_info"],
                        result.get("model_selection", {}),
                        result["elapsed"],
                    )
                    st.markdown("---")
                    _render_predictions(result["predictions"], rel_type)
        else:
            st.markdown("""
            <div class="info-box" style="margin-top:2rem;text-align:center;padding:40px 20px">
                <div style="font-size:2rem;margin-bottom:12px;opacity:0.4">⬡</div>
                Configure the pipeline on the left, then click
                <strong>▶ Run pipeline</strong>.
            </div>
            """, unsafe_allow_html=True)
