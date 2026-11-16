"""
ui/degraders_ui.py
------------------
Onglet « Dégradation » dans le Data Quality Dashboard.
Expose les deux méthodes de DataCompletenessDegrader :
  - degrade_property_completeness
  - degrade_edge_coverage
"""

import streamlit as st
from degraders import DataCompletenessDegrader
from ui.schema import SchemaExplorer


# ── cache helpers (réutilise le même pattern que forms.py) ────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _labels(uri, user, password, database):
    s = SchemaExplorer(uri, user, password, database)
    r = s.get_node_labels(); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _rel_types(uri, user, password, database):
    s = SchemaExplorer(uri, user, password, database)
    r = s.get_relationship_types(); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _node_props(uri, user, password, label, database):
    s = SchemaExplorer(uri, user, password, database)
    r = s.get_node_properties(label); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _edge_props(uri, user, password, rel_type, database):
    s = SchemaExplorer(uri, user, password, database)
    r = s.get_edge_properties(rel_type); s.close(); return r

@st.cache_data(ttl=60, show_spinner=False)
def _rel_node_labels(uri, user, password, rel_type, database):
    s = SchemaExplorer(uri, user, password, database)
    r = s.get_node_labels_for_rel(rel_type); s.close(); return r


# ── résultat ─────────────────────────────────────────────────────────────────

def _render_degrade_result(result: dict, method: str):
    st.markdown("---")
    st.markdown('<div class="section-title">Résultat de la dégradation</div>',
                unsafe_allow_html=True)

    if method == "property":
        cols = st.columns(4)
        metrics = [
            ("Total instances",    result["total"],          "#3968d1"),
            ("Déjà manquantes",    result["already_missing"],"#d99a2b"),
            ("Dégradées",          result["degraded"],        "#e15a5a"),
            ("Total manquantes",   result["final_missing"],   "#00a884"),
        ]
    else:  # edge_coverage
        cols = st.columns(5)
        metrics = [
            ("Nœuds source",       result["total_src"],       "#3968d1"),
            ("Déjà sans arc",      result["already_no_edge"], "#d99a2b"),
            ("Nœuds dégradés",     result["degraded_nodes"],  "#e15a5a"),
            ("Arcs supprimés",     result["edges_deleted"],   "#00a884"),
            ("Total sans arc",     result["final_no_edge"],   "#191c26"),
        ]

    for col, (label, value, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="score-card" style="padding:16px 12px">
                <div class="score-value" style="color:{color};font-size:2rem">{value}</div>
                <div class="score-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    # Taux final
    if method == "property" and result["total"] > 0:
        rate = result["final_missing"] / result["total"]
        color = "#00a884" if rate < 0.2 else ("#d99a2b" if rate < 0.5 else "#e15a5a")
        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem">
            Taux de valeurs manquantes après dégradation :
            <strong style="color:{color}">{rate:.1%}</strong>
        </div>""", unsafe_allow_html=True)
    elif method == "edge" and result["total_src"] > 0:
        rate = result["final_no_edge"] / result["total_src"]
        color = "#00a884" if rate < 0.2 else ("#d99a2b" if rate < 0.5 else "#e15a5a")
        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem">
            Taux de nœuds source sans arc après dégradation :
            <strong style="color:{color}">{rate:.1%}</strong>
        </div>""", unsafe_allow_html=True)


# ── formulaire dégradation de propriété ──────────────────────────────────────

def _form_degrade_property(uri, user, password, database):
    st.markdown("#### Paramètres")

    elem_type = st.selectbox("Type d'élément", ["node", "edge"], key="dg_prop_elem")

    if elem_type == "node":
        labels = _labels(uri, user, password, database)
        label_or_type = st.selectbox("Label du nœud", labels, key="dg_prop_label") \
                        if labels else st.text_input("Label du nœud", key="dg_prop_label")
        props = _node_props(uri, user, password, label_or_type, database)
        prop = st.selectbox("Propriété", props, key="dg_prop_prop") \
               if props else st.text_input("Propriété", key="dg_prop_prop")
        src_label, dst_label = None, None
    else:
        rel_types = _rel_types(uri, user, password, database)
        label_or_type = st.selectbox("Type de relation", rel_types, key="dg_prop_rel") \
                        if rel_types else st.text_input("Type de relation", key="dg_prop_rel")
        src_labels, dst_labels = _rel_node_labels(uri, user, password, label_or_type, database) \
                                  if label_or_type else ([], [])
        src_label = st.selectbox("Nœud source", src_labels, key="dg_prop_src") \
                    if src_labels else st.text_input("Nœud source", key="dg_prop_src")
        dst_label = st.selectbox("Nœud destination", dst_labels, key="dg_prop_dst") \
                    if dst_labels else st.text_input("Nœud destination", key="dg_prop_dst")
        props = _edge_props(uri, user, password, label_or_type, database)
        prop = st.selectbox("Propriété", props, key="dg_prop_prop") \
               if props else st.text_input("Propriété", key="dg_prop_prop")

    lambda_pc = st.slider(
        "λ — Taux cible de valeurs manquantes",
        0.0, 1.0, 0.3, 0.05,
        key="dg_prop_lambda",
        help="Proportion d'instances pour lesquelles la propriété sera supprimée."
    )

    st.markdown(f"""
    <div class="info-box">
        La propriété <strong>{prop}</strong> sera supprimée sur
        <strong>{lambda_pc:.0%}</strong> des instances <strong>{label_or_type}</strong>.
        <br>⚠️ Cette opération modifie la base de données.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        confirm = st.checkbox("Je confirme vouloir dégrader les données", key="dg_prop_confirm")
    with col2:
        run = st.button("💥 Dégrader", use_container_width=True, key="dg_prop_run")

    if run:
        if not confirm:
            st.warning("Cochez la case de confirmation pour procéder.")
            return
        try:
            with st.spinner("Dégradation en cours..."):
                d = DataCompletenessDegrader(uri, user, password, database)
                result = d.degrade_property_completeness(
                    element_type=elem_type,
                    label_or_type=label_or_type,
                    property_name=prop,
                    lambda_pc=lambda_pc,
                    src_label=src_label,
                    dst_label=dst_label,
                )
                d.close()
            st.cache_data.clear()
            _render_degrade_result(result, "property")
        except Exception as e:
            st.error(f"❌ {e}")


# ── formulaire dégradation de couverture d'arcs ───────────────────────────────

def _form_degrade_edge_coverage(uri, user, password, database):
    st.markdown("#### Paramètres")

    rel_types = _rel_types(uri, user, password, database)
    rel_type = st.selectbox("Type de relation", rel_types, key="dg_ec_rel") \
               if rel_types else st.text_input("Type de relation", key="dg_ec_rel")

    src_labels, dst_labels = _rel_node_labels(uri, user, password, rel_type, database) \
                              if rel_type else ([], [])
    src_label = st.selectbox("Nœud source", src_labels, key="dg_ec_src") \
                if src_labels else st.text_input("Nœud source", key="dg_ec_src")
    dst_label = st.selectbox("Nœud destination", dst_labels, key="dg_ec_dst") \
                if dst_labels else st.text_input("Nœud destination", key="dg_ec_dst")

    lambda_ec = st.slider(
        "λ — Taux cible de nœuds source sans arc",
        0.0, 1.0, 0.3, 0.05,
        key="dg_ec_lambda",
        help="Proportion de nœuds source dont tous les arcs de ce type seront supprimés."
    )

    st.markdown(f"""
    <div class="info-box">
        Les arcs <strong>({src_label})-[{rel_type}]→({dst_label})</strong> seront
        supprimés pour <strong>{lambda_ec:.0%}</strong> des nœuds source.
        <br>⚠️ Les nœuds eux-mêmes ne sont pas supprimés.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        confirm = st.checkbox("Je confirme vouloir dégrader les données", key="dg_ec_confirm")
    with col2:
        run = st.button("💥 Dégrader", use_container_width=True, key="dg_ec_run")

    if run:
        if not confirm:
            st.warning("Cochez la case de confirmation pour procéder.")
            return
        try:
            with st.spinner("Suppression des arcs en cours..."):
                d = DataCompletenessDegrader(uri, user, password, database)
                result = d.degrade_edge_coverage(
                    src_label=src_label,
                    rel_type=rel_type,
                    dst_label=dst_label,
                    lambda_ec=lambda_ec,
                )
                d.close()
            st.cache_data.clear()
            _render_degrade_result(result, "edge")
        except Exception as e:
            st.error(f"❌ {e}")


# ── point d'entrée ────────────────────────────────────────────────────────────

def render_degradation_tab(uri: str, user: str, password: str, database: str = "neo4j"):
    """Rendu complet de l'onglet Dégradation."""

    st.markdown("""
    <div class="info-box">
        🧪 Les méthodes de <strong>dégradation</strong> modifient directement la base Neo4j.
        Elles permettent de simuler une baisse de qualité pour tester la robustesse
        des métriques. <strong>Utilisez un environnement de test.</strong>
    </div>
    """, unsafe_allow_html=True)

    method = st.radio(
        "Méthode de dégradation",
        [
            "🗑️ Complétude — Suppression de propriété",
            "✂️ Complétude — Suppression d'arcs (Edge Coverage)",
        ],
        key="dg_method",
        horizontal=True,
    )

    st.markdown('<div class="conn-panel">', unsafe_allow_html=True)

    if method.startswith("🗑️"):
        _form_degrade_property(uri, user, password, database)
    else:
        _form_degrade_edge_coverage(uri, user, password, database)

    st.markdown('</div>', unsafe_allow_html=True)
