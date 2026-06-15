import streamlit as st
import pandas as pd
from statistics import mean
from datetime import date
from ui.charts import make_gauge, make_bar_chart, score_color
from analysers import (
    DataFreshnessAnalyser,
    DataConformityAnalyser,
    DataCompletenessAnalyzer,
)
from ui.schema import SchemaExplorer
# ── Cache helpers ─────────────────────────────────────────────────────────────

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


# ── UI building blocks ────────────────────────────────────────────────────────

def _run_btn():
    return st.button("▶ Évaluer", use_container_width=True)

def _store(result: dict):
    st.session_state.result = result
    st.session_state.error  = None

def _store_err(e: Exception):
    st.session_state.error  = str(e)
    st.session_state.result = None

def _elem_type_selector(key):
    return st.selectbox("Type d'élément", ["node", "edge"], key=key)

def _label_sel(uri, user, password, key, database, label="Label du nœud"):
    opts = _labels(uri, user, password, database)
    return st.selectbox(label, opts, key=key) if opts else st.text_input(label, key=key)

def _rel_sel(uri, user, password, key, database):
    opts = _rel_types(uri, user, password, database)
    return st.selectbox("Type de relation", opts, key=key) if opts else st.text_input("Type de relation", key=key)

def _src_dst_sel(uri, user, password, rel_type, key_src, key_dst, database):
    """Sélecteurs source/destination basés sur les données réelles du graphe."""
    src_labels, dst_labels = _rel_node_labels(uri, user, password, rel_type, database) if rel_type else ([], [])
    src = st.selectbox("Nœud source", src_labels, key=key_src) if src_labels else st.text_input("Nœud source", key=key_src)
    dst = st.selectbox("Nœud destination", dst_labels, key=key_dst) if dst_labels else st.text_input("Nœud destination", key=key_dst)
    return src, dst

def _node_prop_sel(uri, user, password, label, key, database):
    opts = _node_props(uri, user, password, label, database) if label else []
    return st.selectbox("Propriété", opts, key=key) #if opts else st.text_input("Propriété", key=key)

def _edge_prop_sel(uri, user, password, rel_type, key, database):
    opts = _edge_props(uri, user, password, rel_type, database) if rel_type else []
    return st.selectbox("Propriété", opts, key=key) #if opts else st.text_input("Propriété", key=key)


# ── Freshness ─────────────────────────────────────────────────────────────────

def form_freshness_node(uri, user, password, database):
    label = _label_sel(uri, user, password, "fn_label", database)
    if _run_btn():
        try:
            a = DataFreshnessAnalyser(uri, user, password, database=database)
            r = a.avg_age_of_type("node", label)
            a.close()
            _store({"type": "age", "value": r, "label": f"Âge moyen – {label}"})
        except Exception as e:
            _store_err(e)


def form_freshness_edge(uri, user, password, database):
    rel_type    = _rel_sel(uri, user, password, "fe_rel", database)
    src, dst    = _src_dst_sel(uri, user, password, rel_type, "fe_src", "fe_dst", database)
    if _run_btn():
        try:
            a = DataFreshnessAnalyser(uri, user, password, database=database)
            r = a.avg_age_of_type("edge", rel_type, src_label=src, dst_label=dst)
            a.close()
            _store({"type": "age", "value": r,
                    "label": f"Âge moyen – ({src})-[{rel_type}]→({dst})"})
        except Exception as e:
            _store_err(e)


def form_freshness_property(uri, user, password, database):
    elem_type = _elem_type_selector("fp_elem")
    if elem_type == "node":
        label       = _label_sel(uri, user, password, "fp_label", database)
        prop        = _node_prop_sel(uri, user, password, label, "fp_prop", database)
        lbl_or_type = label
        src, dst    = None, None
    else:
        rel_type    = _rel_sel(uri, user, password, "fp_rel", database)
        src, dst    = _src_dst_sel(uri, user, password, rel_type, "fp_src", "fp_dst", database)
        prop        = _edge_prop_sel(uri, user, password, rel_type, "fp_prop", database)
        lbl_or_type = rel_type
    if _run_btn():
        try:
            a = DataFreshnessAnalyser(uri, user, password, database=database)
            r = a.avg_age_of_property(elem_type, lbl_or_type, prop,
                                       src_label=src, dst_label=dst)
            a.close()
            _store({"type": "age", "value": r, "label": f"Âge moyen · {prop}"})
        except Exception as e:
            _store_err(e)


def form_update_frequency(uri, user, password, database):
    elem_type = _elem_type_selector("uf_elem")
    if elem_type == "node":
        label       = _label_sel(uri, user, password, "uf_label", database)
        lbl_or_type = label
        src, dst    = None, None
    else:
        rel_type    = _rel_sel(uri, user, password, "uf_rel", database)
        src, dst    = _src_dst_sel(uri, user, password, rel_type, "uf_src", "uf_dst", database)
        lbl_or_type = rel_type

    days = st.number_input("Période (jours)", min_value=1, max_value=3650, value=30, key="uf_days")

    if _run_btn():
        try:
            a = DataFreshnessAnalyser(uri, user, password, database=database)
            r = a.update_frequency(elem_type, lbl_or_type,
                                    src_label=src, dst_label=dst, days=int(days))
            a.close()
            label_str = lbl_or_type if elem_type == "node" else f"({src})-[{lbl_or_type}]→({dst})"
            _store({"type": "frequency", "value": r,
                    "label": f"Fréquence de MAJ – {label_str}",
                    "days": int(days)})
        except Exception as e:
            _store_err(e)


# ── Completeness ──────────────────────────────────────────────────────────────

def form_completeness_prop(uri, user, password, database):
    elem_type = _elem_type_selector("cp_elem")
    if elem_type == "node":
        label       = _label_sel(uri, user, password, "cp_label", database)
        prop        = _node_prop_sel(uri, user, password, label, "cp_prop", database)
        src, dst    = None, None
        lbl_or_type = label
    else:
        rel_type    = _rel_sel(uri, user, password, "cp_rel", database)
        src, dst    = _src_dst_sel(uri, user, password, rel_type, "cp_src", "cp_dst", database)
        prop        = _edge_prop_sel(uri, user, password, rel_type, "cp_prop", database)
        lbl_or_type = rel_type
    if _run_btn():
        try:
            a = DataCompletenessAnalyzer(uri, user, password, database=database)
            r = a.completeness_of_property(elem_type, lbl_or_type, prop, src_label=src, dst_label=dst)
            a.close()
            _store({"type": "score", "value": r, "label": f"Complétude · {prop}"})
        except Exception as e:
            _store_err(e)


def form_completeness_node_edge(uri, user, password, database):
    rel_type    = _rel_sel(uri, user, password, "cne_rel", database)
    src, dst    = _src_dst_sel(uri, user, password, rel_type, "cne_src", "cne_dst", database)
    if _run_btn():
        try:
            a = DataCompletenessAnalyzer(uri, user, password, database=database)
            r = a.completeness_of_node_edge(src, dst, rel_type)
            a.close()
            _store({"type": "score", "value": r,
                    "label": f"Complétude · ({src})-[{rel_type}]→({dst})"})
        except Exception as e:
            _store_err(e)


def form_completeness_avg(uri, user, password, database):
    elem_type = _elem_type_selector("ca_elem")
    if elem_type == "node":
        label       = _label_sel(uri, user, password, "ca_label", database)
        all_props   = _node_props(uri, user, password, label, database)
        src, dst    = None, None
        lbl_or_type = label
    else:
        rel_type    = _rel_sel(uri, user, password, "ca_rel", database)
        all_props   = _edge_props(uri, user, password, rel_type, database)
        src, dst    = _src_dst_sel(uri, user, password, rel_type, "ca_src", "ca_dst", database)
        lbl_or_type = rel_type

    props = st.multiselect("Propriétés", all_props,
                           default=all_props[:3] if all_props else [], key="ca_props")
    if _run_btn():
        try:
            a = DataCompletenessAnalyzer(uri, user, password, database=database)
            details = {p: a.completeness_of_property(elem_type, lbl_or_type, p, src_label=src, dst_label=dst) for p in props}
            avg = mean(details.values()) if details else 0.0
            a.close()
            _store({"type": "multi", "value": avg, "details": details,
                    "label": f"Complétude moyenne – {lbl_or_type}",
                    "chart_title": "Complétude par propriété",
                    "global_label": "Complétude globale"})
        except Exception as e:
            _store_err(e)


# ── Conformity ────────────────────────────────────────────────────────────────

def form_conformity_prop(uri, user, password, database):
    elem_type = _elem_type_selector("confp_elem")
    if elem_type == "node":
        label       = _label_sel(uri, user, password, "confp_label", database)
        prop        = _node_prop_sel(uri, user, password, label, "confp_prop", database)
        src, dst    = None, None
        lbl_or_type = label
    else:
        rel_type    = _rel_sel(uri, user, password, "confp_rel", database)
        src, dst    = _src_dst_sel(uri, user, password, rel_type, "confp_src", "confp_dst", database)
        prop        = _edge_prop_sel(uri, user, password, rel_type, "confp_prop", database)
        lbl_or_type = rel_type
    regex = st.text_input("Regex", value=r"\d{4}-\d{2}-\d{2}", key="confp_regex")
    if _run_btn():
        try:
            a = DataConformityAnalyser(uri, user, password, database=database)
            r = a.conformity_of_property(elem_type, lbl_or_type, prop, regex, src_label=src, dst_label=dst)
            a.close()
            _store({"type": "score", "value": r, "label": f"Conformité · {prop}"})
        except Exception as e:
            _store_err(e)


def form_conformity_avg(uri, user, password, database):
    elem_type = _elem_type_selector("confa_elem")
    if elem_type == "node":
        label       = _label_sel(uri, user, password, "confa_label", database)
        all_props   = _node_props(uri, user, password, label, database)
        src, dst = None, None
        lbl_or_type = label
    else:
        rel_type    = _rel_sel(uri, user, password, "confa_rel", database)
        all_props   = _edge_props(uri, user, password, rel_type, database)
        src, dst    = _src_dst_sel(uri, user, password, rel_type, "confa_src", "confa_dst", database)
        lbl_or_type = rel_type

    props = st.multiselect("Propriétés", all_props,
                           default=all_props[:2] if all_props else [], key="confa_props")
    regexes = [
        st.text_input(f"Regex pour `{p}`", value=r"\d{4}-\d{2}-\d{2}", key=f"confa_regex_{p}")
        for p in props
    ]
    if _run_btn():
        try:
            a = DataConformityAnalyser(uri, user, password, database=database)
            details = {
                p: a.conformity_of_property(elem_type, lbl_or_type, p, reg, src_label=src, dst_label=dst)
                for p, reg in zip(props, regexes)
            }
            a.close()
            avg = mean(details.values()) if details else 0.0
            _store({"type": "multi", "value": avg, "details": details,
                    "label": f"Conformité moyenne – {lbl_or_type}",
                    "chart_title": "Conformité par propriété",
                    "global_label": "Conformité globale"})
        except Exception as e:
            _store_err(e)


# ── Dispatch ──────────────────────────────────────────────────────────────────

FORM_MAP = {
    "freshness_node":         form_freshness_node,
    "freshness_edge":         form_freshness_edge,
    "freshness_property":     form_freshness_property,
    "update_frequency":       form_update_frequency,
    "completeness_prop":      form_completeness_prop,
    "completeness_node_edge": form_completeness_node_edge,
    "completeness_avg":       form_completeness_avg,
    "conformity_prop":        form_conformity_prop,
    "conformity_avg":         form_conformity_avg,
}

def render_form(metric_id: str, uri: str, user: str, password: str, database: str = "neo4j"):
    fn = FORM_MAP.get(metric_id)
    if fn:
        fn(uri, user, password, database)


# ── Results ───────────────────────────────────────────────────────────────────

def render_result():
    if st.session_state.get("error"):
        st.error(f"❌ {st.session_state.error}")
        return
    res = st.session_state.get("result")
    if not res:
        st.markdown("""
        <div class="info-box" style="margin-top:2rem;text-align:center;padding:40px 20px;">
            <div style="font-size:2rem;margin-bottom:12px;opacity:0.4">◎</div>
            Renseignez les paramètres et cliquez sur <strong>▶ Évaluer</strong>.
        </div>
        """, unsafe_allow_html=True)
        return
    t = res["type"]
    if t == "age":             _render_age(res)
    elif t == "score":         _render_score(res)
    elif t == "multi":         _render_multi(res)
    elif t == "frequency":     _render_frequency(res)
    elif t == "delay":         _render_delay(res)
    elif t == "accuracy":      _render_accuracy(res)
    elif t == "accuracy_multi":_render_accuracy_multi(res)


def _render_age(res):
    val = res["value"]
    if val is None:
        st.warning("Aucune donnée CDC trouvée pour cet élément.")
        return
    days = val / 86400
    freshness = max(0.0, 1.0 - (days / 30))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:#f7ae6a">{days:.2f}</div>
            <div class="score-label">jours (âge moyen)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.plotly_chart(make_gauge(freshness, "Score de fraîcheur (base 30j)"),
                        use_container_width=True, key="gauge_age")


def _render_score(res):
    val   = res["value"]
    color = score_color(val)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:{color}">{val:.1%}</div>
            <div class="score-label">{res['label']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.plotly_chart(make_gauge(val, res["label"]),
                        use_container_width=True, key="gauge_score")


def _render_multi(res):
    val          = res["value"]
    details      = res["details"]
    color        = score_color(val)
    chart_title  = res.get("chart_title", "Score par propriété")
    global_label = res.get("global_label", "Score global")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:{color}">{val:.1%}</div>
            <div class="score-label">{global_label}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.plotly_chart(make_gauge(val, res["label"]),
                        use_container_width=True, key="gauge_multi")
    st.plotly_chart(make_bar_chart(details, chart_title),
                    use_container_width=True, key="bar_multi")
    df = pd.DataFrame(
        [(p, f"{v:.1%}", "✓ Bon" if v >= 0.8 else ("⚠ Moyen" if v >= 0.5 else "✗ Faible"))
         for p, v in details.items()],
        columns=["Propriété", "Score", "Statut"]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_frequency(res):
    val  = res["value"]
    days = res.get("days", 30)
    if val is None:
        st.warning("Aucune donnée CDC trouvée pour cet élément.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:#a89cf7">{val:.2f}</div>
            <div class="score-label">MAJ moyennes / instance sur {days}j</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="info-box" style="margin-top:0">
            <strong>{res['label']}</strong><br><br>
            Chaque instance a été modifiée en moyenne
            <strong>{val:.2f} fois</strong> sur les {days} derniers jours.<br><br>
            {'🟢 Activité élevée' if val >= 5 else ('🟡 Activité modérée' if val >= 1 else '🔴 Activité faible')}
        </div>""", unsafe_allow_html=True)


# ── Nouveaux imports ──────────────────────────────────────────────────────────
from analysers import (
    DataFreshnessAnalyserExt,
    DataConformityAnalyserExt,
    DataUniquenessAnalyser,
    DataAccuracyAnalyser,
)


# ── Freshness extensions ──────────────────────────────────────────────────────

def form_freshness_score(uri, user, password, database):
    """FreshnessScore = 1 / (1 + PropertyAge)"""
    elem_type = _elem_type_selector("fs_elem")
    if elem_type == "node":
        label       = _label_sel(uri, user, password, "fs_label", database)
        prop        = _node_prop_sel(uri, user, password, label, "fs_prop", database)
        lbl_or_type = label; src, dst = None, None
    else:
        rel_type    = _rel_sel(uri, user, password, "fs_rel", database)
        src, dst    = _src_dst_sel(uri, user, password, rel_type, "fs_src", "fs_dst", database)
        prop        = _edge_prop_sel(uri, user, password, rel_type, "fs_eprop", database)
        lbl_or_type = rel_type
    if _run_btn():
        try:
            a = DataFreshnessAnalyserExt(uri, user, password, database)
            r = a.freshness_score(elem_type, lbl_or_type, prop, src_label=src, dst_label=dst)
            a.close()
            _store({"type": "score", "value": r, "label": f"Score de fraîcheur · {prop}"})
        except Exception as e:
            _store_err(e)


def form_update_delay(uri, user, password, database):
    """UpdateDelay : temps moyen entre deux mises à jour successives."""
    elem_type = _elem_type_selector("ud_elem")
    if elem_type == "node":
        label       = _label_sel(uri, user, password, "ud_label", database)
        lbl_or_type = label; src, dst = None, None
    else:
        rel_type    = _rel_sel(uri, user, password, "ud_rel", database)
        src, dst    = _src_dst_sel(uri, user, password, rel_type, "ud_src", "ud_dst", database)
        lbl_or_type = rel_type
    if _run_btn():
        try:
            a = DataFreshnessAnalyserExt(uri, user, password, database)
            r = a.update_delay(elem_type, lbl_or_type, src_label=src, dst_label=dst)
            a.close()
            if r is None:
                _store({"type": "age", "value": None, "label": ""})
            else:
                _store({"type": "delay", "value": r, "label": f"Délai moyen – {lbl_or_type}"})
        except Exception as e:
            _store_err(e)


# ── Completeness extensions ───────────────────────────────────────────────────

def form_completeness_weighted(uri, user, password, database):
    """WeightedTypeCompleteness."""
    elem_type = _elem_type_selector("cw_elem")
    if elem_type == "node":
        label     = _label_sel(uri, user, password, "cw_label", database)
        all_props = _node_props(uri, user, password, label, database)
        lbl_or_type = label
        src, dst = None, None
    else:
        rel_type  = _rel_sel(uri, user, password, "cw_rel", database)
        src, dst = _src_dst_sel(uri, user, password, rel_type, "cw_src", "cw_dst", database)
        all_props = _edge_props(uri, user, password, rel_type, database)
        lbl_or_type = rel_type

    props = st.multiselect("Propriétés", all_props,
                           default=all_props[:3] if all_props else [], key="cw_props")
    weights_raw = [
        st.number_input(f"Poids de `{p}`", min_value=0.0, max_value=1.0,
                        value=round(1/len(props), 2) if props else 0.0,
                        step=0.05, key=f"cw_w_{p}")
        for p in props
    ]
    if props:
        total_w = sum(weights_raw)
        st.caption(f"Somme des poids : **{total_w:.2f}** {'✓' if abs(total_w-1)<0.01 else '⚠ doit être égal à 1'}")
    if _run_btn():
        try:
            a = DataCompletenessAnalyzer(uri, user, password, database)
            r = a.weighted_completeness_of_type(elem_type, lbl_or_type, props, weights_raw, src_label=src, dst_label=dst)
            a.close()
            _store({"type": "score", "value": r, "label": f"Complétude pondérée – {lbl_or_type}"})
        except Exception as e:
            _store_err(e)


# ── Conformity extensions ─────────────────────────────────────────────────────

def form_conformity_conditional(uri, user, password, database):
    """ConditionalPropertyConformity."""
    elem_type = _elem_type_selector("cc_elem")
    if elem_type == "node":
        label       = _label_sel(uri, user, password, "cc_label", database)
        prop        = _node_prop_sel(uri, user, password, label, "cc_prop", database)
        lbl_or_type = label; src, dst = None, None
    else:
        rel_type    = _rel_sel(uri, user, password, "cc_rel", database)
        src, dst    = _src_dst_sel(uri, user, password, rel_type, "cc_src", "cc_dst", database)
        prop        = _edge_prop_sel(uri, user, password, rel_type, "cc_eprop", database)
        lbl_or_type = rel_type
    regex = st.text_input("Regex", value=r"\d{4}-\d{2}-\d{2}", key="cc_regex")
    if _run_btn():
        try:
            a = DataConformityAnalyserExt(uri, user, password, database)
            r = a.conditional_conformity_of_property(elem_type, lbl_or_type, prop, regex,
                                                      src_label=src, dst_label=dst)
            a.close()
            _store({"type": "score", "value": r, "label": f"Conformité conditionnelle · {prop}"})
        except Exception as e:
            _store_err(e)


def form_conformity_weighted(uri, user, password, database):
    """WeightedTypeConformity."""
    elem_type = _elem_type_selector("cfw_elem")
    if elem_type == "node":
        label     = _label_sel(uri, user, password, "cfw_label", database)
        all_props = _node_props(uri, user, password, label, database)
        lbl_or_type = label
        src, dst = None, None
    else:
        rel_type  = _rel_sel(uri, user, password, "cfw_rel", database)
        src, dst = _src_dst_sel(uri, user, password, rel_type, "cfw_src", "cfw_dst", database)
        all_props = _edge_props(uri, user, password, rel_type, database)
        lbl_or_type = rel_type

    props = st.multiselect("Propriétés", all_props,
                           default=all_props[:2] if all_props else [], key="cfw_props")
    regexes = [
        st.text_input(f"Regex pour `{p}`", value=r"\d{4}-\d{2}-\d{2}", key=f"cfw_regex_{p}")
        for p in props
    ]
    weights_raw = [
        st.number_input(f"Poids de `{p}`", min_value=0.0, max_value=1.0,
                        value=round(1/len(props), 2) if props else 0.0,
                        step=0.05, key=f"cfw_w_{p}")
        for p in props
    ]
    if props:
        total_w = sum(weights_raw)
        st.caption(f"Somme des poids : **{total_w:.2f}** {'✓' if abs(total_w-1)<0.01 else '⚠ doit être égal à 1'}")
    if _run_btn():
        try:
            a = DataConformityAnalyserExt(uri, user, password, database)
            r = a.weighted_conformity_of_type(elem_type, lbl_or_type, props, regexes, weights_raw, src_label=src, dst_label=dst)
            a.close()
            _store({"type": "score", "value": r, "label": f"Conformité pondérée – {lbl_or_type}"})
        except Exception as e:
            _store_err(e)


# ── Uniqueness ────────────────────────────────────────────────────────────────

def _uniqueness_base_form(uri, user, password, database, prefix):
    elem_type = _elem_type_selector(f"{prefix}_elem")
    if elem_type == "node":
        label       = _label_sel(uri, user, password, f"{prefix}_label", database)
        lbl_or_type = label; src, dst = None, None
    else:
        rel_type    = _rel_sel(uri, user, password, f"{prefix}_rel", database)
        src, dst    = _src_dst_sel(uri, user, password, rel_type, f"{prefix}_src", f"{prefix}_dst", database)
        lbl_or_type = rel_type
    return elem_type, lbl_or_type, src, dst


def form_uniqueness_exact(uri, user, password, database):
    elem_type, lbl, src, dst = _uniqueness_base_form(uri, user, password, database, "ue")
    if _run_btn():
        try:
            a = DataUniquenessAnalyser(uri, user, password, database)
            r = a.exact_duplicate_rate(elem_type, lbl, src_label=src, dst_label=dst)
            a.close()
            _store({"type": "score", "value": r, "label": f"Taux de duplicats exacts – {lbl}"})
        except Exception as e:
            _store_err(e)


def form_uniqueness_partial(uri, user, password, database):
    elem_type, lbl, src, dst = _uniqueness_base_form(uri, user, password, database, "up")
    threshold = st.slider("Seuil de similarité θ", 0.0, 1.0, 0.8, 0.05, key="up_theta")
    if _run_btn():
        try:
            a = DataUniquenessAnalyser(uri, user, password, database)
            r = a.partial_duplicate_rate(elem_type, lbl, threshold, src_label=src, dst_label=dst)
            a.close()
            _store({"type": "score", "value": r,
                    "label": f"Taux de duplicats partiels (θ={threshold}) – {lbl}"})
        except Exception as e:
            _store_err(e)


def form_uniqueness_score(uri, user, password, database):
    elem_type, lbl, src, dst = _uniqueness_base_form(uri, user, password, database, "us")
    if _run_btn():
        try:
            a = DataUniquenessAnalyser(uri, user, password, database)
            r = a.uniqueness_score(elem_type, lbl, src_label=src, dst_label=dst)
            a.close()
            _store({"type": "score", "value": r, "label": f"Score d'unicité – {lbl}"})
        except Exception as e:
            _store_err(e)


def form_uniqueness_redund(uri, user, password, database):
    elem_type, lbl, src, dst = _uniqueness_base_form(uri, user, password, database, "ur")
    if _run_btn():
        try:
            a = DataUniquenessAnalyser(uri, user, password, database)
            r = a.average_redundancy(elem_type, lbl, src_label=src, dst_label=dst)
            a.close()
            _store({"type": "score", "value": r, "label": f"Redondance moyenne – {lbl}"})
        except Exception as e:
            _store_err(e)


# ── Accuracy ──────────────────────────────────────────────────────────────────

def form_accuracy_property(uri, user, password, database):
    elem_type = _elem_type_selector("ap_elem")
    if elem_type == "node":
        label       = _label_sel(uri, user, password, "ap_label", database)
        prop        = _node_prop_sel(uri, user, password, label, "ap_prop", database)
        lbl_or_type = label; src, dst = None, None
    else:
        rel_type    = _rel_sel(uri, user, password, "ap_rel", database)
        src, dst    = _src_dst_sel(uri, user, password, rel_type, "ap_src", "ap_dst", database)
        prop        = _edge_prop_sel(uri, user, password, rel_type, "ap_eprop", database)
        lbl_or_type = rel_type
    ref_val = st.number_input("Valeur de référence", value=0.0, key="ap_ref")
    if _run_btn():
        try:
            a = DataAccuracyAnalyser(uri, user, password, database)
            r = a.property_accuracy(elem_type, lbl_or_type, prop, ref_val,
                                    src_label=src, dst_label=dst)
            a.close()
            _store({"type": "accuracy", "value": r,
                    "label": f"Exactitude · {prop} (réf={ref_val})"})
        except Exception as e:
            _store_err(e)


def form_accuracy_type(uri, user, password, database):
    elem_type = _elem_type_selector("at_elem")
    if elem_type == "node":
        label     = _label_sel(uri, user, password, "at_label", database)
        all_props = _node_props(uri, user, password, label, database)
        lbl_or_type = label
        src, dst = None, None
    else:
        rel_type  = _rel_sel(uri, user, password, "at_rel", database)
        src, dst = _src_dst_sel(uri, user, password, rel_type, "at_src", "at_dst", database)
        all_props = _edge_props(uri, user, password, rel_type, database)
        lbl_or_type = rel_type

    props = st.multiselect("Propriétés numériques", all_props,
                           default=all_props[:2] if all_props else [], key="at_props")
    refs = [
        st.number_input(f"Référence pour `{p}`", value=0.0, key=f"at_ref_{p}")
        for p in props
    ]
    if _run_btn():
        try:
            a = DataAccuracyAnalyser(uri, user, password, database)
            details = {
                p: a.property_accuracy(elem_type, lbl_or_type, p, ref, src_label=src, dst_label=dst)
                for p, ref in zip(props, refs)
                if a.property_accuracy(elem_type, lbl_or_type, p, ref, src_label=src, dst_label=dst) is not None
            }
            avg = mean(details.values()) if details else None
            a.close()
            _store({"type": "accuracy_multi", "value": avg, "details": details,
                    "label": f"Exactitude moyenne – {lbl_or_type}"})
        except Exception as e:
            _store_err(e)


# ── Mise à jour du dispatch ───────────────────────────────────────────────────

FORM_MAP.update({
    "freshness_score":       form_freshness_score,
    "update_delay":          form_update_delay,
    "completeness_weighted": form_completeness_weighted,
    "conformity_conditional":form_conformity_conditional,
    "conformity_weighted":   form_conformity_weighted,
    "uniqueness_exact":      form_uniqueness_exact,
    "uniqueness_partial":    form_uniqueness_partial,
    "uniqueness_score":      form_uniqueness_score,
    "uniqueness_redund":     form_uniqueness_redund,
    "accuracy_property":     form_accuracy_property,
    "accuracy_type":         form_accuracy_type,
})


def _render_delay(res):
    val = res["value"]
    if val is None:
        st.warning("Pas assez d'événements CDC pour calculer un délai.")
        return
    hours = val / 3600
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:#a89cf7">{hours:.2f}</div>
            <div class="score-label">heures (délai moyen entre MAJ)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="info-box" style="margin-top:0">
            <strong>{res["label"]}</strong><br><br>
            Délai moyen entre deux mises à jour successives :<br>
            <strong>{val:.0f} secondes</strong> ({hours:.2f}h)
        </div>""", unsafe_allow_html=True)


def _render_accuracy(res):
    val = res["value"]
    if val is None:
        st.warning("Aucune donnée trouvée pour cette propriété.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:#f7ae6a">{val:.4f}</div>
            <div class="score-label">Erreur absolue moyenne</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="info-box" style="margin-top:0">
            <strong>{res["label"]}</strong><br><br>
            Différence moyenne entre les valeurs du graphe
            et la valeur de référence.<br>
            Une valeur proche de <strong>0</strong> indique une forte exactitude.
        </div>""", unsafe_allow_html=True)


def _render_accuracy_multi(res):
    val     = res["value"]
    details = res["details"]
    if val is None:
        st.warning("Aucune donnée trouvée.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-value" style="color:#f7ae6a">{val:.4f}</div>
            <div class="score-label">Erreur absolue moyenne globale</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="info-box" style="margin-top:0">
            <strong>{res["label"]}</strong><br><br>
            Moyenne des erreurs absolues par propriété.
        </div>""", unsafe_allow_html=True)
    df = pd.DataFrame(
        [(p, f"{v:.4f}") for p, v in details.items()],
        columns=["Propriété", "Erreur absolue moyenne"]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
