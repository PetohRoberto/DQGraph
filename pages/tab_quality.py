"""
pages/tab_quality.py — Data Quality Insights tab.

Display: one summary table per quality dimension.
Columns: Métrique | Élément | Label | Valeur
Results accumulated in the project and persisted.
"""
import sys, os, html
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from datetime import datetime
from statistics import mean

from analysers import (
    DataCompletenessAnalyzer,
    DataConformityAnalyser,
    DataFreshnessAnalyser,
    DataUniquenessAnalyser,
    DataAccuracyAnalyser,
)
from project_manager import Project, METRIC_TO_DIMENSION
import experiment_jobs

# ── CSS ───────────────────────────────────────────────────────────────────────
QUALITY_CSS = """
<style>
.q-page-title {
  font-family: 'Syne', sans-serif; font-weight: 700;
  font-size: 1.3rem; color: var(--text); margin: 6px 0 4px;
}
.q-page-sub {
  font-family: 'DM Mono', monospace; font-size: 0.7rem;
  color: var(--muted); letter-spacing: .12em; text-transform: uppercase;
  margin-bottom: 20px;
}
.sec-hd {
  font-family: 'DM Mono', monospace; font-size: 0.68rem;
  letter-spacing: .14em; text-transform: uppercase;
  color: var(--muted); margin: 8px 0 10px;
}
.metric-hd {
  font-family: 'DM Mono', monospace; font-size: .72rem;
  letter-spacing: .08em; text-transform: uppercase; margin: 14px 0 8px;
}
.info-strip {
  background: rgba(57,104,209,.06);
  border-left: 3px solid var(--accent);
  border-radius: 0 8px 8px 0;
  padding: 14px 18px;
  font-family: 'DM Mono', monospace; font-size: .78rem;
  color: var(--muted); margin: 12px 0;
}
/* Dimension section header */
.dim-hd {
  font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem;
  padding: 10px 0 6px; margin-top: 16px;
  border-bottom: 1px solid var(--border);
}
/* styled dataframe override */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  overflow: hidden;
}
</style>
"""

DIMENSIONS = {
    "Completeness": {
        "icon": "◫", "color": "#3968d1",
        "metrics": {
            "property_completeness": "Property Completeness",
            "edge_coverage":         "Edge Coverage",
            "type_completeness":     "Average Type Completeness",
        },
    },
    "Conformity": {
        "icon": "◎", "color": "#00a884",
        "metrics": {
            "property_conformity":    "Property Conformity",
            "conditional_conformity": "Conditional Property Conformity",
            "type_conformity":        "Average Type Conformity",
        },
    },
    "Freshness": {
        "icon": "◷", "color": "#d99a2b",
        "metrics": {
            "type_age":         "Average Type Age",
            "property_age":     "Average Property Age",
            "update_frequency": "Update Frequency",
        },
    },
    "Uniqueness": {
        "icon": "◈", "color": "#e15a5a",
        "metrics": {
            "exact_duplicate_rate":   "Exact Duplicate Rate",
            "partial_duplicate_rate": "Partial Duplicate Rate",
            "uniqueness_score":       "Uniqueness Score",
        },
    },
    "Accuracy": {
        "icon": "◉", "color": "#9b5de0",
        "metrics": {
            "property_accuracy": "Property Accuracy",
        },
    },
}


# ── cached schema fetchers ────────────────────────────────────────────────────
from ui.schema_cache import (
    labels as _labels, rels as _rels, node_props as _node_props,
    edge_props as _edge_props, rel_labels as _rel_labels, clear_schema_cache,
    accuracy_props as _accuracy_props,
)


# ── value formatting ──────────────────────────────────────────────────────────

def _format_value(value, display_type: str) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    if display_type == "score":
        return f"{v:.1%}"
    if display_type == "multi":
        return f"{v:.1%}"
    if display_type == "age":
        days = v / 86400
        return f"{days:.1f} d"
    if display_type == "frequency":
        return f"{v:.3f} upd/d"
    if display_type == "accuracy":
        return f"{v:.4f} (MAE)"
    return f"{v:.4f}"


def _label_display(metric_id: str, params: dict) -> tuple[str, str]:
    """Returns (element_type, label_string) for the table."""
    et = params.get("et", "")
    if metric_id == "edge_coverage":
        src = params.get("src", "?")
        rel = params.get("rel", "?")
        dst = params.get("dst", "?")
        return "edge", f"{src} → {rel} → {dst}"
    if et == "edge":
        label = params.get("label", "?")
        return "edge", label
    return et or "node", params.get("label", "?")


def _property_display(params: dict) -> str:
    """Property (or properties) a result was computed on, read from its saved params."""
    if params.get("prop"):
        return params["prop"]
    if params.get("props"):
        return ", ".join(params["props"])
    return "—"


# ── metric param forms ────────────────────────────────────────────────────────

def _metric_params(metric_id, uri, user, pwd, db, prefix):
    labels    = _labels(uri, user, pwd, db)
    rels      = _rels(uri, user, pwd, db)
    elem_types = ["node", "edge"]

    if metric_id == "property_completeness":
        et    = st.selectbox("Element type", elem_types, key=f"{prefix}_et")
        label = st.selectbox("Node label" if et == "node" else "Relation type",
                             labels if et == "node" else rels, key=f"{prefix}_lbl")
        props = _node_props(uri, user, pwd, db, label) if et == "node" else _edge_props(uri, user, pwd, db, label)
        prop  = st.selectbox("Property", props, key=f"{prefix}_prop")
        return {"et": et, "label": label, "prop": prop}

    if metric_id == "edge_coverage":
        rel = st.selectbox("Relation type", rels, key=f"{prefix}_rel")
        srcs, dsts = _rel_labels(uri, user, pwd, db, rel) if rel else ([], [])
        src = st.selectbox("Source label",      srcs, key=f"{prefix}_src")
        dst = st.selectbox("Destination label", dsts, key=f"{prefix}_dst")
        return {"rel": rel, "src": src, "dst": dst}

    if metric_id == "type_completeness":
        et    = st.selectbox("Element type", elem_types, key=f"{prefix}_et")
        label = st.selectbox("Node label" if et == "node" else "Relation type",
                             labels if et == "node" else rels, key=f"{prefix}_lbl")
        props = _node_props(uri, user, pwd, db, label) if et == "node" else _edge_props(uri, user, pwd, db, label)
        sel   = st.multiselect("Properties", props, default=props[:3], key=f"{prefix}_props")
        return {"et": et, "label": label, "props": sel}

    if metric_id in ("property_conformity", "conditional_conformity"):
        et    = st.selectbox("Element type", elem_types, key=f"{prefix}_et")
        label = st.selectbox("Node label" if et == "node" else "Relation type",
                             labels if et == "node" else rels, key=f"{prefix}_lbl")
        props = _node_props(uri, user, pwd, db, label) if et == "node" else _edge_props(uri, user, pwd, db, label)
        prop  = st.selectbox("Property", props, key=f"{prefix}_prop")
        regex = st.text_input("Regex constraint", value=r"\d{4}-\d{2}-\d{2}", key=f"{prefix}_rx")
        return {"et": et, "label": label, "prop": prop, "regex": regex}

    if metric_id == "type_conformity":
        et    = st.selectbox("Element type", elem_types, key=f"{prefix}_et")
        label = st.selectbox("Node label" if et == "node" else "Relation type",
                             labels if et == "node" else rels, key=f"{prefix}_lbl")
        props = _node_props(uri, user, pwd, db, label) if et == "node" else _edge_props(uri, user, pwd, db, label)
        sel   = st.multiselect("Properties", props, default=props[:2], key=f"{prefix}_props")
        regexes = [
            st.text_input(f"Regex for `{p}`", value=r"\d{4}-\d{2}-\d{2}", key=f"{prefix}_rx_{p}")
            for p in sel
        ]
        return {"et": et, "label": label, "props": sel, "regexes": regexes}

    if metric_id == "type_age":
        et    = st.selectbox("Element type", elem_types, key=f"{prefix}_et")
        label = st.selectbox("Node label" if et == "node" else "Relation type",
                             labels if et == "node" else rels, key=f"{prefix}_lbl")
        return {"et": et, "label": label}

    if metric_id == "property_age":
        et    = st.selectbox("Element type", elem_types, key=f"{prefix}_et")
        label = st.selectbox("Node label" if et == "node" else "Relation type",
                             labels if et == "node" else rels, key=f"{prefix}_lbl")
        props = _node_props(uri, user, pwd, db, label) if et == "node" else _edge_props(uri, user, pwd, db, label)
        prop  = st.selectbox("Property", props, key=f"{prefix}_prop")
        return {"et": et, "label": label, "prop": prop}

    if metric_id == "update_frequency":
        et    = st.selectbox("Element type", elem_types, key=f"{prefix}_et")
        label = st.selectbox("Node label" if et == "node" else "Relation type",
                             labels if et == "node" else rels, key=f"{prefix}_lbl")
        days  = st.slider("Period (days)", 1, 90, 30, key=f"{prefix}_days")
        return {"et": et, "label": label, "days": days}

    if metric_id in ("exact_duplicate_rate", "partial_duplicate_rate", "uniqueness_score"):
        et    = st.selectbox("Element type", elem_types, key=f"{prefix}_et")
        label = st.selectbox("Node label" if et == "node" else "Relation type",
                             labels if et == "node" else rels, key=f"{prefix}_lbl")
        theta = None
        if metric_id == "partial_duplicate_rate":
            theta = st.slider("Similarity threshold θ", 0.0, 1.0, 0.8, 0.05, key=f"{prefix}_theta")
        return {"et": et, "label": label, "theta": theta}

    if metric_id == "property_accuracy":
        et    = st.selectbox("Element type", elem_types, key=f"{prefix}_et")
        label = st.selectbox("Node label" if et == "node" else "Relation type",
                             labels if et == "node" else rels, key=f"{prefix}_lbl")
        # Only numeric properties (numbers stored as strings included): on a
        # date or a text, the deviation to a reference value is meaningless.
        props = _accuracy_props(uri, user, pwd, db, et, label)
        prop  = st.selectbox("Numeric property", props, key=f"{prefix}_prop")
        if label and not props:
            st.caption(f"⚠ No numeric property on {label}.")
        ref   = st.number_input("Reference value", value=0.0, key=f"{prefix}_ref")
        return {"et": et, "label": label, "prop": prop, "ref": ref}

    return {}


# ── runner ────────────────────────────────────────────────────────────────────

def _run_metric(metric_id, params, uri, user, pwd, db):
    try:
        if metric_id == "property_completeness":
            a = DataCompletenessAnalyzer(uri, user, pwd, db)
            v = a.completeness_of_property(params["et"], params["label"], params["prop"])
            a.close(); return v, "score"

        if metric_id == "edge_coverage":
            a = DataCompletenessAnalyzer(uri, user, pwd, db)
            v = a.completeness_of_node_edge(params["src"], params["dst"], params["rel"])
            a.close(); return v, "score"

        if metric_id == "type_completeness":
            a = DataCompletenessAnalyzer(uri, user, pwd, db)
            details = {p: a.completeness_of_property(params["et"], params["label"], p)
                       for p in params["props"]}
            a.close()
            v = mean(details.values()) if details else 0.0
            return v, "multi"

        if metric_id == "property_conformity":
            a = DataConformityAnalyser(uri, user, pwd, db)
            v = a.conformity_of_property(params["et"], params["label"],
                                         params["prop"], params["regex"])
            a.close(); return v, "score"

        if metric_id == "conditional_conformity":
            from analysers import DataConformityAnalyserExt
            a = DataConformityAnalyserExt(uri, user, pwd, db)
            v = a.conditional_conformity_of_property(params["et"], params["label"],
                                                     params["prop"], params["regex"])
            a.close(); return v, "score"

        if metric_id == "type_conformity":
            a = DataConformityAnalyser(uri, user, pwd, db)
            details = {p: a.conformity_of_property(params["et"], params["label"], p, rx)
                       for p, rx in zip(params["props"], params["regexes"])}
            a.close()
            v = mean(details.values()) if details else 0.0
            return v, "multi"

        if metric_id == "type_age":
            a = DataFreshnessAnalyser(uri, user, pwd, db)
            v = a.avg_age_of_type(params["et"], params["label"])
            a.close(); return v, "age"

        if metric_id == "property_age":
            a = DataFreshnessAnalyser(uri, user, pwd, db)
            v = a.avg_age_of_property(params["et"], params["label"], params["prop"])
            a.close(); return v, "age"

        if metric_id == "update_frequency":
            a = DataFreshnessAnalyser(uri, user, pwd, db)
            v = a.update_frequency(params["et"], params["label"], days=params["days"])
            a.close(); return v, "frequency"

        if metric_id == "exact_duplicate_rate":
            a = DataUniquenessAnalyser(uri, user, pwd, db)
            v = a.exact_duplicate_rate(params["et"], params["label"])
            a.close(); return v, "score"

        if metric_id == "partial_duplicate_rate":
            a = DataUniquenessAnalyser(uri, user, pwd, db)
            v = a.partial_duplicate_rate(params["et"], params["label"],
                                         threshold=params["theta"])
            a.close(); return v, "score"

        if metric_id == "uniqueness_score":
            a = DataUniquenessAnalyser(uri, user, pwd, db)
            v = a.uniqueness_score(params["et"], params["label"])
            a.close(); return v, "score"

        if metric_id == "property_accuracy":
            a = DataAccuracyAnalyser(uri, user, pwd, db)
            v = a.property_accuracy(params["et"], params["label"],
                                    params["prop"], params["ref"])
            a.close()
            if v is None:
                raise ValueError(f"'{params['prop']}' has no numeric value on "
                                 f"{params['label']}.")
            return v, "accuracy"

    # On error, the message is returned in place of the value (see Evaluate).
    except Exception as e:
        return str(e), "error"
    return f"unknown metric '{metric_id}'.", "error"


# ── summary tables ────────────────────────────────────────────────────────────

def _score_color_hex(val_str: str) -> str:
    if "%" in val_str:
        try:
            v = float(val_str.replace("%", "")) / 100
            if v >= 0.8: return "#00a884"
            if v >= 0.5: return "#d99a2b"
            return "#e15a5a"
        except Exception:
            pass
    return "#191c26"


def _render_dimension_table(dim_name: str, dim_info: dict, entries: list, last_key: str | None = None):
    color = dim_info["color"]
    icon  = dim_info["icon"]

    st.markdown(
        f'<div class="dim-hd" style="color:{color}">'
        f'{icon} &nbsp;{dim_name}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not entries:
        st.markdown(
            '<div class="info-strip">No metrics computed for this dimension yet.</div>',
            unsafe_allow_html=True,
        )
        return

    # Sort: last computed entry first
    sorted_entries = sorted(entries, key=lambda e: (e.get("_key") != last_key))

    # Build HTML table for full formatting control (bold on latest row)
    th_style = ("font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.1em;"
                "text-transform:uppercase;color:#6b7085;padding:6px 12px;"
                "border-bottom:1px solid #e1e4ea;text-align:left;")
    td_style = ("font-family:'DM Mono',monospace;font-size:.78rem;color:#191c26;"
                "padding:6px 12px;border-bottom:1px solid #f4f5f8;")

    # No DB column: the tables only show the results of the selected database.
    headers = ["Metric", "Element", "Label", "Property", "Value", "Computed"]
    thead = "".join(f'<th style="{th_style}">{h}</th>' for h in headers)

    rows_html = ""
    for e in sorted_entries:
        is_latest  = (last_key is not None) and (e.get("_key") == last_key)
        value_str  = _format_value(e.get("value"), e.get("display_type", "score"))
        val_color  = _score_color_hex(value_str)
        row_bg     = "background:rgba(57,104,209,0.07);" if is_latest else ""
        bold       = "font-weight:700;" if is_latest else ""
        marker     = "▶ " if is_latest else ""
        computed   = (e.get("computed_at", "")[:16].replace("T", " ")
                      if e.get("computed_at") else "")

        cells = [
            f'<td style="{td_style}{bold}">{marker}{e.get("metric_name","")}</td>',
            f'<td style="{td_style}{bold}color:#6b7085">{e.get("element_type","")}</td>',
            f'<td style="{td_style}{bold}">{e.get("label_display","")}</td>',
            f'<td style="{td_style}{bold}">{html.escape(_property_display(e.get("params") or {}))}</td>',
            f'<td style="{td_style}{bold}color:{val_color}">{value_str}</td>',
            f'<td style="{td_style}{bold}color:#6b7085;font-size:.65rem">{computed}</td>',
        ]
        rows_html += f'<tr style="{row_bg}">{"".join(cells)}</tr>'

    table_html = f"""
    <div style="overflow-x:auto;border:1px solid #e1e4ea;border-radius:10px;
                margin-bottom:8px;background:#f4f5f8">
      <table style="width:100%;border-collapse:collapse">
        <thead><tr>{thead}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""
    st.markdown(table_html, unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def render_quality_tab():
    st.markdown(QUALITY_CSS, unsafe_allow_html=True)

    uri  = st.session_state.uri
    user = st.session_state.neo_user
    pwd  = st.session_state.neo_pass
    dbs  = st.session_state.databases
    proj: Project = st.session_state.project

    st.markdown('<div class="q-page-title">Data Quality Insights</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="q-page-sub">Quality assessment — results saved in the project</div>',
                unsafe_allow_html=True)

    # ── Database selector ──
    st.markdown('<div class="sec-hd">⬢  Database</div>', unsafe_allow_html=True)
    db = st.selectbox("Database", options=dbs, key="q_db",
                      label_visibility="collapsed")
    if not db:
        st.markdown('<div class="info-strip">No database available.</div>',
                    unsafe_allow_html=True)
        return

    if not _labels(uri, user, pwd, db):
        st.warning(f"No node label found in **{db}**: the database is empty, or "
                   "its schema was read before the data were imported "
                   "(click ↻ Refresh schema).")

    # ── Two-column layout: config 40 %, results 60 % ──
    col_cfg, col_res = st.columns([40, 60], gap="medium")

    # ── LEFT: metric config & evaluate buttons ──
    with col_cfg:
        st.markdown('<div class="sec-hd">⬢  Dimensions &amp; Metrics</div>',
                    unsafe_allow_html=True)

        # Only the selected metric's form is built: each form queries the
        # schema, so building all 13 on every rerun made the page slow.
        dim_name = st.selectbox(
            "Dimension", list(DIMENSIONS.keys()), key="q_dim",
            format_func=lambda d: f"{DIMENSIONS[d]['icon']}  {d}",
        )
        dim_info = DIMENSIONS[dim_name]
        color    = dim_info["color"]
        metric_id = st.selectbox(
            "Metric", list(dim_info["metrics"].keys()), key=f"q_metric_{dim_name}",
            format_func=lambda m: dim_info["metrics"][m],
        )
        metric_label = dim_info["metrics"][metric_id]

        st.markdown(
            f'<div class="metric-hd" style="color:{color}">→ {metric_label}</div>',
            unsafe_allow_html=True,
        )
        params = _metric_params(
            metric_id, uri, user, pwd, db,
            prefix=f"q_{metric_id}",
        )

        # A running Graph ML experiment degrades its database: measuring it now
        # would save the quality of the degraded graph.
        busy = experiment_jobs.busy_db(uri, db) is not None
        if busy:
            st.warning(f"⏳ A Graph ML experiment is degrading {db}: choose another "
                       "database, or wait until the experiment finishes.")

        if st.button(f"▶  Evaluate", key=f"run_{metric_id}", width="stretch",
                     disabled=busy):
            # An empty menu (no label, no property...) gives None: nothing to compute.
            empty = [k for k in ("label", "prop", "rel", "src", "dst")
                     if k in params and not params[k]]
            if empty or ("props" in params and not params["props"]):
                st.error("Select a value in every menu before evaluating "
                         f"({', '.join(empty) or 'properties'} is empty).")
                st.stop()
            with st.spinner(f"Computing {metric_label}..."):
                value, display_type = _run_metric(
                    metric_id, params, uri, user, pwd, db
                )
            if display_type == "error":
                st.error(f"Error on {db}: {value}")
                st.stop()

            elem_type, label_disp = _label_display(metric_id, params)
            entry = {
                "metric_id":    metric_id,
                "metric_name":  metric_label,
                "dimension":    METRIC_TO_DIMENSION.get(metric_id, "Other"),
                "db":           db,
                "element_type": elem_type,
                "label_display": label_disp,
                "value":        value,
                "display_type": display_type,
                "params":       params,
            }
            proj.upsert_quality_result(entry)
            # Track last computed for highlighting
            st.session_state.q_last_key = entry["_key"]

            st.success(f"{metric_label} computed and saved.")
            st.rerun()

        st.markdown(
            "<hr style='border-color:#e1e4ea;margin:10px 0 4px'>",
            unsafe_allow_html=True,
        )
        if st.button("↻  Refresh schema", key="q_refresh_schema", width="stretch"):
            clear_schema_cache()
            st.rerun()

    # ── RIGHT: summary tables per dimension ──
    with col_res:
        st.markdown(f'<div class="sec-hd">⬢  Results by dimension — '
                    f'{html.escape(db)}</div>', unsafe_allow_html=True)

        # Only the results computed on the selected database.
        results_by_dim = proj.results_by_dimension(db=db)
        last_key = st.session_state.get("q_last_key")

        if not results_by_dim:
            st.markdown(f"""
            <div class="info-strip">
              No results for <strong>{html.escape(db)}</strong> yet.<br>
              Select a metric on the left and click
              <strong>▶ Evaluate</strong>.
            </div>""", unsafe_allow_html=True)
            return

        # One table per dimension, in order
        for dim_name, dim_info in DIMENSIONS.items():
            entries = results_by_dim.get(dim_name, [])
            _render_dimension_table(dim_name, dim_info, entries, last_key=last_key)

        # Clears the results of the selected database only
        if st.button(f"🗑  Clear results of {db}", key="clear_q"):
            proj.clear_quality_results(db=db)
            st.session_state.pop("q_last_key", None)
            st.rerun()
