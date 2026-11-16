"""
ui/schema_cache.py — Cached schema lookups shared by every page.

The schema rarely changes during a session, so results are kept 10 minutes;
the "Refresh schema" button calls clear_schema_cache() to force a reload.
"""
import streamlit as st

from ui.schema import SchemaExplorer
from value_types import NUMERIC_TYPES, CATEGORICAL_TYPES

SCHEMA_TTL = 600

# A selectbox with no option returns None (e.g. a database without any node
# label): the lookups below then return nothing instead of sending
# "MATCH (n:None)" to Neo4j.


def _explorer(uri, user, pwd, db) -> SchemaExplorer:
    return SchemaExplorer(uri, user, pwd, db)


@st.cache_data(ttl=SCHEMA_TTL, show_spinner=False)
def labels(uri, user, pwd, db):
    return _explorer(uri, user, pwd, db).get_node_labels()

@st.cache_data(ttl=SCHEMA_TTL, show_spinner=False)
def rels(uri, user, pwd, db):
    return _explorer(uri, user, pwd, db).get_relationship_types()

@st.cache_data(ttl=SCHEMA_TTL, show_spinner=False)
def node_props(uri, user, pwd, db, lbl):
    if not lbl:
        return []
    return _explorer(uri, user, pwd, db).get_node_properties(lbl)

@st.cache_data(ttl=SCHEMA_TTL, show_spinner=False)
def edge_props(uri, user, pwd, db, rel):
    if not rel:
        return []
    return _explorer(uri, user, pwd, db).get_edge_properties(rel)

@st.cache_data(ttl=SCHEMA_TTL, show_spinner=False)
def rel_labels(uri, user, pwd, db, rel):
    if not rel:
        return [], []
    return _explorer(uri, user, pwd, db).get_node_labels_for_rel(rel)

@st.cache_data(ttl=SCHEMA_TTL, show_spinner=False)
def rels_for_src(uri, user, pwd, db, src):
    if not src:
        return []
    return _explorer(uri, user, pwd, db).get_rel_types_for_source(src)

@st.cache_data(ttl=SCHEMA_TTL, show_spinner=False)
def dst_for(uri, user, pwd, db, src, rel):
    if not src or not rel:
        return []
    return _explorer(uri, user, pwd, db).get_dst_labels_for(src, rel)

@st.cache_data(ttl=SCHEMA_TTL, show_spinner=False)
def prop_types(uri, user, pwd, db, lbl):
    if not lbl:
        return {}
    return _explorer(uri, user, pwd, db).get_property_types(lbl)


@st.cache_data(ttl=SCHEMA_TTL, show_spinner=False)
def edge_prop_types(uri, user, pwd, db, rel):
    if not rel:
        return {}
    return _explorer(uri, user, pwd, db).get_edge_property_types(rel)


def accuracy_props(uri, user, pwd, db, et, lbl):
    """Properties PropertyAccuracy can be computed on: numbers, numbers stored
    as strings included; dates, text and booleans excluded (as the accuracy
    degrader does)."""
    types = (prop_types if et == "node" else edge_prop_types)(uri, user, pwd, db, lbl)
    return sorted(p for p, t in types.items() if t in ("numeric", "numeric_discrete"))


def numeric_props(uri, user, pwd, db, lbl):
    types = prop_types(uri, user, pwd, db, lbl)
    return sorted(p for p, t in types.items() if t in NUMERIC_TYPES)


def categorical_props(uri, user, pwd, db, lbl):
    types = prop_types(uri, user, pwd, db, lbl)
    return sorted(p for p, t in types.items() if t in CATEGORICAL_TYPES)


def clear_schema_cache():
    for fn in (labels, rels, node_props, edge_props, rel_labels,
               rels_for_src, dst_for, prop_types, edge_prop_types):
        fn.clear()
