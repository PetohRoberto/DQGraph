"""
app.py — Entry point: login → project selection/creation → tabs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from neo4j import GraphDatabase
from project_manager import Project

st.set_page_config(
    page_title="DQGraph",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── session state defaults ────────────────────────────────────────────────────
DEFAULTS = {
    "connected": False,
    "uri":       "bolt://localhost:7687",
    "neo_user":  "neo4j",
    "neo_pass":  "",
    "databases": [],
    "project":   None,   # Project instance
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --bg:        #080b12;
  --surface:   #0e1320;
  --border:    #1c2333;
  --border2:   #2a3450;
  --text:      #e8eaf0;
  --muted:     #5a6480;
  --accent:    #4f8cff;
  --accent2:   #00e5c0;
  --accent3:   #ff6b6b;
  --accent4:   #ffd166;
  --card:      #0c1018;
}

html, body, [class*="css"], .stApp {
  font-family: 'Syne', sans-serif;
  background: var(--bg) !important;
  color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

.main, .stApp > .main { width: 100% !important; padding: 0 !important; }
.main .block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"] > .main > .block-container {
  max-width: 100% !important;
  padding: 0.5rem 2rem !important;
  margin: 0 !important;
}

h1, h2, h3, h4 { font-family: 'Syne', sans-serif; }

[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-testid="stNumberInput"] input {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.82rem !important;
}
[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}

[data-testid="stButton"] > button {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  transition: all .2s !important;
}
[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent) 0%, #6fa8ff 100%) !important;
  border: none !important;
  color: #fff !important;
  font-size: 0.95rem !important;
  padding: 12px !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 8px;
  border-bottom: 1px solid var(--border);
  padding: 0 0.5rem;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  color: var(--muted) !important;
  padding: 12px 24px !important;
  background: transparent !important;
  border-radius: 8px 8px 0 0 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}

.login-logo {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 2.6rem;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
  margin-bottom: 4px;
}
.login-sub {
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  color: var(--muted);
  letter-spacing: 0.18em;
  text-transform: uppercase;
  text-align: center;
  margin-bottom: 28px;
}

.app-header {
  display: flex; align-items: center; gap: 16px;
  padding: 14px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
}
.app-logo {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 1.3rem;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.conn-badge {
  font-family: 'DM Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.1em;
  background: rgba(0,229,192,.1);
  color: var(--accent2);
  border: 1px solid rgba(0,229,192,.2);
  padding: 3px 10px;
  border-radius: 20px;
}
.proj-badge {
  font-family: 'DM Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.1em;
  background: rgba(79,140,255,.1);
  color: var(--accent);
  border: 1px solid rgba(79,140,255,.2);
  padding: 3px 10px;
  border-radius: 20px;
}

/* Project cards */
.proj-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 10px;
  cursor: pointer;
  transition: border-color .2s;
}
.proj-card:hover { border-color: var(--accent); }
.proj-card-name {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 1rem;
  color: var(--text);
}
.proj-card-meta {
  font-family: 'DM Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
  margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── PAGE 1: LOGIN ─────────────────────────────────────────────────────────────
if not st.session_state.connected:
    st.markdown("<div style='height: 6vh'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown('<div class="login-logo">⬡ DQGraph</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Graph Data Quality Platform</div>',
                    unsafe_allow_html=True)

        uri  = st.text_input("Neo4j URI",  value="bolt://localhost:7687",
                             key="uri_in",  placeholder="bolt://localhost:7687")
        user = st.text_input("Username",   value="neo4j", key="user_in")
        pwd  = st.text_input("Password",   value="",      key="pass_in", type="password")

        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        if st.button("Connect to Neo4j  →", width="stretch", type="primary"):
            try:
                d = GraphDatabase.driver(uri, auth=(user, pwd))
                d.verify_connectivity()
                with d.session(database="system") as s:
                    res = s.run("SHOW DATABASES YIELD name "
                                "WHERE name <> 'system' "
                                "RETURN name ORDER BY name")
                    dbs = [r["name"] for r in res]
                d.close()
                st.session_state.uri       = uri
                st.session_state.neo_user  = user
                st.session_state.neo_pass  = pwd
                st.session_state.databases = dbs
                st.session_state.connected = True
                st.rerun()
            except Exception as e:
                st.error(f"⚠ {e}")
    st.stop()


# ── PAGE 2: PROJECT SELECTION ─────────────────────────────────────────────────
if st.session_state.project is None:
    st.markdown("<div style='height: 4vh'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2.4, 1])
    with col:
        st.markdown('<div class="login-logo">⬡ DQGraph</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Select or create a project</div>',
                    unsafe_allow_html=True)

        existing = Project.list_all()

        tab_new, tab_open = st.tabs(["  New project  ", "  Open project  "])

        with tab_new:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            proj_name = st.text_input("Project name", placeholder="My thesis project",
                                      key="new_proj_name")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("Create project  →", width="stretch", type="primary",
                         key="btn_create"):
                if not proj_name.strip():
                    st.error("Please enter a project name.")
                else:
                    p = Project(proj_name.strip(),
                                st.session_state.uri,
                                st.session_state.neo_user)
                    p.save()
                    st.session_state.project = p
                    st.rerun()

        with tab_open:
            if not existing:
                st.markdown(
                    "<div style='padding:24px 0;font-family:DM Mono,monospace;"
                    "font-size:.78rem;color:#5a6480;text-align:center'>"
                    "No existing projects.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                for p_info in existing:
                    created = p_info["created_at"][:10] if p_info["created_at"] else "?"
                    n_q     = p_info.get("n_quality", 0)
                    col_txt, col_btn = st.columns([5, 1])
                    with col_txt:
                        st.markdown(
                            f'<div class="proj-card">'
                            f'<div class="proj-card-name">{p_info["name"]}</div>'
                            f'<div class="proj-card-meta">'
                            f'Created on {created} &nbsp;·&nbsp; '
                            f'{n_q} metric{"s" if n_q != 1 else ""} computed'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                    with col_btn:
                        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                        if st.button("Open", key=f"open_{p_info['path']}", width="stretch"):
                            p = Project.load(p_info["path"])
                            st.session_state.project = p
                            st.rerun()

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        from project_manager import PROJECTS_ROOT
        st.markdown(
            f'<div style="font-family:DM Mono,monospace;font-size:.65rem;color:#5a6480;'
            f'text-align:center;margin-top:8px">📁 Projects stored in: {PROJECTS_ROOT}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("← Disconnect", key="disc_proj"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    st.stop()


# ── HEADER BAR ────────────────────────────────────────────────────────────────
proj: Project = st.session_state.project

h1, h2, h3, _, h5, h6 = st.columns([3, 5, 4, 3, 2, 2])
with h1:
    st.markdown('<div class="app-logo">⬡ DQGraph</div>', unsafe_allow_html=True)
with h2:
    st.markdown(f'<div class="conn-badge">● {st.session_state.uri}</div>',
                unsafe_allow_html=True)
with h3:
    st.markdown(f'<div class="proj-badge">◈ {proj.name}</div>', unsafe_allow_html=True)
with h5:
    if st.button("Projects", key="back_proj", width="stretch"):
        st.session_state.project = None
        # Clear quality/ML session state
        for k in ["q_results", "ml_results"]:
            st.session_state.pop(k, None)
        st.rerun()
with h6:
    if st.button("Disconnect", key="disconnect", width="stretch"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["◈   Data Quality Insights", "◉   Graph ML Impact"])

with tab1:
    from pages.tab_quality import render_quality_tab
    render_quality_tab()

with tab2:
    from pages.tab_graphml import render_graphml_tab
    render_graphml_tab()
