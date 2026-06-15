import streamlit as st


def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d12;
    color: #e2e2f0;
}
[data-testid="stSidebar"] {
    background: #12121a;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #e2e2f0 !important; }

.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.15rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c6af7 !important;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e2e;
}
.main-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    background: linear-gradient(90deg, #a89cf7 0%, #5ce8e0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.main-subtitle {
    color: #555570;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    margin-bottom: 2rem;
}
.score-card {
    background: linear-gradient(135deg, #1a1a28 0%, #161620 100%);
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 28px 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.score-card::before {
    content: '';
    position: absolute;
    top: -40px; left: -40px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,106,247,0.15), transparent 70%);
}
.score-value {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.5rem;
    line-height: 1;
}
.score-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #555570;
    margin-top: 6px;
}
.info-box {
    background: #12121a;
    border-left: 3px solid #7c6af7;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #8888aa;
    margin-bottom: 1.5rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7c6af7;
    margin: 1.5rem 0 0.75rem 0;
}
.neo-divider { border: none; border-top: 1px solid #1e1e2e; margin: 1.5rem 0; }
.conn-panel {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 1.5rem;
}
[data-testid="stButton"] > button {
    background: transparent;
    border: 1px solid #2a2a3e;
    color: #c0c0d8;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    border-radius: 6px;
    padding: 6px 14px;
    transition: all 0.2s;
    width: 100%;
    text-align: left;
}
[data-testid="stButton"] > button:hover {
    border-color: #7c6af7;
    color: #a89cf7;
    background: #1a1a28;
}
[data-testid="stExpander"] {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
}
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select {
    background: #1a1a28 !important;
    border: 1px solid #2a2a3e !important;
    color: #e2e2f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)
