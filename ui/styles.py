import streamlit as st


def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #ffffff;
    color: #191c26;
}
[data-testid="stSidebar"] {
    background: #f4f5f8;
    border-right: 1px solid #e1e4ea;
}
[data-testid="stSidebar"] * { color: #191c26 !important; }

.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.15rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3968d1 !important;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e1e4ea;
}
.main-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    background: linear-gradient(90deg, #3968d1 0%, #00a884 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.main-subtitle {
    color: #6b7085;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    margin-bottom: 2rem;
}
.score-card {
    background: linear-gradient(135deg, #ffffff 0%, #f4f5f8 100%);
    border: 1px solid #cdd2de;
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
    background: radial-gradient(circle, rgba(57,104,209,0.15), transparent 70%);
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
    color: #6b7085;
    margin-top: 6px;
}
.info-box {
    background: #f4f5f8;
    border-left: 3px solid #3968d1;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #6b7085;
    margin-bottom: 1.5rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #3968d1;
    margin: 1.5rem 0 0.75rem 0;
}
.neo-divider { border: none; border-top: 1px solid #e1e4ea; margin: 1.5rem 0; }
.conn-panel {
    background: #f4f5f8;
    border: 1px solid #e1e4ea;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 1.5rem;
}
[data-testid="stButton"] > button {
    background: transparent;
    border: 1px solid #cdd2de;
    color: #191c26;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    border-radius: 6px;
    padding: 6px 14px;
    transition: all 0.2s;
    width: 100%;
    text-align: left;
}
[data-testid="stButton"] > button:hover {
    border-color: #3968d1;
    color: #3968d1;
    background: #f4f5f8;
}
[data-testid="stExpander"] {
    background: #f4f5f8;
    border: 1px solid #e1e4ea;
    border-radius: 8px;
}
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select {
    background: #f4f5f8 !important;
    border: 1px solid #cdd2de !important;
    color: #191c26 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)
