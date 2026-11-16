import plotly.graph_objects as go


def score_color(score: float) -> str:
    if score >= 0.8: return "#00a884"
    if score >= 0.5: return "#d99a2b"
    return "#e15a5a"


def make_gauge(value: float, title: str, suffix: str = "%", max_val: float = 100):
    color = score_color(value if suffix == "%" else value / max(max_val, 1))
    display_val = value * 100 if suffix == "%" else value

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_val,
        number={"suffix": suffix, "font": {"family": "Syne", "size": 48, "color": color}},
        title={"text": title, "font": {"family": "Syne", "size": 14, "color": "#6b7085"}},
        gauge={
            "axis": {"range": [0, 100 if suffix == "%" else max_val],
                     "tickcolor": "#cdd2de", "tickfont": {"color": "#6b7085", "size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#f4f5f8",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],  "color": "rgba(225,90,90,0.08)"},
                {"range": [50, 80], "color": "rgba(217,154,43,0.08)"},
                {"range": [80, 100],"color": "rgba(0,168,132,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 2},
                "thickness": 0.75,
                "value": display_val,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Syne"},
    )
    return fig


def make_bar_chart(data: dict, title: str):
    props  = list(data.keys())
    vals   = [v * 100 for v in data.values()]
    colors = [score_color(v / 100) for v in vals]

    fig = go.Figure(go.Bar(
        x=props, y=vals,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont={"family": "Space Mono", "size": 11, "color": "#191c26"},
    ))
    fig.update_layout(
        title={"text": title, "font": {"family": "Syne", "size": 14, "color": "#6b7085"}},
        height=320,
        margin=dict(l=10, r=10, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"tickfont": {"family": "Space Mono", "size": 10, "color": "#6b7085"},
               "gridcolor": "#e1e4ea", "zeroline": False},
        yaxis={"range": [0, 110], "tickfont": {"size": 10, "color": "#6b7085"},
               "gridcolor": "#e1e4ea", "zeroline": False},
        showlegend=False,
    )
    return fig
