"""
CFR STIR Dashboard — Dash web app.
Two-tab layout: PRODUCTS (SOFR / FF strip toggle) and MEETINGS (path, spreads, CB LVL).
Fetches live EFFR/SOFR from the NY Fed on each page load.
"""
from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta

import dash
from dash import dcc, html, dash_table, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

# ── PALETTE ──────────────────────────────────────────────────────────────────

CFR = {
    "bg":     "#000000", "panel":     "#080808", "rule":      "#3D2510",
    "orange": "#FE7C04", "orangeHot": "#FF9533", "orangeDim": "#5A2C00",
    "text":   "#D0D0D0", "green":     "#00E676", "red":       "#FF1744",
}

# ── SCHEMAS ──────────────────────────────────────────────────────────────────

_CME_MONTH_CODES = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",
                    7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}

def _cme_symbol(root: str, expiry: date) -> str:
    return f"{root}{_CME_MONTH_CODES[expiry.month]}{expiry.year % 10}"

@dataclass
class Contract:
    symbol: str; root: str; expiry: date; settle: float

def to_strip(contracts: list[Contract]) -> pd.DataFrame:
    return pd.DataFrame([c.__dict__ for c in contracts])

# ── DATA LOADERS ─────────────────────────────────────────────────────────────

def load_ref_rates_nyfed(days: int = 90) -> pd.DataFrame:
    out = {}
    endpoints = {
        "effr": "https://markets.newyorkfed.org/api/rates/unsecured/effr/last/{n}.json",
        "sofr": "https://markets.newyorkfed.org/api/rates/secured/sofr/last/{n}.json",
    }
    for name, url_tpl in endpoints.items():
        r = requests.get(url_tpl.format(n=days), timeout=15)
        r.raise_for_status()
        series = {
            pd.Timestamp(rec["effectiveDate"]): rec["percentRate"]
            for rec in r.json()["refRates"]
            if rec.get("percentRate") is not None
        }
        out[name] = pd.Series(series, dtype=float).sort_index()
    return pd.DataFrame(out).sort_index().ffill()

def load_fomc_dates() -> list[date]:
    return [
        date(2025,1,29), date(2025,3,19), date(2025,5,7),
        date(2025,6,18), date(2025,7,30), date(2025,9,17),
        date(2025,10,29), date(2025,12,10),
        date(2026,1,28), date(2026,3,18), date(2026,4,29),
        date(2026,6,17), date(2026,7,29), date(2026,9,16),
        date(2026,10,28), date(2026,12,9),
    ]

def make_synthetic_strip(today: date) -> pd.DataFrame:
    np.random.seed(42)
    contracts: list[Contract] = []
    sr3_months = [(today.year + (today.month + i*3 - 1) // 12,
                   ((today.month - 1 + i*3) % 12) + 1) for i in range(8)]
    sr3_rates = [4.20, 4.05, 3.92, 3.75, 3.55, 3.42, 3.50, 3.62]
    for (y, m), r in zip(sr3_months, sr3_rates):
        qm = next(q for q in (3,6,9,12) if q >= m) if m not in (3,6,9,12) else m
        exp = date(y, qm, monthrange(y, qm)[1])
        contracts.append(Contract(_cme_symbol("SR3", exp), "SR3", exp, 100.0 - r))
    for i, r in enumerate(
        np.linspace(4.30, 3.30, 18) + np.random.normal(0, 0.015, 18)
    ):
        m = ((today.month - 1 + i) % 12) + 1
        y = today.year + (today.month + i - 1) // 12
        exp = date(y, m, monthrange(y, m)[1])
        contracts.append(Contract(_cme_symbol("ZQ", exp), "ZQ", exp, 100.0 - r))
    return to_strip(contracts).sort_values(["root", "expiry"]).reset_index(drop=True)

# ── COMPUTE ──────────────────────────────────────────────────────────────────

def add_implied(strip: pd.DataFrame, ocr: float) -> pd.DataFrame:
    out = strip.copy()
    out["implied_rate"] = 100.0 - out["settle"]
    out["vs_ocr_bp"]    = (out["implied_rate"] - ocr) * 100.0
    return out

def find_terminal(strip_view: pd.DataFrame, ocr: float) -> pd.Series:
    active = strip_view[strip_view["settle"] > 0].reset_index(drop=True)
    if active.empty:
        return strip_view.iloc[0]
    front  = active.iloc[0]
    hiking = front["implied_rate"] >= ocr
    best   = front
    for _, row in active.iloc[1:].iterrows():
        if hiking and row["implied_rate"] >= best["implied_rate"]:
            best = row
        elif not hiking and row["implied_rate"] <= best["implied_rate"]:
            best = row
        else:
            break
    return best

def post_meeting_rate(contract_rate, prev_rate, meeting_day, days_in_month):
    days_after = days_in_month - meeting_day + 1
    if days_after <= 0:
        return contract_rate
    return (contract_rate * days_in_month - (meeting_day - 1) * prev_rate) / days_after

def build_meeting_path(zq_strip, effr_today, fomc_dates):
    zq_by_month = {(r["expiry"].year, r["expiry"].month): r["implied_rate"]
                   for _, r in zq_strip.iterrows()}
    fomc_keys = {(d.year, d.month) for d in fomc_dates}
    prev = effr_today
    rows = []
    for d in fomc_dates:
        rate = zq_by_month.get((d.year, d.month))
        if rate is None:
            continue
        N = monthrange(d.year, d.month)[1]
        ny, nm = (d.year + (d.month == 12), d.month % 12 + 1)
        next_rate = zq_by_month.get((ny, nm))
        next_has_meeting = (ny, nm) in fomc_keys
        if next_rate is not None and not next_has_meeting:
            post = next_rate
        else:
            post = post_meeting_rate(rate, prev, d.day, N)
        rows.append({"meeting": d, "post_rate": post,
                     "cum_cuts": round((effr_today - post) / 0.25, 2)})
        prev = post
    return pd.DataFrame(rows)

def meeting_probs(post_rate, effr):
    raw = (effr - post_rate) / 0.25
    lower = int(np.floor(raw))
    frac  = raw - lower
    mass = {lower: 1 - frac}
    if frac > 0.001:
        mass[lower + 1] = frac
    return {"hold": 100*mass.get(0,0), "cut25": 100*mass.get(1,0),
            "cut50": 100*mass.get(2,0), "cut75": 100*mass.get(3,0),
            "hike25": 100*mass.get(-1,0)}

def spread_matrix(strip_view, ocr, horizons_m=(3, 6, 9, 12)):
    if strip_view.empty:
        return pd.DataFrame()
    rows = []
    for _, row in strip_view.iterrows():
        row_mo = row["expiry"].month + 12 * row["expiry"].year
        spreads = {}
        for h in horizons_m:
            target = row_mo + h
            forward = strip_view[strip_view["expiry"].apply(
                lambda d: d.month + 12 * d.year >= target)]
            spreads[f"+{h}M"] = (round((forward.iloc[0]["implied_rate"]
                                        - row["implied_rate"]) * 100)
                                  if not forward.empty else float("nan"))
        rows.append({"contract": row["symbol"], **spreads})
    return pd.DataFrame(rows).set_index("contract")

def cb_levels(effr, band_bp=100, step_bp=25):
    settle = round(effr / 0.25) * 0.25
    n = band_bp // step_bp
    return [settle + (i - n) * (step_bp / 100.0) for i in range(2 * n + 1)]

# ── PLOTLY FIGURES ───────────────────────────────────────────────────────────

def _base_layout():
    return dict(
        template="plotly_dark", paper_bgcolor=CFR["bg"], plot_bgcolor="#050505",
        font=dict(family="Segoe UI, system-ui, sans-serif", color=CFR["text"]),
        margin=dict(l=60, r=40, t=60, b=40), height=460,
    )

def fig_strip(strip_view, ocr, title):
    term = find_terminal(strip_view, ocr)
    colors = [CFR["orangeHot"] if s == term["symbol"] else CFR["orangeDim"]
              for s in strip_view["symbol"]]
    fig = go.Figure(go.Bar(
        x=strip_view["symbol"], y=strip_view["implied_rate"],
        marker_color=colors, marker_line_color="#9A4A02",
        hovertemplate="%{x}<br>%{y:.3f}%<extra></extra>",
    ))
    fig.add_hline(y=ocr, line_dash="dash", line_color=CFR["orange"],
                  annotation_text="EFFECTIVE FFR", annotation_position="right",
                  annotation_font=dict(color=CFR["orange"]))
    fig.update_layout(**_base_layout(),
                      title=dict(text=title, font=dict(color=CFR["orange"], size=18)),
                      yaxis_title="Implied rate (%)")
    return fig

def fig_meeting_path(path, effr, title="MEETINGS · IMPLIED POST-MEETING PATH"):
    fig = go.Figure(go.Scatter(
        x=path["meeting"], y=path["post_rate"], mode="lines+markers",
        line=dict(color=CFR["orangeHot"], width=2.4, shape="hv"),
        marker=dict(color=CFR["bg"],
                    line=dict(color=CFR["orangeHot"], width=1.5), size=8),
        hovertemplate="%{x|%b %Y}<br>%{y:.3f}%<extra></extra>",
    ))
    fig.add_hline(y=effr, line_dash="dash", line_color=CFR["orange"],
                  annotation_text="EFFECTIVE FFR", annotation_position="right",
                  annotation_font=dict(color=CFR["orange"]))
    fig.update_layout(**_base_layout(),
                      title=dict(text=title, font=dict(color=CFR["orange"], size=18)),
                      yaxis_title="Implied rate (%)")
    return fig

def fig_cb_lvl(path, effr):
    fig = fig_meeting_path(path, effr, title="MEETINGS · CB LVL")
    settle = round(effr / 0.25) * 0.25
    for lv in cb_levels(effr, band_bp=150):
        is_settle = abs(lv - settle) < 0.01
        fig.add_hline(y=lv,
                      line_color=CFR["orange"] if is_settle else CFR["orangeDim"],
                      line_dash="solid" if is_settle else "dot",
                      line_width=1.4 if is_settle else 0.6)
    return fig

# ── LOAD DATA ONCE AT STARTUP ───────────────────────────────────────────────

TODAY = date.today()
ref_rates  = load_ref_rates_nyfed(days=90)
fomc_dates = [d for d in load_fomc_dates() if d >= TODAY]
strip_raw  = make_synthetic_strip(TODAY)

OCR  = float(ref_rates["effr"].iloc[-1])
SOFR = float(ref_rates["sofr"].iloc[-1])

strip      = add_implied(strip_raw, OCR)
sofr_strip = strip[strip["root"] == "SR3"].reset_index(drop=True)
ff_strip   = strip[strip["root"] == "ZQ"].reset_index(drop=True)
path       = build_meeting_path(ff_strip, OCR, fomc_dates)
sm         = spread_matrix(ff_strip, OCR)

probs_df = pd.DataFrame(
    [meeting_probs(r, OCR) for r in path["post_rate"]],
    index=[d.isoformat() for d in path["meeting"]],
).round(1) if not path.empty else pd.DataFrame()

# ── DASH APP ─────────────────────────────────────────────────────────────────

app = dash.Dash(__name__)
server = app.server  # for gunicorn

HEADER_STYLE = {
    "color": CFR["orange"], "fontFamily": "Segoe UI, system-ui, sans-serif",
    "fontWeight": 300, "letterSpacing": "0.15em", "fontSize": "0.75rem",
    "textTransform": "uppercase", "marginBottom": "4px",
}
CARD = {
    "backgroundColor": CFR["panel"], "borderRadius": "6px",
    "padding": "16px", "marginBottom": "12px",
}
STAT_VAL = {"color": CFR["orange"], "fontSize": "1.6rem", "fontWeight": 600,
            "fontFamily": "IBM Plex Mono, monospace"}
STAT_LBL = {"color": CFR["text"], "fontSize": "0.7rem", "textTransform": "uppercase",
            "letterSpacing": "0.1em", "opacity": 0.7}

app.layout = html.Div(style={"backgroundColor": CFR["bg"], "minHeight": "100vh",
                              "padding": "20px 28px", "fontFamily": "Segoe UI, system-ui, sans-serif"}, children=[

    # ── Banner
    html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "16px",
                     "marginBottom": "20px"}, children=[
        html.H1("US STIR DASHBOARD", style={
            "color": CFR["orange"], "margin": 0, "fontSize": "1.4rem",
            "letterSpacing": "0.2em", "fontWeight": 300}),
        html.Span(f"EFFR {OCR:.2f}%  ·  SOFR {SOFR:.2f}%  ·  basis {(SOFR-OCR)*100:+.1f} bp",
                  style={"color": CFR["text"], "fontSize": "0.8rem", "opacity": 0.6}),
        html.Span(f"as of {TODAY.isoformat()}",
                  style={"color": CFR["text"], "fontSize": "0.7rem", "opacity": 0.4,
                         "marginLeft": "auto"}),
    ]),

    # ── Tabs
    dcc.Tabs(id="main-tabs", value="products", style={"marginBottom": "16px"},
             colors={"border": CFR["rule"], "primary": CFR["orange"],
                     "background": CFR["bg"]}, children=[

        # ──────────────── PRODUCTS TAB ────────────────
        dcc.Tab(label="PRODUCTS", value="products",
                style={"color": CFR["text"], "backgroundColor": CFR["bg"],
                       "border": f"1px solid {CFR['rule']}", "padding": "8px 20px"},
                selected_style={"color": CFR["orange"], "backgroundColor": CFR["panel"],
                                "borderBottom": f"2px solid {CFR['orange']}",
                                "padding": "8px 20px"},
                children=[
            html.Div(style={"display": "flex", "gap": "10px", "margin": "12px 0"}, children=[
                html.Button("SOFR (SR3)", id="btn-sofr", n_clicks=1,
                            style={"backgroundColor": CFR["orange"], "color": "#000",
                                   "border": "none", "padding": "6px 18px",
                                   "borderRadius": "4px", "cursor": "pointer",
                                   "fontWeight": 600, "fontSize": "0.8rem"}),
                html.Button("FED FUNDS (ZQ)", id="btn-ff", n_clicks=0,
                            style={"backgroundColor": CFR["orangeDim"], "color": CFR["text"],
                                   "border": "none", "padding": "6px 18px",
                                   "borderRadius": "4px", "cursor": "pointer",
                                   "fontWeight": 600, "fontSize": "0.8rem"}),
            ]),
            dcc.Graph(id="strip-chart", config={"displayModeBar": False}),
        ]),

        # ──────────────── MEETINGS TAB ────────────────
        dcc.Tab(label="MEETINGS", value="meetings",
                style={"color": CFR["text"], "backgroundColor": CFR["bg"],
                       "border": f"1px solid {CFR['rule']}", "padding": "8px 20px"},
                selected_style={"color": CFR["orange"], "backgroundColor": CFR["panel"],
                                "borderBottom": f"2px solid {CFR['orange']}",
                                "padding": "8px 20px"},
                children=[
            html.Div(style={"display": "flex", "gap": "10px", "margin": "12px 0"}, children=[
                html.Button("STRIP", id="btn-strip", n_clicks=1,
                            style={"backgroundColor": CFR["orange"], "color": "#000",
                                   "border": "none", "padding": "6px 18px",
                                   "borderRadius": "4px", "cursor": "pointer",
                                   "fontWeight": 600, "fontSize": "0.8rem"}),
                html.Button("SPREADS", id="btn-spreads", n_clicks=0,
                            style={"backgroundColor": CFR["orangeDim"], "color": CFR["text"],
                                   "border": "none", "padding": "6px 18px",
                                   "borderRadius": "4px", "cursor": "pointer",
                                   "fontWeight": 600, "fontSize": "0.8rem"}),
                html.Button("CB LVL", id="btn-cblvl", n_clicks=0,
                            style={"backgroundColor": CFR["orangeDim"], "color": CFR["text"],
                                   "border": "none", "padding": "6px 18px",
                                   "borderRadius": "4px", "cursor": "pointer",
                                   "fontWeight": 600, "fontSize": "0.8rem"}),
            ]),
            html.Div(id="meetings-content"),
        ]),
    ]),
])

# ── CALLBACKS ────────────────────────────────────────────────────────────────

@app.callback(
    Output("strip-chart", "figure"),
    Output("btn-sofr", "style"),
    Output("btn-ff", "style"),
    Input("btn-sofr", "n_clicks"),
    Input("btn-ff", "n_clicks"),
)
def update_strip(n_sofr, n_ff):
    active = {"backgroundColor": CFR["orange"], "color": "#000",
              "border": "none", "padding": "6px 18px", "borderRadius": "4px",
              "cursor": "pointer", "fontWeight": 600, "fontSize": "0.8rem"}
    inactive = {"backgroundColor": CFR["orangeDim"], "color": CFR["text"],
                "border": "none", "padding": "6px 18px", "borderRadius": "4px",
                "cursor": "pointer", "fontWeight": 600, "fontSize": "0.8rem"}
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "btn-ff.n_clicks":
        return fig_strip(ff_strip, OCR, "PRODUCTS · FED FUNDS (ZQ) STRIP"), inactive, active
    return fig_strip(sofr_strip, OCR, "PRODUCTS · SOFR (SR3) STRIP"), active, inactive

@app.callback(
    Output("meetings-content", "children"),
    Output("btn-strip", "style"),
    Output("btn-spreads", "style"),
    Output("btn-cblvl", "style"),
    Input("btn-strip", "n_clicks"),
    Input("btn-spreads", "n_clicks"),
    Input("btn-cblvl", "n_clicks"),
)
def update_meetings(n_strip, n_spreads, n_cblvl):
    active = {"backgroundColor": CFR["orange"], "color": "#000",
              "border": "none", "padding": "6px 18px", "borderRadius": "4px",
              "cursor": "pointer", "fontWeight": 600, "fontSize": "0.8rem"}
    inactive = {"backgroundColor": CFR["orangeDim"], "color": CFR["text"],
                "border": "none", "padding": "6px 18px", "borderRadius": "4px",
                "cursor": "pointer", "fontWeight": 600, "fontSize": "0.8rem"}

    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "btn-cblvl" in triggered:
        return (dcc.Graph(figure=fig_cb_lvl(path, OCR), config={"displayModeBar": False}),
                inactive, inactive, active)

    if "btn-spreads" in triggered:
        sm_reset = sm.reset_index()
        tbl = dash_table.DataTable(
            data=sm_reset.to_dict("records"),
            columns=[{"name": c, "id": c} for c in sm_reset.columns],
            style_header={"backgroundColor": CFR["panel"], "color": CFR["orange"],
                          "fontWeight": 600, "border": f"1px solid {CFR['rule']}",
                          "fontFamily": "IBM Plex Mono, monospace", "fontSize": "0.8rem"},
            style_cell={"backgroundColor": CFR["bg"], "color": CFR["text"],
                        "border": f"1px solid {CFR['rule']}", "textAlign": "center",
                        "fontFamily": "IBM Plex Mono, monospace", "fontSize": "0.8rem",
                        "padding": "6px 10px"},
            style_data_conditional=[
                {"if": {"filter_query": f"{{{col}}} < 0", "column_id": col},
                 "color": CFR["green"]}
                for col in ["+3M", "+6M", "+9M", "+12M"]
            ] + [
                {"if": {"filter_query": f"{{{col}}} > 0", "column_id": col},
                 "color": CFR["red"]}
                for col in ["+3M", "+6M", "+9M", "+12M"]
            ],
        )
        return tbl, inactive, active, inactive

    # Default: strip (meeting path + probability table)
    prob_reset = probs_df.reset_index().rename(columns={"index": "meeting"})
    prob_tbl = dash_table.DataTable(
        data=prob_reset.to_dict("records"),
        columns=[{"name": c, "id": c} for c in prob_reset.columns],
        style_header={"backgroundColor": CFR["panel"], "color": CFR["orange"],
                      "fontWeight": 600, "border": f"1px solid {CFR['rule']}",
                      "fontFamily": "IBM Plex Mono, monospace", "fontSize": "0.8rem"},
        style_cell={"backgroundColor": CFR["bg"], "color": CFR["text"],
                    "border": f"1px solid {CFR['rule']}", "textAlign": "center",
                    "fontFamily": "IBM Plex Mono, monospace", "fontSize": "0.8rem",
                    "padding": "6px 10px"},
    )
    return (html.Div([
        dcc.Graph(figure=fig_meeting_path(path, OCR), config={"displayModeBar": False}),
        html.H3("PROBABILITY TABLE", style={**HEADER_STYLE, "marginTop": "16px"}),
        prob_tbl,
    ]), active, inactive, inactive)

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
