# dashboard/app.py
import os
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import yfinance as yf

# ====== Settings ======
API_BASE = os.environ.get("FORECAST_API_BASE", "http://127.0.0.1:8000")
APP_PORT = int(os.environ.get("PORT", 8050))
FAANG = [
    {"label": "Apple (AAPL)", "value": "AAPL"},
    {"label": "Amazon (AMZN)", "value": "AMZN"},
    {"label": "Meta (META)", "value": "META"},
    {"label": "Netflix (NFLX)", "value": "NFLX"},
    {"label": "Alphabet (GOOGL)", "value": "GOOGL"},
]

def fetch_prices(ticker: str, days: int = 365) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=days * 2)
    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
    df = df.rename(columns=str.title).reset_index()[["Date", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def price_and_sentiment_fig(df: pd.DataFrame, ticker: str, predicted_next_return: float | None):
    # Sentiment proxy = normalized daily return (for display only)
    df = df.copy()
    df["ret"] = df["Close"].pct_change()
    df["sent"] = (df["ret"] - df["ret"].rolling(30).mean()) / (df["ret"].rolling(30).std() + 1e-9)
    df["sent"] = df["sent"].clip(-3, 3)

    fig = go.Figure()
    # Price line
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"], mode="lines",
        name="Close", line=dict(width=2)
    ))
    # Sentiment bars (secondary style)
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["sent"], name="Sentiment (proxy)",
        opacity=0.35, yaxis="y2"
    ))

    # Optional next-day marker (draw as last point/tag)
    if predicted_next_return is not None and len(df) > 0:
        last_close = df["Close"].iloc[-1]
        forecast_close = last_close * (1 + predicted_next_return)
        fig.add_trace(go.Scatter(
            x=[df["Date"].iloc[-1] + pd.Timedelta(days=1)],
            y=[forecast_close],
            mode="markers+text",
            name="Forecast",
            marker=dict(size=10),
            text=[f"+{predicted_next_return*100:.2f}%"],
            textposition="top center"
        ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        yaxis2=dict(overlaying="y", side="right", title="Sentiment (z)", showgrid=False),
        legend=dict(orientation="h", y=1.1, x=0),
        title=f"{ticker} - Price History & Forecast",
    )
    return fig

app = Dash(__name__, title="AI Stock Sentiment Dashboard", suppress_callback_exceptions=True)
server = app.server  # so you can deploy Dash on a WSGI/ASGI host if needed

# ====== Layout (dark, blue/green accents) ======
app.layout = html.Div(
    style={"backgroundColor": "#0f1220", "minHeight": "100vh", "color": "#e8f0fe", "padding": "24px"},
    children=[
        html.Div(
            style={"textAlign": "center", "marginBottom": "16px"},
            children=[
                html.H1("📊 AI Stock Sentiment & Forecast Dashboard",
                        style={"margin": 0, "fontWeight": 700}),
            ],
        ),

        # Controls row
        html.Div(
            style={"display": "flex", "gap": "12px", "justifyContent": "center", "alignItems": "center",
                   "flexWrap": "wrap", "marginBottom": "12px"},
            children=[
                html.Div([
                    html.Label("Select Stock:", style={"fontWeight": 600}),
                    dcc.Dropdown(
                        id="ticker-dd",
                        options=FAANG,
                        value="AAPL",
                        clearable=False,
                        style={"width": "320px", "color": "#111"}
                    )
                ]),

                html.Button("🔮 Forecast", id="predict-btn",
                            style={"background": "#3b82f6", "border": "none", "color": "white",
                                   "padding": "10px 16px", "borderRadius": "8px", "cursor": "pointer",
                                   "fontWeight": 600})
            ]
        ),

        # Status row
        html.Div(
            style={"display": "flex", "gap": "18px", "justifyContent": "center",
                   "alignItems": "center", "flexWrap": "wrap", "marginBottom": "16px"},
            children=[
                html.Div(id="api-status", children="🔄 Checking API...", style={"color": "#22c55e"}),
                html.Div(id="forecast-status", style={"color": "#a3e635"}),
                html.Div(id="predicted-text", style={"fontWeight": 700}),
            ],
        ),

        # Graph
        dcc.Graph(id="price-graph", figure=go.Figure(),
                  style={"height": "68vh", "backgroundColor": "#0b0e19", "borderRadius": "12px"}),

        # Stores & interval
        dcc.Store(id="pred-store"),
        dcc.Interval(id="ping-api", interval=4_000, n_intervals=0, disabled=False),
    ]
)

# ====== Callbacks ======
@app.callback(
    Output("api-status", "children"),
    Output("ping-api", "disabled"),
    Input("ping-api", "n_intervals")
)
def ping_api(_):
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2.5)
        if r.ok:
            return "🟢 API Connected", True
        return "🔴 API Unreachable", False
    except Exception:
        return "🔴 API Unreachable", False

@app.callback(
    Output("pred-store", "data"),
    Output("forecast-status", "children"),
    Input("predict-btn", "n_clicks"),
    State("ticker-dd", "value"),
    prevent_initial_call=True
)
def get_prediction(n_clicks, ticker):
    if not ticker:
        return no_update, "⚠️ Select a ticker"
    try:
        r = requests.get(f"{API_BASE}/predict/{ticker}", timeout=10)
        if r.ok:
            data = r.json()
            return data, "✅ Forecast updated successfully!"
        return no_update, "❌ Forecast failed."
    except Exception:
        return no_update, "❌ Forecast failed."

@app.callback(
    Output("predicted-text", "children"),
    Output("price-graph", "figure"),
    Input("pred-store", "data"),
    Input("ticker-dd", "value"),
)
def update_graph(pred_data, ticker):
    # fetch prices for graph on any change
    try:
        df = fetch_prices(ticker)
    except Exception:
        return "⚠️ Could not load prices.", go.Figure()

    predicted_ret = None
    if isinstance(pred_data, dict) and pred_data.get("ticker", "").upper() == ticker:
        predicted_ret = float(pred_data.get("predicted_return", 0.0))
        conf = float(pred_data.get("confidence", 0.0))
        text = f"Predicted next return: {predicted_ret:+.4f}  •  Confidence: {conf:.0%}"
    else:
        text = "Predicted next return: —  •  Confidence: —"

    fig = price_and_sentiment_fig(df, ticker, predicted_ret)
    return text, fig

if __name__ == "__main__":
    # Run the Dash app
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
