import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf
import datetime
import requests
import sys, os

# Ensure src is importable if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Initialize app ---
app = dash.Dash(__name__, title="AI Stock Sentiment Dashboard")
server = app.server  # Needed if you later deploy to Render or Heroku

# --- Layout ---
app.layout = html.Div(
    style={"fontFamily": "Segoe UI", "backgroundColor": "#f8fafc", "padding": "30px"},
    children=[
        html.H1("📊 AI Stock Sentiment & Forecast Dashboard", style={"textAlign": "center"}),

        html.Div(
            [
                html.Label("Select Stock:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="ticker-dropdown",
                    options=[
                        {"label": "Apple (AAPL)", "value": "AAPL"},
                        {"label": "Amazon (AMZN)", "value": "AMZN"},
                        {"label": "Meta (META)", "value": "META"},
                        {"label": "Netflix (NFLX)", "value": "NFLX"},
                        {"label": "Google (GOOGL)", "value": "GOOGL"},
                    ],
                    value="AAPL",
                    style={"width": "50%"},
                ),
                html.Button(
                    "🔮 Forecast",
                    id="predict-btn",
                    n_clicks=0,
                    style={
                        "marginLeft": "15px",
                        "backgroundColor": "#2563eb",
                        "color": "white",
                        "border": "none",
                        "padding": "10px 20px",
                        "cursor": "pointer",
                        "borderRadius": "5px",
                    },
                ),
            ],
            style={"display": "flex", "alignItems": "center", "justifyContent": "center", "gap": "10px"},
        ),

        html.Br(),
        html.Div(id="api-status", style={"textAlign": "center", "marginBottom": "10px"}),
        html.Div(id="alert-box", style={"textAlign": "center"}),

        dcc.Loading(
            id="loading-spinner",
            type="circle",
            color="#2563eb",
            children=html.Div(id="forecast-output", style={"textAlign": "center", "marginTop": "20px"}),
        ),

        html.Hr(),
        dcc.Graph(id="price-graph", style={"height": "70vh"}),

        dcc.Interval(
            id="auto-refresh",
            interval=5 * 60 * 1000,  # 5 minutes
            n_intervals=0,
        ),
    ],
)


# --- Callbacks ---
@app.callback(
    [Output("forecast-output", "children"),
     Output("price-graph", "figure"),
     Output("alert-box", "children"),
     Output("api-status", "children")],
    [Input("predict-btn", "n_clicks"),
     Input("auto-refresh", "n_intervals")],
    [State("ticker-dropdown", "value")],
)
def update_forecast(n_clicks, n_intervals, ticker):
    api_url = f"http://127.0.0.1:8000/predict/{ticker}"

    # Check API status
    try:
        status_check = requests.get("http://127.0.0.1:8000/")
        api_status = html.Span("🟢 API Connected", style={"color": "green", "fontWeight": "bold"})
    except Exception:
        api_status = html.Span("🔴 API Offline", style={"color": "red", "fontWeight": "bold"})
        return "", go.Figure(), html.Div("❌ API is not running."), api_status

    try:
        # Download historical data
        df = yf.download(ticker, start="2023-01-01", end=datetime.date.today())
        if df.empty:
            return "", go.Figure(), html.Div("⚠️ No stock data found."), api_status

        # Call prediction API
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            forecast = data.get("prediction", None)
            forecast_str = f"Predicted next return: **{forecast:.4f}**" if forecast else "Prediction unavailable."
        else:
            forecast_str = "⚠️ API returned an invalid response."

        # Build price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Closing Price", line=dict(color="#2563eb")))
        fig.update_layout(
            title=f"{ticker} - Price History & Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
        )

        alert = html.Div("✅ Forecast updated successfully!", style={"color": "green"})
        return dcc.Markdown(forecast_str), fig, alert, api_status

    except Exception as e:
        return "", go.Figure(), html.Div(f"❌ Error: {str(e)}", style={"color": "red"}), api_status


# --- Run ---
if __name__ == "__main__":
    app.run(debug=True, port=8050)
