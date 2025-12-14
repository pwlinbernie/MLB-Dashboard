# src/layout_home.py
from dash import dcc, html, Input, Output, callback

from containers import (
    page_overview, 
    page_performance,
    page_contribution
    )


layout = html.Div(
    [
        # ===== Top navigation tabs =====
        dcc.Tabs(
            id="top-tabs",
            value="overview",
            children=[
                dcc.Tab(label="Overview", value="overview"),
                dcc.Tab(label="Performance", value="performance"),
                dcc.Tab(label="Contribution", value="contribution"),
            ],
        ),

        # ===== Page content =====
        html.Div(id="page-content"),
    ],
    style={"padding": "0px"},
)

@callback(
    Output("page-content", "children"),
    Input("top-tabs", "value"),
)
def render_page(tab):
    if tab == "overview":
        return page_overview()
    if tab == "performance":
        return page_performance()
    if tab == "contribution":
        return page_contribution()
    return page_overview()
