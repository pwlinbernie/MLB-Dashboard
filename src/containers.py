from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go

from charts import (
    plot_contribution_salary_scatter,
    plot_laa_batter_radar,
    plot_laa_pitcher_radar
)

def radar_container():
    """雷達圖區塊：只放圖，不放控制元件。"""
    return html.Div([
        html.H3(
            "Team Radar (PR values)",
            style={"textAlign": "center"}
        ),
        dcc.Graph(id="player-radar-graph"),
    ])

@callback(
    Output("player-radar-graph", "figure"),
    Input("apply-button", "n_clicks"),
    State("player-type-radio", "value"),
    State("sub-type-dropdown", "value"),
)
def update_radar(n_clicks, player_type, sub_type):
    if n_clicks == 0 or not sub_type:
        return go.Figure()

    
    group_value = sub_type[0] if isinstance(sub_type, list) else sub_type

    if player_type == "batter":
        
        return plot_laa_batter_radar(group_code=group_value)

    group_code = group_value.replace(" ", "_")
    return plot_laa_pitcher_radar(group_code=group_code)

def contribution_salary_container():
    container = html.Div([
        # 建立 checkbox
        dcc.RadioItems(
            id="player-type-radio",
            options=[
                {"label": "batter", "value": "batter"},
                {"label": "pitcher", "value": "pitcher"},
            ],
            value="batter",   # 預設選項
            inline=True       # 兩個選項排成一列
        ),
        # 建立 dropdown，會根據 checkbox 的值而改變
        dcc.Dropdown(
            id="sub-type-dropdown",
            value=None,
            placeholder="please choose",
            multi=True
        ),
        html.Button("Apply", id="apply-button", n_clicks=0),
        # 取得 scatter plot，根據 checkbox 跟 dropdown 所選的值而變化
        html.H3(
            "Salary vs. Player Contribution",
            style={"textAlign": "center"}
        ),
        dcc.Graph(id="player-scatter-graph")
    ])
    return container


@callback(
    Output("sub-type-dropdown", "options"),
    Output("sub-type-dropdown", "value"),
    Input("player-type-radio", "value")
)
def update_dropdown(player_type):
    if player_type == "batter":
        options = [
            {"label": "1B", "value": "1B"},
            {"label": "2B", "value": "2B"},
            {"label": "3B", "value": "3B"},
            {"label": "SS", "value": "SS"},
            {"label": "C", "value": "C"},
            {"label": "OF", "value": "OF"},
            {"label": "DH", "value": "DH"},
        ]
        default_value = None
    else:
        options = [
            {"label": "SP R", "value": "SP R"},
            {"label": "SP L", "value": "SP L"},
            {"label": "RP R", "value": "RP R"},
            {"label": "RP L", "value": "RP L"},
        ]
        default_value = None

    return options, default_value


@callback(
    Output("player-scatter-graph", "figure"),
    Input("apply-button", "n_clicks"),
    State("player-type-radio", "value"),
    State("sub-type-dropdown", "value")
)
def update_scatter(n_clicks, player_type, sub_type):
    if n_clicks == 0 or sub_type is None:
        return px.scatter()
    return plot_contribution_salary_scatter(
        player_type=player_type,
        roles=sub_type
    )
    