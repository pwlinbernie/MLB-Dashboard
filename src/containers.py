from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.express as px

from charts import (
    plot_contribution_salary_scatter,
    plot_laa_batter_radar,
    plot_laa_pitcher_radar,
    plot_overview_breakdown,
    TEAM_ID,
    plot_performance_radar,
    plot_performance_bar,
    get_overview_tiles,
    get_team_record,
    get_player_list
)

#--------------------------------------------------------------#
# Radar Chart                                                  #
#--------------------------------------------------------------#

# 雷達圖區塊
def radar_container():
    """雷達圖區塊：只放圖，不放控制元件。"""
    return html.Div([
        html.H3(
            "Team Radar (PR values)",
            style={"textAlign": "center"}
        ),
        html.Div(
            id="radar-grid",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
                "gap": "16px",
                "alignItems": "start",
            }
        )
    ])

@callback(
    Output("radar-grid", "children"),
    Input("apply-button", "n_clicks"),
    State("player-type-radio", "value"),
    State("sub-type-dropdown", "value"),
)

# 雷達圖更新
def update_radar_grid(n_clicks, player_type, sub_type):
    if n_clicks == 0 or not sub_type:
        return []

    # sub_type 因為 multi=True，會是 list
    selected = sub_type if isinstance(sub_type, list) else [sub_type]

    cards = []
    for group_value in selected:
        if player_type == "batter":
            fig = plot_laa_batter_radar(group_code=group_value)
            title = f"{group_value} Radar"
        else:
            group_code = group_value.replace(" ", "_")
            fig = plot_laa_pitcher_radar(group_code=group_code)
            title = f"{group_value} Radar"

        fig.update_layout(height=320, margin=dict(l=40, r=40, t=50, b=40))

        cards.append(
            html.Div(
                [
                    html.H4(title, style={"textAlign": "center", "margin": "8px 0"}),
                    dcc.Graph(
                        figure=fig,
                        config={"displayModeBar": False},
                        style={"height": "320px"},
                    ),
                ],
                style={
                    "border": "1px solid #ddd",
                    "borderRadius": "12px",
                    "padding": "8px",
                    "backgroundColor": "white",
                    "overflow": "hidden",
                },
            )
        )

    return cards

#--------------------------------------------------------------#
# Contribution vs. Salary Scatter Plot                         #
#--------------------------------------------------------------#

# Contribution vs. Salary 區塊
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
        html.H2(
            "Salary vs. Player Contribution",
            style={"textAlign": "center"}
        ),
        dcc.Graph(id="player-scatter-graph"),
        dcc.Dropdown(
            id="action-dropdown",
            options=[
                {"label": "Retain", "value": "retain"},
                {"label": "Trade", "value": "trade"},
                {"label": "Extend", "value": "extend"},
                {"label": "Option", "value": "option"},
            ],
            value="retain",
            placeholder="please choose"
        ),
        dash_table.DataTable(
            id="player-list",
            data=[],
            page_size=10,
            sort_action="native",
            style_cell={
                "fontSize": "20px"
            },
        )
    ])
    return container


@callback(
    Output("sub-type-dropdown", "options"),
    Output("sub-type-dropdown", "value"),
    Input("player-type-radio", "value")
)

# Contribution vs. Salary dropdown 更新
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
# Contribution vs. Salary scatter plot 更新
def update_scatter(n_clicks, player_type, sub_type):
    if n_clicks == 0 or sub_type is None:
        return px.scatter()
    return plot_contribution_salary_scatter(
        player_type=player_type,
        roles=sub_type
    )


@callback(
    Output("player-list", "data"),
    Output("player-list", "columns"),
    Input("apply-button", "n_clicks"),
    Input("action-dropdown", "value"),
    State("player-type-radio", "value"),
    State("sub-type-dropdown", "value"),
    prevent_initial_call=True
)
# 球員列表更新
def update_player_list(n_clicks, action, player_type, sub_type):
    if n_clicks == 0 or sub_type is None:
        return [], []
    players = get_player_list(
        player_type=player_type,
        roles=sub_type,
        action=action
    )
    return players.to_dict("records"), [{"name": col.upper(), "id": col} for col in players.columns]


#--------------------------------------------------------------#
# Overview Container                                           #
#--------------------------------------------------------------#

# Overview 區塊
def overview_container():
    tiles = get_overview_tiles(TEAM_ID)
    record = get_team_record(TEAM_ID)

    def tile_box(title, diff, metric_label, metric_value):
        diff_text = "N/A" if diff is None else f"{diff:+.1f}"
        metric_text = "N/A" if metric_value is None else f"{metric_label}: {metric_value:.1f}"

        if diff is None:
            bg = "#f2f2f2"
        else:
            bg = "#d0f0c0" if diff >= 0 else "#f4cccc"

        return html.Div(
            [
                html.H4(title, style={"margin": "0 0 4px 0"}),
                html.Div(diff_text, style={"fontSize": "24px", "fontWeight": "600"}),
                html.Div(metric_text, style={"fontSize": "12px"}),
            ],
            style={
                "border": "2px solid black",
                "borderRadius": "10px",
                "padding": "10px",
                "width": "100px",
                "textAlign": "center",
                "backgroundColor": bg,
            },
        )

    # 先在這裡算好文字（不要放進 children list）
    wl_text = "W/L: N/A" if record["W"] is None or record["L"] is None else f"W/L: {record['W']} / {record['L']}"
    rank_text = "Standing: N/A" if record["Rank"] is None else f"Standing: {record['Rank']}"

    return html.Div(
        [
            html.H2(f"Team Overview – {TEAM_ID}"),

            html.Div(
                [
                    tile_box("SP", tiles["SP"]["diff"], "FIP-", tiles["SP"]["metric"]),
                    tile_box("RP", tiles["RP"]["diff"], "FIP-", tiles["RP"]["metric"]),
                    tile_box("H",  tiles["H"]["diff"],  "OPS+", tiles["H"]["metric"]),

                    html.Div(
                        [
                            html.P(wl_text, style={"margin": "0 0 6px 0"}),
                            html.P(rank_text, style={"margin": "0"}),
                        ],
                        style={"marginLeft": "40px", "fontSize": "16px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "20px"},
            ),

            html.Div(
                [
                    html.Label("Select group:"),
                    dcc.Dropdown(
                        id="overview-group-dropdown",
                        options=[
                            {"label": "Starting Pitcher (SP)", "value": "SP"},
                            {"label": "Relief Pitcher (RP)", "value": "RP"},
                            {"label": "Hitters (H)", "value": "H"},
                        ],
                        value="SP",
                        clearable=False,
                        style={"width": "320px"},
                    ),
                ],
                style={"marginBottom": "10px"},
            ),

            dcc.Graph(id="overview-breakdown-chart"),
        ],
        style={"padding": "20px"},
    )

@callback(
    Output("overview-breakdown-chart", "figure"),
    Input("overview-group-dropdown", "value")
)

# Overview breakdown chart 更新
def overview_breakdown_real(group):
    return plot_overview_breakdown(team_id=TEAM_ID, group=group)

#--------------------------------------------------------------#
# Layout Pages                                                 #
#--------------------------------------------------------------#

# Layout: Overview Page
def page_overview():
    return html.Div(
        [
            overview_container(), 
        ],
        style={"padding": "16px"},
    )

# Layout: Performance Page
def page_performance():
    return html.Div(
        [
            html.H2("Performance"),
            html.Div(
                [
                    dcc.RadioItems(
                        id="perf-player-type-radio",
                        options=[
                            {"label": "Batter", "value": "batter"},
                            {"label": "Pitcher", "value": "pitcher"},
                        ],
                        value="batter",
                        inline=True,
                    ),
                    dcc.Dropdown(
                        id="perf-subtype-dropdown",
                        value=None,
                        placeholder="Please choose",
                        multi=True,
                        style={"width": "420px", "marginLeft": "12px"},
                    ),
                    html.Button("Apply", id="perf-apply-button", n_clicks=0, style={"marginLeft": "12px"}),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "12px"},
            ),

            # Charts zone
            html.Div(
                [
                    dcc.Graph(id="perf-bar-chart"),
                    html.Div(id="perf-radar-grid"),
                ],
                style={"display": "flex", "flexDirection": "column", "gap": "14px"},
            ),
        ],
        style={"padding": "16px"},
    )

@callback(
    Output("perf-subtype-dropdown", "options"),
    Output("perf-subtype-dropdown", "value"),
    Input("perf-player-type-radio", "value"),
)

# Performance dropdown 更新
def perf_update_dropdown(player_type):
    """
    Batter: defensive positions
    Pitcher: SP/RP
    """
    if player_type == "batter":
        options = [
            {"label": "C", "value": "C"},
            {"label": "1B", "value": "1B"},
            {"label": "2B", "value": "2B"},
            {"label": "3B", "value": "3B"},
            {"label": "SS", "value": "SS"},
            {"label": "OF", "value": "OF"},
            {"label": "DH", "value": "DH"},
        ]
    else:
        options = [
            {"label": "Starter (SP)", "value": "SP"},
            {"label": "Reliever (RP)", "value": "RP"}
        ]

    return options, None

@callback(
    Output("perf-bar-chart", "figure"),
    Output("perf-radar-grid", "children"),
    Input("perf-apply-button", "n_clicks"),
    State("perf-player-type-radio", "value"),
    State("perf-subtype-dropdown", "value"),
)
# Performance charts 更新
def perf_update_charts(n_clicks, player_type, sub_types):
    if n_clicks == 0 or not sub_types:
        return px.bar(), []

    # 1) Bar chart：team vs league
    fig_bar = plot_performance_bar(team_id=TEAM_ID, player_type=player_type, groups=sub_types)

    # 2) Radar charts：多選 → 多張雷達圖
    radar_cards = []
    for g in sub_types:
        fig_radar = plot_performance_radar(player_type=player_type, group_code=g)
        radar_cards.append(
            html.Div(
                [
                    dcc.Graph(figure=fig_radar, config={"displayModeBar": False}),
                ],
                style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "8px"},
            )
        )

    radar_grid = html.Div(
        radar_cards,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, minmax(320px, 1fr))",
            "gap": "12px",
        },
    )

    return fig_bar, radar_grid

# Layout: Contribution Page
def page_contribution():
    return html.Div(
        [
            html.H2("Contribution"),
            contribution_salary_container(),
        ],
        style={"padding": "16px"},
    )