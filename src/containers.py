from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.express as px

from charts import (
    plot_contribution_salary_scatter,
    plot_laa_batter_radar,
    plot_laa_hitter_team_radar,
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

    sp_radar = plot_laa_pitcher_radar("SP")
    rp_radar = plot_laa_pitcher_radar("RP")
    h_radar  = plot_laa_hitter_team_radar()

    # 左卡文字
    w = record["W"]
    l = record["L"]
    rank = record["Rank"]

    wl_big = "N/A" if w is None or l is None else f"{w} - {l}"
    rank_text = "Division Rank: N/A" if rank is None else f"Division Rank: {rank}th"
    games_text = "GAME: 162"  # 先固定；你之後若要從 DB 算也可以

    # 右卡：SP/RP/H 三列（用你現有 fip- / ops+）
    def trend_symbol(diff):
        if diff is None:
            return "–"
        return "▲" if diff >= 0 else "▼"

    def summary_row(label, metric_label, metric_value, diff):
        metric_value_text = "N/A" if metric_value is None else f"{metric_value:.1f}"
        diff_color = "#0B2D5C" if diff is None else ("#0B2D5C" if diff >= 0 else "#B00020")

        return html.Div(
            [
                html.Div(trend_symbol(diff), style={"width": "28px", "fontSize": "24px", "color": diff_color}),
                html.Div(label, style={"width": "70px", "fontSize": "34px", "fontWeight": "800", "color": "#0B2D5C"}),
                html.Div(
                    [
                        html.Span(f"{metric_label}: ", style={"fontWeight": "700", "color": "#555"}),
                        html.Span(metric_value_text, style={"fontWeight": "800", "color": "#B00020"}),
                        html.Span("   "),
                        html.Span("DIFF: ", style={"fontWeight": "700", "color": "#555"}),
                        html.Span("N/A" if diff is None else f"{diff:+.1f}", style={"fontWeight": "800", "color": "#B00020"}),
                    ],
                    style={"fontSize": "26px"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "border": "2px solid #E24A4A",
                "padding": "18px 18px",
            },
        )

    conclusion_text = overview_conclusion(tiles)

    html.Div(
        conclusion_text,
        style={
            "marginTop": "10px",
            "fontSize": "18px",
            "color": "#333",
            "fontWeight": "600",
        },
    ),

    # 先把「Overview 下拉 + bar chart」暫時移除（下一步改成三張 radar）
    return html.Div(
        [
            # 上排：左戰績卡 + 右概覽卡
            html.Div(
                [
                    # 左：戰績卡
                    html.Div(
                        [
                            html.Div(games_text, style={"fontSize": "34px", "fontWeight": "800", "color": "#777"}),
                            html.Div(
                                [
                                    html.Span(w if w is not None else "N/A",
                                              style={"fontSize": "110px", "fontWeight": "900", "color": "#B00020"}),
                                    html.Span(" Wins ", style={"fontSize": "28px", "color": "#B00020"}),
                                    html.Span("  -  ", style={"fontSize": "60px", "fontWeight": "900", "color": "#222"}),
                                    html.Span(l if l is not None else "N/A",
                                              style={"fontSize": "110px", "fontWeight": "900", "color": "#222"}),
                                    html.Span(" Loses", style={"fontSize": "28px", "color": "#222"}),
                                ],
                                style={"display": "flex", "alignItems": "baseline", "gap": "8px"},
                            ),
                            html.Div(rank_text, style={"fontSize": "54px", "fontWeight": "900", "color": "#0B2D5C"}),
                        ],
                        style={
                            "flex": "1",
                            "border": "1px solid #DDD",
                            "backgroundColor": "white",
                            "padding": "24px",
                            "minHeight": "260px",
                        },
                    ),

                    # 右：SP/RP/H 概覽 + 結論
                    html.Div(
                        [
                            summary_row("SP", "FIP-", tiles["SP"]["metric"], tiles["SP"]["diff"]),
                            summary_row("RP", "FIP-", tiles["RP"]["metric"], tiles["RP"]["diff"]),
                            summary_row("H",  "OPS+", tiles["H"]["metric"],  tiles["H"]["diff"]),
                            html.Div(
                                conclusion_text,
                                style={
                                    "marginTop": "10px",
                                    "fontSize": "18px",
                                    "color": "#333",
                                    "fontWeight": "600",
                                },
                            ),
                        ],
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "gap": "12px"},
                    ),
                ],
                style={"display": "flex", "gap": "20px", "alignItems": "stretch"},
            ),

            # 下排：三張 radar（Step 2 會補）
            html.Div(
                [
                    dcc.Graph(figure=sp_radar, config={"displayModeBar": False}),
                    dcc.Graph(figure=rp_radar, config={"displayModeBar": False}),
                    dcc.Graph(figure=h_radar,  config={"displayModeBar": False}),
                ],
                style={
                    "marginTop": "26px",
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3, minmax(320px, 1fr))",
                    "gap": "18px",
                },
            )
        ],
        style={"padding": "20px", "backgroundColor": "#F3F5F7", "minHeight": "100vh"},
    )

@callback(
    Output("overview-breakdown-chart", "figure"),
    Input("overview-group-dropdown", "value")
)

# Overview breakdown chart 更新
def overview_breakdown_real(group):
    return plot_overview_breakdown(team_id=TEAM_ID, group=group)

def overview_conclusion(tiles: dict) -> str:
    """
    Generate a short diagnostic conclusion based on SP/RP/H diffs.
    diff rules:
      - SP/RP (FIP-): diff = 100 - FIP-  (positive is good)
      - H (OPS+)   : diff = OPS+ - 100   (positive is good)
    """
    def d(key: str):
        v = tiles.get(key, {}).get("diff", None)
        return None if v is None else float(v)

    sp = d("SP")
    rp = d("RP")
    h  = d("H")

    # missing data
    if sp is None or rp is None or h is None:
        return "Overall: insufficient data to generate a reliable conclusion."

    pitching_avg = (sp + rp) / 2.0

    # severity helper
    def severity(x: float) -> str:
        ax = abs(x)
        if ax >= 10:
            return "severely"
        if ax >= 5:
            return "clearly"
        if ax >= 2:
            return "slightly"
        return "roughly"

    # 判断主弱点（选 diff 最负的那一项）
    weakest_key, weakest_val = min([("SP", sp), ("RP", rp), ("H", h)], key=lambda t: t[1])

    # both sides status
    pitching_bad = (sp < 0) and (rp < 0)
    hitting_bad  = (h < 0)

    if pitching_bad and hitting_bad:
        return (
            f"Overall weakness: both pitching and hitting are below league average. "
            f"Primary concern: {weakest_key} ({severity(weakest_val)} below average)."
        )

    if pitching_bad and not hitting_bad:
        return (
            f"Overall weakness: pitching is below league average "
            f"(SP {sp:+.1f}, RP {rp:+.1f}). "
            f"Primary concern: {weakest_key} ({severity(weakest_val)} below average)."
        )

    if not pitching_bad and hitting_bad:
        return (
            f"Overall weakness: hitting is below league average (H {h:+.1f}). "
            f"Primary concern: H ({severity(h)} below average)."
        )

    # neither is clearly bad
    if pitching_avg > 0 and h > 0:
        return "Overall strength: both pitching and hitting are above league average."
    if pitching_avg > 0 and h <= 0:
        return "Overall: pitching is solid, but hitting is around league average."
    if pitching_avg <= 0 and h > 0:
        return "Overall: hitting is solid, but pitching is around league average."

    return "Overall: performance is around league average across pitching and hitting."

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

def overview_conclusion(tiles: dict) -> str:
    sp = tiles["SP"]["diff"]
    rp = tiles["RP"]["diff"]
    h  = tiles["H"]["diff"]

    # None 防呆
    if sp is None or rp is None or h is None:
        return "Not enough data to generate conclusion."

    pitching = (sp + rp) / 2
    hitting = h

    if pitching < 0 and hitting < 0:
        return "Overall weakness: both pitching and hitting are below league average."
    if pitching < 0 and hitting >= 0:
        return "Key issue: pitching is below league average. Consider upgrading rotation/bullpen depth."
    if pitching >= 0 and hitting < 0:
        return "Key issue: offense is below league average. Consider upgrading lineup production."
    return "Overall: team performance is above league average in both pitching and hitting."