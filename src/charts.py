import operator
from typing import Literal, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from db_access import query
from db_access import load_batter_raw, load_pitcher_raw

#--------------------------------------------------------------#
# Data Processing Functions                                    #
#--------------------------------------------------------------#

# 打者指標計算
def compute_batter_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 基本打擊指標計算
    df["1B"] = df["H"] - df["2B"] - df["3B"] - df["HR"]
    df["PA"] = df["AB"] + df["BB"] + df["HBP"] + df["SF"] + df["SH"]

    df["AVG"] = df["H"] / df["AB"].where(df["AB"] > 0, 1)
    df["OBP"] = (df["H"] + df["BB"] + df["HBP"]) / df["PA"].where(df["PA"] > 0, 1)
    df["SLG"] = (df["1B"] + 2*df["2B"] + 3*df["3B"] + 4*df["HR"]) / df["AB"].where(df["AB"] > 0, 1)
    df["BB_rate"] = df["BB"] / df["PA"].where(df["PA"] > 0, 1)
    df["K_rate"] = df["SO"] / df["PA"].where(df["PA"] > 0, 1)
    df["OPS_plus"] = pd.to_numeric(df["OPS_plus"], errors="coerce")

    return df

# 打者 PR 計算
def add_batter_pr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    league = df[df["PA"] >= 50].copy()  # 設門檻，避免樣本太小

    # 每個指標的 PR
    for col in ["AVG", "OBP", "SLG", "BB_rate", "OPS_plus"]:
        rank = league[col].rank(pct=True) * 100
        df[f"{col}_PR"] = rank.reindex(df.index)

    # K_rate 越低越好，所以反向
    rank_k = (1 - league["K_rate"].rank(pct=True)) * 100
    df["K_rate_PR"] = rank_k.reindex(df.index)

    return df

# 投手指標計算
def compute_pitcher_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 將 IPouts 轉成局數
    df["IP"] = df["IPouts"] / 3.0

    # 避免除以 0 的情況：IP <= 0 時，把分母設成 1（結果不會被我們當成有意義的樣本）
    ip_safe = df["IP"].where(df["IP"] > 0, 1)

    # K/9, BB/9, H/9
    df["K9"] = df["SO"] * 9 / ip_safe
    df["BB9"] = df["BB"] * 9 / ip_safe
    df["H9"] = df["H"] * 9 / ip_safe

    # WHIP = (BB + H) / IP
    df["WHIP"] = (df["BB"] + df["H"]) / ip_safe

    # 確保 ERA、fip 是數值
    df["ERA"] = pd.to_numeric(df["ERA"], errors="coerce")
    df["fip"] = pd.to_numeric(df["fip"], errors="coerce")

    return df

# 投手 PR 計算
def add_pitcher_pr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 設定聯盟樣本門檻：例如 IP >= 20
    league = df[df["IP"] >= 20].copy()

    # 「越高越好」的指標：K9
    for col in ["K9"]:
        rank = league[col].rank(pct=True) * 100  # 0~100
        df[f"{col}_PR"] = rank.reindex(df.index)

    # 「越低越好」的指標：ERA, FIP, WHIP, BB9, H9
    for col in ["ERA", "fip", "WHIP", "BB9", "H9"]:
        rank = (1 - league[col].rank(pct=True)) * 100  # 0~100，數值越好 PR 越高
        df[f"{col}_PR"] = rank.reindex(df.index)

    return df

# 指定 team
TEAM_ID = "LAA"

#--------------------------------------------------------------#
# Scatter plot: Salary vs. Player Contribution                 #
#--------------------------------------------------------------#

# Contribution vs Salary 散布圖
def get_players(player_type: Literal["batter", "pitcher"], roles: List[str]) -> pd.DataFrame:
    """
    取得特定位置的球員數據
    """
    # 取資料（分打者跟投手）
    roles_str = "','".join(roles)
    if player_type == "batter":
        # 篩選 team 跟 position
        pos_filter_sql = f"""
            SELECT playerID, `ops+`, salary
            FROM batter
            WHERE teamID = '{TEAM_ID}'
                AND POS IN ('{roles_str}')
        """

    elif player_type == "pitcher":
        # 篩選 team, position 跟 throws（position 跟 throws concat）
        pos_filter_sql = f"""
            SELECT playerID, `fip-`, salary
            FROM pitcher
            WHERE teamID = '{TEAM_ID}'
                AND POS || ' ' || throws IN ('{roles_str}')
        """

    # query db
    pos_filter_result = query(sql=pos_filter_sql)
    return pos_filter_result


def get_salary_median(player_type: Literal["batter", "pitcher"]) -> float:
    """
    取得該 player type 的薪水中位數
    """
    salary_sql = f"""
        SELECT salary
        FROM {player_type}
    """
    # query db
    salary_result = query(sql=salary_sql)
    salary_median = salary_result['salary'].median()
    return salary_median


def get_metric_name(player_type: Literal["batter", "pitcher"]) -> str:
    """
    根據 player type 決定要看什麼指標（ops+ or fip-）
    """
    metrics = {
        "batter": "ops+",
        "pitcher": "fip-"
    }
    metric = metrics[player_type]
    return metric

def plot_contribution_salary_scatter(player_type: Literal["batter", "pitcher"], roles: List[str]) -> go.Figure:
    """
    畫選手貢獻和薪資的散布圖
    """
    y_axis = get_metric_name(player_type=player_type)
    # 取資料
    players = get_players(
        player_type=player_type,
        roles=roles
    )
    salary_median = get_salary_median(player_type=player_type)
    # 畫圖
    fig = px.scatter(
        data_frame=players,
        x="salary",
        y=y_axis,
        hover_data=["playerID"]
    )
    fig.add_shape(
        type="line",
        x0=min(players['salary'].min(), salary_median),
        y0=100,
        x1=max(players['salary'].max(), salary_median),
        y1=100,
        line=dict(width=2, dash="dash", color="black")
    )
    fig.add_annotation(
        x=players["salary"].max(),
        y=100,
        text=f"{y_axis.upper()} = 100",
        showarrow=False,
        yanchor="bottom",
        xanchor="right",
        font=dict(size=12, color="black")
    )
    fig.add_shape(
        type="line",
        y0=players[y_axis].min(),
        x0=salary_median,
        y1=players[y_axis].max(),
        x1=salary_median,
        line=dict(width=2, dash="dash", color="black")
    )
    fig.add_annotation(
        x=salary_median,
        y=players[y_axis].max(),
        text=f"Median salary = {salary_median:,.0f}",
        showarrow=False,
        yanchor="top",
        xanchor="left",
        font=dict(size=12, color="black")
    )

    return fig


def get_player_list(player_type: Literal["batter", "pitcher"], roles: List[str], action: Literal["retain", "trade", "extend", "option"]) -> pd.DataFrame:
    """
    根據 action 及位置篩選球員
    """
    op_map = {
        "batter": operator.ge,  # >=
        "pitcher": operator.le  # <=
    }
    op = op_map[player_type]
    players = get_players(
        player_type=player_type,
        roles=roles
    )
    salary_median = get_salary_median(player_type=player_type)
    metric = get_metric_name(player_type=player_type)

    if action == "retain":
        condition = (players["salary"] >= salary_median) & (op(players[metric], 100))
    elif action == "trade":
        condition = (players["salary"] >= salary_median) & (~op(players[metric], 100))
    elif action == "extend":
        condition = (players["salary"] < salary_median) & (op(players[metric], 100))
    elif action == "option":
        condition = (players["salary"] < salary_median) & (~op(players[metric], 100))
    
    filtered_players = players[condition]
    filtered_players["salary"] = filtered_players["salary"].apply(lambda x: f"{x:,.0f}")
    filtered_players[metric] = filtered_players[metric].apply(lambda x: round(float(x), 2))
    return filtered_players


#--------------------------------------------------------------#
# Radar Chart: Batter Group Radar (LAA only)                   #
#--------------------------------------------------------------#

# 打者群組雷達圖指標
BATTER_RADAR_METRICS = [
    "AVG_PR",
    "OBP_PR",
    "SLG_PR",
    "BB_rate_PR",
    "K_rate_PR",
    "OPS_plus_PR"
]

# 投手群組雷達圖指標
PITCHER_RADAR_METRICS = [
    "ERA_PR",   # ERA 表現（反向成高分好）
    "fip_PR",   # FIP 表現
    "WHIP_PR",  # WHIP
    "K9_PR",    # 三振能力
    "BB9_PR",   # 保送控制
    "H9_PR",    # 被安打抑制
]

# 打者群組資料建構
def build_laa_batter_group_profile(group_code: str) -> pd.Series | None:
    df_raw = load_batter_raw()
    df = compute_batter_rates(df_raw)
    df = add_batter_pr(df)

    # 只看 LAA
    df = df[df["teamID"] == TEAM_ID].copy()

    # 根據 group_code 過濾守位
    df = df[df["POS"] == group_code]

    if df.empty:
        return None

    profile = df[BATTER_RADAR_METRICS].mean()
    return profile

# 投手群組資料建構
def build_laa_pitcher_group_profile(group_code: str) -> pd.Series | None:
    """
    回傳洛杉磯天使隊 (TEAM_ID) 某投手群組的 6 個 PR 平均值。

    group_code 可能是：
        "SP"    : 全隊先發投手
        "RP"    : 全隊中繼+後援投手
        "SP_L"  : 先發左投
        "SP_R"  : 先發右投
        "RP_L"  : 中繼+後援左投
        "RP_R"  : 中繼+後援右投
    """

    # 1. 撈 raw 投手資料 + 算 rate + PR
    df_raw = load_pitcher_raw()
    df = compute_pitcher_rates(df_raw)
    df = add_pitcher_pr(df)

    # 2. 只看 LAA
    df = df[df["teamID"] == TEAM_ID].copy()

    # 3. 根據 group_code 過濾 POS / throws
    if group_code == "SP":
        df = df[df["POS"] == "SP"]
    elif group_code == "RP":
        df = df[df["POS"] == "RP"]
    elif group_code == "SP_L":
        df = df[(df["POS"] == "SP") & (df["throws"] == "L")]
    elif group_code == "SP_R":
        df = df[(df["POS"] == "SP") & (df["throws"] == "R")]
    elif group_code == "RP_L":
        df = df[(df["POS"] == "RP") & (df["throws"] == "L")]
    elif group_code == "RP_R":
        df = df[(df["POS"] == "RP") & (df["throws"] == "R")]
    else:
        # 未知 group，直接回傳 None
        return None

    if df.empty:
        return None

    # 4. 計算這個群組在六個指標上的平均 PR
    profile = df[PITCHER_RADAR_METRICS].mean()

    return profile

# 打者雷達圖
def plot_laa_batter_radar(group_code: str) -> go.Figure:
    """
    畫出 LAA 在指定打者群組 (group_code) 的雷達圖。

    group_code:
        "C", "1B", "2B", "3B", "SS", "OF", "DH"
    """
    profile = build_laa_batter_group_profile(group_code)

    if profile is None:
        return go.Figure(
            layout_title_text = f"{TEAM_ID} {group_code} – No data for selected group"
        )
    
    metrics = profile.index.tolist()
    values = profile.values.tolist()

    # 雷達圖需要閉合
    metrics += [metrics[0]]
    values += [values[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r = values,
            theta = metrics,
            fill = 'toself',
            name = f"{TEAM_ID} {group_code}"
        )
    )
    
    fig.update_layout(
        polar = dict(
            radialaxis = dict(
                visible = True,
                range = [0, 100]
            )
        ),
        showlegend = True,
        title = f"{TEAM_ID} {group_code} - Batter Radar (PR Scores)"
    )

    return fig

# 投手雷達圖
def plot_laa_pitcher_radar(group_code: str) -> go.Figure:
    """
    畫出 LAA 在指定投手群組 (group_code) 的雷達圖。
    """
    profile = build_laa_pitcher_group_profile(group_code)

    if profile is None:
        return go.Figure(
            layout_title_text=f"{TEAM_ID} {group_code} – No data for selected pitcher group"
        )

    metrics = profile.index.tolist()
    values = profile.values.tolist()

    # 雷達圖首尾相接
    metrics += [metrics[0]]
    values += [values[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=metrics,
            fill="toself",
            name=f"{TEAM_ID} {group_code}",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],  # PR 值 0~100
            )
        ),
        showlegend=True,
        title=f"{TEAM_ID} {group_code} – Pitcher Radar (PR values)",
    )

    return fig

#--------------------------------------------------------------#
# Overview Breakdown Bar Chart                                 #
#--------------------------------------------------------------#

# Overview breakdown: Team vs League
def plot_overview_breakdown(team_id: str, group: str) -> go.Figure:
    """
    Overview breakdown: Team vs League
    group:
      - "SP" / "RP" : pitcher, breakdown by throws (R/L), metric = fip-
      - "H"         : batter, breakdown by POS, metric = ops+
    """

    if group in ["SP", "RP"]:
        df = query(f"""
            SELECT
                throws AS category,
                AVG(`fip-`) AS league_metric,
                AVG(CASE WHEN teamID = '{team_id}' THEN `fip-` END) AS team_metric
            FROM pitcher
            WHERE POS = '{group}'
              AND throws IN ('R','L')
            GROUP BY throws
            ORDER BY category
        """)
        metric_name = "FIP-"
        x_title = "Throws"

    else:  # group == "H"
        hitter_pos = ["1B","2B","3B","SS","OF","C","DH"]
        pos_str = "','".join(hitter_pos)

        df = query(f"""
            SELECT
                POS AS category,
                AVG(`ops+`) AS league_metric,
                AVG(CASE WHEN teamID = '{team_id}' THEN `ops+` END) AS team_metric
            FROM batter
            WHERE POS IN ('{pos_str}')
            GROUP BY POS
            ORDER BY category
        """)
        metric_name = "OPS+"
        x_title = "Position"

    # 本隊沒有資料的類別會是 NULL，先排除
    df = df.dropna(subset=["team_metric"])

    fig = go.Figure()
    fig.add_bar(
        x=df["category"],
        y=df["league_metric"],
        name="League Average"
    )
    fig.add_bar(
        x=df["category"],
        y=df["team_metric"],
        name="Team Average"
    )

    fig.update_layout(
        barmode="group",
        xaxis_title=x_title,
        yaxis_title=metric_name,
        legend_title="",
        margin=dict(l=40, r=20, t=40, b=40),
        title=f"{team_id} vs League – {group}",
    )

    return fig

#--------------------------------------------------------------#
# Page performance                                             #
#--------------------------------------------------------------#

# Performance page Bar Chart
def plot_performance_bar(team_id: str, player_type: str, groups: list[str]) -> go.Figure:
    """
    Bar chart for Performance page:
    - batter: compare OPS+ by POS (team vs league)
    - pitcher: compare FIP- by POS (SP/RP) (team vs league)
    groups: selected categories from dropdown
    """
    fig = go.Figure()

    if player_type == "batter":
        pos_str = "','".join(groups)
        df = query(f"""
            SELECT
                POS AS category,
                AVG(`ops+`) AS league_metric,
                AVG(CASE WHEN teamID = '{team_id}' THEN `ops+` END) AS team_metric
            FROM batter
            WHERE POS IN ('{pos_str}')
            GROUP BY POS
            ORDER BY category
        """).dropna(subset=["team_metric"])

        metric_name = "OPS+"
        x_title = "Position"

    else:

        pos_str = "','".join(groups)

        df = query(f"""
            SELECT
                POS AS category,
                AVG(`fip-`) AS league_metric,
                AVG(CASE WHEN teamID = '{team_id}' THEN `fip-` END) AS team_metric
            FROM pitcher
            WHERE POS IN ('{pos_str}')
            GROUP BY POS
            ORDER BY category
        """).dropna(subset=["team_metric"])

        metric_name = "FIP-"
        x_title = "Pitcher Role"

    fig.add_bar(x=df["category"], y=df["league_metric"], name="League Average")
    fig.add_bar(x=df["category"], y=df["team_metric"], name="Team Average")

    fig.update_layout(
        barmode="group",
        xaxis_title=x_title,
        yaxis_title=metric_name,
        legend_title="",
        margin=dict(l=40, r=20, t=40, b=40),
        title=f"{team_id} vs League – {metric_name}",
    )
    return fig

#--------------------------------------------------------------#
# Page functions                                               #
#--------------------------------------------------------------#

# Performance page 雷達圖入口
def plot_performance_radar(player_type: str, group_code: str) -> go.Figure:
    """
    Performance page 用的統一雷達入口（LAA only）
    player_type: "batter" / "pitcher"
    group_code:
      - batter: "C","1B","2B","3B","SS","OF","DH"
      - pitcher: "SP","RP"
    """
    if player_type == "batter":
        return plot_laa_batter_radar(group_code)

    if player_type == "pitcher":
        return plot_laa_pitcher_radar(group_code)

    return go.Figure(layout_title_text=f"Unknown player_type: {player_type}")

# Overview tiles data 入口
def get_overview_tiles(team_id: str) -> dict:
    """
    回傳 Overview tiles 需要的數值（從 DB 計算）：
    {
      "SP": {"metric": float, "diff": float},  # diff: 100 - fip-
      "RP": {"metric": float, "diff": float},
      "H":  {"metric": float, "diff": float},  # diff: ops+ - 100
    }
    """
    BASELINE = 100

    # SP (pitcher POS = SP), metric = avg(fip-)
    sp_df = query(f"""
        SELECT AVG(`fip-`) AS metric
        FROM pitcher
        WHERE teamID = '{team_id}' AND POS = 'SP'
    """)
    sp_metric = float(sp_df.iloc[0]["metric"]) if not sp_df.empty and sp_df.iloc[0]["metric"] is not None else None

    # RP (pitcher POS = RP), metric = avg(fip-)
    rp_df = query(f"""
        SELECT AVG(`fip-`) AS metric
        FROM pitcher
        WHERE teamID = '{team_id}' AND POS = 'RP'
    """)
    rp_metric = float(rp_df.iloc[0]["metric"]) if not rp_df.empty and rp_df.iloc[0]["metric"] is not None else None

    # H (batters), metric = avg(ops+)
    h_df = query(f"""
        SELECT AVG(`ops+`) AS metric
        FROM batter
        WHERE teamID = '{team_id}'
          AND POS IN ('C','1B','2B','3B','SS','OF','DH')
    """)
    h_metric = float(h_df.iloc[0]["metric"]) if not h_df.empty and h_df.iloc[0]["metric"] is not None else None

    # diffs (依你的定義)
    sp_diff = (BASELINE - sp_metric) if sp_metric is not None else None
    rp_diff = (BASELINE - rp_metric) if rp_metric is not None else None
    h_diff  = (h_metric - BASELINE) if h_metric is not None else None

    return {
        "SP": {"metric": sp_metric, "diff": sp_diff},
        "RP": {"metric": rp_metric, "diff": rp_diff},
        "H":  {"metric": h_metric, "diff": h_diff},
    }

# Overview team record 入口
def get_team_record(team_id: str) -> dict:
    """
    從 DB 的 team table 取戰績與排名
    回傳: {"name": str|None, "W": int|None, "L": int|None, "Rank": int|None}
    """
    df = query(f"""
        SELECT name, W, L, Rank
        FROM team
        WHERE teamID = '{team_id}'
        LIMIT 1
    """)

    if df.empty:
        return {"name": None, "W": None, "L": None, "Rank": None}

    row = df.iloc[0]
    return {
        "name": row["name"] if "name" in df.columns else None,
        "W": int(row["W"]) if row["W"] is not None else None,
        "L": int(row["L"]) if row["L"] is not None else None,
        "Rank": int(row["Rank"]) if row["Rank"] is not None else None,
    }