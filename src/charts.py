import pandas as pd
from typing import Literal, List

import plotly.express as px
import plotly.graph_objects as go

from db_access import query
from db_access import load_batter_raw, load_pitcher_raw

#--------------------------------------------------------------#
# Data Processing Functions                                    #
#--------------------------------------------------------------#

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

def plot_contribution_salary_scatter(player_type: Literal["batter", "pitcher"], roles: List[str]) -> go.Figure:
    """
    畫選手貢獻和薪資的散布圖
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
        y_axis = "ops+"

    elif player_type == "pitcher":
        # 篩選 team, position 跟 throws（position 跟 throws concat）
        pos_filter_sql = f"""
            SELECT playerID, `fip-`, salary
            FROM pitcher
            WHERE teamID = '{TEAM_ID}'
                AND POS || ' ' || throws IN ('{roles_str}')
        """
        y_axis = "fip-"

    salary_sql = f"""
        SELECT salary
        FROM {player_type}
    """
    # query db
    pos_filter_result = query(sql=pos_filter_sql)
    salary_result = query(sql=salary_sql)
    salary_median = salary_result['salary'].median()
    # 畫圖
    fig = px.scatter(
        data_frame=pos_filter_result,
        x="salary",
        y=y_axis,
        hover_data=["playerID"]
    )
    fig.add_shape(
        type="line",
        x0=min(pos_filter_result['salary'].min(), salary_median),
        y0=100,
        x1=max(pos_filter_result['salary'].max(), salary_median),
        y1=100,
        line=dict(width=2, dash="dash", color="black")
    )
    fig.add_annotation(
        x=pos_filter_result["salary"].max(),
        y=100,
        text=f"{y_axis.upper()} = 100",
        showarrow=False,
        yanchor="bottom",
        xanchor="right",
        font=dict(size=12, color="black")
    )
    fig.add_shape(
        type="line",
        y0=pos_filter_result[y_axis].min(),
        x0=salary_median,
        y1=pos_filter_result[y_axis].max(),
        x1=salary_median,
        line=dict(width=2, dash="dash", color="black")
    )
    fig.add_annotation(
        x=salary_median,
        y=pos_filter_result[y_axis].max(),
        text=f"Median salary = {salary_median:,.0f}",
        showarrow=False,
        yanchor="top",
        xanchor="left",
        font=dict(size=12, color="black")
    )

    return fig

#--------------------------------------------------------------#
# Radar Chart: Batter Group Radar (LAA only)                   #
#--------------------------------------------------------------#

BATTER_RADAR_METRICS = [
    "AVG_PR",
    "OBP_PR",
    "SLG_PR",
    "BB_rate_PR",
    "K_rate_PR",
    "OPS_plus_PR"
]

PITCHER_RADAR_METRICS = [
    "ERA_PR",   # ERA 表現（反向成高分好）
    "fip_PR",   # FIP 表現
    "WHIP_PR",  # WHIP
    "K9_PR",    # 三振能力
    "BB9_PR",   # 保送控制
    "H9_PR",    # 被安打抑制
]

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