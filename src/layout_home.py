# src/layout_home.py
from dash import html

from containers import contribution_salary_container, radar_container


layout = html.Div(
    [
        html.H1(
            "MLB Team Weakness Diagnosis Dashboard",
            style={"textAlign": "center"},
        ),
        html.Hr(),
        # 未來可以在這裡放 Radio/Dropdown 控制「選哪一隊」
        contribution_salary_container(),

        radar_container(),
    ]
)
