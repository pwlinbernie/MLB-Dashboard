from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "db" / "MLBDashboard.db"



def query(sql: str) -> pd.DataFrame:
    connection = sqlite3.connect(database=DB_PATH)
    cur = connection.cursor()
    cur.execute(sql, ())
    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]
    result = pd.DataFrame(data=rows, columns=col_names)
    connection.close()
    return result

def load_batter_raw():
    sql = """
    SELECT playerID, yearID, teamID, POS,
           AB, H, "2B", "3B", HR, BB, SO, HBP, SF, SH, salary,
           `ops+` AS OPS_plus
    FROM batter
    """
    return query(sql)

def load_pitcher_raw():
    sql = """
    SELECT playerID, yearID, teamID, POS, throws,
           IPouts, H, ER, HR, BB, SO, ERA, fip, "fip-", salary
    FROM pitcher
    """
    return query(sql)