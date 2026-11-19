"""
Data loading and preprocessing module for Thai Equity Portfolio.
"""
import sqlite3
from pathlib import Path
import pandas as pd

# กำหนดที่อยู่ของฐานข้อมูลเริ่มต้น (อิงจากตำแหน่งไฟล์ src/data.py โดยตรง จะรันจากโฟลเดอร์ไหนก็เจอ)
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "prices.db"


def load_price_matrix(db_path=None, price_col="adj_close"):
    """
    ดึงข้อมูลราคาจาก SQLite แล้ว pivot เป็น Price Matrix (Index: Date, Columns: Ticker)
    คืนค่าเป็น DataFrame พร้อมสำหรับการคำนวณผลตอบแทนและความเสี่ยง
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    conn = sqlite3.connect(str(db_path))
    query = f"SELECT date, ticker, {price_col} FROM prices ORDER BY date, ticker"
    df = pd.read_sql(query, conn)
    conn.close()

    # แปลงจาก Long format เป็น Price Matrix (Date x Tickers)
    price_matrix = df.pivot(index="date", columns="ticker", values=price_col)
    price_matrix.index = pd.to_datetime(price_matrix.index)

    return price_matrix


def get_tickers_info(db_path=None):
    """ดึงข้อมูลรายชื่อหุ้นและ Sector จากตาราง tickers"""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql("SELECT * FROM tickers ORDER BY sector, ticker", conn)
    conn.close()
    return df
