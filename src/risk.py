"""
Risk and performance analytics module for Thai Equity Portfolio.
"""
import numpy as np
import pandas as pd


def calculate_annualized_return(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """คำนวณผลตอบแทนเฉลี่ยต่อปี (Annualized Return)"""
    return returns.mean() * periods_per_year


def calculate_annualized_volatility(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """คำนวณความผันผวนต่อปี (Annualized Volatility) ด้วยกฎ Square Root of Time"""
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: pd.DataFrame, risk_free_rate: float = 0.025, periods_per_year: int = 252
) -> pd.Series:
    """
    คำนวณ Sharpe Ratio: (Annualized Return - Risk Free Rate) / Annualized Volatility
    กำหนด default Risk-Free Rate = 2.5% ตามตั๋วเงินคลัง/ดอกเบี้ยนโยบายไทย
    """
    ann_ret = calculate_annualized_return(returns, periods_per_year)
    ann_vol = calculate_annualized_volatility(returns, periods_per_year)
    return (ann_ret - risk_free_rate) / ann_vol


def calculate_drawdowns(returns: pd.DataFrame) -> pd.DataFrame:
    """คำนวณเส้นทาง Drawdown สะสมข้ามเวลา (Peak-to-Trough)"""
    wealth_index = (1 + returns).cumprod()
    running_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - running_peaks) / running_peaks
    return drawdowns


def calculate_max_drawdown(returns: pd.DataFrame) -> pd.Series:
    """คำนวณ Maximum Drawdown (การขาดทุนลึกที่สุดจากจุดสูงสุดเดิม)"""
    drawdowns = calculate_drawdowns(returns)
    return drawdowns.min()
