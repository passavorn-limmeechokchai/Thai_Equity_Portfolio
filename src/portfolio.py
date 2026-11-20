"""
Portfolio Construction Module
Provides 4 asset allocation strategies:
1. Equal-Weight (EW)
2. Inverse-Volatility (IV)
3. Minimum-Variance (Min-Var)
4. Maximum-Sharpe (Max-Sharpe)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def get_equal_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Equal-Weight Strategy: Allocates 1/N weight to each asset.
    """
    n = returns.shape[1]
    weights = np.repeat(1.0 / n, n)
    return pd.Series(weights, index=returns.columns, name="Equal_Weight")


def get_inverse_vol_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Inverse-Volatility Strategy: Weights inversely proportional to asset volatility.
    Lower volatility assets get higher weights.
    """
    vols = returns.std() * np.sqrt(252)
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return pd.Series(weights, index=returns.columns, name="Inverse_Vol")


def get_min_variance_weights(returns: pd.DataFrame, max_weight: float = 0.20) -> pd.Series:
    """
    Minimum-Variance Strategy: Minimizes portfolio variance w^T * Sigma * w.
    Constraints: sum(w) = 1, 0 <= w_i <= max_weight.
    """
    n = returns.shape[1]
    cov_matrix = returns.cov().values * 252

    def portfolio_variance(weights):
        return weights @ cov_matrix @ weights

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = tuple((0.0, max_weight) for _ in range(n))
    w0 = np.repeat(1.0 / n, n)

    result = minimize(portfolio_variance, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=returns.columns, name="Min_Variance")


def get_max_sharpe_weights(
    returns: pd.DataFrame, risk_free_rate: float = 0.025, max_weight: float = 0.20
) -> pd.Series:
    """
    Maximum-Sharpe (Tangency Portfolio) Strategy:
    Maximizes (R_p - R_f) / sigma_p by minimizing negative Sharpe ratio.
    Constraints: sum(w) = 1, 0 <= w_i <= max_weight.
    """
    n = returns.shape[1]
    mean_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252

    def negative_sharpe(weights):
        p_ret = weights @ mean_returns
        p_vol = np.sqrt(weights @ cov_matrix @ weights)
        if p_vol == 0:
            return 0.0
        return -(p_ret - risk_free_rate) / p_vol

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = tuple((0.0, max_weight) for _ in range(n))
    w0 = np.repeat(1.0 / n, n)

    result = minimize(negative_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=returns.columns, name="Max_Sharpe")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data import load_price_matrix

    print("Testing portfolio construction module...")
    prices = load_price_matrix()
    stock_prices = prices.drop(columns=["TDEX.BK"])
    stock_returns = stock_prices.pct_change().dropna()
    sample_returns = stock_returns.loc["2015-01-01":"2017-12-31"]

    ew = get_equal_weights(sample_returns)
    iv = get_inverse_vol_weights(sample_returns)
    mv = get_min_variance_weights(sample_returns)
    ms = get_max_sharpe_weights(sample_returns)

    df = pd.DataFrame({"EW (%)": ew, "IV (%)": iv, "MinVar (%)": mv, "MaxSharpe (%)": ms}) * 100
    print(df.round(2))
    print("\nWeight Sums:")
    print(df.sum().round(2))

    # =========================================================================
    # DEV NOTES: ทำความเข้าใจ if __name__ == "__main__": และ Path Resolution
    # =========================================================================
    # 1. if __name__ == "__main__":
    #    - ตัวแปร __name__ มี 2 สถานะ:
    #      a) ถ้ารันไฟล์นี้ตรงๆ (python src/portfolio.py) -> __name__ จะมีค่าเป็น "__main__"
    #         บล็อกนี้จึงทำงาน เพื่อให้เราใช้รัน Test ดูผลลัพธ์ได้ทันที
    #      b) ถ้าถูกไฟล์อื่น import ไปใช้ (เช่น from src.portfolio import ...)
    #         __name__ จะมีค่าเป็น "src.portfolio" บล็อกนี้จะไม่ทำงาน
    #         ช่วยป้องกันไม่ให้โค้ดทดสอบข้างล่างนี้แอบรันอัตโนมัติ (No Side Effect)
    #
    # 2. sys.path.append(str(Path(__file__).resolve().parent.parent)):
    #    - เมื่อรันไฟล์ในโฟลเดอร์ย่อย (src/) ทาง CLI ตัว Python จะมอง src/ เป็น root ชั่วคราว
    #    - ทำให้คำสั่ง "from src.data import ..." หาโฟลเดอร์ "src" ไม่เจอ
    #    - การใช้ pathlib ถอยกลับไป 2 ชั้น (.parent.parent) คือการนำโฟลเดอร์โปรเจกต์หลัก
    #      (Thai_Equity_Portfolio) ไปเพิ่มลงใน sys.path ทำให้ import ได้ถูกต้องเสมอ
    # =========================================================================

