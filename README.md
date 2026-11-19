# Thai Equity Portfolio Backtest & Risk Dashboard

A quantitative portfolio construction, walk-forward backtesting, and risk management pipeline for Thai equities (SET).

## Project Objectives
- **Portfolio Construction:** Implement and evaluate multiple allocation strategies (Equal-Weight, Inverse-Volatility, Minimum-Variance, and Maximum-Sharpe).
- **Realistic Backtesting:** Execute an out-of-sample walk-forward backtest with transaction costs and strict look-ahead bias prevention against the SET benchmark.
- **Risk Analytics & Stress Testing:** Measure tail risk using Historical VaR/CVaR (95%), simulate historical crisis scenarios (e.g., COVID-19 crash), and compute block bootstrap confidence intervals for Sharpe ratios.
- **Predictive Volatility Modeling:** Explore realized volatility forecasting (GARCH, LightGBM, LSTM via PyTorch) to drive volatility-targeted allocation strategies.
- **Interactive Reporting:** Deliver risk and performance insights through an intuitive analytics dashboard.