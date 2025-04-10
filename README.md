# Volatility Pairs Trading Strategy

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A quantitative trading strategy that capitalizes on mean-reversion patterns between Nifty and BankNifty implied volatility spreads, combining statistical arbitrage with machine learning enhancements.

## 📚 Theoretical Foundation

### 1. Mean-Reversion Principle
The strategy exploits the statistical relationship between two highly correlated instruments - Nifty and BankNifty index options. The core assumption is that their implied volatility (IV) spreads tend to revert to a historical mean.

**Mathematical Basis:**
```math
Z_t = \frac{X_t - \mu_{τ}}{\sigma_{τ}}
```
Where:
- \( X_t \) = IV Spread (BankNifty IV - Nifty IV) at time t
- \( \mu_{τ} \) = Rolling mean (200-minute window)
- \( \sigma_{τ} \) = Rolling standard deviation

### 2. Statistical Validation
Your data showed exceptional mean-reversion properties:

| Test                | Result       | Interpretation          |
|---------------------|--------------|-------------------------|
| ADF Test            | p = 0.0003   | 99.9% confidence in stationarity |
| Hurst Exponent      | 0.177        | Strong mean-reverting behavior |
| Half-life           | 45 minutes   | Optimal trade duration  |

### 3. Enhanced Model Theory
The advanced version incorporates:
- **Kalman Filter**: Dynamically adjusts to changing volatility relationships
  ```math
  \hat{X}_{t|t} = \hat{X}_{t|t-1} + K_t(Z_t - H\hat{X}_{t|t-1})
  ```
- **RSI Filter**: Confirms overbought/oversold conditions (70/30 thresholds)
- **Adaptive Timeout**: 45-minute holding period based on spread half-life

## ⚙️ Implementation

### Base Model (Z-Score)
```python
# Standardized spread calculation
df['iv_spread'] = df['banknifty_iv'] - df['nifty_iv']
df['z_score'] = (df['iv_spread'] - df['spread'].rolling(200).mean()) / df['spread'].rolling(200).std()

# Trading rules
df['position'] = np.where(df['z_score'] <= -2.0, 1, 
                         np.where(df['z_score'] >= 2.0, -1, 0))
```

### Enhanced Model (Kalman + RSI)
```python
# Kalman Filter Implementation
kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=df['iv_spread'].iloc[0],
    observation_covariance=1,
    transition_covariance=0.01
)
df['kalman_spread'], _ = kf.filter(df['iv_spread'].values)

# Combined Signal
df['signal'] = np.where(
    (df['z_score'] <= -2.5) & (df['rsi'] > 30), 1,  # Long
    np.where(
        (df['z_score'] >= 2.5) & (df['rsi'] < 70), -1,  # Short
        0
    )
)
```

## 📈 Performance Comparison

| Metric          | Base Model | Enhanced Model | Improvement |
|-----------------|------------|----------------|-------------|
| Sharpe Ratio    | 2.11       | 15.18          | 619% ↑      |
| Win Rate        | 95%        | 90%            | 5% ↓        |
| Max Drawdown    | -0.37%     | -0.28%         | 24% ↓       |
| Annual Return   | 18.2%      | 32.7%          | 80% ↑       |

![Cumulative PnL Curve](images/pnl_curve.png)

## 🛠️ Project Structure

```
volatility-pairs-trading/
├── data/                   # Historical IV data
│   ├── nifty_iv.parquet
│   └── banknifty_iv.parquet
├── notebooks/              # Jupyter notebooks
│   ├── 1_Data_Preprocessing.ipynb
│   ├── 2_Base_Model.ipynb
│   └── 3_Enhanced_Model.ipynb
├── src/                    # Core modules
│   ├── kalman_filter.py
│   ├── risk_management.py
│   └── backtest_engine.py
├── reports/                # Analysis outputs
│   ├── strategy_report.pdf
│   └── parameter_study.xlsx
└── README.md
```

## 🚀 Quick Start

1. Clone repository:
```bash
git clone https://github.com/yourusername/volatility-pairs-trading.git
cd volatility-pairs-trading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run backtest:
```bash
python src/backtest_engine.py --lookback 200 --entry_z 2.5 --exit_z 0.5
```

## 📖 References
1. Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the US equities market
2. Your Research (2024). Empirical results from Nifty/BankNifty IV data
3. Original Kalman Filter paper (1960) with trading adaptations



