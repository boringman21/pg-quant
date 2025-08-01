# 📈 Logarithm trong Finance - First Principles

## 🧠 First Principles: Tại Sao Log Returns?

**Câu hỏi cốt lõi**: Tại sao không dùng simple returns (R = P1/P0 - 1) mà lại dùng log returns (r = ln(P1/P0))?

### 💡 Hiểu Bản Chất

**Logarithm không phải là trick toán học** - nó giải quyết **fundamental problems** trong finance:

1. **Asymmetry Problem**: 50% gain cần 33.3% loss để về lại gốc
2. **Compounding Issue**: Phép cộng thay vì phép nhân phức tạp  
3. **Statistical Properties**: Normal distribution assumptions
4. **Time Aggregation**: Easy scaling across timeframes

## 📊 The Problems with Simple Returns

### ⚠️ Problem 1: Asymmetry

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example: $100 investment
initial_price = 100

# Scenario A: +50% then -33.33%
price_up = 100 * 1.5      # = $150
price_back = 150 * 0.6667 # = $100 (back to original)

simple_return_1 = (150 - 100) / 100    # = 0.5 (50%)
simple_return_2 = (100 - 150) / 150    # = -0.333 (-33.33%)
total_simple = simple_return_1 + simple_return_2  # = 0.167 (16.7%!) ❌

# But actual return is 0%!
actual_return = (100 - 100) / 100  # = 0%

print(f"Simple returns sum: {total_simple:.1%}")
print(f"Actual return: {actual_return:.1%}")
print(f"Error: {total_simple - actual_return:.1%}")
```

### ⚠️ Problem 2: Bounds Issue

```python
# Simple returns have asymmetric bounds
# Gains: 0% to +∞ (unlimited upside)
# Losses: -100% to 0% (limited downside)

# This creates statistical problems
gains = [0.1, 0.2, 0.5, 1.0, 2.0]  # 10% to 200%
losses = [-0.1, -0.2, -0.5, -0.9]  # 10% to 90% loss (can't go below -100%)

print("Simple returns distribution is skewed by design!")
```

## ✅ Log Returns: The Solution

### 🎯 Definition & Properties

```python
import math

def simple_return(p0, p1):
    """Simple return: (P1 - P0) / P0"""
    return (p1 - p0) / p0

def log_return(p0, p1):
    """Log return: ln(P1/P0)"""
    return math.log(p1 / p0)

# Example prices
p0 = 100
p1 = 150

simple_ret = simple_return(p0, p1)
log_ret = log_return(p0, p1)

print(f"Simple return: {simple_ret:.4f} ({simple_ret:.1%})")
print(f"Log return: {log_ret:.4f} ({log_ret:.1%})")
```

### 🎯 Key Properties of Log Returns

#### 1. **Additive Property** 🔥

```python
# Multi-period returns
prices = [100, 110, 121, 133.1]  # 10% growth each period

# Simple returns (multiplicative)
simple_returns = []
for i in range(1, len(prices)):
    simple_returns.append(simple_return(prices[i-1], prices[i]))

total_simple = 1
for r in simple_returns:
    total_simple *= (1 + r)
total_simple -= 1

print(f"Simple returns: {simple_returns}")
print(f"Total simple return (multiplicative): {total_simple:.1%}")

# Log returns (additive) ✨
log_returns = []
for i in range(1, len(prices)):
    log_returns.append(log_return(prices[i-1], prices[i]))

total_log = sum(log_returns)

print(f"Log returns: {[f'{r:.4f}' for r in log_returns]}")
print(f"Total log return (additive): {total_log:.4f}")
print(f"Converting back: {(math.exp(total_log) - 1):.1%}")
```

#### 2. **Symmetric Bounds** ⚖️

```python
# Log returns are symmetric around 0
price_changes = [0.5, 0.8, 1.0, 1.25, 2.0]  # 50% loss to 100% gain

print("Price Ratio | Simple Return | Log Return")
print("-" * 40)
for ratio in price_changes:
    simple = ratio - 1
    log_ret = math.log(ratio)
    print(f"{ratio:9.2f} | {simple:11.1%} | {log_ret:8.4f}")
```

#### 3. **Time Scaling** ⏰

```python
# Easy aggregation across time periods
daily_log_return = 0.001    # 0.1% daily
trading_days = 252

# Annual return (additive property)
annual_log_return = daily_log_return * trading_days
annual_simple_return = math.exp(annual_log_return) - 1

print(f"Daily log return: {daily_log_return:.3f}")
print(f"Annual log return: {annual_log_return:.3f}")
print(f"Annual simple return: {annual_simple_return:.1%}")
```

## 📈 Real-World Applications

### 🎯 Portfolio Returns

```python
import pandas as pd
import yfinance as yf

# Download stock data
symbols = ['AAPL', 'GOOGL', 'TSLA']
data = yf.download(symbols, period='1y')['Adj Close']

# Calculate different return types
simple_returns = data.pct_change().dropna()
log_returns = np.log(data / data.shift(1)).dropna()

# Portfolio weights
weights = np.array([0.4, 0.3, 0.3])

# Portfolio returns
portfolio_simple = (simple_returns * weights).sum(axis=1)
portfolio_log = (log_returns * weights).sum(axis=1)

# Compare properties
print("Portfolio Return Statistics:")
print(f"Simple Returns - Mean: {portfolio_simple.mean():.4f}, Std: {portfolio_simple.std():.4f}")
print(f"Log Returns - Mean: {portfolio_log.mean():.4f}, Std: {portfolio_log.std():.4f}")

# Cumulative returns
cumulative_simple = (1 + portfolio_simple).cumprod()
cumulative_log = np.exp(portfolio_log.cumsum())

print(f"\nFinal cumulative return (Simple): {cumulative_simple.iloc[-1] - 1:.1%}")
print(f"Final cumulative return (Log): {cumulative_log.iloc[-1] - 1:.1%}")
```

### 🔬 Statistical Analysis

```python
from scipy import stats
import seaborn as sns

# Generate sample data
np.random.seed(42)
prices = [100]
for _ in range(1000):
    # Random walk with log returns
    log_return = np.random.normal(0.0005, 0.02)  # 0.05% mean, 2% vol
    new_price = prices[-1] * math.exp(log_return)
    prices.append(new_price)

prices = np.array(prices)
simple_rets = (prices[1:] - prices[:-1]) / prices[:-1]
log_rets = np.log(prices[1:] / prices[:-1])

# Test normality
simple_shapiro = stats.shapiro(simple_rets)[1]  # p-value
log_shapiro = stats.shapiro(log_rets)[1]        # p-value

print(f"Normality test (p-value):")
print(f"Simple returns: {simple_shapiro:.6f}")
print(f"Log returns: {log_shapiro:.6f}")
print(f"Log returns are {'more' if log_shapiro > simple_shapiro else 'less'} normal")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Simple returns distribution
ax1.hist(simple_rets, bins=50, alpha=0.7, density=True)
ax1.set_title('Simple Returns Distribution')
ax1.set_xlabel('Simple Return')
ax1.set_ylabel('Density')

# Log returns distribution  
ax2.hist(log_rets, bins=50, alpha=0.7, density=True)
ax2.set_title('Log Returns Distribution')
ax2.set_xlabel('Log Return')
ax2.set_ylabel('Density')

plt.tight_layout()
plt.show()
```

### 🎲 Monte Carlo Simulation

```python
def simulate_price_path(S0, mu, sigma, T, dt, use_log=True):
    """
    Simulate stock price path using geometric Brownian motion
    
    S0: Initial price
    mu: Expected return (annualized)
    sigma: Volatility (annualized)
    T: Time horizon (years)
    dt: Time step (years)
    use_log: Use log returns (True) or simple returns (False)
    """
    n_steps = int(T / dt)
    prices = [S0]
    
    for _ in range(n_steps):
        if use_log:
            # Log-normal model (standard GBM)
            log_return = np.random.normal(mu * dt, sigma * np.sqrt(dt))
            new_price = prices[-1] * math.exp(log_return)
        else:
            # Simple return model (non-standard)
            simple_return = np.random.normal(mu * dt, sigma * np.sqrt(dt))
            new_price = prices[-1] * (1 + simple_return)
            
        prices.append(max(new_price, 0.01))  # Prevent negative prices
    
    return np.array(prices)

# Compare simulations
S0 = 100
mu = 0.08    # 8% expected return
sigma = 0.2  # 20% volatility
T = 1        # 1 year
dt = 1/252   # Daily steps

# Run multiple simulations
n_sims = 1000
final_prices_log = []
final_prices_simple = []

for _ in range(n_sims):
    path_log = simulate_price_path(S0, mu, sigma, T, dt, use_log=True)
    path_simple = simulate_price_path(S0, mu, sigma, T, dt, use_log=False)
    
    final_prices_log.append(path_log[-1])
    final_prices_simple.append(path_simple[-1])

# Compare results
print(f"Monte Carlo Results (1000 simulations):")
print(f"Log model - Mean final price: ${np.mean(final_prices_log):.2f}")
print(f"Simple model - Mean final price: ${np.mean(final_prices_simple):.2f}")
print(f"Theoretical expectation: ${S0 * math.exp(mu * T):.2f}")
```

## 🤖 Advanced Applications

### 📊 Volatility Modeling (GARCH)

```python
from arch import arch_model

# Get real data
symbol = 'AAPL'
data = yf.download(symbol, period='2y')['Adj Close']
log_returns = np.log(data / data.shift(1)).dropna() * 100  # Convert to percentage

# Fit GARCH model (requires log returns)
garch_model = arch_model(log_returns, vol='GARCH', p=1, q=1)
garch_fit = garch_model.fit()

print("GARCH Model Summary:")
print(garch_fit.summary())

# Forecast volatility
forecast = garch_fit.forecast(horizon=5)
print(f"\nVolatility forecast (next 5 days): {forecast.variance.iloc[-1].values}")
```

### 🎯 Value at Risk (VaR) Calculation

```python
def calculate_var(returns, confidence_level=0.05, use_log=True):
    """
    Calculate Value at Risk
    """
    if use_log:
        # Convert log returns to simple returns for VaR interpretation
        simple_rets = np.exp(returns) - 1
        var = np.percentile(simple_rets, confidence_level * 100)
    else:
        var = np.percentile(returns, confidence_level * 100)
    
    return var

# Example with portfolio data
portfolio_log_returns = portfolio_log  # From previous example
portfolio_simple_returns = portfolio_simple

var_log = calculate_var(portfolio_log_returns, use_log=True)
var_simple = calculate_var(portfolio_simple_returns, use_log=False)

print(f"VaR (95% confidence):")
print(f"From log returns: {var_log:.2%}")
print(f"From simple returns: {var_simple:.2%}")
```

## 🚀 2025 Modern Applications

### 🤖 Machine Learning with Returns

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def prepare_ml_features(returns, lags=5):
    """Prepare lagged features for ML"""
    features = []
    targets = []
    
    for i in range(lags, len(returns)):
        # Features: past returns
        feature_vector = returns[i-lags:i].tolist()
        features.append(feature_vector)
        
        # Target: next period return
        targets.append(returns[i])
    
    return np.array(features), np.array(targets)

# Prepare data with log returns (better for ML)
X, y = prepare_ml_features(log_returns['AAPL'].values)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

print(f"ML Model Performance (using log returns):")
print(f"Training R²: {train_score:.4f}")
print(f"Testing R²: {test_score:.4f}")
```

### 📡 Real-time Log Returns

```python
import asyncio
import websockets
import json

class LogReturnCalculator:
    def __init__(self):
        self.prices = {}
        self.log_returns = {}
    
    async def process_price_update(self, symbol, price):
        """Process real-time price updates"""
        if symbol in self.prices:
            # Calculate log return
            log_return = math.log(price / self.prices[symbol])
            
            # Store log return
            if symbol not in self.log_returns:
                self.log_returns[symbol] = []
            self.log_returns[symbol].append(log_return)
            
            # Real-time risk metrics
            if len(self.log_returns[symbol]) > 20:
                recent_returns = self.log_returns[symbol][-20:]
                volatility = np.std(recent_returns) * math.sqrt(252)  # Annualized
                
                print(f"{symbol}: Price={price:.2f}, Log Return={log_return:.4f}, Vol={volatility:.1%}")
        
        self.prices[symbol] = price

# Usage example
calculator = LogReturnCalculator()

# Simulate price updates
async def simulate_price_stream():
    symbols = ['AAPL', 'GOOGL', 'TSLA']
    base_prices = {'AAPL': 150, 'GOOGL': 2500, 'TSLA': 800}
    
    for symbol in symbols:
        calculator.prices[symbol] = base_prices[symbol]
    
    for _ in range(100):
        for symbol in symbols:
            # Simulate price movement
            log_return = np.random.normal(0, 0.02)  # 2% daily vol
            new_price = calculator.prices[symbol] * math.exp(log_return)
            
            await calculator.process_price_update(symbol, new_price)
            await asyncio.sleep(0.1)

# Run simulation
# asyncio.run(simulate_price_stream())
```

## 💎 Key Takeaways

### 🎯 When to Use Log Returns

✅ **Always use for:**
- Multi-period analysis
- Statistical modeling (GARCH, etc.)
- Machine learning features
- Risk calculations
- Performance attribution

✅ **Advantages:**
- Additive across time
- Symmetric distribution
- Better statistical properties
- Easy aggregation

### ⚠️ When Simple Returns Still Matter

- **Interpretation**: Easier to understand (10% gain vs 0.0953 log return)
- **Benchmarking**: Industry standards often use simple returns
- **Short periods**: Minimal difference for small returns

### 🔧 Practical Implementation

```python
# Standard workflow
def analyze_returns(prices):
    """Complete return analysis workflow"""
    
    # Calculate both types
    simple_returns = prices.pct_change().dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    # Use log returns for analysis
    volatility = log_returns.std() * np.sqrt(252)
    
    # Convert back to simple for interpretation
    total_simple_return = np.exp(log_returns.sum()) - 1
    
    return {
        'log_returns': log_returns,
        'simple_returns': simple_returns,
        'volatility': volatility,
        'total_return': total_simple_return
    }

# Example usage
# results = analyze_returns(data['AAPL'])
```

---

**Next**: [[01-Toán-học-và-thống-kê/Tỷ Suất Sinh Lời - Hiểu Sâu|Understanding Returns]]

**Advanced**: [[01-Toán-học-và-thống-kê/Black Scholes - Từ Đầu|Black-Scholes Math]]

---

*"In finance, the logarithm is not just a mathematical tool - it's the natural language of growth"* 📈

*"Log returns: Where addition becomes multiplication, and complexity becomes simplicity"* ✨
