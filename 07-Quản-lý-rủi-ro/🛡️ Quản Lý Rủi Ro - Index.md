# 🛡️ Quản Lý Rủi Ro - Index

## 🧠 First Principles: Survival First, Profit Second

**Câu hỏi cốt lõi**: Tại sao 90% traders thua lỗ? Không phải vì họ không biết kiếm tiền - mà vì họ không biết **bảo vệ tiền**!

### 💡 Philosophy của Risk Management

**Risk không phải là enemy** - nó là **dance partner**:

- **Risk = Uncertainty** về future outcomes
- **Reward** chỉ tồn tại khi có risk
- **Management** = Control what you can control
- **Mathematical edge** + Risk control = Long-term success

## 🎯 Mục Tiêu Chương Này

### 🛡️ From Gambling to Trading

```
Gambling → Risk Awareness → Risk Measurement → Risk Control → Risk Optimization
```

**Không tránh risk** - học cách **dance with risk**!

## 📚 Risk Management Mastery (6 Tuần)

### 🏗️ Week 1: Risk Fundamentals

#### 1.1 🎯 Understanding Risk Types
- [[07-Quản-lý-rủi-ro/Market Risk - Systematic Unsystematic|Market Risk Breakdown]]
- [[07-Quản-lý-rủi-ro/Credit Risk - Counterparty Default|Credit & Counterparty Risk]]
- [[07-Quản-lý-rủi-ro/Liquidity Risk - Market Impact|Liquidity Risk Analysis]]
- [[07-Quản-lý-rủi-ro/Operational Risk - System Failures|Operational Risk Control]]

#### 1.2 📊 Risk Measurement Basics
- [[07-Quản-lý-rủi-ro/Volatility - Standard Deviation Deep|Volatility Analysis]]
- [[07-Quản-lý-rủi-ro/Correlation - Portfolio Diversification|Correlation & Diversification]]
- [[07-Quản-lý-rủi-ro/Beta - Market Sensitivity|Beta & Market Risk]]

### 🏗️ Week 2: Position Sizing & Capital Allocation

#### 2.1 💰 Position Sizing Methods
- [[07-Quản-lý-rủi-ro/Fixed Fractional - Classic Approach|Fixed Fractional Method]]
- [[07-Quản-lý-rủi-ro/Kelly Criterion - Optimal Sizing|Kelly Criterion Deep Dive]]
- [[07-Quản-lý-rủi-ro/Volatility Scaling - Dynamic Sizing|Volatility-Based Sizing]]

#### 2.2 🎯 Capital Allocation Strategies
- [[07-Quản-lý-rủi-ro/Equal Weight vs Risk Parity|Portfolio Allocation Methods]]
- [[07-Quản-lý-rủi-ro/Maximum Drawdown Control|Drawdown-Based Allocation]]
- [[07-Quản-lý-rủi-ro/Dynamic Hedging - Options Futures|Dynamic Hedging Strategies]]

### 🏗️ Week 3: Value at Risk (VaR) & Risk Metrics

#### 3.1 📊 VaR Methodologies
- [[07-Quản-lý-rủi-ro/Historical VaR - Empirical Approach|Historical VaR]]
- [[07-Quản-lý-rủi-ro/Monte Carlo VaR - Simulation|Monte Carlo VaR]]
- [[07-Quản-lý-rủi-ro/Parametric VaR - Normal Distribution|Parametric VaR]]

#### 3.2 🔍 Advanced Risk Metrics
- [[07-Quản-lý-rủi-ro/Expected Shortfall - CVaR|Conditional VaR (CVaR)]]
- [[07-Quản-lý-rủi-ro/Maximum Drawdown - Path Dependent|Maximum Drawdown Analysis]]
- [[07-Quản-lý-rủi-ro/Sharpe Sortino Calmar - Risk Adjusted|Risk-Adjusted Returns]]

### 🏗️ Week 4: Portfolio Risk Management

#### 4.1 🔄 Diversification Strategies
- [[07-Quản-lý-rủi-ro/Asset Class Diversification|Asset Class Diversification]]
- [[07-Quản-lý-rủi-ro/Geographic Diversification|Geographic Diversification]]
- [[07-Quản-lý-rủi-ro/Time Diversification - Dollar Cost|Time-Based Diversification]]

#### 4.2 📈 Portfolio Optimization
- [[07-Quản-lý-rủi-ro/Mean Variance Optimization - Markowitz|Markowitz Optimization]]
- [[07-Quản-lý-rủi-ro/Black Litterman - Bayesian Approach|Black-Litterman Model]]
- [[07-Quản-lý-rủi-ro/Risk Budgeting - Contribution|Risk Budgeting Framework]]

### 🏗️ Week 5: Derivatives & Hedging

#### 5.1 🛡️ Options for Risk Management
- [[07-Quản-lý-rủi-ro/Protective Puts - Downside Protection|Protective Put Strategies]]
- [[07-Quản-lý-rủi-ro/Covered Calls - Income Generation|Covered Call Strategies]]
- [[07-Quản-lý-rủi-ro/Collars - Range Bound Protection|Collar Strategies]]

#### 5.2 📈 Futures & Swaps Hedging
- [[07-Quản-lý-rủi-ro/Index Futures - Market Hedging|Index Futures Hedging]]
- [[07-Quản-lý-rủi-ro/Currency Hedging - FX Risk|Currency Risk Management]]
- [[07-Quản-lý-rủi-ro/Interest Rate Swaps - Duration|Interest Rate Hedging]]

### 🏗️ Week 6: Behavioral Risk & Psychology

#### 6.1 🧠 Behavioral Biases
- [[07-Quản-lý-rủi-ro/Loss Aversion - Prospect Theory|Loss Aversion Psychology]]
- [[07-Quản-lý-rủi-ro/Overconfidence - Dunning Kruger|Overconfidence Bias]]
- [[07-Quản-lý-rủi-ro/Anchoring - Reference Points|Anchoring & Mental Accounting]]

#### 6.2 🎯 Risk Psychology Management
- [[07-Quản-lý-rủi-ro/Systematic Decision Making|Systematic Decision Frameworks]]
- [[07-Quản-lý-rủi-ro/Stress Testing - Scenario Analysis|Stress Testing Methods]]
- [[07-Quản-lý-rủi-ro/Risk Culture - Organization|Building Risk Culture]]

## 🛠️ Risk Management Toolkit

### 📊 Python Libraries for Risk

```python
# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis
import scipy.stats as stats
import scipy.optimize as optimize
from sklearn.covariance import LedoitWolf

# Risk-specific libraries
import pyfolio               # Portfolio performance analysis
import empyrical            # Financial risk metrics
import quantlib as ql       # Quantitative finance library
import riskfolio as rp      # Portfolio optimization

# Monte Carlo & simulation
import monte_carlo          # Monte Carlo methods
import arch                 # GARCH models for volatility

# Backtesting with risk focus
import zipline              # Algorithmic trading
import backtrader          # Alternative backtesting
import vectorbt            # High-performance backtesting

# Options & derivatives
import mibian              # Options pricing
import py_vollib           # Volatility & Greeks
```

### 🎯 Core Risk Functions

```python
def calculate_var(returns, confidence_level=0.05):
    """
    Calculate Value at Risk using historical method
    """
    return np.percentile(returns, confidence_level * 100)

def calculate_cvar(returns, confidence_level=0.05):
    """
    Calculate Conditional Value at Risk (Expected Shortfall)
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_maximum_drawdown(returns):
    """
    Calculate Maximum Drawdown
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

def kelly_fraction(win_rate, avg_win, avg_loss):
    """
    Calculate Kelly Criterion optimal position size
    """
    win_loss_ratio = abs(avg_win / avg_loss)
    return (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

def portfolio_var(weights, cov_matrix, confidence_level=0.05):
    """
    Calculate portfolio VaR using covariance matrix
    """
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return stats.norm.ppf(confidence_level) * portfolio_std
```

## 📈 Real-World Applications

### 🎯 Project 1: Risk Dashboard

```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

class RiskDashboard:
    def __init__(self):
        self.portfolio = {}
        
    def create_dashboard(self):
        st.title("🛡️ Portfolio Risk Dashboard")
        
        # Portfolio input
        symbols = st.sidebar.multiselect(
            "Select stocks:", 
            ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "SPY"]
        )
        
        if symbols:
            # Fetch data
            data = yf.download(symbols, period="2y")['Adj Close']
            returns = data.pct_change().dropna()
            
            # Risk metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                portfolio_var = self.calculate_portfolio_var(returns)
                st.metric("Portfolio VaR (95%)", f"{portfolio_var:.2%}")
                
            with col2:
                portfolio_cvar = self.calculate_portfolio_cvar(returns)
                st.metric("Portfolio CVaR", f"{portfolio_cvar:.2%}")
                
            with col3:
                max_dd = self.calculate_max_drawdown(returns)
                st.metric("Max Drawdown", f"{max_dd:.2%}")
                
            with col4:
                sharpe_ratio = self.calculate_sharpe_ratio(returns)
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            # Risk decomposition chart
            self.plot_risk_contribution(returns)
            
            # Correlation heatmap
            self.plot_correlation_matrix(returns)
    
    def calculate_portfolio_var(self, returns, confidence=0.05):
        """Calculate equal-weighted portfolio VaR"""
        portfolio_returns = returns.mean(axis=1)
        return np.percentile(portfolio_returns, confidence * 100)
    
    def plot_risk_contribution(self, returns):
        """Plot individual stock risk contributions"""
        individual_vars = []
        for col in returns.columns:
            var = np.percentile(returns[col], 5)
            individual_vars.append(var)
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(returns.columns, individual_vars)
        ax.set_title("Individual Stock VaR Contributions")
        ax.set_ylabel("VaR (5%)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
```

### 🤖 Project 2: Dynamic Position Sizing

```python
class DynamicPositionSizer:
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.positions = {}
        self.risk_per_trade = 0.02  # 2% risk per trade
        
    def calculate_position_size(self, symbol, entry_price, stop_loss, method='fixed_risk'):
        """
        Calculate position size based on different methods
        """
        if method == 'fixed_risk':
            return self.fixed_risk_sizing(entry_price, stop_loss)
        elif method == 'kelly':
            return self.kelly_sizing(symbol)
        elif method == 'volatility':
            return self.volatility_sizing(symbol)
        else:
            return self.fixed_fractional_sizing()
    
    def fixed_risk_sizing(self, entry_price, stop_loss):
        """Fixed risk per trade sizing"""
        risk_amount = self.capital * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        position_size = risk_amount / price_risk
        return min(position_size, self.capital * 0.1)  # Max 10% of capital
    
    def kelly_sizing(self, symbol):
        """Kelly criterion sizing based on historical performance"""
        # Get historical data for the symbol
        data = yf.download(symbol, period="2y")
        returns = data['Adj Close'].pct_change().dropna()
        
        # Calculate win rate and avg win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / len(returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        if avg_loss == 0:
            return 0
            
        kelly_f = kelly_fraction(win_rate, avg_win, avg_loss)
        return min(max(kelly_f * self.capital, 0), self.capital * 0.25)  # Cap at 25%
    
    def volatility_sizing(self, symbol):
        """Size position inverse to volatility"""
        data = yf.download(symbol, period="6m")
        returns = data['Adj Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Inverse volatility sizing
        target_volatility = 0.15  # 15% target
        size_multiplier = target_volatility / volatility
        return min(size_multiplier * self.capital * 0.1, self.capital * 0.2)
```

### 📊 Project 3: Stress Testing Framework

```python
import scipy.stats as stats

class StressTester:
    def __init__(self, portfolio_returns):
        self.returns = portfolio_returns
        
    def run_stress_tests(self):
        """Run comprehensive stress testing"""
        results = {}
        
        # Historical stress scenarios
        results['2008_crisis'] = self.historical_stress('2008-09-01', '2009-03-01')
        results['covid_crash'] = self.historical_stress('2020-02-01', '2020-04-01')
        results['dot_com_bubble'] = self.historical_stress('2000-01-01', '2002-12-31')
        
        # Monte Carlo stress scenarios
        results['monte_carlo'] = self.monte_carlo_stress(num_simulations=10000)
        
        # Tail risk scenarios
        results['tail_risk'] = self.tail_risk_scenarios()
        
        return results
    
    def historical_stress(self, start_date, end_date):
        """Apply historical stress scenario"""
        # Get market data for stress period
        spy_data = yf.download('SPY', start=start_date, end=end_date)
        stress_returns = spy_data['Adj Close'].pct_change().dropna()
        
        # Apply correlation to portfolio
        correlation = np.corrcoef(self.returns, stress_returns[self.returns.index])[0,1]
        stressed_returns = self.returns * correlation
        
        return {
            'total_return': (1 + stressed_returns).prod() - 1,
            'volatility': stressed_returns.std() * np.sqrt(252),
            'max_drawdown': self.calculate_max_drawdown(stressed_returns),
            'var_95': np.percentile(stressed_returns, 5)
        }
    
    def monte_carlo_stress(self, num_simulations=10000):
        """Monte Carlo stress testing"""
        mean_return = self.returns.mean()
        std_return = self.returns.std()
        
        # Generate scenarios
        scenarios = np.random.normal(mean_return, std_return * 2, 
                                   (num_simulations, len(self.returns)))
        
        # Calculate metrics for each scenario
        scenario_results = []
        for scenario in scenarios:
            total_return = (1 + scenario).prod() - 1
            max_dd = self.calculate_max_drawdown(pd.Series(scenario))
            scenario_results.append({'return': total_return, 'max_dd': max_dd})
        
        results_df = pd.DataFrame(scenario_results)
        
        return {
            'worst_1_percent': results_df['return'].quantile(0.01),
            'worst_5_percent': results_df['return'].quantile(0.05),
            'average_worst_case': results_df['return'].mean(),
            'max_drawdown_99': results_df['max_dd'].quantile(0.99)
        }
    
    def tail_risk_scenarios(self):
        """Extreme tail risk scenarios"""
        # Fat tail simulation using Student's t-distribution
        df = 3  # Degrees of freedom for heavy tails
        
        tail_scenarios = stats.t.rvs(df, size=1000) * self.returns.std()
        
        return {
            'extreme_loss_1_in_100': np.percentile(tail_scenarios, 1),
            'extreme_loss_1_in_1000': np.percentile(tail_scenarios, 0.1),
            'expected_shortfall_1%': tail_scenarios[tail_scenarios <= np.percentile(tail_scenarios, 1)].mean()
        }
```

## 🚀 2025 Advanced Risk Management

### 🤖 AI-Enhanced Risk Management

```python
import tensorflow as tf
from sklearn.ensemble import IsolationForest

class AIRiskManager:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.risk_predictor = self.build_risk_model()
        
    def build_risk_model(self):
        """Build LSTM model for risk prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 5)),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Risk probability
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def detect_market_anomalies(self, market_data):
        """Detect unusual market conditions"""
        features = self.extract_risk_features(market_data)
        anomaly_scores = self.anomaly_detector.decision_function(features)
        
        # Flag high-risk periods
        risk_threshold = np.percentile(anomaly_scores, 10)  # Bottom 10%
        high_risk_periods = anomaly_scores < risk_threshold
        
        return high_risk_periods, anomaly_scores
    
    def predict_portfolio_risk(self, portfolio_data):
        """Predict portfolio risk using AI"""
        features = self.prepare_features(portfolio_data)
        risk_probability = self.risk_predictor.predict(features)
        
        # Convert to risk categories
        risk_levels = np.where(risk_probability > 0.7, 'HIGH',
                      np.where(risk_probability > 0.4, 'MEDIUM', 'LOW'))
        
        return risk_levels, risk_probability
```

### 📡 Real-Time Risk Monitoring

```python
import asyncio
import websocket
import json

class RealTimeRiskMonitor:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.risk_limits = {
            'var_limit': 0.05,      # 5% daily VaR limit
            'drawdown_limit': 0.15,  # 15% max drawdown limit
            'concentration_limit': 0.20  # 20% max single position
        }
        self.alerts = []
        
    async def monitor_portfolio_risk(self):
        """Real-time portfolio risk monitoring"""
        while True:
            try:
                # Get real-time prices
                current_prices = await self.fetch_real_time_prices()
                
                # Calculate current portfolio metrics
                current_var = self.calculate_real_time_var(current_prices)
                current_drawdown = self.calculate_current_drawdown(current_prices)
                position_concentrations = self.calculate_concentrations(current_prices)
                
                # Check risk limits
                await self.check_risk_limits(current_var, current_drawdown, position_concentrations)
                
                # Update risk dashboard
                await self.update_risk_dashboard({
                    'var': current_var,
                    'drawdown': current_drawdown,
                    'concentrations': position_concentrations,
                    'timestamp': pd.Timestamp.now()
                })
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                print(f"Risk monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def check_risk_limits(self, var, drawdown, concentrations):
        """Check if any risk limits are breached"""
        alerts = []
        
        if abs(var) > self.risk_limits['var_limit']:
            alerts.append({
                'type': 'VAR_BREACH',
                'current': var,
                'limit': self.risk_limits['var_limit'],
                'severity': 'HIGH'
            })
        
        if abs(drawdown) > self.risk_limits['drawdown_limit']:
            alerts.append({
                'type': 'DRAWDOWN_BREACH',
                'current': drawdown,
                'limit': self.risk_limits['drawdown_limit'],
                'severity': 'CRITICAL'
            })
        
        max_concentration = max(concentrations.values())
        if max_concentration > self.risk_limits['concentration_limit']:
            alerts.append({
                'type': 'CONCENTRATION_BREACH',
                'current': max_concentration,
                'limit': self.risk_limits['concentration_limit'],
                'severity': 'MEDIUM'
            })
        
        if alerts:
            await self.send_risk_alerts(alerts)
    
    async def send_risk_alerts(self, alerts):
        """Send risk alerts via multiple channels"""
        for alert in alerts:
            # Email alert
            await self.send_email_alert(alert)
            
            # Slack/Discord notification
            await self.send_chat_alert(alert)
            
            # SMS for critical alerts
            if alert['severity'] == 'CRITICAL':
                await self.send_sms_alert(alert)
            
            # Auto-hedging for extreme cases
            if alert['severity'] == 'CRITICAL':
                await self.execute_emergency_hedge()
```

## ✅ Risk Management Progression

### Week 1: Foundation ✅
- [ ] Understand different risk types
- [ ] Calculate basic risk metrics
- [ ] Identify portfolio risk sources
- [ ] Build risk measurement tools

### Week 2: Position Sizing ✅
- [ ] Master Kelly Criterion
- [ ] Implement dynamic sizing
- [ ] Optimize capital allocation
- [ ] Build position sizing calculator

### Week 3: VaR & Advanced Metrics ✅
- [ ] Calculate different VaR methods
- [ ] Understand CVaR and drawdowns
- [ ] Implement risk-adjusted returns
- [ ] Build comprehensive risk metrics

### Week 4: Portfolio Risk ✅
- [ ] Apply Markowitz optimization
- [ ] Implement diversification strategies
- [ ] Build risk budgeting framework
- [ ] Create portfolio risk analyzer

### Week 5: Derivatives Hedging ✅
- [ ] Use options for protection
- [ ] Implement futures hedging
- [ ] Build dynamic hedging strategies
- [ ] Create hedging calculator

### Week 6: Behavioral & Advanced ✅
- [ ] Understand behavioral biases
- [ ] Implement stress testing
- [ ] Build AI risk models
- [ ] Create real-time monitoring

## 💎 Risk Management Principles

### 🎯 Core Principles

1. **Risk First, Return Second** - Protect capital before seeking profits
2. **Measure What Matters** - Focus on actionable risk metrics
3. **Diversification is Free** - The only free lunch in finance
4. **Position Size = Risk Control** - Size positions based on risk, not conviction
5. **Prepare for Black Swans** - Expect the unexpected

### 🚀 2025 Advanced Principles

1. **AI-Enhanced Detection** - Use AI to spot risks humans miss
2. **Real-Time Monitoring** - Risk changes by the second
3. **Alternative Data Integration** - New risk signals from new data
4. **Quantum Computing** - Solve complex risk optimization problems
5. **Decentralized Risk** - New risks from DeFi and crypto

---

**Next**: [[07-Quản-lý-rủi-ro/Market Risk - Systematic Unsystematic|Start with Market Risk]]

**Advanced**: [[08-Backtest-và-optimization/🔍 Explainable AI (XAI)|XAI for Risk]]

---

*"Risk comes from not knowing what you're doing"* - Warren Buffett 🛡️

*"In investing, what is comfortable is rarely profitable"* - Robert Arnott 📈
