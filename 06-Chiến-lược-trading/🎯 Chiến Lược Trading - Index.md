# 🎯 Chiến Lược Trading - Index

## 🧠 First Principles: Every Strategy is a Hypothesis About Market Behavior

**Câu hỏi cốt lõi**: Tại sao 90% retail traders lose money? Không phải vì họ không biết technical analysis - mà vì họ **không hiểu market microstructure và psychology**!

### 💡 Philosophy của Trading Strategy Development

**Strategy không phải là rules** - nó là **systematic exploitation của market inefficiencies**:

- **Market Inefficiency** = Information asymmetry + behavioral bias
- **Strategy** = Systematic method to capture alpha
- **Edge** = Statistical advantage over time
- **Risk Management** = Preserving capital to compound returns
- **Psychology** = Managing emotions to execute systematically

## 🎯 Mục Tiêu Chương Này

### 🔄 From Pattern Chaser to Strategy Developer

```
Pattern Recognition → Market Understanding → Strategy Development → Systematic Execution → Alpha Generation
```

**Không trade patterns** - **trade probabilities với defined edge**!

## 📚 Trading Strategy Mastery (10 Tuần)

### 🏗️ Week 1-2: Strategy Development Foundations

#### 1.1 🔍 Market Structure Understanding
- [[06-Chiến-lược-trading/Market Microstructure - Order Flow|Market Microstructure & Order Flow]]
- [[06-Chiến-lược-trading/Bid-Ask Spread - Transaction Costs|Transaction Costs Analysis]]
- [[06-Chiến-lược-trading/Market Regimes - Trending vs Mean Reverting|Market Regime Detection]]

#### 1.2 📊 Strategy Classification & Framework
- [[06-Chiến-lược-trading/Strategy Taxonomy - Alpha Sources|Strategy Classification Framework]]
- [[06-Chiến-lược-trading/Statistical Arbitrage Concepts|Statistical Arbitrage]]
- [[06-Chiến-lược-trading/Market Making vs Taking|Market Making Strategies]]

### 🏗️ Week 3-4: Traditional Systematic Strategies

#### 3.1 📈 Trend Following Systems
- [[06-Chiến-lược-trading/Momentum Strategies - CTA Style|Momentum & Trend Following]]
- [[06-Chiến-lược-trading/Breakout Systems - Donchian|Breakout Trading Systems]]
- [[06-Chiến-lược-trading/Moving Average Systems|Moving Average Strategies]]

#### 3.2 🔄 Mean Reversion Strategies
- [[06-Chiến-lược-trading/Pairs Trading - Statistical Arbitrage|Pairs Trading Implementation]]
- [[06-Chiến-lược-trading/RSI Mean Reversion|RSI-Based Mean Reversion]]
- [[06-Chiến-lược-trading/Bollinger Band Strategies|Bollinger Band Systems]]

### 🏗️ Week 5-6: Factor-Based Strategies

#### 5.1 📊 Multi-Factor Models
- [[06-Chiến-lược-trading/Fama French Factors|Fama-French Factor Strategies]]
- [[06-Chiến-lược-trading/Quality Factor Investing|Quality Factor Strategies]]
- [[06-Chiến-lược-trading/Low Volatility Anomaly|Low Volatility Strategies]]

#### 5.2 🎯 Smart Beta Strategies
- [[06-Chiến-lược-trading/Momentum Factor Strategy|Momentum Factor Implementation]]
- [[06-Chiến-lược-trading/Value Factor Strategy|Value Factor Implementation]]
- [[06-Chiến-lược-trading/Size Factor Strategy|Size Factor Implementation]]

### 🏗️ Week 7: Alternative Strategies

#### 7.1 🤖 AI-Powered Strategies
- [[06-Chiến-lược-trading/Machine Learning Trading|ML-Based Trading Systems]]
- [[06-Chiến-lược-trading/NLP Sentiment Trading|NLP Sentiment Strategies]]
- [[06-Chiến-lược-trading/Reinforcement Learning Trading|RL Trading Agents]]

#### 7.2 🔗 DeFi & Blockchain Strategies
- [[06-Chiến-lược-trading/🔗 DeFi và Blockchain Trading|DeFi Trading Strategies]]
- [[06-Chiến-lược-trading/Yield Farming Strategies|Yield Farming Optimization]]
- [[06-Chiến-lược-trading/Arbitrage DeFi Protocols|DeFi Arbitrage]]

### 🏗️ Week 8: High-Frequency & Microstructure

#### 8.1 ⚡ High-Frequency Trading
- [[06-Chiến-lược-trading/Market Making HFT|HFT Market Making]]
- [[06-Chiến-lược-trading/Latency Arbitrage|Latency Arbitrage Strategies]]
- [[06-Chiến-lược-trading/Order Book Dynamics|Order Book Analysis]]

#### 8.2 🎯 Statistical Arbitrage
- [[06-Chiến-lược-trading/Cointegration Trading|Cointegration-Based Trading]]
- [[06-Chiến-lược-trading/PCA Trading Strategies|PCA Statistical Arbitrage]]
- [[06-Chiến-lược-trading/Kalman Filter Trading|Kalman Filter Applications]]

### 🏗️ Week 9: Portfolio & Multi-Asset Strategies

#### 9.1 🌐 Cross-Asset Strategies
- [[06-Chiến-lược-trading/Currency Carry Trade|Currency Carry Strategies]]
- [[06-Chiến-lược-trading/Commodity Momentum|Commodity Trading]]
- [[06-Chiến-lược-trading/Fixed Income Arbitrage|Fixed Income Strategies]]

#### 9.2 📊 Portfolio Construction
- [[06-Chiến-lược-trading/Risk Parity Strategies|Risk Parity Implementation]]
- [[06-Chiến-lược-trading/Dynamic Asset Allocation|Dynamic Allocation Strategies]]
- [[06-Chiến-lược-trading/Volatility Targeting|Volatility Targeting Systems]]

### 🏗️ Week 10: Strategy Integration & Production

#### 10.1 🔄 Multi-Strategy Systems
- [[06-Chiến-lược-trading/Strategy Ensemble Methods|Strategy Ensemble]]
- [[06-Chiến-lược-trading/Strategy Allocation Models|Strategy Allocation]]
- [[06-Chiến-lược-trading/Regime-Based Strategy Selection|Regime-Based Selection]]

#### 10.2 🚀 Production Implementation
- [[06-Chiến-lược-trading/Strategy Monitoring Systems|Strategy Monitoring]]
- [[06-Chiến-lược-trading/Performance Attribution|Performance Attribution]]
- [[06-Chiến-lược-trading/Strategy Decay Detection|Strategy Decay Management]]

## 🛠️ Trading Strategy Toolkit

### 📊 Core Strategy Libraries

```python
# Strategy Development Framework
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Libraries
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Financial Data & Backtesting
import yfinance as yf
import quantlib as ql
import zipline
import pyfolio
import backtrader as bt

# Technical Analysis
import talib
import ta
import pandas_ta as pta

# Portfolio & Risk Management
import pypfopt
from pypfopt import EfficientFrontier, risk_models, expected_returns
import empyrical as ep

# High-Frequency & Microstructure
import arctic                 # Time series database
import ib_insync             # Interactive Brokers API
import ccxt                  # Cryptocurrency exchange APIs

# Alternative Data
import tweepy                # Twitter API
import newspaper3k           # News scraping
import gdelt                 # GDELT news database

# DeFi & Blockchain
import web3                  # Ethereum blockchain
from uniswap import Uniswap  # DeFi protocols
import requests              # API calls
```

### 🎯 Core Strategy Implementation Framework

```python
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import logging

class TradingStrategy(ABC):
    """
    Base class for all trading strategies
    Follows first principles: Every strategy must define edge, risk, and execution
    """
    
    def __init__(self, name: str, parameters: Dict):
        self.name = name
        self.parameters = parameters
        self.positions = pd.Series(dtype=float)
        self.signals = pd.DataFrame()
        self.performance_metrics = {}
        self.logger = logging.getLogger(f'Strategy.{name}')
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on strategy logic"""
        pass
    
    @abstractmethod
    def calculate_position_sizes(self, signals: pd.DataFrame, 
                               portfolio_value: float) -> pd.Series:
        """Calculate position sizes based on risk management rules"""
        pass
    
    def validate_strategy_assumptions(self, data: pd.DataFrame) -> Dict:
        """Validate that market conditions support strategy assumptions"""
        
        returns = data['close'].pct_change().dropna()
        
        validation_results = {
            'has_trend': self._check_trend_presence(returns),
            'has_mean_reversion': self._check_mean_reversion(returns),
            'volatility_regime': self._identify_volatility_regime(returns),
            'correlation_stability': self._check_correlation_stability(data),
            'liquidity_sufficient': self._check_liquidity(data)
        }
        
        return validation_results
    
    def _check_trend_presence(self, returns: pd.Series) -> bool:
        """Check if significant trend is present"""
        # Hurst exponent test
        def hurst_exponent(ts):
            lags = range(2, 100)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        hurst = hurst_exponent(returns.cumsum())
        return hurst > 0.5  # H > 0.5 indicates trend
    
    def _check_mean_reversion(self, returns: pd.Series) -> bool:
        """Check for mean reversion using ADF test"""
        from statsmodels.tsa.stattools import adfuller
        
        cumulative_returns = returns.cumsum()
        adf_stat, p_value, _, _, _, _ = adfuller(cumulative_returns)
        
        return p_value < 0.05  # Stationary = mean reverting
    
    def _identify_volatility_regime(self, returns: pd.Series) -> str:
        """Identify current volatility regime"""
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        historical_vol = returns.rolling(252).std().iloc[-1] * np.sqrt(252)
        
        if current_vol > historical_vol * 1.5:
            return 'HIGH_VOLATILITY'
        elif current_vol < historical_vol * 0.5:
            return 'LOW_VOLATILITY'
        else:
            return 'NORMAL_VOLATILITY'
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Comprehensive backtesting framework"""
        
        # Generate signals
        signals = self.generate_signals(data)
        
        # Validate strategy assumptions
        validation = self.validate_strategy_assumptions(data)
        
        # Calculate positions
        positions = self.calculate_position_sizes(signals, initial_capital)
        
        # Calculate returns
        returns = self._calculate_strategy_returns(data, positions)
        
        # Performance metrics
        metrics = self._calculate_performance_metrics(returns)
        
        # Store results
        self.signals = signals
        self.positions = positions
        self.performance_metrics = metrics
        
        return {
            'returns': returns,
            'positions': positions,
            'signals': signals,
            'metrics': metrics,
            'validation': validation
        }
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, 
                                   positions: pd.Series) -> pd.Series:
        """Calculate strategy returns based on positions"""
        
        # Price returns
        price_returns = data['close'].pct_change()
        
        # Strategy returns = position * price_return
        strategy_returns = positions.shift(1) * price_returns
        
        # Account for transaction costs
        position_changes = positions.diff().abs()
        transaction_costs = position_changes * 0.001  # 0.1% transaction cost
        
        net_returns = strategy_returns - transaction_costs
        
        return net_returns.fillna(0)
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Advanced metrics
        sortino_ratio = self._calculate_sortino_ratio(returns)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': (returns > 0).mean(),
            'avg_win': returns[returns > 0].mean(),
            'avg_loss': returns[returns < 0].mean(),
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else np.inf
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        annual_return = returns.mean() * 252
        
        return annual_return / downside_deviation if downside_deviation > 0 else 0

class MomentumStrategy(TradingStrategy):
    """
    Momentum strategy implementation
    Edge: Trending markets exhibit continuation
    """
    
    def __init__(self, lookback_period: int = 20, entry_threshold: float = 0.02):
        parameters = {
            'lookback_period': lookback_period,
            'entry_threshold': entry_threshold
        }
        super().__init__('Momentum', parameters)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals"""
        
        lookback = self.parameters['lookback_period']
        threshold = self.parameters['entry_threshold']
        
        # Calculate momentum
        momentum = data['close'].pct_change(lookback)
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['momentum'] = momentum
        signals['signal'] = np.where(momentum > threshold, 1,
                            np.where(momentum < -threshold, -1, 0))
        
        # Signal confidence based on momentum strength
        signals['confidence'] = np.abs(momentum) / momentum.rolling(252).std()
        
        return signals
    
    def calculate_position_sizes(self, signals: pd.DataFrame, 
                               portfolio_value: float) -> pd.Series:
        """Calculate position sizes using volatility targeting"""
        
        # Base position size
        base_position = 0.1  # 10% of portfolio
        
        # Adjust for signal confidence
        confidence_adjusted = base_position * signals['confidence'].clip(0, 2)
        
        # Volatility targeting
        returns = signals.index.to_series().pct_change()
        vol_target = 0.15  # 15% annual volatility target
        current_vol = returns.rolling(20).std() * np.sqrt(252)
        vol_scalar = vol_target / current_vol.clip(0.01, 1)  # Prevent division by zero
        
        # Final position sizes
        positions = signals['signal'] * confidence_adjusted * vol_scalar
        positions = positions.clip(-0.5, 0.5)  # Max 50% position
        
        return positions

class MeanReversionStrategy(TradingStrategy):
    """
    Mean reversion strategy implementation
    Edge: Prices revert to statistical mean over time
    """
    
    def __init__(self, lookback_period: int = 20, entry_z_score: float = 2.0):
        parameters = {
            'lookback_period': lookback_period,
            'entry_z_score': entry_z_score
        }
        super().__init__('MeanReversion', parameters)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals"""
        
        lookback = self.parameters['lookback_period']
        z_threshold = self.parameters['entry_z_score']
        
        # Calculate rolling statistics
        rolling_mean = data['close'].rolling(lookback).mean()
        rolling_std = data['close'].rolling(lookback).std()
        
        # Z-score
        z_score = (data['close'] - rolling_mean) / rolling_std
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['z_score'] = z_score
        signals['signal'] = np.where(z_score < -z_threshold, 1,    # Buy oversold
                            np.where(z_score > z_threshold, -1,     # Sell overbought
                                   np.where(np.abs(z_score) < 0.5, 0, 0)))  # Exit when near mean
        
        # Signal strength
        signals['strength'] = np.abs(z_score) / z_threshold
        
        return signals
    
    def calculate_position_sizes(self, signals: pd.DataFrame, 
                               portfolio_value: float) -> pd.Series:
        """Calculate position sizes based on z-score strength"""
        
        # Base position size
        base_position = 0.1
        
        # Scale by signal strength
        strength_adjusted = base_position * signals['strength'].clip(0, 2)
        
        # Final positions
        positions = signals['signal'] * strength_adjusted
        positions = positions.clip(-0.3, 0.3)  # Max 30% position
        
        return positions

class PairsTradingtrategy(TradingStrategy):
    """
    Statistical arbitrage pairs trading strategy
    Edge: Cointegrated pairs revert to statistical relationship
    """
    
    def __init__(self, symbol1: str, symbol2: str, lookback: int = 60):
        parameters = {
            'symbol1': symbol1,
            'symbol2': symbol2,
            'lookback': lookback,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5
        }
        super().__init__('PairsTrading', parameters)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pairs trading signals"""
        
        # Assuming data has columns for both symbols
        price1 = data[f'{self.parameters["symbol1"]}_close']
        price2 = data[f'{self.parameters["symbol2"]}_close']
        
        # Calculate hedge ratio using rolling regression
        lookback = self.parameters['lookback']
        hedge_ratios = []
        
        for i in range(lookback, len(data)):
            y = price1.iloc[i-lookback:i]
            x = price2.iloc[i-lookback:i]
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
            hedge_ratios.append(model.coef_[0])
        
        hedge_ratio_series = pd.Series(hedge_ratios, index=data.index[lookback:])
        
        # Calculate spread
        spread = price1 - hedge_ratio_series * price2
        
        # Z-score of spread
        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['hedge_ratio'] = hedge_ratio_series
        
        entry_threshold = self.parameters['entry_threshold']
        exit_threshold = self.parameters['exit_threshold']
        
        signals['signal'] = np.where(z_score > entry_threshold, -1,   # Short spread
                            np.where(z_score < -entry_threshold, 1,   # Long spread
                                   np.where(np.abs(z_score) < exit_threshold, 0, 0)))
        
        return signals
    
    def calculate_position_sizes(self, signals: pd.DataFrame, 
                               portfolio_value: float) -> pd.Series:
        """Calculate position sizes for pairs trade"""
        
        base_position = 0.1
        
        # Position in symbol1
        pos1 = signals['signal'] * base_position
        
        # Position in symbol2 (hedged)
        pos2 = -signals['signal'] * signals['hedge_ratio'] * base_position
        
        # Return as combined position (simplified)
        return pos1  # In practice, would return both positions
```

## 📈 Advanced Strategy Applications

### 🤖 Project 1: AI-Enhanced Multi-Strategy System

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

class AIMultiStrategySystem:
    """
    AI-enhanced system that dynamically allocates between strategies
    based on market regime detection and strategy performance prediction
    """
    
    def __init__(self):
        self.strategies = {}
        self.regime_classifier = RandomForestClassifier(n_estimators=100)
        self.performance_predictor = None
        self.regime_features = None
        self.allocation_model = None
        
    def add_strategy(self, name: str, strategy: TradingStrategy):
        """Add a strategy to the system"""
        self.strategies[name] = strategy
    
    def extract_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime classification"""
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        returns = data['close'].pct_change()
        features['volatility'] = returns.rolling(20).std()
        features['skewness'] = returns.rolling(60).skew()
        features['kurtosis'] = returns.rolling(60).kurt()
        
        # Trend features
        features['trend_strength'] = data['close'] / data['close'].rolling(50).mean() - 1
        features['momentum'] = returns.rolling(20).mean()
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_trend'] = data['volume'] / data['volume'].rolling(20).mean()
            features['price_volume_corr'] = returns.rolling(20).corr(data['volume'].pct_change())
        
        # Market microstructure (if available)
        if 'high' in data.columns and 'low' in data.columns:
            features['high_low_ratio'] = (data['high'] - data['low']) / data['close']
            features['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        
        # Regime indicators
        features['vix_level'] = self._get_vix_proxy(returns)  # VIX proxy
        features['correlation_regime'] = self._calculate_correlation_regime(data)
        
        return features.dropna()
    
    def _get_vix_proxy(self, returns: pd.Series) -> pd.Series:
        """Calculate VIX proxy using GARCH model"""
        from arch import arch_model
        
        # Fit GARCH(1,1) model
        try:
            model = arch_model(returns.dropna() * 100, vol='Garch', p=1, q=1)
            fitted = model.fit(disp='off')
            vix_proxy = fitted.conditional_volatility / 100
            return vix_proxy
        except:
            # Fallback to rolling volatility
            return returns.rolling(20).std() * np.sqrt(252)
    
    def _calculate_correlation_regime(self, data: pd.DataFrame) -> pd.Series:
        """Calculate cross-asset correlation regime indicator"""
        
        # This would require multiple asset data
        # For now, use autocorrelation as proxy
        returns = data['close'].pct_change()
        autocorr = returns.rolling(60).apply(lambda x: x.autocorr(lag=1))
        
        return autocorr
    
    def train_regime_classifier(self, historical_data: pd.DataFrame, 
                              regime_labels: pd.Series):
        """Train regime classification model"""
        
        # Extract features
        features = self.extract_market_features(historical_data)
        
        # Align with labels
        aligned_data = pd.concat([features, regime_labels], axis=1).dropna()
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train classifier
        self.regime_classifier.fit(X_scaled, y)
        self.feature_scaler = scaler
        self.regime_features = X.columns
        
        return self.regime_classifier.score(X_scaled, y)
    
    def predict_current_regime(self, current_data: pd.DataFrame) -> int:
        """Predict current market regime"""
        
        # Extract features
        features = self.extract_market_features(current_data)
        
        # Use most recent observation
        current_features = features.iloc[-1:][self.regime_features]
        
        # Scale and predict
        current_scaled = self.feature_scaler.transform(current_features)
        regime_proba = self.regime_classifier.predict_proba(current_scaled)[0]
        
        return {
            'predicted_regime': self.regime_classifier.predict(current_scaled)[0],
            'regime_probabilities': regime_proba,
            'confidence': max(regime_proba)
        }
    
    def build_strategy_allocation_model(self, historical_data: pd.DataFrame):
        """Build deep learning model for strategy allocation"""
        
        # Prepare training data
        strategy_returns = {}
        
        for name, strategy in self.strategies.items():
            backtest_results = strategy.backtest(historical_data)
            strategy_returns[name] = backtest_results['returns']
        
        # Create features and targets
        features = self.extract_market_features(historical_data)
        
        # Target: optimal allocation that maximizes Sharpe ratio
        optimal_allocations = self._calculate_optimal_allocations(
            strategy_returns, features.index
        )
        
        # Align data
        aligned_data = pd.concat([features, optimal_allocations], axis=1).dropna()
        
        X = aligned_data.iloc[:, :len(features.columns)].values
        y = aligned_data.iloc[:, len(features.columns):].values
        
        # Build neural network
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(self.strategies), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.allocation_model = model
        return history
    
    def _calculate_optimal_allocations(self, strategy_returns: Dict, 
                                     index: pd.Index) -> pd.DataFrame:
        """Calculate optimal strategy allocations using mean-variance optimization"""
        
        from pypfopt import EfficientFrontier, expected_returns, risk_models
        
        # Create returns matrix
        returns_df = pd.DataFrame(strategy_returns, index=index).dropna()
        
        optimal_allocations = []
        lookback = 252  # 1 year
        
        for i in range(lookback, len(returns_df)):
            window_returns = returns_df.iloc[i-lookback:i]
            
            try:
                # Calculate expected returns and covariance
                mu = expected_returns.mean_historical_return(window_returns)
                S = risk_models.sample_cov(window_returns)
                
                # Optimize for maximum Sharpe ratio
                ef = EfficientFrontier(mu, S)
                weights = ef.max_sharpe()
                
                optimal_allocations.append(pd.Series(weights, index=returns_df.columns))
                
            except:
                # Equal weight if optimization fails
                equal_weight = 1 / len(self.strategies)
                optimal_allocations.append(
                    pd.Series([equal_weight] * len(self.strategies), 
                             index=returns_df.columns)
                )
        
        return pd.DataFrame(optimal_allocations, 
                          index=returns_df.index[lookback:])
    
    def generate_combined_signals(self, current_data: pd.DataFrame) -> Dict:
        """Generate combined signals from all strategies with AI allocation"""
        
        # Get individual strategy signals
        strategy_signals = {}
        for name, strategy in self.strategies.items():
            signals = strategy.generate_signals(current_data)
            strategy_signals[name] = signals
        
        # Predict regime and allocation
        regime_info = self.predict_current_regime(current_data)
        
        if self.allocation_model is not None:
            current_features = self.extract_market_features(current_data).iloc[-1:].values
            predicted_allocation = self.allocation_model.predict(current_features)[0]
        else:
            # Equal allocation if model not trained
            predicted_allocation = np.ones(len(self.strategies)) / len(self.strategies)
        
        # Combine signals
        combined_signal = 0
        allocation_dict = {}
        
        for i, (name, signals) in enumerate(strategy_signals.items()):
            weight = predicted_allocation[i]
            latest_signal = signals['signal'].iloc[-1] if len(signals) > 0 else 0
            
            combined_signal += weight * latest_signal
            allocation_dict[name] = weight
        
        return {
            'combined_signal': combined_signal,
            'strategy_allocations': allocation_dict,
            'regime_info': regime_info,
            'individual_signals': {name: signals['signal'].iloc[-1] 
                                 for name, signals in strategy_signals.items()}
        }
```

### 🔗 Project 2: DeFi Yield Farming Strategy

```python
import requests
import json
from web3 import Web3
import pandas as pd
import numpy as np

class DeFiYieldFarmingStrategy:
    """
    Automated DeFi yield farming strategy
    Dynamically allocates capital across different DeFi protocols
    """
    
    def __init__(self, web3_provider_url: str):
        self.w3 = Web3(Web3.HTTPProvider(web3_provider_url))
        self.protocols = {}
        self.current_positions = {}
        self.yield_history = pd.DataFrame()
        
    def add_protocol(self, name: str, contract_address: str, abi: dict):
        """Add a DeFi protocol to monitor"""
        
        contract = self.w3.eth.contract(
            address=Web3.toChecksumAddress(contract_address),
            abi=abi
        )
        
        self.protocols[name] = {
            'contract': contract,
            'address': contract_address
        }
    
    def fetch_protocol_metrics(self) -> pd.DataFrame:
        """Fetch current metrics for all protocols"""
        
        metrics = []
        
        for name, protocol in self.protocols.items():
            try:
                # This would be protocol-specific
                # Example for a generic lending protocol
                
                # APY
                apy = self._get_protocol_apy(name)
                
                # TVL
                tvl = self._get_protocol_tvl(name)
                
                # Risk metrics
                risk_score = self._calculate_protocol_risk(name)
                
                # Impermanent loss risk (for AMM protocols)
                il_risk = self._calculate_impermanent_loss_risk(name)
                
                metrics.append({
                    'protocol': name,
                    'apy': apy,
                    'tvl': tvl,
                    'risk_score': risk_score,
                    'il_risk': il_risk,
                    'risk_adjusted_yield': apy * (1 - risk_score),
                    'timestamp': pd.Timestamp.now()
                })
                
            except Exception as e:
                print(f"Error fetching data for {name}: {e}")
        
        return pd.DataFrame(metrics)
    
    def _get_protocol_apy(self, protocol_name: str) -> float:
        """Get current APY for protocol"""
        
        # This would integrate with protocol-specific APIs
        # For demonstration, using DeFi Pulse API structure
        
        api_endpoints = {
            'compound': 'https://api.compound.finance/api/v2/ctoken',
            'aave': 'https://api.aave.com/data/rates',
            'uniswap': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'
        }
        
        if protocol_name.lower() in api_endpoints:
            try:
                response = requests.get(api_endpoints[protocol_name.lower()])
                data = response.json()
                
                # Parse protocol-specific response
                if protocol_name.lower() == 'compound':
                    return float(data['cToken'][0]['supply_rate']['value']) * 100
                
                # Add other protocol parsers as needed
                
            except:
                pass
        
        # Fallback to mock data
        return np.random.uniform(5, 25)  # Mock APY between 5-25%
    
    def _get_protocol_tvl(self, protocol_name: str) -> float:
        """Get Total Value Locked for protocol"""
        
        try:
            # Use DeFi Llama API
            response = requests.get(f'https://api.llama.fi/protocol/{protocol_name.lower()}')
            data = response.json()
            
            if 'tvl' in data:
                return float(data['tvl'][-1]['totalLiquidityUSD'])
                
        except:
            pass
        
        return np.random.uniform(1e6, 1e9)  # Mock TVL
    
    def _calculate_protocol_risk(self, protocol_name: str) -> float:
        """Calculate risk score for protocol (0-1, higher = riskier)"""
        
        risk_factors = {
            'smart_contract_risk': 0.1,  # Based on audit history
            'liquidity_risk': 0.1,       # Based on TVL and volume
            'market_risk': 0.1,          # Based on token volatility
            'regulatory_risk': 0.05,     # Based on jurisdiction
            'governance_risk': 0.05      # Based on token distribution
        }
        
        # This would be based on actual analysis
        # For demo, using random weights
        total_risk = sum(risk_factors.values()) * np.random.uniform(0.5, 1.5)
        
        return min(total_risk, 1.0)
    
    def _calculate_impermanent_loss_risk(self, protocol_name: str) -> float:
        """Calculate impermanent loss risk for AMM protocols"""
        
        # Only relevant for AMM protocols
        amm_protocols = ['uniswap', 'sushiswap', 'balancer', 'curve']
        
        if not any(amm in protocol_name.lower() for amm in amm_protocols):
            return 0.0
        
        # Simplified IL calculation based on correlation
        # Real implementation would use actual pool compositions
        
        correlation = np.random.uniform(0.3, 0.9)  # Asset correlation
        volatility_diff = np.random.uniform(0.1, 0.5)  # Volatility difference
        
        # Higher IL risk when correlation is low and volatility diff is high
        il_risk = (1 - correlation) * volatility_diff
        
        return min(il_risk, 0.5)  # Cap at 50%
    
    def optimize_allocation(self, available_capital: float, 
                          max_protocols: int = 5) -> Dict:
        """Optimize capital allocation across protocols"""
        
        # Fetch current metrics
        metrics = self.fetch_protocol_metrics()
        
        if len(metrics) == 0:
            return {}
        
        # Filter and rank protocols
        # Remove protocols with excessive risk
        filtered_metrics = metrics[metrics['risk_score'] < 0.5]
        
        # Sort by risk-adjusted yield
        top_protocols = filtered_metrics.nlargest(max_protocols, 'risk_adjusted_yield')
        
        # Mean-variance optimization
        expected_returns = top_protocols['risk_adjusted_yield'].values / 100
        
        # Simplified covariance (would need historical data)
        n_assets = len(expected_returns)
        correlation_matrix = np.full((n_assets, n_assets), 0.3)
        np.fill_diagonal(correlation_matrix, 1.0)
        
        volatilities = top_protocols['risk_score'].values * 0.5  # Convert risk to volatility
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Optimize using mean-variance
        try:
            from pypfopt import EfficientFrontier
            
            ef = EfficientFrontier(expected_returns, cov_matrix)
            weights = ef.max_sharpe()
            
            # Create allocation dictionary
            allocation = {}
            for i, protocol in enumerate(top_protocols['protocol']):
                allocation[protocol] = weights[i] * available_capital
                
        except:
            # Fallback to equal allocation
            equal_weight = available_capital / len(top_protocols)
            allocation = {
                protocol: equal_weight 
                for protocol in top_protocols['protocol']
            }
        
        return {
            'allocations': allocation,
            'expected_apy': np.sum(expected_returns * list(weights.values())) * 100,
            'total_risk': np.sqrt(np.dot(list(weights.values()), 
                                       np.dot(cov_matrix, list(weights.values())))),
            'protocols_data': top_protocols.to_dict('records')
        }
    
    def execute_rebalancing(self, target_allocation: Dict) -> Dict:
        """Execute rebalancing transactions"""
        
        execution_results = {}
        
        for protocol, target_amount in target_allocation.items():
            current_amount = self.current_positions.get(protocol, 0)
            difference = target_amount - current_amount
            
            if abs(difference) > 100:  # Only rebalance if difference > $100
                try:
                    if difference > 0:
                        # Deposit more
                        tx_hash = self._deposit_to_protocol(protocol, difference)
                        execution_results[protocol] = {
                            'action': 'deposit',
                            'amount': difference,
                            'tx_hash': tx_hash,
                            'status': 'pending'
                        }
                    else:
                        # Withdraw
                        tx_hash = self._withdraw_from_protocol(protocol, abs(difference))
                        execution_results[protocol] = {
                            'action': 'withdraw',
                            'amount': abs(difference),
                            'tx_hash': tx_hash,
                            'status': 'pending'
                        }
                        
                    # Update positions
                    self.current_positions[protocol] = target_amount
                    
                except Exception as e:
                    execution_results[protocol] = {
                        'action': 'failed',
                        'error': str(e)
                    }
        
        return execution_results
    
    def _deposit_to_protocol(self, protocol_name: str, amount: float) -> str:
        """Execute deposit transaction to protocol"""
        
        # This would contain protocol-specific deposit logic
        # Using mock implementation
        
        protocol = self.protocols[protocol_name]
        contract = protocol['contract']
        
        # Mock transaction hash
        return f"0x{''.join(np.random.choice(list('0123456789abcdef'), 64))}"
    
    def _withdraw_from_protocol(self, protocol_name: str, amount: float) -> str:
        """Execute withdrawal transaction from protocol"""
        
        # This would contain protocol-specific withdrawal logic
        # Using mock implementation
        
        protocol = self.protocols[protocol_name]
        contract = protocol['contract']
        
        # Mock transaction hash
        return f"0x{''.join(np.random.choice(list('0123456789abcdef'), 64))}"
    
    def monitor_and_rebalance(self, available_capital: float, 
                            rebalance_threshold: float = 0.05):
        """Continuous monitoring and rebalancing"""
        
        while True:
            try:
                # Get optimal allocation
                optimal_allocation = self.optimize_allocation(available_capital)
                
                # Check if rebalancing is needed
                needs_rebalancing = False
                current_total = sum(self.current_positions.values())
                
                if current_total == 0:
                    needs_rebalancing = True
                else:
                    for protocol, target_amount in optimal_allocation['allocations'].items():
                        current_weight = self.current_positions.get(protocol, 0) / current_total
                        target_weight = target_amount / available_capital
                        
                        if abs(current_weight - target_weight) > rebalance_threshold:
                            needs_rebalancing = True
                            break
                
                if needs_rebalancing:
                    print("Rebalancing needed...")
                    results = self.execute_rebalancing(optimal_allocation['allocations'])
                    print(f"Rebalancing results: {results}")
                
                # Wait before next check (e.g., 1 hour)
                time.sleep(3600)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
```

## ✅ Trading Strategy Progression

### Week 1-2: Strategy Foundations ✅
- [ ] Understand market microstructure
- [ ] Master transaction cost analysis
- [ ] Build strategy classification framework
- [ ] Implement basic trend/mean-reversion strategies

### Week 3-4: Traditional Systematic ✅
- [ ] Develop momentum systems
- [ ] Build mean reversion strategies
- [ ] Implement pairs trading
- [ ] Master technical indicator strategies

### Week 5-6: Factor-Based Strategies ✅
- [ ] Multi-factor model implementation
- [ ] Smart beta strategies
- [ ] Factor timing models
- [ ] Performance attribution

### Week 7: Alternative Strategies ✅
- [ ] AI-powered trading systems
- [ ] NLP sentiment strategies
- [ ] DeFi yield farming
- [ ] Cross-chain arbitrage

### Week 8: High-Frequency & Microstructure ✅
- [ ] Market making strategies
- [ ] Statistical arbitrage
- [ ] Order book analysis
- [ ] Latency optimization

### Week 9: Multi-Asset Strategies ✅
- [ ] Cross-asset momentum
- [ ] Currency carry trades
- [ ] Commodity strategies
- [ ] Fixed income arbitrage

### Week 10: Strategy Integration ✅
- [ ] Multi-strategy systems
- [ ] Dynamic allocation
- [ ] Regime-based selection
- [ ] Production deployment

## 💎 Key Strategy Principles

### 🎯 Strategy Development Rules

1. **Edge Identification** - Every strategy must have a clear statistical edge
2. **Risk-First Design** - Risk management is not optional
3. **Transaction Cost Awareness** - Include all costs in backtests
4. **Regime Awareness** - Strategies fail when regimes change
5. **Systematic Execution** - Remove emotion from trading

### 🚀 2025 Strategy Trends

1. **AI-Enhanced Strategies** - ML augments traditional methods
2. **Multi-Asset Integration** - Strategies span asset classes
3. **Real-Time Adaptation** - Strategies adapt to changing conditions
4. **DeFi Integration** - Traditional meets decentralized finance
5. **ESG Integration** - Sustainability as alpha source

---

**Next**: [[06-Chiến-lược-trading/Market Microstructure - Order Flow|Market Structure Basics]]

**Advanced**: [[06-Chiến-lược-trading/🔗 DeFi và Blockchain Trading|DeFi Strategies]]

---

*"Strategy without tactics is the slowest route to victory. Tactics without strategy is the noise before defeat."* - Sun Tzu 🎯

*"In trading, your worst enemy is yourself"* - Trading Psychology 101 🧠
