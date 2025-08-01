# 🔬 Phân Tích Định Lượng - Index

## 🧠 First Principles: Numbers Don't Lie, But They Don't Tell the Whole Truth

**Câu hỏi cốt lõi**: Tại sao quant models fail trong crisis? Không phải vì math sai - mà vì **assumptions về human behavior sai**!

### 💡 Philosophy của Quantitative Analysis

**Quant không phải là pure math** - nó là **applied psychology through mathematics**:

- **Models** = Simplified representations of complex reality
- **Assumptions** = What we believe about market behavior  
- **Parameters** = Quantified human emotions (fear, greed, hope)
- **Backtests** = Historical fiction, not future prediction
- **Alpha** = Systematic exploitation of human irrationality

## 🎯 Mục Tiêu Chương Này

### 🔄 From Formula Memorizer to Model Builder

```
Math Theory → Market Application → Model Building → Strategy Development → Alpha Generation
```

**Không áp dụng công thức** - học cách **build models từ first principles**!

## 📚 Quantitative Analysis Mastery (8 Tuần)

### 🏗️ Week 1-2: Statistical Foundations for Finance

#### 1.1 📊 Descriptive Statistics in Market Context
- [[05-Phân-tích-định-lượng/Returns Distribution - Normal vs Reality|Returns Distribution Analysis]]
- [[05-Phân-tích-định-lượng/Volatility Clustering - GARCH Models|Volatility Clustering Phenomenon]]
- [[05-Phân-tích-định-lượng/Correlation vs Cointegration|Correlation vs Cointegration]]

#### 1.2 🎲 Probability Theory for Trading
- [[05-Phân-tích-định-lượng/Monte Carlo Methods Finance|Monte Carlo in Finance]]
- [[05-Phân-tích-định-lượng/Bayesian Inference Trading|Bayesian Methods for Trading]]
- [[05-Phân-tích-định-lượng/Extreme Value Theory - Tail Risk|Extreme Value Theory]]

### 🏗️ Week 3-4: Time Series Analysis

#### 3.1 📈 Univariate Time Series
- [[05-Phân-tích-định-lượng/ARMA ARIMA Models|ARMA/ARIMA Modeling]]
- [[05-Phân-tích-định-lượng/GARCH Volatility Models|GARCH Family Models]]
- [[05-Phân-tích-định-lượng/Unit Root Tests - Stationarity|Stationarity Testing]]

#### 3.2 🔄 Multivariate Time Series
- [[05-Phân-tích-định-lượng/Vector Autoregression VAR|VAR Models]]
- [[05-Phân-tích-định-lượng/Cointegration Analysis - Pairs Trading|Cointegration for Pairs Trading]]
- [[05-Phân-tích-định-lượng/State Space Models - Kalman Filter|Kalman Filter Applications]]

### 🏗️ Week 5-6: Machine Learning for Finance

#### 5.1 🤖 Supervised Learning Applications
- [[05-Phân-tích-định-lượng/🤖 AI và Machine Learning Hiện Đại|AI & ML Modern Applications]]
- [[05-Phân-tích-định-lượng/Feature Engineering Finance|Financial Feature Engineering]]
- [[05-Phân-tích-định-lượng/Cross Validation Time Series|Time Series Cross-Validation]]

#### 5.2 🧠 Deep Learning for Markets
- [[05-Phân-tích-định-lượng/LSTM Stock Prediction|LSTM for Stock Prediction]]
- [[05-Phân-tích-định-lượng/Transformer Models Finance|Transformer Models in Finance]]
- [[05-Phân-tích-định-lượng/GANs Market Simulation|GANs for Market Simulation]]

### 🏗️ Week 7: Advanced Quantitative Methods

#### 7.1 ⚛️ Quantum Computing Applications
- [[05-Phân-tích-định-lượng/🔬 Quantum Computing trong Finance|Quantum Computing in Finance]]
- [[05-Phân-tích-định-lượng/Quantum Portfolio Optimization|Quantum Portfolio Optimization]]
- [[05-Phân-tích-định-lượng/Quantum Monte Carlo|Quantum Monte Carlo Methods]]

#### 7.2 🌱 ESG Quantitative Analysis
- [[05-Phân-tích-định-lượng/🌱 ESG và Sustainable Investing|ESG Quantitative Framework]]
- [[05-Phân-tích-định-lượng/ESG Scoring Models|ESG Scoring Methodologies]]
- [[05-Phân-tích-định-lượng/Climate Risk Modeling|Climate Risk Quantification]]

### 🏗️ Week 8: Model Validation & Implementation

#### 8.1 🔍 Model Validation Techniques
- [[05-Phân-tích-định-lượng/Backtesting Best Practices|Backtesting Methodology]]
- [[05-Phân-tích-định-lượng/Walk Forward Analysis|Walk-Forward Analysis]]
- [[05-Phân-tích-định-lượng/Monte Carlo Validation|Monte Carlo Validation]]

#### 8.2 🚀 Production Model Deployment
- [[05-Phân-tích-định-lượng/Model Monitoring - Performance Decay|Model Performance Monitoring]]
- [[05-Phân-tích-định-lượng/A-B Testing Strategies|A/B Testing for Strategies]]
- [[05-Phân-tích-định-lượng/Model Ensemble Methods|Ensemble Trading Models]]

## 🛠️ Quantitative Analysis Toolkit

### 📊 Python Libraries for Quant Analysis

```python
# Statistical Analysis
import numpy as np              # Numerical computing
import pandas as pd             # Data manipulation
import scipy.stats as stats     # Statistical functions
import statsmodels.api as sm    # Econometric models
import arch                     # GARCH models

# Time Series Analysis
import statsmodels.tsa as tsa   # Time series analysis
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import coint, adfuller
import pmdarima as pm           # Auto ARIMA

# Machine Learning
import scikit-learn as sklearn  # Classical ML
import tensorflow as tf        # Deep learning
import torch                   # Alternative DL framework
import xgboost as xgb          # Gradient boosting
import lightgbm as lgb         # Light gradient boosting

# Specialized Finance ML
import mlfinlab                # Advances in Financial ML
import pyfolio                 # Portfolio performance analysis
import empyrical               # Risk metrics
import zipline                 # Backtesting framework

# Quantum Computing
import qiskit                  # IBM Quantum
import cirq                    # Google Quantum
import pennylane               # Quantum ML

# Alternative Data Processing
import textblob                # Text sentiment
import transformers            # NLP models
import opencv-cv2              # Computer vision
import satellite_image         # Satellite data processing

# High-Performance Computing
import numba                   # JIT compilation
import dask                    # Parallel computing
import ray                     # Distributed computing
import cudf                    # GPU DataFrames
```

### 🎯 Core Quantitative Functions

```python
def build_factor_model(returns_data, factors_data):
    """
    Build multi-factor model using Fama-French approach
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    
    # Align data
    aligned_data = pd.concat([returns_data, factors_data], axis=1).dropna()
    
    # Separate dependent and independent variables
    y = aligned_data.iloc[:, 0]  # Returns
    X = aligned_data.iloc[:, 1:]  # Factors
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate metrics
    r_squared = model.score(X, y)
    predictions = model.predict(X)
    residuals = y - predictions
    
    # Factor loadings
    loadings = pd.Series(model.coef_, index=X.columns)
    
    results = {
        'model': model,
        'r_squared': r_squared,
        'loadings': loadings,
        'alpha': model.intercept_,
        'residuals': residuals,
        'fitted_values': predictions
    }
    
    return results

def regime_detection_hmm(returns, n_states=2):
    """
    Detect market regimes using Hidden Markov Model
    """
    from hmmlearn import hmm
    import numpy as np
    
    # Prepare features
    features = np.column_stack([
        returns.values,
        returns.rolling(5).std().values,
        returns.rolling(20).mean().values
    ])
    
    # Remove NaN values
    features = features[~np.isnan(features).any(axis=1)]
    
    # Fit HMM model
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full")
    model.fit(features)
    
    # Predict states
    states = model.predict(features)
    
    # Calculate state probabilities
    state_probs = model.predict_proba(features)
    
    results = {
        'states': states,
        'state_probabilities': state_probs,
        'model': model,
        'transition_matrix': model.transmat_,
        'means': model.means_,
        'covariances': model.covars_
    }
    
    return results

def pairs_trading_analysis(price1, price2):
    """
    Comprehensive pairs trading analysis
    """
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.api import OLS
    
    # Ensure same length
    data = pd.concat([price1, price2], axis=1).dropna()
    p1, p2 = data.iloc[:, 0], data.iloc[:, 1]
    
    # Cointegration test
    coint_stat, coint_pvalue, coint_critical = coint(p1, p2)
    
    # Hedge ratio via OLS
    hedge_model = OLS(p1, p2).fit()
    hedge_ratio = hedge_model.params[0]
    
    # Spread calculation
    spread = p1 - hedge_ratio * p2
    
    # Spread stationarity test
    adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(spread.dropna())
    
    # Half-life of mean reversion
    spread_lag = spread.shift(1)
    spread_diff = spread.diff()
    
    mean_rev_model = OLS(spread_diff.dropna(), spread_lag.dropna()).fit()
    half_life = -np.log(2) / mean_rev_model.params[0]
    
    # Z-score for entry/exit signals
    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / spread_std
    
    results = {
        'cointegration_pvalue': coint_pvalue,
        'hedge_ratio': hedge_ratio,
        'spread': spread,
        'z_score': z_score,
        'half_life': half_life,
        'adf_pvalue': adf_pvalue,
        'entry_threshold': 2.0,  # Standard z-score threshold
        'exit_threshold': 0.5,
        'is_cointegrated': coint_pvalue < 0.05,
        'is_stationary': adf_pvalue < 0.05
    }
    
    return results
```

## 📈 Real-World Applications

### 🎯 Project 1: Multi-Factor Alpha Model

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

class MultiFactorAlphaModel:
    def __init__(self):
        self.factors = {}
        self.models = {}
        self.predictions = {}
        
    def build_factor_universe(self, price_data, volume_data, fundamental_data):
        """Build comprehensive factor universe"""
        
        # Technical factors
        technical_factors = self.calculate_technical_factors(price_data, volume_data)
        
        # Fundamental factors
        fundamental_factors = self.calculate_fundamental_factors(fundamental_data)
        
        # Macro factors
        macro_factors = self.calculate_macro_factors()
        
        # Alternative data factors
        alt_factors = self.calculate_alternative_factors()
        
        # Combine all factors
        all_factors = pd.concat([
            technical_factors,
            fundamental_factors,
            macro_factors,
            alt_factors
        ], axis=1)
        
        self.factors = all_factors
        return all_factors
    
    def calculate_technical_factors(self, prices, volumes):
        """Calculate technical analysis factors"""
        factors = pd.DataFrame(index=prices.index)
        
        # Momentum factors
        factors['momentum_1m'] = prices.pct_change(20)  # 1-month momentum
        factors['momentum_3m'] = prices.pct_change(60)  # 3-month momentum
        factors['momentum_12m'] = prices.pct_change(252) # 12-month momentum
        
        # Mean reversion factors
        factors['mean_reversion_short'] = (prices / prices.rolling(20).mean()) - 1
        factors['mean_reversion_long'] = (prices / prices.rolling(60).mean()) - 1
        
        # Volatility factors
        returns = prices.pct_change()
        factors['volatility_20d'] = returns.rolling(20).std()
        factors['volatility_60d'] = returns.rolling(60).std()
        
        # Volume factors
        factors['volume_trend'] = volumes.rolling(20).mean() / volumes.rolling(60).mean()
        factors['price_volume_trend'] = (returns * volumes).rolling(20).sum()
        
        # Technical indicators
        factors['rsi'] = self.calculate_rsi(prices)
        factors['bollinger_position'] = self.calculate_bollinger_position(prices)
        
        return factors
    
    def calculate_fundamental_factors(self, fundamentals):
        """Calculate fundamental analysis factors"""
        factors = pd.DataFrame(index=fundamentals.index)
        
        # Valuation factors
        factors['pe_ratio'] = fundamentals['market_cap'] / fundamentals['earnings']
        factors['pb_ratio'] = fundamentals['market_cap'] / fundamentals['book_value']
        factors['ps_ratio'] = fundamentals['market_cap'] / fundamentals['revenue']
        
        # Quality factors
        factors['roe'] = fundamentals['net_income'] / fundamentals['shareholders_equity']
        factors['roa'] = fundamentals['net_income'] / fundamentals['total_assets']
        factors['debt_to_equity'] = fundamentals['total_debt'] / fundamentals['shareholders_equity']
        
        # Growth factors
        factors['revenue_growth'] = fundamentals['revenue'].pct_change(4)  # YoY
        factors['earnings_growth'] = fundamentals['earnings'].pct_change(4)
        
        return factors
    
    def build_ml_model(self, factors, returns, lookback_periods=[20, 60, 120]):
        """Build machine learning model for return prediction"""
        
        # Prepare training data
        X_list = []
        y_list = []
        
        for period in lookback_periods:
            # Features: factors over lookback period
            X_period = factors.rolling(period).mean().dropna()
            
            # Target: forward returns
            y_period = returns.shift(-period).dropna()
            
            # Align data
            aligned_data = pd.concat([X_period, y_period], axis=1).dropna()
            
            if len(aligned_data) > 0:
                X_list.append(aligned_data.iloc[:, :-1])
                y_list.append(aligned_data.iloc[:, -1])
        
        # Combine all periods
        X_combined = pd.concat(X_list, axis=0)
        y_combined = pd.concat(y_list, axis=0)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42
        )
        
        # Cross-validation
        cv_scores = []
        for train_idx, test_idx in tscv.split(X_combined):
            X_train, X_test = X_combined.iloc[train_idx], X_combined.iloc[test_idx]
            y_train, y_test = y_combined.iloc[train_idx], y_combined.iloc[test_idx]
            
            rf_model.fit(X_train, y_train)
            score = rf_model.score(X_test, y_test)
            cv_scores.append(score)
        
        # Final model training
        rf_model.fit(X_combined, y_combined)
        
        # Feature importance
        feature_importance = pd.Series(
            rf_model.feature_importances_,
            index=X_combined.columns
        ).sort_values(ascending=False)
        
        self.models['ml_model'] = rf_model
        
        return {
            'model': rf_model,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'mean_cv_score': np.mean(cv_scores)
        }
    
    def generate_alpha_signals(self, current_factors):
        """Generate alpha signals from trained models"""
        
        if 'ml_model' not in self.models:
            raise ValueError("Model not trained yet!")
        
        # Get predictions from ML model
        ml_predictions = self.models['ml_model'].predict(current_factors)
        
        # Convert to signals (-1, 0, 1)
        signals = np.where(ml_predictions > 0.02, 1,  # Buy signal
                  np.where(ml_predictions < -0.02, -1, 0))  # Sell signal, Hold
        
        # Create signal DataFrame
        signal_df = pd.DataFrame({
            'predictions': ml_predictions,
            'signals': signals,
            'confidence': np.abs(ml_predictions)
        }, index=current_factors.index)
        
        return signal_df
```

### 🤖 Project 2: Deep Learning Trading System

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class DeepLearningTradingSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.predictions = {}
    
    def prepare_lstm_data(self, data, lookback=60, forecast_horizon=1):
        """Prepare data for LSTM training"""
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data) - forecast_horizon + 1):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i:i+forecast_horizon, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y, scaler
    
    def build_lstm_model(self, input_shape, forecast_horizon=1):
        """Build LSTM model for price prediction"""
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(forecast_horizon)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_ensemble_models(self, price_data, symbols):
        """Train ensemble of models for multiple symbols"""
        
        for symbol in symbols:
            print(f"Training model for {symbol}...")
            
            # Prepare data
            symbol_data = price_data[symbol].dropna()
            X, y, scaler = self.prepare_lstm_data(symbol_data)
            
            # Split train/validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build and train model
            model = self.build_lstm_model((X.shape[1], 1))
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            print(f"✅ {symbol} model trained. Final loss: {history.history['loss'][-1]:.6f}")
    
    def generate_predictions(self, current_data, symbols):
        """Generate predictions for current market data"""
        
        predictions = {}
        
        for symbol in symbols:
            if symbol in self.models:
                # Prepare current data
                symbol_data = current_data[symbol].tail(60)  # Last 60 days
                scaled_data = self.scalers[symbol].transform(
                    symbol_data.values.reshape(-1, 1)
                )
                
                # Reshape for prediction
                X_current = scaled_data.reshape((1, 60, 1))
                
                # Make prediction
                scaled_prediction = self.models[symbol].predict(X_current, verbose=0)
                
                # Inverse transform
                prediction = self.scalers[symbol].inverse_transform(scaled_prediction)
                
                # Calculate expected return
                current_price = symbol_data.iloc[-1]
                expected_return = (prediction[0][0] - current_price) / current_price
                
                predictions[symbol] = {
                    'current_price': current_price,
                    'predicted_price': prediction[0][0],
                    'expected_return': expected_return,
                    'signal': 'BUY' if expected_return > 0.02 else 'SELL' if expected_return < -0.02 else 'HOLD'
                }
        
        return predictions
    
    def create_portfolio_allocation(self, predictions, risk_budget=0.1):
        """Create portfolio allocation based on predictions"""
        
        # Filter high-confidence predictions
        high_conf_predictions = {
            symbol: pred for symbol, pred in predictions.items()
            if abs(pred['expected_return']) > 0.01  # 1% threshold
        }
        
        if not high_conf_predictions:
            return {}  # No allocation if no confident predictions
        
        # Calculate position sizes using Kelly Criterion approximation
        allocations = {}
        total_weight = 0
        
        for symbol, pred in high_conf_predictions.items():
            expected_return = pred['expected_return']
            
            # Simplified Kelly fraction (assuming 60% win rate, 1.5 win/loss ratio)
            kelly_fraction = 0.6 * 1.5 - 0.4  # = 0.5
            
            # Adjust by expected return and risk budget
            position_size = kelly_fraction * expected_return * risk_budget
            position_size = max(-0.2, min(0.2, position_size))  # Cap at ±20%
            
            allocations[symbol] = position_size
            total_weight += abs(position_size)
        
        # Normalize to risk budget
        if total_weight > 0:
            scale_factor = risk_budget / total_weight
            allocations = {k: v * scale_factor for k, v in allocations.items()}
        
        return allocations
```

## 🚀 2025 Cutting-Edge Applications

### ⚛️ Quantum-Enhanced Portfolio Optimization

```python
from qiskit import QuantumCircuit, execute, Aer
from qiskit.optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np

class QuantumPortfolioOptimizer:
    def __init__(self):
        self.quantum_backend = Aer.get_backend('qasm_simulator')
        
    def quantum_portfolio_optimization(self, expected_returns, cov_matrix, risk_aversion=1.0):
        """
        Quantum-enhanced portfolio optimization using QAOA
        """
        n_assets = len(expected_returns)
        
        # Create quadratic program
        qp = QuadraticProgram()
        
        # Add variables (portfolio weights)
        for i in range(n_assets):
            qp.binary_var(f'x_{i}')
        
        # Objective: maximize return - risk penalty
        # This is a simplified binary version
        linear_coeffs = expected_returns.values
        quadratic_coeffs = risk_aversion * cov_matrix.values
        
        qp.maximize(linear=linear_coeffs, quadratic=quadratic_coeffs)
        
        # Constraint: sum of weights = 1 (simplified for binary case)
        qp.linear_constraint(
            linear={f'x_{i}': 1 for i in range(n_assets)},
            sense='==',
            rhs=1
        )
        
        # Solve using quantum algorithm
        qaoa = MinimumEigenOptimizer()
        result = qaoa.solve(qp)
        
        return result
    
    def hybrid_quantum_classical_optimization(self, returns_data, lookback=252):
        """
        Hybrid quantum-classical approach for dynamic portfolio optimization
        """
        
        # Classical preprocessing
        rolling_returns = returns_data.rolling(lookback)
        
        portfolio_allocations = []
        
        for window in rolling_returns:
            if len(window.dropna()) == lookback:
                # Calculate expected returns and covariance
                expected_returns = window.mean() * 252  # Annualized
                cov_matrix = window.cov() * 252
                
                # Quantum optimization
                quantum_result = self.quantum_portfolio_optimization(
                    expected_returns, cov_matrix
                )
                
                portfolio_allocations.append(quantum_result)
        
        return portfolio_allocations
```

## ✅ Quantitative Analysis Progression

### Week 1-2: Statistical Foundations ✅
- [ ] Master probability distributions in finance
- [ ] Implement Monte Carlo simulations
- [ ] Build Bayesian inference models
- [ ] Create statistical testing framework

### Week 3-4: Time Series Mastery ✅
- [ ] ARIMA/GARCH model implementation
- [ ] Cointegration analysis for pairs trading
- [ ] Regime detection algorithms
- [ ] Kalman filter applications

### Week 5-6: Machine Learning Integration ✅
- [ ] Feature engineering for financial data
- [ ] Cross-validation for time series
- [ ] Ensemble methods for prediction
- [ ] Deep learning model development

### Week 7: Advanced Methods ✅
- [ ] Quantum computing applications
- [ ] ESG quantitative frameworks
- [ ] Alternative data integration
- [ ] Climate risk modeling

### Week 8: Model Validation ✅
- [ ] Backtesting methodologies
- [ ] Walk-forward analysis
- [ ] Model performance monitoring
- [ ] Production deployment strategies

## 💎 Key Principles

### 🎯 Model Building Principles

1. **Simplicity over Complexity** - Simple models often outperform complex ones
2. **Economic Intuition** - Models should make economic sense
3. **Out-of-Sample Validation** - Never trust in-sample results
4. **Regime Awareness** - Models fail when regimes change
5. **Human Factor** - Account for behavioral biases

### 🚀 2025 Quantitative Trends

1. **AI-First Modeling** - AI augments traditional quant methods
2. **Quantum Advantage** - Quantum computing for optimization
3. **Alternative Data Integration** - Non-traditional data sources
4. **Real-time Adaptation** - Models adapt in real-time
5. **Explainable AI** - Interpretable complex models

---

**Next**: [[05-Phân-tích-định-lượng/Returns Distribution - Normal vs Reality|Statistical Foundations]]

**Advanced**: [[05-Phân-tích-định-lượng/🤖 AI và Machine Learning Hiện Đại|AI/ML Applications]]

---

*"All models are wrong, but some are useful"* - George Box 📊

*"The goal is not to predict the future, but to prepare for it"* - Quant Wisdom 2025 🔮
