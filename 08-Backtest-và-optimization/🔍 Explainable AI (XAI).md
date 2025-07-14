# 🔍 Explainable AI (XAI) 

## 📝 Tổng Quan

Explainable AI (XAI) đã trở thành yêu cầu bắt buộc trong quant finance. Với Colorado AI Act 2026 và EU AI Act yêu cầu transparency, việc hiểu được tại sao AI models đưa ra quyết định là crucial. XAI giúp regulators, investors, và traders hiểu được "black box" của AI systems.

## 🎯 Tại Sao XAI Quan Trọng?

### 1. Regulatory Compliance
- **Colorado AI Act 2026**: Transparency requirements
- **EU AI Act**: Explainable AI mandates
- **SEC Oversight**: Algorithmic trading regulations
- **GDPR**: Right to explanation

### 2. Risk Management
- **Model Validation**: Hiểu model behavior
- **Bias Detection**: Identify unfair predictions
- **Failure Analysis**: Debug model failures
- **Stress Testing**: Understand model limits

### 3. Business Value
- **Trust Building**: Stakeholder confidence
- **Model Improvement**: Identify weaknesses
- **Feature Engineering**: Better input selection
- **Regulatory Approval**: Faster compliance

## 🛠️ XAI Techniques

### 1. SHAP (SHapley Additive exPlanations)
```python
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.explainer = shap.TreeExplainer(model)
        
    def explain_prediction(self, X_test, sample_idx=0):
        """
        Giải thích prediction cho một sample
        """
        # Tính SHAP values
        shap_values = self.explainer.shap_values(X_test)
        
        # Lấy sample cụ thể
        sample_shap = shap_values[sample_idx]
        sample_features = X_test.iloc[sample_idx]
        
        # Tạo explanation
        explanation = {
            'prediction': self.model.predict(X_test.iloc[[sample_idx]])[0],
            'base_value': self.explainer.expected_value,
            'shap_values': dict(zip(X_test.columns, sample_shap)),
            'feature_values': dict(zip(X_test.columns, sample_features))
        }
        
        return explanation
    
    def plot_waterfall(self, explanation, title="SHAP Waterfall Plot"):
        """
        Vẽ waterfall plot cho explanation
        """
        shap.waterfall_plot(
            shap.Explanation(
                values=list(explanation['shap_values'].values()),
                base_values=explanation['base_value'],
                data=list(explanation['feature_values'].values()),
                feature_names=list(explanation['shap_values'].keys())
            )
        )
        plt.title(title)
        plt.show()
    
    def plot_force_plot(self, explanation):
        """
        Vẽ force plot cho explanation
        """
        shap.force_plot(
            base_value=explanation['base_value'],
            shap_values=list(explanation['shap_values'].values()),
            features=list(explanation['feature_values'].values()),
            feature_names=list(explanation['shap_values'].keys())
        )
    
    def global_feature_importance(self, X_test):
        """
        Tính global feature importance
        """
        shap_values = self.explainer.shap_values(X_test)
        
        # Mean absolute SHAP values
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_summary(self, X_test, plot_type='dot'):
        """
        Vẽ summary plot
        """
        shap_values = self.explainer.shap_values(X_test)
        
        shap.summary_plot(
            shap_values, X_test, 
            plot_type=plot_type,
            feature_names=X_test.columns
        )

# Example usage
def explain_stock_prediction():
    # Load data
    data = pd.read_csv('stock_features.csv')
    X = data.drop(['target', 'date'], axis=1)
    y = data['target']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = SHAPExplainer(model, X)
    
    # Explain prediction
    explanation = explainer.explain_prediction(X, sample_idx=0)
    
    print("Prediction:", explanation['prediction'])
    print("Base value:", explanation['base_value'])
    print("\nFeature contributions:")
    for feature, shap_value in explanation['shap_values'].items():
        print(f"{feature}: {shap_value:.4f}")
    
    # Plot explanations
    explainer.plot_waterfall(explanation)
    explainer.plot_summary(X)
    
    return explanation
```

### 2. LIME (Local Interpretable Model-agnostic Explanations)
```python
import lime
import lime.lime_tabular
from sklearn.neural_network import MLPRegressor

class LIMEExplainer:
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        
        # Create LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=['Stock Return'],
            mode='regression'
        )
    
    def explain_instance(self, instance, num_features=10):
        """
        Giải thích một instance cụ thể
        """
        # Get explanation
        explanation = self.explainer.explain_instance(
            instance.values,
            self.model.predict,
            num_features=num_features
        )
        
        # Parse explanation
        explanation_dict = {}
        for feature, importance in explanation.as_list():
            explanation_dict[feature] = importance
        
        return {
            'explanation': explanation_dict,
            'prediction': self.model.predict([instance.values])[0],
            'lime_explanation': explanation
        }
    
    def plot_explanation(self, explanation):
        """
        Vẽ LIME explanation
        """
        explanation['lime_explanation'].show_in_notebook(show_table=True)
        
        # Create matplotlib plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(explanation['explanation'].keys())
        importances = list(explanation['explanation'].values())
        
        colors = ['red' if x < 0 else 'green' for x in importances]
        
        ax.barh(features, importances, color=colors)
        ax.set_xlabel('Feature Importance')
        ax.set_title('LIME Explanation')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_predictions(self, instances):
        """
        So sánh explanations cho multiple instances
        """
        explanations = []
        
        for i, instance in instances.iterrows():
            explanation = self.explain_instance(instance)
            explanations.append(explanation)
        
        return explanations

# Example usage
def explain_trading_model():
    # Load data
    data = pd.read_csv('trading_features.csv')
    X = data.drop(['target', 'date'], axis=1)
    y = data['target']
    
    # Train neural network
    model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = LIMEExplainer(model, X, X.columns.tolist())
    
    # Explain specific instance
    explanation = explainer.explain_instance(X.iloc[0])
    
    print("Prediction:", explanation['prediction'])
    print("Feature explanations:")
    for feature, importance in explanation['explanation'].items():
        print(f"{feature}: {importance:.4f}")
    
    # Plot explanation
    explainer.plot_explanation(explanation)
    
    return explanation
```

### 3. Integrated Gradients
```python
import torch
import torch.nn as nn
import numpy as np

class IntegratedGradients:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def integrated_gradients(self, inputs, target_label_idx, baseline=None, steps=50):
        """
        Tính Integrated Gradients
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        
        # Generate alphas
        alphas = torch.linspace(0, 1, steps)
        
        # Initialize gradients
        gradients = []
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated)
            
            # Backward pass
            self.model.zero_grad()
            output[target_label_idx].backward()
            
            # Store gradients
            gradients.append(interpolated.grad.clone())
        
        # Average gradients
        avg_gradients = torch.mean(torch.stack(gradients), dim=0)
        
        # Multiply by input difference
        integrated_gradients = (inputs - baseline) * avg_gradients
        
        return integrated_gradients
    
    def explain_prediction(self, inputs, target_label_idx, feature_names):
        """
        Giải thích prediction sử dụng Integrated Gradients
        """
        # Calculate integrated gradients
        attributions = self.integrated_gradients(inputs, target_label_idx)
        
        # Convert to numpy
        attributions_np = attributions.detach().numpy()
        
        # Create explanation dictionary
        explanation = {}
        for i, feature_name in enumerate(feature_names):
            explanation[feature_name] = attributions_np[i]
        
        return explanation
    
    def plot_attributions(self, explanation, title="Integrated Gradients"):
        """
        Vẽ attributions
        """
        features = list(explanation.keys())
        attributions = list(explanation.values())
        
        plt.figure(figsize=(12, 6))
        colors = ['red' if x < 0 else 'green' for x in attributions]
        
        plt.bar(features, attributions, color=colors)
        plt.xlabel('Features')
        plt.ylabel('Attribution')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.show()

# Example usage with PyTorch model
class StockPredictionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockPredictionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def explain_pytorch_model():
    # Create model
    model = StockPredictionNet(input_size=20, hidden_size=50, output_size=1)
    
    # Load pretrained weights (example)
    # model.load_state_dict(torch.load('model_weights.pth'))
    
    # Create explainer
    explainer = IntegratedGradients(model)
    
    # Example input
    inputs = torch.randn(1, 20)
    feature_names = [f'feature_{i}' for i in range(20)]
    
    # Explain prediction
    explanation = explainer.explain_prediction(inputs, 0, feature_names)
    
    print("Feature attributions:")
    for feature, attribution in explanation.items():
        print(f"{feature}: {attribution:.4f}")
    
    # Plot attributions
    explainer.plot_attributions(explanation)
    
    return explanation
```

### 4. Attention Mechanisms
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionExplainableModel(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length):
        super(AttentionExplainableModel, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layer
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Attention weights
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output
        output = self.output(context)
        
        return output, attention_weights
    
    def explain_prediction(self, x, feature_names):
        """
        Giải thích prediction bằng attention weights
        """
        with torch.no_grad():
            output, attention_weights = self.forward(x)
        
        # Convert to numpy
        attention_weights_np = attention_weights.squeeze().numpy()
        
        # Create explanation
        explanation = {
            'prediction': output.item(),
            'time_step_importance': attention_weights_np.tolist(),
            'most_important_timestep': np.argmax(attention_weights_np),
            'attention_distribution': dict(zip(
                range(len(attention_weights_np)), 
                attention_weights_np
            ))
        }
        
        return explanation
    
    def plot_attention(self, explanation, title="Attention Weights"):
        """
        Vẽ attention weights
        """
        time_steps = list(explanation['attention_distribution'].keys())
        weights = list(explanation['attention_distribution'].values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(time_steps, weights, color='blue', alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('Attention Weight')
        plt.title(title)
        plt.axhline(y=np.mean(weights), color='red', linestyle='--', 
                   label=f'Average: {np.mean(weights):.4f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Example usage
def explain_time_series_model():
    # Create model
    model = AttentionExplainableModel(
        input_size=10, 
        hidden_size=50, 
        sequence_length=20
    )
    
    # Example input (batch_size=1, seq_len=20, features=10)
    x = torch.randn(1, 20, 10)
    
    # Explain prediction
    explanation = model.explain_prediction(x, feature_names=None)
    
    print("Prediction:", explanation['prediction'])
    print("Most important timestep:", explanation['most_important_timestep'])
    
    # Plot attention
    model.plot_attention(explanation)
    
    return explanation
```

## 📊 XAI trong Trading Systems

### Explainable Trading Signals
```python
class ExplainableTradingSystem:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = shap.TreeExplainer(model)
    
    def generate_explainable_signal(self, features):
        """
        Tạo trading signal với explanation
        """
        # Make prediction
        signal = self.model.predict([features])[0]
        
        # Generate explanation
        shap_values = self.shap_explainer.shap_values([features])
        
        # Create detailed explanation
        explanation = {
            'signal': signal,
            'confidence': self.calculate_confidence(features),
            'feature_contributions': dict(zip(self.feature_names, shap_values[0])),
            'top_reasons': self.get_top_reasons(shap_values[0]),
            'risk_factors': self.identify_risk_factors(shap_values[0])
        }
        
        return explanation
    
    def get_top_reasons(self, shap_values, top_n=5):
        """
        Lấy top reasons cho signal
        """
        feature_importance = list(zip(self.feature_names, shap_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_reasons = []
        for feature, importance in feature_importance[:top_n]:
            direction = "bullish" if importance > 0 else "bearish"
            top_reasons.append({
                'feature': feature,
                'importance': importance,
                'direction': direction,
                'explanation': self.generate_feature_explanation(feature, importance)
            })
        
        return top_reasons
    
    def generate_feature_explanation(self, feature, importance):
        """
        Tạo human-readable explanation cho feature
        """
        explanations = {
            'rsi': f"RSI indicates {'oversold' if importance > 0 else 'overbought'} conditions",
            'macd': f"MACD suggests {'bullish' if importance > 0 else 'bearish'} momentum",
            'volume': f"Volume shows {'high' if importance > 0 else 'low'} trading activity",
            'price_momentum': f"Price momentum is {'positive' if importance > 0 else 'negative'}",
            'volatility': f"Volatility is {'increasing' if importance > 0 else 'decreasing'}"
        }
        
        return explanations.get(feature, f"{feature} contributes {'positively' if importance > 0 else 'negatively'}")
    
    def create_trading_report(self, explanation):
        """
        Tạo trading report với explanations
        """
        report = f"""
        TRADING SIGNAL REPORT
        =====================
        
        Signal: {explanation['signal']:.4f}
        Confidence: {explanation['confidence']:.2%}
        
        TOP CONTRIBUTING FACTORS:
        """
        
        for i, reason in enumerate(explanation['top_reasons'], 1):
            report += f"""
        {i}. {reason['feature'].upper()}: {reason['explanation']}
           Impact: {reason['importance']:.4f} ({reason['direction']})
        """
        
        if explanation['risk_factors']:
            report += "\n\nRISK FACTORS:"
            for risk in explanation['risk_factors']:
                report += f"\n- {risk}"
        
        return report
    
    def identify_risk_factors(self, shap_values):
        """
        Identify potential risk factors
        """
        risk_factors = []
        
        # High volatility risk
        if 'volatility' in self.feature_names:
            vol_idx = self.feature_names.index('volatility')
            if shap_values[vol_idx] > 0.1:
                risk_factors.append("High volatility detected")
        
        # Conflicting signals
        positive_signals = sum(1 for val in shap_values if val > 0)
        negative_signals = sum(1 for val in shap_values if val < 0)
        
        if abs(positive_signals - negative_signals) < 2:
            risk_factors.append("Conflicting signals detected")
        
        # Low confidence
        if abs(max(shap_values) - min(shap_values)) < 0.05:
            risk_factors.append("Low signal strength")
        
        return risk_factors
```

### Regulatory Compliance Dashboard
```python
class RegulatoryComplianceDashboard:
    def __init__(self):
        self.audit_log = []
        self.explanation_cache = {}
    
    def log_prediction(self, model_id, inputs, prediction, explanation):
        """
        Log prediction cho regulatory audit
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_id': model_id,
            'inputs': inputs.tolist() if hasattr(inputs, 'tolist') else inputs,
            'prediction': prediction,
            'explanation': explanation,
            'model_version': self.get_model_version(model_id)
        }
        
        self.audit_log.append(log_entry)
        
        # Cache explanation
        prediction_id = f"{model_id}_{len(self.audit_log)}"
        self.explanation_cache[prediction_id] = explanation
    
    def generate_compliance_report(self, start_date, end_date):
        """
        Tạo compliance report cho regulators
        """
        filtered_logs = [
            log for log in self.audit_log
            if start_date <= log['timestamp'] <= end_date
        ]
        
        report = {
            'period': f"{start_date} to {end_date}",
            'total_predictions': len(filtered_logs),
            'models_used': list(set(log['model_id'] for log in filtered_logs)),
            'explanation_coverage': self.calculate_explanation_coverage(filtered_logs),
            'bias_analysis': self.analyze_bias(filtered_logs),
            'performance_metrics': self.calculate_performance_metrics(filtered_logs)
        }
        
        return report
    
    def calculate_explanation_coverage(self, logs):
        """
        Tính explanation coverage
        """
        total_predictions = len(logs)
        explained_predictions = sum(1 for log in logs if log['explanation'])
        
        return explained_predictions / total_predictions if total_predictions > 0 else 0
    
    def analyze_bias(self, logs):
        """
        Phân tích bias trong predictions
        """
        # Group by demographic factors (if available)
        predictions_by_group = {}
        
        for log in logs:
            # Example: group by market sector
            sector = log['inputs'].get('sector', 'unknown')
            if sector not in predictions_by_group:
                predictions_by_group[sector] = []
            predictions_by_group[sector].append(log['prediction'])
        
        # Calculate statistics by group
        bias_analysis = {}
        for sector, predictions in predictions_by_group.items():
            bias_analysis[sector] = {
                'mean_prediction': np.mean(predictions),
                'std_prediction': np.std(predictions),
                'count': len(predictions)
            }
        
        return bias_analysis
    
    def export_for_auditors(self, output_path):
        """
        Export data cho external auditors
        """
        audit_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_records': len(self.audit_log),
                'date_range': {
                    'start': min(log['timestamp'] for log in self.audit_log),
                    'end': max(log['timestamp'] for log in self.audit_log)
                }
            },
            'predictions': self.audit_log,
            'explanations': self.explanation_cache
        }
        
        with open(output_path, 'w') as f:
            json.dump(audit_data, f, indent=2)
```

## 🔮 Future of XAI in Finance

### Emerging Trends
1. **Causal AI**: Understanding cause-effect relationships
2. **Contrastive Explanations**: "Why this and not that?"
3. **Multi-modal Explanations**: Text, visual, audio explanations
4. **Interactive Explanations**: User-driven exploration
5. **Federated XAI**: Privacy-preserving explanations

### Regulatory Evolution
- **Global Standards**: International XAI standards
- **Real-time Monitoring**: Continuous explanation validation
- **Automated Compliance**: AI-driven regulatory reporting
- **Stakeholder Transparency**: Public explanation requirements

### Technical Advances
- **Quantum XAI**: Quantum-enhanced explanations
- **Neuromorphic XAI**: Brain-inspired explanations
- **Graph-based XAI**: Relationship explanations
- **Temporal XAI**: Time-aware explanations

---

**Tags:** #explainable-ai #xai #regulatory-compliance #interpretability #transparency
**Ngày tạo:** 2024-12-19
**Trạng thái:** #critical-importance