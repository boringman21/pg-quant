# 🌱 ESG và Sustainable Investing

## 📝 Tổng Quan

ESG (Environmental, Social, Governance) đã trở thành xu hướng chủ đạo trong đầu tư. Đến tháng 12/2020, tài sản ESG đã chiếm 1/3 trong tổng số 51 nghìn tỷ USD tài sản được quản lý chuyên nghiệp tại Mỹ, với dự báo có thể vượt 50 nghìn tỷ USD vào năm 2025.

## 🎯 Xu Hướng ESG 2025

### 1. Climate Reality Check
- **Extreme Weather Impact**: 84% participants tin rằng extreme weather sẽ ảnh hưởng xấu đến kinh tế
- **Climate Adaptation**: Đầu tư vào resilience solutions
- **Carbon Markets**: Chất lượng standards tăng cao

### 2. AI's Data Dilemma
- **Data Quality**: Access to training data ngày càng thu hẹp
- **AI Ethics**: Transparency trong AI decision-making
- **Privacy Concerns**: Human capital management risks

### 3. Social Risks Rise
- **Tech Dominance**: IT và communication services tăng gấp đôi weight
- **Human Capital**: Quản lý nhân lực trở nên quan trọng
- **Data Privacy**: Risks mới từ technology adoption

## 📊 ESG Metrics và Measurement

### Environmental Metrics
```python
import pandas as pd
import numpy as np

class ESGAnalyzer:
    def __init__(self):
        self.environmental_metrics = [
            'carbon_emissions', 'energy_efficiency', 'water_usage',
            'waste_management', 'renewable_energy_adoption'
        ]
    
    def calculate_environmental_score(self, company_data):
        """
        Tính environmental score cho công ty
        """
        scores = {}
        
        # Carbon footprint score
        carbon_score = self.normalize_carbon_emissions(
            company_data['carbon_emissions']
        )
        scores['carbon'] = carbon_score
        
        # Energy efficiency score
        energy_score = self.calculate_energy_efficiency(
            company_data['energy_consumption'],
            company_data['revenue']
        )
        scores['energy'] = energy_score
        
        # Renewable energy adoption
        renewable_score = company_data['renewable_energy_percentage'] / 100
        scores['renewable'] = renewable_score
        
        # Composite environmental score
        weights = {'carbon': 0.4, 'energy': 0.3, 'renewable': 0.3}
        environmental_score = sum(
            scores[metric] * weights[metric] 
            for metric in weights
        )
        
        return environmental_score
    
    def normalize_carbon_emissions(self, emissions):
        """
        Normalize carbon emissions (lower is better)
        """
        # Industry benchmarks
        industry_avg = 1000  # tons CO2/year
        if emissions <= industry_avg * 0.5:
            return 1.0  # Excellent
        elif emissions <= industry_avg:
            return 0.7  # Good
        elif emissions <= industry_avg * 1.5:
            return 0.4  # Average
        else:
            return 0.1  # Poor
```

### Social Metrics
```python
def calculate_social_score(self, company_data):
    """
    Tính social score dựa trên human capital metrics
    """
    social_factors = {
        'employee_satisfaction': company_data.get('employee_satisfaction', 0),
        'diversity_index': company_data.get('diversity_index', 0),
        'safety_incidents': company_data.get('safety_incidents', 0),
        'community_investment': company_data.get('community_investment', 0),
        'data_privacy_score': company_data.get('data_privacy_score', 0)
    }
    
    # Normalize each factor
    normalized_scores = {}
    
    # Employee satisfaction (0-100 scale)
    normalized_scores['employee'] = social_factors['employee_satisfaction'] / 100
    
    # Diversity index (0-1 scale)
    normalized_scores['diversity'] = social_factors['diversity_index']
    
    # Safety (inverse - fewer incidents is better)
    max_incidents = 50  # benchmark
    normalized_scores['safety'] = max(0, 1 - social_factors['safety_incidents'] / max_incidents)
    
    # Community investment (as % of revenue)
    normalized_scores['community'] = min(1, social_factors['community_investment'] / 0.02)
    
    # Data privacy compliance
    normalized_scores['privacy'] = social_factors['data_privacy_score'] / 100
    
    # Weighted average
    weights = {
        'employee': 0.25,
        'diversity': 0.20,
        'safety': 0.20,
        'community': 0.15,
        'privacy': 0.20
    }
    
    social_score = sum(
        normalized_scores[factor] * weights[factor]
        for factor in weights
    )
    
    return social_score
```

### Governance Metrics
```python
def calculate_governance_score(self, company_data):
    """
    Tính governance score
    """
    governance_factors = {
        'board_independence': company_data.get('independent_directors_ratio', 0),
        'executive_compensation': company_data.get('ceo_pay_ratio', 0),
        'audit_quality': company_data.get('audit_score', 0),
        'shareholder_rights': company_data.get('shareholder_rights_score', 0),
        'transparency': company_data.get('transparency_score', 0),
        'ai_oversight': company_data.get('ai_governance_score', 0)
    }
    
    # Board independence (>50% is good)
    board_score = min(1, governance_factors['board_independence'] / 0.5)
    
    # Executive compensation (lower ratio is better)
    max_ratio = 500  # benchmark CEO-to-worker pay ratio
    comp_score = max(0, 1 - governance_factors['executive_compensation'] / max_ratio)
    
    # Audit quality (0-100 scale)
    audit_score = governance_factors['audit_quality'] / 100
    
    # Shareholder rights (0-100 scale)
    shareholder_score = governance_factors['shareholder_rights'] / 100
    
    # Transparency (0-100 scale)
    transparency_score = governance_factors['transparency'] / 100
    
    # AI oversight (new metric for 2025)
    ai_score = governance_factors['ai_oversight'] / 100
    
    # Weighted governance score
    weights = {
        'board': 0.20,
        'compensation': 0.15,
        'audit': 0.20,
        'shareholder': 0.15,
        'transparency': 0.15,
        'ai': 0.15  # Growing importance
    }
    
    governance_score = (
        board_score * weights['board'] +
        comp_score * weights['compensation'] +
        audit_score * weights['audit'] +
        shareholder_score * weights['shareholder'] +
        transparency_score * weights['transparency'] +
        ai_score * weights['ai']
    )
    
    return governance_score
```

## 🚀 Alternative Data cho ESG

### Satellite Data Analysis
```python
import requests
import numpy as np
from datetime import datetime

class SatelliteESGAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.satellite-data.com"
    
    def analyze_environmental_impact(self, company_coordinates, time_period):
        """
        Phân tích environmental impact qua satellite data
        """
        # Lấy dữ liệu satellite
        satellite_data = self.fetch_satellite_data(
            company_coordinates, time_period
        )
        
        # Phân tích pollution levels
        pollution_analysis = self.analyze_pollution(satellite_data)
        
        # Deforestation tracking
        deforestation_score = self.track_deforestation(satellite_data)
        
        # Water quality monitoring
        water_quality = self.monitor_water_quality(satellite_data)
        
        return {
            'pollution_score': pollution_analysis,
            'deforestation_risk': deforestation_score,
            'water_impact': water_quality,
            'overall_environmental_risk': self.calculate_env_risk(
                pollution_analysis, deforestation_score, water_quality
            )
        }
    
    def fetch_satellite_data(self, coordinates, time_period):
        """
        Lấy dữ liệu từ satellite APIs
        """
        # Example API call
        response = requests.get(
            f"{self.base_url}/environmental-data",
            params={
                'lat': coordinates['lat'],
                'lon': coordinates['lon'],
                'start_date': time_period['start'],
                'end_date': time_period['end'],
                'api_key': self.api_key
            }
        )
        return response.json()
```

### Social Media Sentiment for ESG
```python
from transformers import pipeline
import pandas as pd

class ESGSentimentAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    
    def analyze_esg_sentiment(self, company_ticker, news_articles):
        """
        Phân tích ESG sentiment từ news và social media
        """
        esg_keywords = {
            'environmental': ['climate', 'carbon', 'emissions', 'renewable', 'green'],
            'social': ['diversity', 'inclusion', 'safety', 'workers', 'community'],
            'governance': ['board', 'transparency', 'ethics', 'compliance', 'ai governance']
        }
        
        esg_sentiments = {'environmental': [], 'social': [], 'governance': []}
        
        for article in news_articles:
            # Classify ESG category
            category = self.classify_esg_category(article['text'], esg_keywords)
            
            if category:
                # Analyze sentiment
                sentiment = self.sentiment_analyzer(article['text'])
                esg_sentiments[category].append(sentiment[0]['score'])
        
        # Calculate average sentiment for each ESG pillar
        esg_scores = {}
        for category, sentiments in esg_sentiments.items():
            if sentiments:
                esg_scores[category] = np.mean(sentiments)
            else:
                esg_scores[category] = 0.5  # neutral
        
        return esg_scores
    
    def classify_esg_category(self, text, keywords):
        """
        Phân loại text vào ESG category
        """
        text_lower = text.lower()
        
        for category, kw_list in keywords.items():
            if any(keyword in text_lower for keyword in kw_list):
                return category
        
        return None
```

## 📈 ESG Investment Strategies

### ESG Integration Strategy
```python
import yfinance as yf
import pandas as pd

class ESGIntegrationStrategy:
    def __init__(self):
        self.esg_analyzer = ESGAnalyzer()
        self.weights = {'E': 0.4, 'S': 0.3, 'G': 0.3}
    
    def create_esg_portfolio(self, universe, esg_data, min_esg_score=0.7):
        """
        Tạo ESG-compliant portfolio
        """
        # Filter stocks based on ESG criteria
        esg_eligible = []
        
        for ticker in universe:
            if ticker in esg_data:
                company_data = esg_data[ticker]
                
                # Calculate composite ESG score
                esg_score = self.calculate_composite_esg_score(company_data)
                
                if esg_score >= min_esg_score:
                    esg_eligible.append({
                        'ticker': ticker,
                        'esg_score': esg_score,
                        'environmental': company_data['environmental_score'],
                        'social': company_data['social_score'],
                        'governance': company_data['governance_score']
                    })
        
        # Sort by ESG score
        esg_eligible.sort(key=lambda x: x['esg_score'], reverse=True)
        
        # Portfolio construction
        portfolio = self.construct_esg_portfolio(esg_eligible)
        
        return portfolio
    
    def calculate_composite_esg_score(self, company_data):
        """
        Tính composite ESG score
        """
        e_score = company_data['environmental_score']
        s_score = company_data['social_score']
        g_score = company_data['governance_score']
        
        composite_score = (
            e_score * self.weights['E'] +
            s_score * self.weights['S'] +
            g_score * self.weights['G']
        )
        
        return composite_score
    
    def construct_esg_portfolio(self, esg_eligible):
        """
        Xây dựng portfolio với ESG constraints
        """
        # Top ESG stocks
        top_esg = esg_eligible[:20]  # Top 20 stocks
        
        # Equal weight initially
        equal_weight = 1.0 / len(top_esg)
        
        portfolio = {}
        for stock in top_esg:
            portfolio[stock['ticker']] = {
                'weight': equal_weight,
                'esg_score': stock['esg_score'],
                'esg_breakdown': {
                    'environmental': stock['environmental'],
                    'social': stock['social'],
                    'governance': stock['governance']
                }
            }
        
        return portfolio
```

### ESG Factor Investing
```python
def esg_factor_analysis(returns_data, esg_scores):
    """
    Phân tích ESG như một factor
    """
    # Merge returns with ESG scores
    merged_data = pd.merge(
        returns_data, esg_scores, 
        left_index=True, right_index=True
    )
    
    # Create ESG factor
    # High ESG minus Low ESG portfolio
    high_esg = merged_data[merged_data['esg_score'] > merged_data['esg_score'].quantile(0.8)]
    low_esg = merged_data[merged_data['esg_score'] < merged_data['esg_score'].quantile(0.2)]
    
    high_esg_returns = high_esg.groupby('date')['returns'].mean()
    low_esg_returns = low_esg.groupby('date')['returns'].mean()
    
    # ESG factor = High ESG - Low ESG
    esg_factor = high_esg_returns - low_esg_returns
    
    # Calculate factor statistics
    factor_stats = {
        'mean_return': esg_factor.mean(),
        'volatility': esg_factor.std(),
        'sharpe_ratio': esg_factor.mean() / esg_factor.std(),
        'max_drawdown': calculate_max_drawdown(esg_factor.cumsum())
    }
    
    return esg_factor, factor_stats
```

## 🌍 Climate Risk Modeling

### Physical Risk Assessment
```python
import numpy as np
from scipy import stats

class ClimateRiskModel:
    def __init__(self):
        self.risk_factors = [
            'temperature_change', 'precipitation_change', 
            'sea_level_rise', 'extreme_weather_frequency'
        ]
    
    def assess_physical_risk(self, company_locations, climate_scenarios):
        """
        Đánh giá physical climate risk
        """
        risk_scores = {}
        
        for location in company_locations:
            location_risk = 0
            
            for scenario in climate_scenarios:
                # Temperature risk
                temp_risk = self.calculate_temperature_risk(
                    location, scenario['temperature_change']
                )
                
                # Precipitation risk
                precip_risk = self.calculate_precipitation_risk(
                    location, scenario['precipitation_change']
                )
                
                # Sea level risk
                sea_risk = self.calculate_sea_level_risk(
                    location, scenario['sea_level_rise']
                )
                
                # Extreme weather risk
                extreme_risk = self.calculate_extreme_weather_risk(
                    location, scenario['extreme_weather_frequency']
                )
                
                # Composite risk for this scenario
                scenario_risk = np.mean([temp_risk, precip_risk, sea_risk, extreme_risk])
                location_risk += scenario_risk * scenario['probability']
            
            risk_scores[location] = location_risk
        
        return risk_scores
    
    def calculate_temperature_risk(self, location, temp_change):
        """
        Tính toán risk từ temperature change
        """
        # Industry-specific temperature sensitivity
        sensitivity_mapping = {
            'agriculture': 0.8,
            'energy': 0.6,
            'tourism': 0.7,
            'manufacturing': 0.4,
            'technology': 0.2
        }
        
        industry = location.get('industry', 'technology')
        sensitivity = sensitivity_mapping.get(industry, 0.5)
        
        # Risk increases exponentially with temperature
        risk = 1 - np.exp(-sensitivity * temp_change)
        return min(1, max(0, risk))
```

### Transition Risk Assessment
```python
def assess_transition_risk(self, company_data, policy_scenarios):
    """
    Đánh giá transition risk từ policy changes
    """
    transition_factors = {
        'carbon_pricing': company_data.get('carbon_intensity', 0),
        'regulatory_change': company_data.get('regulatory_exposure', 0),
        'technology_shift': company_data.get('green_tech_adoption', 0),
        'market_preference': company_data.get('esg_demand_exposure', 0)
    }
    
    risk_scores = {}
    
    for scenario in policy_scenarios:
        scenario_risk = 0
        
        # Carbon pricing impact
        carbon_price = scenario['carbon_price']  # $/ton CO2
        carbon_risk = (
            transition_factors['carbon_pricing'] * carbon_price / 1000
        )
        
        # Regulatory change impact
        reg_stringency = scenario['regulatory_stringency']  # 0-1 scale
        reg_risk = (
            transition_factors['regulatory_change'] * reg_stringency
        )
        
        # Technology disruption
        tech_disruption = scenario['technology_disruption']  # 0-1 scale
        tech_risk = (
            (1 - transition_factors['technology_shift']) * tech_disruption
        )
        
        # Market preference shift
        market_shift = scenario['market_preference_shift']  # 0-1 scale
        market_risk = (
            (1 - transition_factors['market_preference']) * market_shift
        )
        
        # Composite transition risk
        scenario_risk = np.mean([carbon_risk, reg_risk, tech_risk, market_risk])
        risk_scores[scenario['name']] = scenario_risk
    
    return risk_scores
```

## 🎯 Future of ESG Investing

### 2025 Trends
1. **AI-Enhanced ESG Analysis**: ML models cho ESG scoring
2. **Real-time ESG Monitoring**: Satellite data và IoT sensors
3. **Climate Stress Testing**: Scenario-based risk assessment
4. **ESG Integration**: ESG factors trong traditional models
5. **Sustainable Finance Regulation**: Stricter disclosure requirements

### Emerging Metrics
- **Carbon Footprint Tracking**: Real-time emissions monitoring
- **Biodiversity Impact**: Ecosystem health assessment
- **Social Impact Measurement**: Community benefit quantification
- **AI Ethics Scores**: Responsible AI deployment
- **Supply Chain ESG**: End-to-end sustainability

### Technology Integration
```python
# ESG-enhanced portfolio optimization
def esg_optimized_portfolio(expected_returns, covariance_matrix, esg_scores, 
                          min_esg_score=0.7, esg_weight=0.3):
    """
    Portfolio optimization với ESG constraints
    """
    from scipy.optimize import minimize
    
    n_assets = len(expected_returns)
    
    def objective(weights):
        # Traditional mean-variance objective
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        
        # ESG penalty
        portfolio_esg = np.sum(esg_scores * weights)
        esg_penalty = (1 - portfolio_esg) * esg_weight
        
        # Combined objective: maximize return - variance - ESG penalty
        return -(portfolio_return - portfolio_variance - esg_penalty)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        {'type': 'ineq', 'fun': lambda w: np.sum(esg_scores * w) - min_esg_score}  # Min ESG
    ]
    
    bounds = [(0, 1) for _ in range(n_assets)]
    
    result = minimize(objective, np.ones(n_assets)/n_assets, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x
```

---

**Tags:** #esg #sustainable-investing #climate-risk #alternative-data #ai-ethics
**Ngày tạo:** 2024-12-19
**Trạng thái:** #cutting-edge