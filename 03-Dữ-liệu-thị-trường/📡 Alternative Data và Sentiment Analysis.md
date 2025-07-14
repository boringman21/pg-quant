# 📡 Alternative Data và Sentiment Analysis

## 📝 Tổng Quan

Alternative Data đã trở thành competitive advantage trong quant trading. Từ satellite imagery đến social media sentiment, credit card transactions đến weather data - những nguồn dữ liệu phi truyền thống này đang tạo ra alpha trong thị trường hiện đại. Theo JPMorgan, 89% hedge funds hiện đang sử dụng alternative data.

## 🎯 Các Loại Alternative Data

### 1. Satellite Data
- **Retail Foot Traffic**: Parking lot analysis cho retail stocks
- **Oil Storage**: Tank levels cho energy companies
- **Agriculture**: Crop yield predictions
- **Real Estate**: Construction activity monitoring

### 2. Social Media và News Sentiment
- **Twitter Sentiment**: Real-time market mood
- **Reddit WallStreetBets**: Retail investor sentiment
- **News Analysis**: NLP trên financial news
- **Executive Communications**: Earnings call sentiment

### 3. Credit Card và Transaction Data
- **Consumer Spending**: Real-time economic indicators
- **Company Revenue**: Predictive revenue analytics
- **Sector Trends**: Industry performance insights
- **Geographic Analysis**: Regional economic health

### 4. Weather và Environmental Data
- **Agriculture Trading**: Weather impact on crops
- **Energy Consumption**: Temperature effects on utilities
- **Supply Chain**: Weather disruption analysis
- **Insurance**: Natural disaster risk assessment

## 🛰️ Satellite Data Analysis

### Retail Foot Traffic Analysis
```python
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SatelliteRetailAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.satellite-analytics.com"
    
    def analyze_retail_footfall(self, store_locations, time_period):
        """
        Phân tích foot traffic từ satellite data
        """
        footfall_data = []
        
        for location in store_locations:
            # Lấy dữ liệu parking lot từ satellite
            parking_data = self.get_parking_data(
                location['coordinates'], time_period
            )
            
            # Convert parking occupancy to foot traffic
            foot_traffic = self.parking_to_footfall(
                parking_data, location['store_size']
            )
            
            footfall_data.append({
                'store_id': location['store_id'],
                'coordinates': location['coordinates'],
                'foot_traffic': foot_traffic,
                'yoy_change': self.calculate_yoy_change(foot_traffic)
            })
        
        return footfall_data
    
    def get_parking_data(self, coordinates, time_period):
        """
        Lấy dữ liệu parking lot từ satellite images
        """
        response = requests.get(
            f"{self.base_url}/parking-analysis",
            params={
                'lat': coordinates['lat'],
                'lon': coordinates['lon'],
                'start_date': time_period['start'],
                'end_date': time_period['end'],
                'resolution': 'daily',
                'api_key': self.api_key
            }
        )
        
        return response.json()['parking_occupancy']
    
    def parking_to_footfall(self, parking_data, store_size):
        """
        Convert parking occupancy thành foot traffic estimate
        """
        # Conversion factors based on store size
        conversion_factors = {
            'small': 2.5,    # 2.5 customers per car
            'medium': 3.0,   # 3.0 customers per car
            'large': 3.5     # 3.5 customers per car
        }
        
        factor = conversion_factors.get(store_size, 3.0)
        
        foot_traffic = []
        for data_point in parking_data:
            estimated_customers = data_point['occupied_spots'] * factor
            foot_traffic.append({
                'date': data_point['date'],
                'estimated_customers': estimated_customers,
                'confidence': data_point['confidence']
            })
        
        return foot_traffic
    
    def predict_quarterly_revenue(self, footfall_data, historical_revenue):
        """
        Dự đoán quarterly revenue từ foot traffic
        """
        # Tính average daily foot traffic
        avg_daily_traffic = np.mean([
            point['estimated_customers'] for point in footfall_data
        ])
        
        # Historical correlation: foot traffic vs revenue
        traffic_revenue_correlation = self.calculate_correlation(
            footfall_data, historical_revenue
        )
        
        # Predict revenue
        predicted_revenue = avg_daily_traffic * 90 * traffic_revenue_correlation
        
        return predicted_revenue
```

### Oil Storage Analysis
```python
class OilStorageAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def analyze_oil_storage(self, tank_locations, time_period):
        """
        Phân tích oil storage levels từ satellite
        """
        storage_data = []
        
        for location in tank_locations:
            # Lấy dữ liệu tank từ satellite
            tank_data = self.get_tank_data(location, time_period)
            
            # Calculate storage levels
            storage_levels = self.calculate_storage_levels(tank_data)
            
            storage_data.append({
                'location': location['name'],
                'storage_levels': storage_levels,
                'trend': self.calculate_trend(storage_levels),
                'oil_type': location['oil_type']
            })
        
        return storage_data
    
    def get_tank_data(self, location, time_period):
        """
        Lấy dữ liệu tank từ satellite imagery
        """
        # Floating roof tanks: roof height indicates oil level
        response = requests.get(
            f"{self.base_url}/oil-tanks",
            params={
                'lat': location['coordinates']['lat'],
                'lon': location['coordinates']['lon'],
                'start_date': time_period['start'],
                'end_date': time_period['end'],
                'analysis_type': 'floating_roof',
                'api_key': self.api_key
            }
        )
        
        return response.json()['tank_measurements']
    
    def calculate_storage_levels(self, tank_data):
        """
        Tính storage levels từ tank measurements
        """
        storage_levels = []
        
        for measurement in tank_data:
            # Floating roof height indicates oil level
            roof_height = measurement['roof_height']
            tank_height = measurement['tank_height']
            
            # Storage percentage
            storage_pct = (tank_height - roof_height) / tank_height
            
            storage_levels.append({
                'date': measurement['date'],
                'storage_percentage': storage_pct,
                'estimated_barrels': storage_pct * measurement['tank_capacity']
            })
        
        return storage_levels
    
    def predict_oil_prices(self, storage_data, market_data):
        """
        Dự đoán oil prices từ storage levels
        """
        # Aggregate storage levels
        total_storage = sum([
            location['storage_levels'][-1]['estimated_barrels']
            for location in storage_data
        ])
        
        # Historical correlation: storage vs prices
        price_correlation = self.calculate_price_correlation(
            storage_data, market_data
        )
        
        # Predict price direction
        if total_storage > historical_average * 1.1:
            price_direction = 'DOWN'  # High storage = lower prices
        elif total_storage < historical_average * 0.9:
            price_direction = 'UP'    # Low storage = higher prices
        else:
            price_direction = 'STABLE'
        
        return {
            'price_direction': price_direction,
            'storage_level': total_storage,
            'confidence': price_correlation
        }
```

## 📱 Social Media Sentiment Analysis

### Twitter Sentiment Analysis
```python
import tweepy
from transformers import pipeline
import pandas as pd
import numpy as np

class TwitterSentimentAnalyzer:
    def __init__(self, twitter_api_key, twitter_secret):
        # Twitter API setup
        auth = tweepy.OAuthHandler(twitter_api_key, twitter_secret)
        self.api = tweepy.API(auth)
        
        # Sentiment analysis model
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    
    def analyze_stock_sentiment(self, stock_symbol, num_tweets=1000):
        """
        Phân tích sentiment cho một stock từ Twitter
        """
        # Collect tweets
        tweets = self.collect_tweets(stock_symbol, num_tweets)
        
        # Analyze sentiment
        sentiment_scores = []
        for tweet in tweets:
            sentiment = self.sentiment_pipeline(tweet['text'])
            sentiment_scores.append({
                'tweet_id': tweet['id'],
                'text': tweet['text'],
                'sentiment': sentiment[0]['label'],
                'confidence': sentiment[0]['score'],
                'created_at': tweet['created_at'],
                'followers': tweet['user']['followers_count'],
                'retweets': tweet['retweet_count'],
                'likes': tweet['favorite_count']
            })
        
        # Calculate weighted sentiment
        weighted_sentiment = self.calculate_weighted_sentiment(sentiment_scores)
        
        return {
            'stock_symbol': stock_symbol,
            'overall_sentiment': weighted_sentiment,
            'total_tweets': len(tweets),
            'sentiment_breakdown': self.get_sentiment_breakdown(sentiment_scores),
            'top_influencers': self.get_top_influencers(sentiment_scores)
        }
    
    def collect_tweets(self, stock_symbol, num_tweets):
        """
        Collect tweets about a stock
        """
        search_terms = [
            f"${stock_symbol}",
            f"#{stock_symbol}",
            f"{stock_symbol} stock",
            f"{stock_symbol} earnings"
        ]
        
        tweets = []
        for term in search_terms:
            try:
                tweet_data = tweepy.Cursor(
                    self.api.search_tweets,
                    q=term,
                    lang="en",
                    result_type="recent"
                ).items(num_tweets // len(search_terms))
                
                for tweet in tweet_data:
                    tweets.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'user': {
                            'followers_count': tweet.user.followers_count,
                            'verified': tweet.user.verified
                        },
                        'retweet_count': tweet.retweet_count,
                        'favorite_count': tweet.favorite_count
                    })
                    
            except Exception as e:
                print(f"Error collecting tweets for {term}: {e}")
        
        return tweets
    
    def calculate_weighted_sentiment(self, sentiment_scores):
        """
        Calculate weighted sentiment based on user influence
        """
        total_weight = 0
        weighted_sum = 0
        
        for score in sentiment_scores:
            # Weight based on followers, retweets, and likes
            influence_weight = (
                score['followers'] * 0.5 +
                score['retweets'] * 10 +
                score['likes'] * 5
            )
            
            # Convert sentiment to numeric score
            sentiment_score = self.sentiment_to_numeric(
                score['sentiment'], score['confidence']
            )
            
            weighted_sum += sentiment_score * influence_weight
            total_weight += influence_weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def sentiment_to_numeric(self, sentiment, confidence):
        """
        Convert sentiment label to numeric score
        """
        if sentiment == 'POSITIVE':
            return confidence
        elif sentiment == 'NEGATIVE':
            return -confidence
        else:  # NEUTRAL
            return 0
```

### Reddit Sentiment Analysis
```python
import praw
import pandas as pd
from collections import Counter

class RedditSentimentAnalyzer:
    def __init__(self, reddit_client_id, reddit_client_secret, reddit_user_agent):
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
    
    def analyze_wallstreetbets_sentiment(self, limit=1000):
        """
        Phân tích sentiment từ r/wallstreetbets
        """
        subreddit = self.reddit.subreddit('wallstreetbets')
        
        posts = []
        for submission in subreddit.hot(limit=limit):
            posts.append({
                'title': submission.title,
                'text': submission.selftext,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc,
                'author': str(submission.author),
                'upvote_ratio': submission.upvote_ratio
            })
        
        # Extract stock mentions
        stock_mentions = self.extract_stock_mentions(posts)
        
        # Analyze sentiment for each stock
        stock_sentiments = {}
        for stock in stock_mentions:
            stock_posts = [p for p in posts if stock in p['title'].upper() or stock in p['text'].upper()]
            
            if stock_posts:
                sentiment_analysis = self.analyze_posts_sentiment(stock_posts)
                stock_sentiments[stock] = sentiment_analysis
        
        return stock_sentiments
    
    def extract_stock_mentions(self, posts):
        """
        Extract stock tickers from posts
        """
        import re
        
        stock_pattern = r'\$([A-Z]{1,5})\b'
        stock_mentions = []
        
        for post in posts:
            # Search in title and text
            text = post['title'] + ' ' + post['text']
            matches = re.findall(stock_pattern, text)
            stock_mentions.extend(matches)
        
        # Count occurrences and return top mentions
        stock_counts = Counter(stock_mentions)
        return [stock for stock, count in stock_counts.most_common(20)]
    
    def analyze_posts_sentiment(self, posts):
        """
        Analyze sentiment for posts about a specific stock
        """
        sentiments = []
        
        for post in posts:
            text = post['title'] + ' ' + post['text']
            if len(text) > 10:  # Skip very short posts
                try:
                    sentiment = self.sentiment_pipeline(text[:512])  # Truncate for model
                    sentiments.append({
                        'sentiment': sentiment[0]['label'],
                        'confidence': sentiment[0]['score'],
                        'weight': post['score'] * post['upvote_ratio']  # Weight by Reddit score
                    })
                except Exception as e:
                    continue
        
        if not sentiments:
            return {'overall_sentiment': 'NEUTRAL', 'confidence': 0}
        
        # Calculate weighted average
        weighted_sentiment = self.calculate_weighted_reddit_sentiment(sentiments)
        
        return {
            'overall_sentiment': weighted_sentiment,
            'total_posts': len(posts),
            'sentiment_breakdown': self.get_sentiment_breakdown(sentiments)
        }
```

## 💳 Credit Card Transaction Analysis

### Consumer Spending Analysis
```python
class ConsumerSpendingAnalyzer:
    def __init__(self, data_provider_api):
        self.api = data_provider_api
        self.spending_categories = [
            'retail', 'restaurants', 'gas_stations', 'grocery',
            'entertainment', 'travel', 'healthcare', 'utilities'
        ]
    
    def analyze_sector_spending(self, sector, time_period):
        """
        Phân tích consumer spending cho một sector
        """
        spending_data = self.get_spending_data(sector, time_period)
        
        # Calculate key metrics
        metrics = {
            'total_spending': sum(spending_data['amounts']),
            'transaction_count': len(spending_data['transactions']),
            'avg_transaction_size': np.mean(spending_data['amounts']),
            'yoy_growth': self.calculate_yoy_growth(spending_data),
            'geographic_breakdown': self.analyze_geographic_spending(spending_data),
            'demographic_breakdown': self.analyze_demographic_spending(spending_data)
        }
        
        return metrics
    
    def get_spending_data(self, sector, time_period):
        """
        Lấy dữ liệu spending từ data provider
        """
        response = self.api.get_spending_data(
            sector=sector,
            start_date=time_period['start'],
            end_date=time_period['end'],
            anonymized=True
        )
        
        return response
    
    def predict_company_revenue(self, company_ticker, spending_data):
        """
        Dự đoán company revenue từ spending data
        """
        # Historical correlation between spending and revenue
        correlation = self.get_spending_revenue_correlation(company_ticker)
        
        # Current spending trend
        current_spending = spending_data['total_spending']
        spending_growth = spending_data['yoy_growth']
        
        # Predict revenue
        predicted_revenue = current_spending * correlation['coefficient']
        revenue_growth = spending_growth * correlation['elasticity']
        
        return {
            'predicted_revenue': predicted_revenue,
            'predicted_growth': revenue_growth,
            'confidence': correlation['r_squared']
        }
    
    def analyze_economic_indicators(self, spending_data):
        """
        Tạo economic indicators từ spending data
        """
        indicators = {}
        
        # Consumer confidence indicator
        discretionary_spending = spending_data.get('entertainment', 0) + spending_data.get('travel', 0)
        essential_spending = spending_data.get('grocery', 0) + spending_data.get('utilities', 0)
        
        if essential_spending > 0:
            confidence_ratio = discretionary_spending / essential_spending
            indicators['consumer_confidence'] = min(1, confidence_ratio)
        
        # Economic health score
        total_spending = sum(spending_data.values())
        if total_spending > 0:
            health_components = {
                'retail_health': spending_data.get('retail', 0) / total_spending,
                'travel_recovery': spending_data.get('travel', 0) / total_spending,
                'dining_activity': spending_data.get('restaurants', 0) / total_spending
            }
            indicators['economic_health'] = np.mean(list(health_components.values()))
        
        return indicators
```

## 🌤️ Weather Data Analysis

### Weather Impact on Trading
```python
import requests
import pandas as pd
from datetime import datetime, timedelta

class WeatherTradingAnalyzer:
    def __init__(self, weather_api_key):
        self.api_key = weather_api_key
        self.base_url = "https://api.weather.com"
    
    def analyze_weather_impact(self, sector, geographic_regions, time_period):
        """
        Phân tích impact của weather lên sector
        """
        weather_data = self.get_weather_data(geographic_regions, time_period)
        
        # Sector-specific weather impacts
        impact_analysis = {}
        
        if sector == 'agriculture':
            impact_analysis = self.analyze_agriculture_impact(weather_data)
        elif sector == 'energy':
            impact_analysis = self.analyze_energy_impact(weather_data)
        elif sector == 'retail':
            impact_analysis = self.analyze_retail_impact(weather_data)
        elif sector == 'transportation':
            impact_analysis = self.analyze_transportation_impact(weather_data)
        
        return impact_analysis
    
    def get_weather_data(self, regions, time_period):
        """
        Lấy weather data cho các regions
        """
        weather_data = {}
        
        for region in regions:
            response = requests.get(
                f"{self.base_url}/historical",
                params={
                    'lat': region['lat'],
                    'lon': region['lon'],
                    'start_date': time_period['start'],
                    'end_date': time_period['end'],
                    'api_key': self.api_key
                }
            )
            
            weather_data[region['name']] = response.json()
        
        return weather_data
    
    def analyze_agriculture_impact(self, weather_data):
        """
        Phân tích weather impact lên agriculture
        """
        impacts = {}
        
        for region, data in weather_data.items():
            # Calculate Growing Degree Days (GDD)
            gdd = self.calculate_gdd(data['temperature'])
            
            # Precipitation analysis
            precipitation = data['precipitation']
            drought_risk = self.calculate_drought_risk(precipitation)
            flood_risk = self.calculate_flood_risk(precipitation)
            
            # Extreme weather events
            extreme_events = self.identify_extreme_events(data)
            
            impacts[region] = {
                'growing_degree_days': gdd,
                'drought_risk': drought_risk,
                'flood_risk': flood_risk,
                'extreme_events': extreme_events,
                'crop_stress_index': self.calculate_crop_stress(gdd, drought_risk, extreme_events)
            }
        
        return impacts
    
    def analyze_energy_impact(self, weather_data):
        """
        Phân tích weather impact lên energy sector
        """
        impacts = {}
        
        for region, data in weather_data.items():
            # Heating/Cooling Degree Days
            hdd = self.calculate_hdd(data['temperature'])
            cdd = self.calculate_cdd(data['temperature'])
            
            # Wind speed for renewable energy
            wind_speed = data['wind_speed']
            wind_power_potential = self.calculate_wind_power(wind_speed)
            
            # Solar radiation for solar energy
            solar_radiation = data.get('solar_radiation', 0)
            solar_power_potential = self.calculate_solar_power(solar_radiation)
            
            impacts[region] = {
                'heating_demand': hdd,
                'cooling_demand': cdd,
                'wind_power_potential': wind_power_potential,
                'solar_power_potential': solar_power_potential,
                'energy_demand_index': self.calculate_energy_demand(hdd, cdd)
            }
        
        return impacts
    
    def predict_commodity_prices(self, commodity, weather_impacts):
        """
        Dự đoán commodity prices từ weather impacts
        """
        # Historical weather-price correlations
        correlations = {
            'corn': {'drought_risk': -0.7, 'flood_risk': -0.4, 'gdd': 0.6},
            'natural_gas': {'hdd': 0.8, 'cdd': 0.7},
            'crude_oil': {'hurricane_risk': 0.5, 'extreme_cold': 0.3}
        }
        
        if commodity not in correlations:
            return None
        
        commodity_correlations = correlations[commodity]
        
        # Calculate predicted price impact
        price_impact = 0
        for factor, correlation in commodity_correlations.items():
            if factor in weather_impacts:
                impact_value = weather_impacts[factor]
                price_impact += impact_value * correlation
        
        return {
            'predicted_price_impact': price_impact,
            'confidence': self.calculate_prediction_confidence(weather_impacts),
            'key_factors': commodity_correlations
        }
```

## 🔮 Integration với Trading Strategies

### Multi-Source Alpha Generation
```python
class AlternativeDataAlphaGenerator:
    def __init__(self):
        self.satellite_analyzer = SatelliteRetailAnalyzer(api_key)
        self.twitter_analyzer = TwitterSentimentAnalyzer(api_key, secret)
        self.spending_analyzer = ConsumerSpendingAnalyzer(api_key)
        self.weather_analyzer = WeatherTradingAnalyzer(api_key)
    
    def generate_alpha_signals(self, universe, current_date):
        """
        Tạo alpha signals từ multiple alternative data sources
        """
        signals = {}
        
        for ticker in universe:
            # Satellite data signal
            satellite_signal = self.get_satellite_signal(ticker, current_date)
            
            # Social sentiment signal
            sentiment_signal = self.get_sentiment_signal(ticker, current_date)
            
            # Consumer spending signal
            spending_signal = self.get_spending_signal(ticker, current_date)
            
            # Weather impact signal
            weather_signal = self.get_weather_signal(ticker, current_date)
            
            # Combine signals
            combined_signal = self.combine_signals(
                satellite_signal, sentiment_signal, 
                spending_signal, weather_signal
            )
            
            signals[ticker] = combined_signal
        
        return signals
    
    def combine_signals(self, satellite, sentiment, spending, weather):
        """
        Combine multiple alternative data signals
        """
        # Weight each signal based on reliability
        weights = {
            'satellite': 0.3,
            'sentiment': 0.25,
            'spending': 0.3,
            'weather': 0.15
        }
        
        # Normalize signals to [-1, 1] range
        normalized_signals = {
            'satellite': self.normalize_signal(satellite),
            'sentiment': self.normalize_signal(sentiment),
            'spending': self.normalize_signal(spending),
            'weather': self.normalize_signal(weather)
        }
        
        # Weighted combination
        combined_signal = sum(
            normalized_signals[source] * weights[source]
            for source in weights
        )
        
        # Signal confidence
        confidence = self.calculate_signal_confidence(normalized_signals)
        
        return {
            'signal': combined_signal,
            'confidence': confidence,
            'components': normalized_signals
        }
    
    def create_trading_strategy(self, alpha_signals, risk_tolerance=0.1):
        """
        Tạo trading strategy từ alternative data signals
        """
        positions = {}
        
        for ticker, signal_data in alpha_signals.items():
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            # Position sizing based on signal strength and confidence
            if confidence > 0.7:  # High confidence threshold
                if signal > 0.3:  # Strong positive signal
                    position = risk_tolerance * signal * confidence
                elif signal < -0.3:  # Strong negative signal
                    position = risk_tolerance * signal * confidence
                else:  # Weak signal
                    position = 0
            else:  # Low confidence
                position = 0
            
            positions[ticker] = {
                'position': position,
                'signal': signal,
                'confidence': confidence,
                'reasoning': self.explain_position(signal_data)
            }
        
        return positions
```

## 🔗 Liên Kết và Tài Nguyên

### Data Providers
- **Satellite**: Planet Labs, Maxar, Airbus
- **Social Media**: Twitter API, Reddit API, StockTwits
- **Transaction Data**: Yodlee, Plaid, Mastercard SpendingPulse
- **Weather**: Weather.com, AccuWeather, NOAA

### Tools và Libraries
- **NLP**: transformers, spaCy, NLTK
- **Image Analysis**: OpenCV, PIL, TensorFlow
- **APIs**: requests, tweepy, praw
- **Data Processing**: pandas, numpy, scipy

### Tiếp Theo
- [[05-Phân-tích-định-lượng/🤖 AI và Machine Learning Hiện Đại|AI và Machine Learning]]
- [[06-Chiến-lược-trading/📊 Data-Driven Strategies|Data-Driven Strategies]]
- [[07-Quản-lý-rủi-ro/🔍 Advanced Risk Analytics|Advanced Risk Analytics]]

---

**Tags:** #alternative-data #sentiment-analysis #satellite-data #social-media #consumer-spending
**Ngày tạo:** 2024-12-19
**Trạng thái:** #cutting-edge