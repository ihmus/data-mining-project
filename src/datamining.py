import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import os
import json
import numpy as np
import warnings
import signal
import sys
from urllib.parse import quote
import random
from typing import Optional, Tuple, List, Dict
import threading
import asyncio
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

class OptimizedFinancialScraper:
    def __init__(self, min_days_required=1001, max_workers=5, request_timeout=30):
        self.min_days_required = min_days_required
        self.max_workers = max_workers
        self.request_timeout = request_timeout
        self.session = self._create_session()
        self.interrupt_flag = False
        self.lock = threading.Lock()
        
        # Interrupt handler
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Graceful shutdown handler"""
        print("\n🛑 İşlem durduruldu! Mevcut veriler kaydediliyor...")
        self.interrupt_flag = True
        
    def _create_session(self) -> requests.Session:
        """Optimized session creation with better retry strategy"""
        session = requests.Session()
        
        # Premium user agents for better success rate
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        session.headers.update({
            'User-Agent': random.choice(user_agents),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9,tr;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none'
        })
        
        # Enhanced connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=15,
            pool_maxsize=30,
            max_retries=3
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    def _safe_request(self, url: str, params: dict = None, headers: dict = None, retries: int = 5) -> Optional[requests.Response]:
        """Enhanced safe HTTP request with exponential backoff"""
        for attempt in range(retries):
            try:
                if self.interrupt_flag:
                    return None
                
                # Dynamic User-Agent rotation
                if attempt > 0:
                    self.session.headers['User-Agent'] = random.choice([
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
                        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    ])
                
                response = self.session.get(
                    url, 
                    params=params, 
                    headers=headers or {}, 
                    timeout=self.request_timeout,
                    verify=True,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limit
                    wait_time = min(2 ** attempt, 30)  # Max 30 seconds
                    print(f"⏳ Rate limit, {wait_time}s bekleniyor...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code in [502, 503, 504]:  # Server errors
                    wait_time = min(2 ** attempt, 15)
                    print(f"⚠️  Server hatası {response.status_code}, {wait_time}s bekleniyor...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ HTTP {response.status_code} for {url}")
                    
            except requests.exceptions.Timeout:
                print(f"⏰ Timeout at attempt {attempt + 1}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    print(f"❌ Request failed after {retries} attempts: {str(e)}")
                    return None
                time.sleep(min(2 ** attempt, 10))
                
        return None
    
    # Enhanced Yahoo Finance with multiple endpoints
    def _get_yahoo_finance_data(self, symbol: str, days: int = 1500) -> Tuple[Optional[pd.DataFrame], str]:
        """Enhanced Yahoo Finance with multiple fallback endpoints"""
        try:
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=days)
            
            # Multiple Yahoo endpoints with different query parameters
            endpoint_configs = [
                {
                    "url": "https://query1.finance.yahoo.com/v8/finance/chart/{}",
                    "params": {
                        'period1': int(start_date.timestamp()),
                        'period2': int(end_date.timestamp()),
                        'interval': '1d',
                        'includePrePost': 'false',
                        'events': 'div%2Csplits'
                    }
                },
                {
                    "url": "https://query2.finance.yahoo.com/v8/finance/chart/{}",
                    "params": {
                        'period1': int(start_date.timestamp()),
                        'period2': int(end_date.timestamp()),
                        'interval': '1d',
                        'includePrePost': 'false'
                    }
                },
                {
                    "url": "https://finance.yahoo.com/quote/{}/history",
                    "params": {
                        'period1': int(start_date.timestamp()),
                        'period2': int(end_date.timestamp()),
                        'interval': '1d',
                        'filter': 'history',
                        'frequency': '1d'
                    }
                }
            ]
            
            for config in endpoint_configs:
                if self.interrupt_flag:
                    break
                    
                url = config["url"].format(symbol)
                params = config["params"]
                
                # Add random delay to avoid rate limiting
                time.sleep(random.uniform(0.1, 0.3))
                
                response = self._safe_request(url, params)
                if not response:
                    continue
                    
                try:
                    data = response.json()
                    
                    if 'chart' in data and data['chart']['result'] and len(data['chart']['result']) > 0:
                        result = data['chart']['result'][0]
                        
                        if 'timestamp' not in result or not result['timestamp']:
                            continue
                        
                        timestamps = result['timestamp']
                        dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
                        
                        indicators = result.get('indicators', {})
                        quote = indicators.get('quote', [{}])[0] if indicators.get('quote') else {}
                        
                        # Handle adjusted close if available
                        adjclose = indicators.get('adjclose', [{}])[0] if indicators.get('adjclose') else {}
                        
                        df = pd.DataFrame({
                            'Date': dates,
                            'Open': quote.get('open', [None] * len(dates)),
                            'High': quote.get('high', [None] * len(dates)),
                            'Low': quote.get('low', [None] * len(dates)),
                            'Close': adjclose.get('adjclose', quote.get('close', [None] * len(dates))),
                            'Volume': quote.get('volume', [None] * len(dates)),
                            'Symbol': symbol
                        })
                        
                        # Enhanced data cleaning
                        df = df.dropna(subset=['Close'])
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.sort_values('Date').reset_index(drop=True)
                        
                        # Remove invalid data points
                        df = df[df['Close'] > 0]
                        
                        if len(df) >= 50:  # Minimum reasonable data
                            return df, f"Yahoo: {len(df)} gün (gerçek)"
                        
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    continue
            
            return None, "Yahoo: Tüm endpoint'ler başarısız"
            
        except Exception as e:
            return None, f"Yahoo: {str(e)}"
    
    def _get_enhanced_coingecko_data(self, symbol: str, days: int = 1500) -> Tuple[Optional[pd.DataFrame], str]:
        """Enhanced CoinGecko with pro endpoints and better error handling"""
        try:
            if not self._is_crypto_symbol(symbol):
                return None, "CoinGecko: Kripto değil"
            
            coin_id = self._convert_to_coingecko_id(symbol)
            if not coin_id:
                return None, "CoinGecko: ID bulunamadı"
            
            # CoinGecko allows up to 365 days for free tier, but we can make multiple requests
            max_days_per_request = 365
            
            endpoint_configs = [
                {
                    "url": f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
                    "params": {
                        'vs_currency': 'usd',
                        'days': min(days, max_days_per_request)
                    }
                },
                {
                    "url": f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                    "params": {
                        'vs_currency': 'usd',
                        'days': min(days, max_days_per_request),
                        'interval': 'daily'
                    }
                },
                {
                    "url": f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
                    "params": {
                        'vs_currency': 'usd',
                        'days': min(days, 1000),  # Pro allows more
                        'x_cg_pro_api_key': 'demo'  # Replace with real key if available
                    }
                }
            ]
            
            for config in endpoint_configs:
                if self.interrupt_flag:
                    break
                    
                # Add delay between requests
                time.sleep(random.uniform(0.5, 1.0))
                
                response = self._safe_request(config["url"], config["params"])
                if not response:
                    continue
                    
                try:
                    data = response.json()
                    
                    if 'ohlc' in config["url"] and isinstance(data, list) and len(data) > 0:
                        df_data = []
                        for item in data:
                            if len(item) >= 5:
                                timestamp = item[0] / 1000
                                date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                                
                                df_data.append({
                                    'Date': date,
                                    'Open': float(item[1]),
                                    'High': float(item[2]),
                                    'Low': float(item[3]),
                                    'Close': float(item[4]),
                                    'Volume': 0,
                                    'Symbol': symbol
                                })
                        
                        if df_data:
                            df = pd.DataFrame(df_data)
                            df['Date'] = pd.to_datetime(df['Date'])
                            df = df.sort_values('Date').reset_index(drop=True)
                            return df, f"CoinGecko OHLC: {len(df)} gün (gerçek)"
                    
                    elif 'market_chart' in config["url"] and 'prices' in data and data['prices']:
                        df_data = []
                        prices = data['prices']
                        volumes = data.get('total_volumes', [])
                        
                        for i, price_item in enumerate(prices):
                            timestamp = price_item[0] / 1000
                            date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                            close_price = float(price_item[1])
                            volume = float(volumes[i][1]) if i < len(volumes) else 0
                            
                            df_data.append({
                                'Date': date,
                                'Open': close_price,
                                'High': close_price,
                                'Low': close_price,
                                'Close': close_price,
                                'Volume': volume,
                                'Symbol': symbol
                            })
                        
                        if df_data:
                            df = pd.DataFrame(df_data)
                            df['Date'] = pd.to_datetime(df['Date'])
                            df = df.sort_values('Date').reset_index(drop=True)
                            return df, f"CoinGecko Market: {len(df)} gün"
                        
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            
            return None, "CoinGecko: Tüm endpoint'ler başarısız"
            
        except Exception as e:
            return None, f"CoinGecko: {str(e)}"
    
    def _get_enhanced_binance_data(self, symbol: str, days: int = 1500) -> Tuple[Optional[pd.DataFrame], str]:
        """Enhanced Binance with multiple endpoints and better handling"""
        try:
            if not self._is_crypto_symbol(symbol):
                return None, "Binance: Kripto değil"
            
            binance_symbol = self._convert_to_binance_symbol(symbol)
            if not binance_symbol:
                return None, "Binance: Sembol desteklenmiyor"
            
            # Binance allows max 1000 klines per request, so we might need multiple requests
            endpoints = [
                "https://api.binance.com/api/v3/klines",
                "https://api1.binance.com/api/v3/klines",
                "https://api2.binance.com/api/v3/klines",
                "https://api3.binance.com/api/v3/klines"
            ]
            
            end_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=min(days, 1000))).timestamp() * 1000)
            
            for url in endpoints:
                if self.interrupt_flag:
                    break
                    
                params = {
                    'symbol': binance_symbol,
                    'interval': '1d',
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 1000
                }
                
                time.sleep(random.uniform(0.2, 0.5))
                
                response = self._safe_request(url, params)
                if not response:
                    continue
                
                try:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        df_data = []
                        
                        for kline in data:
                            timestamp = int(kline[0])
                            date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
                            
                            df_data.append({
                                'Date': date,
                                'Open': float(kline[1]),
                                'High': float(kline[2]),
                                'Low': float(kline[3]),
                                'Close': float(kline[4]),
                                'Volume': float(kline[5]),
                                'Symbol': symbol
                            })
                        
                        if df_data:
                            df = pd.DataFrame(df_data)
                            df['Date'] = pd.to_datetime(df['Date'])
                            df = df.sort_values('Date').reset_index(drop=True)
                            return df, f"Binance: {len(df)} gün (gerçek)"
                    
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue
            
            return None, "Binance: Tüm endpoint'ler başarısız"
            
        except Exception as e:
            return None, f"Binance: {str(e)}"
    
    # Enhanced helper methods with expanded symbol mappings
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Enhanced crypto symbol detection"""
        crypto_indicators = ['-USD', '-EUR', '-BTC', '-ETH', 'USD', 'BTC', 'ETH']
        crypto_symbols = {'BTC', 'ETH', 'XRP', 'SOL', 'BNB', 'DOGE', 'LINK', 'DOT', 'UNI', 'AVAX'}
        
        return (any(indicator in symbol for indicator in crypto_indicators) or 
                symbol.replace('-USD', '').replace('-EUR', '') in crypto_symbols)
    
    def _convert_to_coingecko_id(self, symbol: str) -> Optional[str]:
        """Enhanced CoinGecko ID mapping with more symbols"""
        conversions = {
            'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'XRP-USD': 'ripple',
            'SOL-USD': 'solana', 'BNB-USD': 'binancecoin', 'DOGE-USD': 'dogecoin',
            'LINK-USD': 'chainlink', 'DOT-USD': 'polkadot', 'UNI-USD': 'uniswap',
            'AVAX-USD': 'avalanche-2', 'SHIB-USD': 'shiba-inu', 'XLM-USD': 'stellar',
            'HBAR-USD': 'hedera-hashgraph', 'TRX-USD': 'tron', 'BCH-USD': 'bitcoin-cash',
            'ETC-USD': 'ethereum-classic', 'NEAR-USD': 'near', 'APT-USD': 'aptos',
            'FIL-USD': 'filecoin', 'API3-USD': 'api3', 'USDT-USD': 'tether',
            'USDC-USD': 'usd-coin', 'DAI-USD': 'dai', 'WETH-USD': 'weth',
            'WBTC-USD': 'wrapped-bitcoin', 'PEPE-USD': 'pepe', 'SUI-USD': 'sui',
            'CFX-USD': 'conflux-token', 'XTZ-USD': 'tezos', 'ARB-USD': 'arbitrum',
            'OP-USD': 'optimism', 'MNT-USD': 'mantle', 'SEI-USD': 'sei-network',
            'TON-USD': 'the-open-network', 'LDO-USD': 'lido-dao', 'APE-USD': 'apecoin',
            'USDT-EUR': 'tether', 'XAUT-USD': 'tether-gold'
        }
        return conversions.get(symbol)
    
    def _convert_to_binance_symbol(self, symbol: str) -> Optional[str]:
        """Enhanced Binance symbol mapping"""
        conversions = {
            'BTC-USD': 'BTCUSDT', 'ETH-USD': 'ETHUSDT', 'XRP-USD': 'XRPUSDT',
            'SOL-USD': 'SOLUSDT', 'BNB-USD': 'BNBUSDT', 'DOGE-USD': 'DOGEUSDT',
            'LINK-USD': 'LINKUSDT', 'DOT-USD': 'DOTUSDT', 'UNI-USD': 'UNIUSDT',
            'AVAX-USD': 'AVAXUSDT', 'SHIB-USD': 'SHIBUSDT', 'XLM-USD': 'XLMUSDT',
            'TRX-USD': 'TRXUSDT', 'BCH-USD': 'BCHUSDT', 'ETC-USD': 'ETCUSDT',
            'NEAR-USD': 'NEARUSDT', 'APT-USD': 'APTUSDT', 'FIL-USD': 'FILUSDT',
            'API3-USD': 'API3USDT', 'CFX-USD': 'CFXUSDT', 'XTZ-USD': 'XTZUSDT',
            'ARB-USD': 'ARBUSDT', 'OP-USD': 'OPUSDT', 'MNT-USD': 'MNTUSDT',
            'SEI-USD': 'SEIUSDT', 'LDO-USD': 'LDOUSDT', 'APE-USD': 'APEUSDT'
        }
        return conversions.get(symbol)
    
    def _get_realtime_price(self, symbol: str) -> Optional[float]:
        """Enhanced real-time price fetching with multiple fallbacks"""
        # Priority: Yahoo Finance → Binance → CoinGecko
        
        # 1. Yahoo Finance - Most reliable for all assets
        try:
            endpoints = [
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
            ]
            
            for url in endpoints:
                params = {'interval': '1d', 'range': '1d'}
                response = self._safe_request(url, params)
                if response:
                    data = response.json()
                    result = data.get('chart', {}).get('result', [{}])[0]
                    if result and 'meta' in result:
                        price = result['meta'].get('regularMarketPrice') or result['meta'].get('previousClose')
                        if price:
                            return float(price)
        except Exception:
            pass
        
        # 2. Binance for crypto
        if self._is_crypto_symbol(symbol):
            binance_symbol = self._convert_to_binance_symbol(symbol)
            if binance_symbol:
                try:
                    endpoints = [
                        "https://api.binance.com/api/v3/ticker/price",
                        "https://api1.binance.com/api/v3/ticker/price"
                    ]
                    
                    for url in endpoints:
                        params = {'symbol': binance_symbol}
                        response = self._safe_request(url, params)
                        if response:
                            data = response.json()
                            if 'price' in data:
                                return float(data['price'])
                except Exception:
                    pass
        
        # 3. CoinGecko for crypto
        if self._is_crypto_symbol(symbol):
            coingecko_id = self._convert_to_coingecko_id(symbol)
            if coingecko_id:
                try:
                    url = "https://api.coingecko.com/api/v3/simple/price"
                    params = {'ids': coingecko_id, 'vs_currencies': 'usd'}
                    response = self._safe_request(url, params)
                    if response:
                        data = response.json()
                        if coingecko_id in data and 'usd' in data[coingecko_id]:
                            return float(data[coingecko_id]['usd'])
                except Exception:
                    pass
        
        return None
    
    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Enhanced data validation and cleaning"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        original_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Date']).reset_index(drop=True)
        
        # Convert numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove invalid prices
        df = df[df['Close'] > 0].reset_index(drop=True)
        df = df.dropna(subset=['Close']).reset_index(drop=True)
        
        # Remove extreme outliers (daily change > 300%)
        if len(df) > 1:
            df['pct_change'] = df['Close'].pct_change()
            df = df[abs(df['pct_change']) < 3.0].reset_index(drop=True)
            df = df.drop('pct_change', axis=1)
        
        # Ensure OHLC consistency
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High should be >= max(Open, Close)
            df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
            # Low should be <= min(Open, Close)
            df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        cleaned_count = len(df)
        
        if cleaned_count != original_count:
            print(f"   🧹 {symbol}: {original_count} → {cleaned_count} kayıt (temizlendi)")
        
        return df
    
    def _try_all_sources_optimized(self, symbol: str, target_days: int = 1500) -> Tuple[Optional[pd.DataFrame], str, int]:
        """Optimized sequential data fetching with smart prioritization"""
        
        # Prioritize sources based on symbol type
        if self._is_crypto_symbol(symbol):
            sources = [
                ('Yahoo Finance', self._get_yahoo_finance_data),
                ('Binance', self._get_enhanced_binance_data),
                ('CoinGecko', self._get_enhanced_coingecko_data),
            ]
        else:
            sources = [
                ('Yahoo Finance', self._get_yahoo_finance_data),
            ]
        
        all_source_dfs = []
        used_sources = []
        
        # Sequential processing with smart early termination
        for source_name, source_func in sources:
            if self.interrupt_flag:
                break
                
            try:
                # Add small delay between requests to avoid rate limiting
                time.sleep(random.uniform(0.2, 0.5))
                
                df, message = source_func(symbol, target_days)
                
                if df is not None and len(df) > 0:
                    df = self._validate_and_clean_data(df, symbol)
                    if len(df) > 0:
                        all_source_dfs.append((source_name, df))
                        used_sources.append(source_name)
                        print(f"   ✅ {source_name}: {len(df)} gün ✓")
                        
                        # Early termination if we have enough data from primary source
                        if len(df) >= target_days * 0.8 and source_name == 'Yahoo Finance':
                            print(f"   🚀 {source_name} yeterli veri sağladı, diğer kaynaklar atlanıyor")
                            break
                    else:
                        print(f"   ❌ {source_name}: Temizlik sonrası veri kalmadı")
                else:
                    print(f"   ❌ {source_name}: {message}")
                    
            except Exception as e:
                print(f"   ❌ {source_name}: Exception - {str(e)}")
                continue
        
        if not all_source_dfs:
            return None, "Tüm kaynaklar başarısız", 0
        
        # Merge all data sources, prioritizing actual values over forward-filled ones
        all_dates = set()
        for _, df in all_source_dfs:
            all_dates.update(df['Date'])
        
        all_dates = sorted(all_dates)
        
        # Build combined dataset
        records = []
        for date in all_dates:
            row = {'Date': date}
            found = False
            
            # Try each source in priority order
            for _, df in all_source_dfs:
                match = df[df['Date'] == date]
                if not match.empty:
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']:
                        if col in match.columns:
                            row[col] = match.iloc[0][col]
                    found = True
                    break
            
            if found:
                records.append(row)
        
        if records:
            combined_df = pd.DataFrame(records)
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            
            # Ensure we have enough data
            if len(combined_df) >= self.min_days_required:
                # _try_all_sources_parallel method continued from part 1
                print(f"   📊 Kombine veriler: {len(combined_df)} gün")
                return combined_df, '+'.join(used_sources), len(combined_df)
            
        return None, "Yetersiz veri", 0
    
    def _fill_weekend_gaps_enhanced(self, df: pd.DataFrame, target_days: int = None) -> pd.DataFrame:
        """Enhanced weekend gap filling with better logic"""
        if df.empty:
            return df
        
        # Ensure we have the target number of days
        end_date = df['Date'].max()
        if target_days is not None:
            start_date = end_date - pd.Timedelta(days=target_days-1)
        else:
            start_date = df['Date'].min()
        
        # Generate all dates in range
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        complete_df = pd.DataFrame({'Date': all_dates})
        complete_df = complete_df.merge(df, on='Date', how='left')
        
        # Smart forward fill for OHLC data
        fill_columns = ['Open', 'High', 'Low', 'Close']
        for col in fill_columns:
            if col in complete_df.columns:
                complete_df[col] = complete_df[col].fillna(method='ffill')
        
        # Volume should be 0 for non-trading days
        if 'Volume' in complete_df.columns:
            complete_df['Volume'] = complete_df['Volume'].fillna(0)
        
        # Fill symbol
        if 'Symbol' in complete_df.columns:
            complete_df['Symbol'] = complete_df['Symbol'].fillna(df['Symbol'].iloc[0] if len(df) > 0 else 'UNKNOWN')
        
        # Remove rows where we couldn't fill critical data
        complete_df = complete_df.dropna(subset=['Close'])
        
        # Trim to target days if specified
        if target_days is not None and len(complete_df) > target_days:
            complete_df = complete_df.iloc[-target_days:]
        
        return complete_df.reset_index(drop=True)
    
    def download_all_symbols_optimized(self, symbols: List[str], target_days: int = 1500, 
                                     output_file: Optional[str] = None, 
                                     batch_processing: bool = True) -> Tuple[List[str], List[str]]:
        """Optimized bulk download with batch processing and smart delays"""
        
        successful_symbols = []
        failed_symbols = []
        all_dfs = {}
        
        symbols = list(symbols)  # Ensure list
        print(f"\n🚀 {len(symbols)} sembol için optimized veri çekme başlatılıyor...")
        print(f"🎯 Minimum {self.min_days_required} günlük veri gerekli, hedef: {target_days} gün")
        
        # Batch processing to avoid overwhelming servers
        batch_size = 1 if batch_processing else 1
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        total_processed = 0
        
        for batch_idx, batch in enumerate(batches):
            if self.interrupt_flag:
                break
                
            print(f"\n🔄 Batch {batch_idx + 1}/{len(batches)} işleniyor ({len(batch)} sembol)")
            
            for idx_in_batch, symbol in enumerate(batch):
                if self.interrupt_flag:
                    break
                
                global_idx = total_processed + idx_in_batch + 1
                print(f"\n[{global_idx}/{len(symbols)}] 🔍 {symbol} işleniyor...")
                
                try:
                    df, source, count = self._try_all_sources_optimized(symbol, target_days)
                    
                    if df is not None and len(df) >= self.min_days_required:
                        # Apply enhanced gap filling
                        df = self._fill_weekend_gaps_enhanced(df, target_days)
                        
                        if len(df) >= self.min_days_required:
                            # Final validation
                            df = self._validate_and_clean_data(df, symbol)
                            
                            if len(df) >= self.min_days_required:
                                # Keep only Date and Close for CSV output
                                result_df = df[['Date', 'Close']].copy()
                                result_df = result_df.rename(columns={'Close': symbol})
                                
                                all_dfs[symbol] = result_df
                                successful_symbols.append(symbol)
                                print(f"✅ {symbol}: {len(result_df)} gün - {source}")
                            else:
                                failed_symbols.append(symbol)
                                print(f"❌ {symbol}: Final validation failed ({len(df)} < {self.min_days_required})")
                        else:
                            failed_symbols.append(symbol)
                            print(f"❌ {symbol}: Gap filling sonrası yetersiz veri ({len(df)} < {self.min_days_required})")
                    else:
                        failed_symbols.append(symbol)
                        print(f"❌ {symbol}: Yetersiz veri - {count if df is not None else 0} gün")
                        
                except Exception as e:
                    failed_symbols.append(symbol)
                    print(f"❌ {symbol}: İşlem hatası - {str(e)}")
                
                # Smart delay based on success/failure
                if symbol in successful_symbols:
                    delay = random.uniform(0.5, 1.0)  # Shorter delay for successful requests
                else:
                    delay = random.uniform(1.0, 2.0)  # Longer delay after failures
                
                time.sleep(delay)
            
            total_processed += len(batch)
            
            # Longer delay between batches
            if batch_idx < len(batches) - 1:  # Don't delay after last batch
                batch_delay = random.uniform(2.0, 3.0)
                print(f"   ⏳ Batch arası {batch_delay:.1f}s bekle...")
                time.sleep(batch_delay)
        
        # Save results to CSV if we have successful data
        if all_dfs and output_file:
            print(f"\n💾 CSV dosyası oluşturuluyor: {output_file}")
            
            # Merge all DataFrames on Date (outer join for maximum coverage)
            merged_df = None
            for symbol, df in all_dfs.items():
                if merged_df is None:
                    merged_df = df
                else:
                    merged_df = pd.merge(merged_df, df, on='Date', how='outer')
            
            # Sort by date
            merged_df = merged_df.sort_values('Date').reset_index(drop=True)
            
            # Remove future dates (keep only up to yesterday)
            yesterday = pd.to_datetime((datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'))
            merged_df = merged_df[merged_df['Date'] <= yesterday]
            
            # Add real-time prices for today
            print("📡 Anlık fiyatlar ekleniyor...")
            today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
            realtime_row = {'Date': today}
            
            realtime_count = 0
            for symbol in successful_symbols:
                try:
                    price = self._get_realtime_price(symbol)
                    if price is not None:
                        realtime_row[symbol] = price
                        realtime_count += 1
                    else:
                        realtime_row[symbol] = None
                except Exception:
                    realtime_row[symbol] = None
            
            # Only add realtime row if we got at least some prices
            if realtime_count > 0:
                merged_df = pd.concat([merged_df, pd.DataFrame([realtime_row])], ignore_index=True)
                print(f"   ✅ {realtime_count}/{len(successful_symbols)} anlık fiyat eklendi")
            
            # Save to CSV
            merged_df.to_csv(output_file, index=False)
            print(f"   💾 {len(merged_df)} satır, {len(merged_df.columns)-1} sembol kaydedildi")
            
            # Data quality report
            print(f"\n📊 VERİ KALİTE RAPORU:")
            print(f"   • Toplam satır: {len(merged_df)}")
            print(f"   • Tarih aralığı: {merged_df['Date'].min().strftime('%Y-%m-%d')} - {merged_df['Date'].max().strftime('%Y-%m-%d')}")
            print(f"   • Başarılı semboller: {len(successful_symbols)}")
            
            # Missing data analysis
            missing_data = merged_df.isnull().sum()
            if missing_data.sum() > 0:
                print(f"   • Eksik veri olan semboller:")
                for symbol in missing_data.index:
                    if symbol != 'Date' and missing_data[symbol] > 0:
                        missing_pct = (missing_data[symbol] / len(merged_df)) * 100
                        print(f"     - {symbol}: {missing_data[symbol]} kayıt (%{missing_pct:.1f})")
        
        return successful_symbols, failed_symbols
    
    def validate_symbol_coverage(self, symbols: List[str]) -> Dict[str, Dict[str, bool]]:
        """Validate which data sources support which symbols"""
        coverage = {}
        
        print("\n🔍 SEMBOL KAYNAK KAPSAMASI ANALİZİ")
        print(f"{'Sembol':<15} {'Yahoo':<8} {'Binance':<10} {'CoinGecko':<12} {'Desteklenen':<12}")
        print("-" * 65)
        
        for symbol in symbols:
            yahoo_support = True  # Yahoo generally supports most symbols
            binance_support = self._convert_to_binance_symbol(symbol) is not None
            coingecko_support = self._convert_to_coingecko_id(symbol) is not None
            
            total_support = sum([yahoo_support, binance_support, coingecko_support])
            
            coverage[symbol] = {
                'yahoo': yahoo_support,
                'binance': binance_support,
                'coingecko': coingecko_support,
                'total_sources': total_support
            }
            
            yahoo_str = "✅" if yahoo_support else "❌"
            binance_str = "✅" if binance_support else "❌"
            coingecko_str = "✅" if coingecko_support else "❌"
            
            print(f"{symbol:<15} {yahoo_str:<8} {binance_str:<10} {coingecko_str:<12} {total_support}/3")
        
        return coverage


# MAIN EXECUTION SCRIPT
if __name__ == "__main__":
    # Extended and validated symbol list with real market symbols
    veriler = set([
    # 🔹 Major Stock Indices
    "^GSPC", "^DJI", "^IXIC", "^NDX", "^FTSE", "^GDAXI", "^N225", "^STOXX50E", "000001.SS", "^HSI", "XU100.IS",
    "^RUT", "^VIX", "^AXJO", "^BSESN", "^KS11", "^TWII", "IMOEX.ME", "^BVSP", "^MXX",
    "^NSEI", "^J203.JO", "^CASE30", "^TA125.TA", "XKID.JK", "^KLSE",

    # 🔹 Individual Stocks (High Volume)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPGI", "JPM", "JNJ",
    "BRK-B", "V", "MA", "UNH", "PG", "HD", "PEP", "COST", "DIS", "NKE", "PFE", "BAC", "KO", "INTC", "ORCL",
    "TSM", "BABA", "NVS", "TM", "SAP",

    # 🔹 Currency & Commodities
    "DX-Y.NYB", "EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F", "ZW=F",
    "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X",
    "PL=F", "PA=F", "LE=F", "HE=F", "KC=F", "CC=F", "SB=F", "LBS=F", "OJ=F",

    # 🔹 Bonds & Interest Rates
    "^TNX", "^FVX", "^TYX", "TLT",
    "IEI", "SHY", "IEF", "TIP",

    # 🔹 Major Cryptocurrencies
    "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "BNB-USD", "AVAX-USD", "DOT-USD", "UNI-USD",
    "BCH-USD", "XLM-USD", "TRX-USD", "ETC-USD", "NEAR-USD", "APT-USD", "FIL-USD", "API3-USD",
    "SHIB-USD", "HBAR-USD", "INJ-USD", "MKR-USD", "RUNE-USD","SNX-USD", "CRV-USD",
    "TON11419-USD", "MINA-USD", "MANA-USD", "ATOM-USD", "FTM-USD", "KAVA-USD", "EGLD-USD", "AR-USD",
    "GMX-USD", "AAVE-USD", "1INCH-USD", "ZEC-USD", "LTC-USD", "CRO-USD", "DASH-USD",

    # 🔹 Stablecoins & Wrapped Tokens
    "USDT-USD", "USDC-USD", "DAI-USD", "WETH-USD", "WBTC-USD", "FRAX-USD", "USDN-USD",
    "TUSD-USD", "SUSD-USD", "ALUSD-USD", "GUSD-USD", "LUSD-USD", "USDP-USD",

    # 🔹 Emerging Crypto Projects
    "PEPE-USD", "SUI-USD", "CFX-USD", "XTZ-USD", "ARB-USD", "OP-USD", "MNT-USD", "SEI-USD",
    "TON-USD", "LDO-USD", "APE-USD", "JTO-USD", "PYTH-USD", "ZETA-USD", "JUP-USD", "METIS-USD",
    "AKT-USD", "NKN-USD", "FET-USD", "AGIX-USD", "OCEAN-USD", "FRIEND-USD", "DESO-USD",
    "BLUR-USD", "TIA-USD", "SYN-USD", "AIOZ-USD", "DYDX-USD", "JOE-USD", "NYM-USD",
    "CKB-USD", "CANTO-USD", "VRA-USD", "VELO-USD", "ID-USD",
    "BTC-USD",    # BTCUSDT
    "ETH-USD",    # ETHUSDT
    "XRP-USD",    # XRPUSDT
    "BNB-USD",    # BNBUSDT
    "SOL-USD",    # SOLUSDT
    "DOGE-USD",   # DOGEUSDT
    "TRX-USD",    # TRXUSDT
    "ADA-USD",    # ADAUSDT
    "XLM-USD",    # XLMUSDT
    "LINK-USD",   # LINKUSDT
    "WBETH-USD",  # WBETHUSDT (Not: Yahoo'da doğrudan karşılığı olmayabilir)
    "WBTC-USD",   # WBTCUSDT
    "SUI-USD",    # SUIUSDT
    "HBAR-USD",   # HBARUSDT
    "BCH-USD",    # BCHUSDT
    "AVAX-USD",   # AVAXUSDT
    "UNI-USD",    # UNIUSDT
    "LTC-USD",    # LTCUSDT
    "TON-USD",    # TONUSDT (Not: Toncoin)
    "SHIB-USD",   # SHIBUSDT
    "DOT-USD",    # DOTUSDT
    "PAXG-USD",   # PAXGUSDT
    "AAVE-USD",   # AAVEUSDT
    "ONDO-USD",   # ONDOUSDT (Not: Yahoo'da listelenmeyebilir)
    "PEPE-USD",   # PEPEUSDT
    "ARB-USD",    # ARBUSDT
    "ENA-USD",    # ENAUSDT (Not: Yahoo'da listelenmeyebilir)
    "WTAO-USD",    # TAOUSDT (Bittensor)
    "WLD-USD",    # WLDUSDT
    "ETC-USD",    # ETCUSDT
    "NEAR-USD",   # NEARUSDT
    "APT-USD",    # APTUSDT
    "OP-USD",     # OPUSDT
    "ICP-USD",    # ICPUSDT
    "FIL-USD",    # FILUSDT
    "POL-USD",    # POLUSDT (Not: Polygon Ecosystem Token)
    "ENS-USD",    # ENSUSDT
    "BNSOL-USD",  # BNSOLUSDT (Not: Yahoo'da karşılığı yok)
    "ALGO-USD",   # ALGOUSDT
    "VET-USD",    # VETUSDT
    "ATOM-USD",   # ATOMUSDT
    "RAY-USD",    # RAYUSDT (Not: Raydium)
    "PENGU34466-USD",  # PENGUUSDT (Not: Yahoo'da listelenmeyebilir)
    "RNDR-USD",   # RENDERUSDT
    "SEI-USD",    # SEIUSDT
    "FET-USD",    # FETUSDT
    "BONK-USD",   # BONKUSDT
    "BFUSD-USD",  # BFUSDUSDT (Not: Yahoo'da karşılığı yok)
    "JTO-USD",    # JTOUSDT (Not: Jito)
    "JUP-USD",    # JUPUSDT
    "TIA-USD",    # TIAUSDT
    "PENDLE-USD", # PENDLEUSDT
    "FDUSD-USD",  # FDUSDUSDT (Not: Yahoo'da karşılığı yok)
    "FORM-USD",   # FORMUSDT (Not: Yahoo'da listelenmeyebilir)
    "YGG-USD",    # YGGUSDT
    "LDO-USD",    # LDOUSDT
    "AXL-USD"     # AXLUSDT

    # 🔹 Gaming / Metaverse / NFT Related
    "SAND-USD", "AXS-USD", "GALA-USD", "ENJ-USD", "ILV-USD", "IMX-USD", "RARI-USD",
    "ALICE-USD", "YGG-USD", "MAGIC-USD", "GHST-USD", "HIGH-USD",

    # 🔹 AI & Big Data Tokens
    "RNDR-USD", "NMR-USD", "CTX-USD", "ALI-USD",

    # 🔹 Layer 2 / Rollups / Infrastructure
    "ZKS-USD", "STRK22691-USD", "LOOM-USD", "OMG-USD",

    # 🔹 Oracle & Data Sharing
    "BAND-USD", "DIA-USD", "UMA-USD", "TRB-USD",

    # 🔹 Privacy Coins
    "XMR-USD", "SCRT-USD", "BEAM-USD",

    # 🔹 Meme & Community Tokens
    "FLOKI-USD", "BONK-USD", "WIF-USD", "HOGE-USD",

    # 🔹 RWA (Real World Asset) Tokenleri
    "ONDO-USD", "CFG-USD", "RIO-USD", "SKY-USD", "OM-USD",

    # 🔹 DeFi Lending / Borrowing
    "COMP-USD", "MORPHO34104-USD", "ALCX-USD", "TRU-USD", "RDNT-USD", "SLND-USD", "BZRX-USD",

    # 🔹 Liquid Staking (LST) tokenleri
    "RPL-USD", "ANKR-USD", "ETHFI-USD", "REZ-USD", "PSTAKE-USD", "SWETH-USD", "PZETH-USD",

    # 🔹 İşlem Hacmine Göre İlk 83 Coin (24‑s saatlik hacme göre)
    'ADA-USD', 'BTC-USD', 'XRP-USD', 'BNB-USD', 'SOL-USD', 'USDT-USD', 'LTC-USD', 'ETH-USD',
      'DOGE-USD', 'TRX-USD', 'SSWP-USD', 'PUMP36507-USD', 'PEPE-USD', 'AVAX-USD',
        'ENA-USD', 'LINK-USD', 'BCH-USD', 'USD.AX', 'MNT27075-USD', 'APT-USD', 'MAGIC14783-USD', 'XLM-USD',
          'BONK-USD', 'FLR-USD', 'HBAR-USD', 'AAVE-USD', 'ILV-USD', 'CRV-USD', 'TON-USD', 'HYPE32196-USD', 
          'RARE11294-USD', 'ECOIN-USD', 'FARTCOIN-USD', 'CFX-USD', 'DOGWIFHAT-USD', 'ARB-USD', 'DOT-USD', 
          'WLD-USD', 'OP-USD', 'KERNEL-USD', 'NEAR-USD', 'RED21707-USD', 'ZEC-USD', 'SOL28825-USD', 'SPC-USD',
            'SHIB-USD', 'SEI-USD', 'INJ-USD', 'GAS-USD', 'XMR-USD', 'G7-USD', 'ATOM-USD', 'TOWNS-USD',
              'ONDO-USD', 'ZORA35931-USD', 'DAI-USD', 'FIL-USD', 'TAO22974-USD', 'ERA37374-USD', 'FLOKI-USD', 'WETH-USD',
                'LUNA33543-USD', 'TIA-USD', 'IONX-USD', 'ALGO-USD', 'OM-USD', 'MKR-USD', 'ETHFI-USD', 
                  'SAHARA-USD', 'CAKE-USD', 'SPX28081-USD', 'UGOLD-USD', 'HYPER36281-USD', 'RENDER-USD', 'RUNE-USD',
                    'GALA-USD', 'PEANUT-USD',   'EGL1-USD', 'PACUSA.SW', 'AB36894-USD', 'LIDO-USD.AS',
                      'MAPLE-USD', 'PENDLE-USD', 'KAS-USD', 'FTN-USD',  'XTZ-USD', 'RAY-USD', 
                      'PIN34017-USD', 'MYRUSD=X','USDT-USD', 'HYPE32196-USD', 'XLM-USD', 'SSWP-USD', 'HBAR-USD',
                        'USD.AX', 'XMR-USD', 'ENA-USD', 'CRO-USD', 'ETH-USD', 'MNT27075-USD', 'TAO22974-USD',
                          'PIN34017-USD', 'APT-USD', 'PENGU34525-USD', 'ARB-USD', 'ALGO-USD', 'DOT-USD', 'ATOM-USD',
                            'KAS-USD', 'WLD-USD', 'FTN-USD', 'ECOIN-USD', 'VET-USD', 'GT-USD', 'SPX28081-USD', 'FIL-USD',
                              'QNT-USD', 'JUP-USD', 'INJ-USD', 'OP-USD', 'CRV-USD', 'TIA-USD', 'LIDO-USD.AS', 'FLR-USD',
                                'PUMP36507-USD', 'CFX-USD', 'SOL28825-USD', 'IMX10603-USD', 'CAKE-USD', 'XTZ-USD', 'PACUSA.SW', 
                                'TDROP-USD', 'STX4847-USD', 'SUPERGROK-USD', 'LUNA33543-USD', 'RAY-USD', 'JASMY-USD', 'M35491-USD',
                                  'MORPHO34104-USD', 'ZEC-USD', 'JTO-USD', 'CHRETT-USD', 'BTTOLD-USD', 'SD-USD', 'B-USD', 'WAL36119-USD',
                                    'MANA-USD', 'KTA-USD', 'HNT-USD', 'RYOSHI11283-USD', 'BTC-USD',  'COMP5692-USD',
                                      'STRK22691-USD', 'TEL-USD', 'ETHFI-USD', 'EFR-USD', 'AR-USD', 'RUNE-USD', 'BDX-USD', 'APE-USD',
                                        'SUPER8290-USD', 'XCN18679-USD', 'NFT9816-USD', 'TREE37495-USD', 'EGLD-USD', 'XEC-USD', 'ZK24091-USD',
                                          'TRIP35555-USD', 'CHZ-USD', 'RON14101-USD', 'ATH30083-USD', 'MOVE32452-USD', 'GNO-USD', 'AXL17799-USD',
                                            'DOGE-USD', 'BABYPOPCAT-USD', 'TRC-USD', 'SNEK25264-USD', 'CTC-USD', 'STMATIC-USD', 'DRIFT31278-USD',
                                              'CAT32724-USD', 'ZORA35931-USD', 'TOSHI27750-USD', 'DCR-USD', 'OM-USD', 'LPT-USD', '0P00019LJQ', 'WIOTX-USD',
                                                'KSM-USD', 'ABTC-USD.SW', 'BORG-USD', 'BERA-USD', 'GLM-USD', 'PROVE-USD', 'SFP-USD', 'GRASS32276-USD',
                                                 'ARKM-USD', 'CBU-USD', 'GLS-USD', 'BABI-USD', 'TRAC-USD', 'ZIL-USD', 'AGENTFUN-USD', 'SNX-USD', '0XBTC-USD',
                                                   'NXS-USD', 'RVN-USD', 'FTMO-USD', 'ORO14950-USD', 'EUL-USD', 'NOT-USD', 'ZRO26997-USD', 'ZETA-USD',
                                                     'WASTR-USD', 'YFI-USD', 'UBFC.HA', 'XMT-USD', 'GMX11857-USD', 'SGR-U.TO', 'ETHEREUM27840-USD',
                                                       'ILV-USD', 'SC-USD', 'T-USD', 'ETHW-USD', 'POLYX-USD', 'AURA36544-USD', 'CFG-USD',
                                                         'MGG36318-USD', 'VRSC-USD', 'ELF-USD', 'FBTC-USD', 'ONE3945-USD', 'DGB-USD', 'GIGA30063-USD'
    # 🔹 Alternative Assets
    "USDT-EUR", "XAUT-USD","UBFC.HA",
    "WTAO-USD",
    "^CASE30",
    "MMM",
    "AXL-USD",
    "SAND-USD",
    "USDJPY=X",
    "STX-USD"
    "BAR", "IAU", "SLV", "GLDM", "UUP", "PAXG-USD", "SPY","GLD", "KRBN",

    # 🔹 Temettü Hisseleri (Long-Term)
    "CVX", "T", "VZ", "MMM", "IBM", "XOM",

    # 🔹 Volatil Altcoinler (Short-Term)
    "FLOKI-USD", "BONK-USD", "LUNC-USD", "VRA-USD", "WIF-USD",
    "JASMY-USD", "REQ-USD", "MASK-USD", "CHZ-USD", "CVC-USD",

    # 🔹 Hedge Fon İlgi Alanı (Spekülatif Hisseler)
    "PLTR", "RIVN", "HOOD", "LCID", "SOFI", "AFRM", "DNA", "FUBO",
     "GME", "AMC", "BB"
    ])
    # Initialize optimized scraper
    scraper = OptimizedFinancialScraper(
        min_days_required=100,  # Minimum 1001 days required
        max_workers=7,           # Parallel workers
        request_timeout=30       # 30 second timeout
    )
    
    # Create output directory
    output_dir = "optimized_financial_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 OPTİMİZE EDİLMİŞ FİNANSAL VERİ ÇEKME SİSTEMİ v2.0")
    print(f"🎯 {len(veriler)} sembol için veri çekilecek")
    print(f"📊 Minimum {scraper.min_days_required} günlük veri gerekli")
    print("🔄 Paralel işleme ve gelişmiş hata yönetimi aktif")
    
    # Validate symbol coverage before starting
    coverage = scraper.validate_symbol_coverage(veriler)
    
    # Filter symbols with at least 2 source support for better success rate
    supported_symbols = [
        symbol for symbol, info in coverage.items() 
        if info['total_sources'] >= 2
    ]
    
    unsupported_symbols = [
        symbol for symbol, info in coverage.items() 
        if info['total_sources'] < 2
    ]
    
    if unsupported_symbols:
        print(f"\n⚠️  Düşük destek seviyeli semboller: {len(unsupported_symbols)}")
        for symbol in unsupported_symbols:
            print(f"   - {symbol}: {coverage[symbol]['total_sources']}/3 kaynak")
    
    print(f"\n✅ Yüksek başarı şansı olan semboller: {len(supported_symbols)}")
    
    # Start optimized download process
    successful, failed = scraper.download_all_symbols_optimized(
        symbols=veriler,  # Use all symbols, let the system handle fallbacks
        target_days=8000,  # Target 2000 days for comprehensive historical data
        output_file=f"{output_dir}/comprehensive_market_data_400_plus_features.csv",
        batch_processing=True  # Enable batch processing instead of parallel
    )
    
    # Final results
    print(f"\n🏁 OPTİMİZE EDİLMİŞ İŞLEM TAMAMLANDI!")
    print(f"✅ Başarılı semboller: {len(successful)}")
    print(f"❌ Başarısız semboller: {len(failed)}")
    print(f"📈 Başarı oranı: %{(len(successful)/(len(successful)+len(failed)))*100:.1f}")
    
    if successful:
        print(f"\n🎯 BAŞARILI SEMBOLLER:")
        for i, symbol in enumerate(successful, 1):
            print(f"   {i:2d}. {symbol}")
    
    if failed:
        print(f"\n❌ BAŞARISIZ SEMBOLLER:")
        for i, symbol in enumerate(failed, 1):
            print(f"   {i:2d}. {symbol}")
    
    print(f"\n💾 Veri dosyası: {output_dir}/comprehensive_market_data_400_plus_features.csv")
    print("🔥 Optimized scraper tamamlandı!")
