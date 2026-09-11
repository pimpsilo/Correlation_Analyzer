import re
import os
import json
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta

# Create the local data storage folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "data_storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

def parse_urls_from_text(text: str) -> dict:
    """Parse Composer symphony URLs or IDs from text."""
    symphony_urls = {}
    url_pattern = r'(?:https?://)?(?:app\.)?composer\.trade/symphony/([A-Za-z0-9_-]+)'
    id_pattern = r'\b([A-Za-z0-9_-]{12,})\b'
    
    found_ids = set()
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        url_match = re.search(url_pattern, line)
        if url_match:
            sym_id = url_match.group(1)
            if sym_id not in found_ids:
                found_ids.add(sym_id)
                symphony_urls[f"Symphony_{len(symphony_urls) + 1}"] = f"https://app.composer.trade/symphony/{sym_id}/details"
            continue
            
        id_match = re.search(id_pattern, line)
        if id_match:
            sym_id = id_match.group(1)
            if sym_id not in found_ids and len(sym_id) >= 12:
                found_ids.add(sym_id)
                symphony_urls[f"Symphony_{len(symphony_urls) + 1}"] = f"https://app.composer.trade/symphony/{sym_id}/details"
                
    return symphony_urls

def convert_trading_date(date_int):
    """Convert integer trading date to datetime."""
    return datetime.strptime("01/01/1970", "%m/%d/%Y") + timedelta(days=int(date_int))

def fetch_composer_symphony(symphony_url: str, start_date: str, end_date: str):
    """Fetch backtest data from Composer API."""
    symphony_id = symphony_url.split('/')[-2] if symphony_url.endswith('/details') else symphony_url.split('/')[-1]
    
    payload = {
        "capital": 100000, "apply_reg_fee": True, "apply_taf_fee": True,
        "backtest_version": "v2", "slippage_percent": 0.0005,
        "start_date": start_date, "end_date": end_date,
    }
    
    response = requests.post(
        f"https://backtest-api.composer.trade/api/v2/public/symphonies/{symphony_id}/backtest", 
        json=payload
    )
    response.raise_for_status()
    data = response.json()
    
    symphony_name = data['legend'][symphony_id]['name']
    tickers = list(data["last_market_days_holdings"].keys())
    allocations = data["tdvm_weights"]
    
    date_range = pd.date_range(start=start_date, end=end_date)
    allocations_df = pd.DataFrame(0.0, index=date_range, columns=tickers)
    
    for ticker in allocations:
        for date_int, percent in allocations[ticker].items():
            allocations_df.at[convert_trading_date(date_int), ticker] = percent
            
    return allocations_df, symphony_name, tickers

def calculate_symphony_returns(allocations_df: pd.DataFrame, tickers: list):
    """Calculate daily returns from allocations and Yahoo Finance prices."""
    allocations_df = allocations_df.loc[(allocations_df != 0).any(axis=1)] * 100.0
    if '$USD' not in allocations_df.columns:
        allocations_df['$USD'] = 0
    allocations_df.index = pd.to_datetime(allocations_df.index).normalize()
    
    unique_tickers = {t for t in tickers if t != '$USD'}
    start_str = (allocations_df.index.min() - timedelta(days=10)).strftime('%Y-%m-%d')
    end_str = (allocations_df.index.max() + timedelta(days=10)).strftime('%Y-%m-%d')
    
    prices_data = {}
    for ticker in unique_tickers:
        t_obj = yf.Ticker(ticker.replace('BRK/B', 'BRK-B'))
        hist = t_obj.history(start=start_str, end=end_str, auto_adjust=True)
        if not hist.empty:
            prices_data[ticker] = hist['Close'].tz_localize(None)

    prices = pd.DataFrame(prices_data)
    prices.index = pd.to_datetime(prices.index).normalize()
    prices['$USD'] = 1.0
    
    for ticker in tickers:
        if ticker not in prices.columns and ticker != '$USD':
            prices[ticker] = np.nan
            
    prices = prices.ffill().bfill().fillna(1.0)[tickers]
    allocations_df.sort_index(inplace=True)
    prices.sort_index(inplace=True)
    
    daily_returns, dates = [], []
    prev_alloc = None
    
    for date in allocations_df.index:
        if date not in prices.index:
            continue
        current_prices = prices.loc[date]
        
        if prev_alloc is not None:
            prev_date = dates[-1]
            if prev_date in prices.index:
                prev_prices = prices.loc[prev_date]
                d_return = sum(
                    (prev_alloc[t] / 100.0) * ((current_prices[t] - prev_prices[t]) / prev_prices[t])
                    for t in tickers if prev_alloc[t] > 0.0001 and t in current_prices and t in prev_prices and prev_prices[t] > 0
                )
                daily_returns.append(d_return)
                dates.append(date)
        else:
            daily_returns.append(0.0)
            dates.append(date)
            
        prev_alloc = allocations_df.loc[date].copy()
        
    return pd.Series(daily_returns, index=dates), dates

def load_symphonies(urls: dict, start_date: str, end_date: str):
    """Load symphonies from cache or Composer API."""
    symphony_data = {}
    req_start = pd.to_datetime(start_date)
    req_end = pd.to_datetime(end_date)
    
    for name, url in urls.items():
        try:
            sym_id = url.split('/')[-2] if url.endswith('/details') else url.split('/')[-1]
            cache_csv = os.path.join(STORAGE_DIR, f"{sym_id}_returns.csv")
            cache_meta = os.path.join(STORAGE_DIR, f"{sym_id}_meta.json")
            
            use_cache = False
            
            # Check local storage for existing data
            if os.path.exists(cache_csv) and os.path.exists(cache_meta):
                with open(cache_meta, 'r') as f:
                    meta = json.load(f)
                
                cached_start = pd.to_datetime(meta['earliest_date'])
                cached_end = pd.to_datetime(meta['latest_date'])
                
                # Use cache if it covers the requested dates
                if req_start >= cached_start and req_end <= cached_end:
                    actual_name = meta['symphony_name']
                    df = pd.read_csv(cache_csv, index_col='date', parse_dates=True)
                    
                    df = df.loc[req_start:req_end]
                    
                    df.attrs['metadata'] = {
                        'id': sym_id,
                        'earliest_date': df.index.min(),
                        'latest_date': df.index.max()
                    }
                    symphony_data[actual_name] = df
                    use_cache = True
            
            # Fetch new data if cache is missing or insufficient
            if not use_cache:
                alloc_df, actual_name, tickers = fetch_composer_symphony(url, start_date, end_date)
                returns, dates = calculate_symphony_returns(alloc_df, tickers)
                
                df = pd.DataFrame({'returns': returns}, index=dates)
                df.index.name = 'date'
                
                # Save new data to storage
                df.to_csv(cache_csv)
                meta_data = {
                    'symphony_name': actual_name,
                    'id': sym_id,
                    'earliest_date': dates[0].strftime('%Y-%m-%d') if dates else None,
                    'latest_date': dates[-1].strftime('%Y-%m-%d') if dates else None
                }
                with open(cache_meta, 'w') as f:
                    json.dump(meta_data, f)
                
                df.attrs['metadata'] = {
                    'id': sym_id,
                    'earliest_date': dates[0],
                    'latest_date': dates[-1]
                }
                symphony_data[actual_name] = df
                
        except Exception as e:
            pass # UI handles missing data alerts
            
    return symphony_data
