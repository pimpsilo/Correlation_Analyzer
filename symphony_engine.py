import re
import os
import json
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta

# Create the local data storage folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "data_storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

def normalize_ticker_symbol(symbol: str) -> str:
    """Normalize ticker symbol for Yahoo Finance."""
    sym = symbol.strip().upper().lstrip('$')
    if not sym.startswith('^'):
        sym = sym.replace('/', '-').replace('.', '-')
    return sym

def parse_urls_from_text(text: str) -> dict:
    """Parse Composer symphony URLs, symphony IDs, or ticker symbols from text,
    supporting newlines, commas, spaces, and comments (#)."""
    parsed_items = {}
    url_pattern = r'(?:https?://)?(?:app\.)?composer\.trade/symphony/([A-Za-z0-9_-]+)'
    
    found_ids = set()
    clean_lines = []
    for line in text.split('\n'):
        line_clean = line.split('#')[0].strip()
        if line_clean:
            clean_lines.append(line_clean)
            
    tokens = re.split(r'[\s,]+', ' '.join(clean_lines))
    for token in tokens:
        token = token.strip()
        if not token:
            continue
            
        lower_token = token.lower()
        # Explicit ticker prefixes (e.g. ticker:SPY, etf:QQQ, mf:FSELX)
        if lower_token.startswith(('ticker:', 'stock:', 'etf:', 'mf:', 'fund:')):
            raw_sym = token.split(':', 1)[1].strip()
            sym = normalize_ticker_symbol(raw_sym)
            if sym and sym not in found_ids:
                found_ids.add(sym)
                parsed_items[f"Ticker_{sym}"] = f"ticker:{sym}"
            continue
            
        # Explicit symphony prefix (e.g. symphony:IxUYGLhjD2rF1Xi2GEmI)
        if lower_token.startswith(('symphony:', 'sym:')):
            sym_id = token.split(':', 1)[1].strip()
            if sym_id and sym_id not in found_ids:
                found_ids.add(sym_id)
                parsed_items[f"Symphony_{len(parsed_items) + 1}"] = f"https://app.composer.trade/symphony/{sym_id}/details"
            continue
            
        # Match Composer URLs
        url_match = re.search(url_pattern, token)
        if url_match:
            sym_id = url_match.group(1)
            if sym_id not in found_ids:
                found_ids.add(sym_id)
                parsed_items[f"Symphony_{len(parsed_items) + 1}"] = f"https://app.composer.trade/symphony/{sym_id}/details"
            continue
            
        # Match Composer ID (alphanumeric string with length >= 12)
        if re.match(r'^[A-Za-z0-9_-]{12,}$', token):
            sym_id = token
            if sym_id not in found_ids:
                found_ids.add(sym_id)
                parsed_items[f"Symphony_{len(parsed_items) + 1}"] = f"https://app.composer.trade/symphony/{sym_id}/details"
            continue
            
        # Match ticker symbols (1-6 letters, optional classes like BRK-B or BRK.B, or indexes like ^GSPC)
        clean_token = token.lstrip('$')
        if re.match(r'^\^?[A-Za-z]{1,6}(?:[\.\-\/][A-Za-z]{1,2})?$', clean_token):
            sym = normalize_ticker_symbol(clean_token)
            if sym and sym not in found_ids:
                found_ids.add(sym)
                parsed_items[f"Ticker_{sym}"] = f"ticker:{sym}"
            continue
                
    return parsed_items

def convert_trading_date(date_int):
    """Convert integer trading date to datetime."""
    return datetime.strptime("01/01/1970", "%m/%d/%Y") + timedelta(days=int(date_int))

def fetch_composer_symphony(symphony_url: str, start_date: str, end_date: str, max_retries: int = 3):
    """Fetch backtest data from Composer API with retry logic."""
    symphony_id = symphony_url.split('/')[-2] if symphony_url.endswith('/details') else symphony_url.split('/')[-1]
    
    payload = {
        "capital": 100000, "apply_reg_fee": True, "apply_taf_fee": True,
        "backtest_version": "v2", "slippage_percent": 0.0005,
        "start_date": start_date, "end_date": end_date,
    }
    
    last_err = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"https://backtest-api.composer.trade/api/v2/public/symphonies/{symphony_id}/backtest", 
                json=payload,
                timeout=30
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
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                raise last_err

def calculate_symphony_returns(allocations_df: pd.DataFrame, tickers: list):
    """Calculate daily returns from allocations and Yahoo Finance prices."""
    allocations_df = allocations_df.loc[(allocations_df != 0).any(axis=1)] * 100.0
    if '$USD' not in allocations_df.columns:
        allocations_df['$USD'] = 0
    allocations_df.index = pd.to_datetime(allocations_df.index).normalize()
    
    unique_tickers = [t for t in tickers if t != '$USD']
    start_str = (allocations_df.index.min() - timedelta(days=10)).strftime('%Y-%m-%d')
    end_str = (allocations_df.index.max() + timedelta(days=10)).strftime('%Y-%m-%d')
    
    prices_data = {}
    
    # Batch download to minimize HTTP requests and avoid Yahoo Finance 429 rate limits
    clean_map = {t: t.replace('BRK/B', 'BRK-B') for t in unique_tickers}
    download_list = list(clean_map.values())
    
    if download_list:
        try:
            raw_prices = yf.download(download_list, start=start_str, end=end_str, auto_adjust=True, progress=False)
            if not raw_prices.empty and 'Close' in raw_prices:
                close_df = raw_prices['Close']
                for orig_t, clean_t in clean_map.items():
                    if clean_t in close_df.columns:
                        series = close_df[clean_t].dropna()
                        if not series.empty:
                            prices_data[orig_t] = series.tz_localize(None) if series.index.tz is not None else series
        except Exception:
            pass
            
    # Fallback to single-ticker fetch for any missing ticker
    for orig_t in unique_tickers:
        if orig_t not in prices_data:
            try:
                t_obj = yf.Ticker(orig_t.replace('BRK/B', 'BRK-B'))
                hist = t_obj.history(start=start_str, end=end_str, auto_adjust=True)
                if not hist.empty:
                    series = hist['Close']
                    prices_data[orig_t] = series.tz_localize(None) if series.index.tz is not None else series
            except Exception:
                pass

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
    """Load symphonies or individual tickers from cache or APIs.
    
    Returns:
        tuple: (symphony_data: dict, failed_symphonies: dict)
    """
    symphony_data = {}
    failed_symphonies = {}
    req_start = pd.to_datetime(start_date)
    req_end = pd.to_datetime(end_date)
    
    composer_items = {}
    ticker_items = []
    
    for name, target in urls.items():
        if target.startswith("ticker:"):
            sym = target.split("ticker:")[1].upper()
            if sym not in ticker_items:
                ticker_items.append(sym)
        elif target.startswith("http") or "composer.trade" in target or len(target) >= 12:
            composer_items[name] = target
        else:
            norm = normalize_ticker_symbol(target)
            if len(norm) <= 8 and not re.match(r'^[A-Za-z0-9_-]{12,}$', target):
                if norm not in ticker_items:
                    ticker_items.append(norm)
            else:
                composer_items[name] = target

    # --- 1. PROCESS TICKERS (Stocks, ETFs, Mutual Funds) ---
    uncached_tickers = []
    
    for ticker in ticker_items:
        cache_csv = os.path.join(STORAGE_DIR, f"ticker_{ticker}_returns.csv")
        cache_meta = os.path.join(STORAGE_DIR, f"ticker_{ticker}_meta.json")
        use_cache = False
        
        if os.path.exists(cache_csv) and os.path.exists(cache_meta):
            try:
                with open(cache_meta, 'r') as f:
                    cached_meta = json.load(f)
                cached_start = pd.to_datetime(cached_meta.get('earliest_date'))
                cached_end = pd.to_datetime(cached_meta.get('latest_date'))
                
                if cached_start is not None and cached_end is not None:
                    if req_end <= cached_end + pd.Timedelta(days=5) and req_start <= cached_end:
                        df = pd.read_csv(cache_csv, index_col='date', parse_dates=True)
                        actual_start = max(req_start, cached_start)
                        actual_end = min(req_end, cached_end)
                        df_sliced = df.loc[actual_start:actual_end]
                        
                        if len(df_sliced) >= 2:
                            df_sliced.attrs['metadata'] = {
                                'id': ticker,
                                'earliest_date': df_sliced.index.min(),
                                'latest_date': df_sliced.index.max(),
                                'is_ticker': True
                            }
                            display_name = ticker
                            suffix = 2
                            while display_name in symphony_data:
                                display_name = f"{ticker}_{suffix}"
                                suffix += 1
                            symphony_data[display_name] = df_sliced
                            use_cache = True
            except Exception:
                use_cache = False
                
        if not use_cache:
            uncached_tickers.append(ticker)

    if uncached_tickers:
        fetch_start = (req_start - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
        fetch_end = (req_end + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        
        batch_prices = {}
        try:
            raw_prices = yf.download(
                uncached_tickers,
                start=fetch_start,
                end=fetch_end,
                auto_adjust=True,
                progress=False
            )
            if not raw_prices.empty:
                close_df = raw_prices['Close'] if 'Close' in raw_prices else raw_prices
                if isinstance(close_df, pd.DataFrame):
                    for t in uncached_tickers:
                        if t in close_df.columns:
                            s = close_df[t].dropna()
                            if len(s) >= 2:
                                batch_prices[t] = s
                elif isinstance(close_df, pd.Series):
                    s = close_df.dropna()
                    if len(s) >= 2 and len(uncached_tickers) == 1:
                        batch_prices[uncached_tickers[0]] = s
        except Exception:
            pass
            
        for ticker in uncached_tickers:
            series = batch_prices.get(ticker)
            if series is None or len(series) < 2:
                try:
                    t_obj = yf.Ticker(ticker)
                    hist = t_obj.history(start=fetch_start, end=fetch_end, auto_adjust=True)
                    if not hist.empty and 'Close' in hist:
                        s = hist['Close'].dropna()
                        if len(s) >= 2:
                            series = s
                except Exception:
                    pass
                    
            if series is not None and len(series) >= 2:
                series.index = pd.to_datetime(series.index).normalize()
                if series.index.tz is not None:
                    series.index = series.index.tz_localize(None)
                returns = series.pct_change().dropna()
                returns_sliced = returns.loc[req_start:req_end]
                
                if len(returns_sliced) >= 2:
                    df = pd.DataFrame({'returns': returns_sliced})
                    df.index.name = 'date'
                    
                    cache_csv = os.path.join(STORAGE_DIR, f"ticker_{ticker}_returns.csv")
                    cache_meta = os.path.join(STORAGE_DIR, f"ticker_{ticker}_meta.json")
                    try:
                        df.to_csv(cache_csv)
                        meta_data = {
                            'symphony_name': ticker,
                            'id': ticker,
                            'is_ticker': True,
                            'earliest_date': returns_sliced.index.min().strftime('%Y-%m-%d'),
                            'latest_date': returns_sliced.index.max().strftime('%Y-%m-%d')
                        }
                        with open(cache_meta, 'w') as f:
                            json.dump(meta_data, f)
                    except Exception:
                        pass
                        
                    df.attrs['metadata'] = {
                        'id': ticker,
                        'earliest_date': returns_sliced.index.min(),
                        'latest_date': returns_sliced.index.max(),
                        'is_ticker': True
                    }
                    display_name = ticker
                    suffix = 2
                    while display_name in symphony_data:
                        display_name = f"{ticker}_{suffix}"
                        suffix += 1
                    symphony_data[display_name] = df
                else:
                    failed_symphonies[ticker] = f"Insufficient trading days for {ticker} within the selected date range."
            else:
                fallback_success = False
                cache_csv = os.path.join(STORAGE_DIR, f"ticker_{ticker}_returns.csv")
                cache_meta = os.path.join(STORAGE_DIR, f"ticker_{ticker}_meta.json")
                if os.path.exists(cache_csv) and os.path.exists(cache_meta):
                    try:
                        df = pd.read_csv(cache_csv, index_col='date', parse_dates=True)
                        if len(df) >= 2:
                            df.attrs['metadata'] = {
                                'id': ticker,
                                'earliest_date': df.index.min(),
                                'latest_date': df.index.max(),
                                'is_ticker': True
                            }
                            display_name = ticker
                            suffix = 2
                            while display_name in symphony_data:
                                display_name = f"{ticker}_{suffix}"
                                suffix += 1
                            symphony_data[display_name] = df
                            fallback_success = True
                    except Exception:
                        pass
                if not fallback_success:
                    failed_symphonies[ticker] = f"No price data found on Yahoo Finance for ticker '{ticker}'."

    # --- 2. PROCESS COMPOSER SYMPHONIES ---
    for name, url in composer_items.items():
        sym_id = url.split('/')[-2] if url.endswith('/details') else url.split('/')[-1]
        cache_csv = os.path.join(STORAGE_DIR, f"{sym_id}_returns.csv")
        cache_meta = os.path.join(STORAGE_DIR, f"{sym_id}_meta.json")
        
        use_cache = False
        cached_meta = None
        
        # Check local storage for existing data
        if os.path.exists(cache_csv) and os.path.exists(cache_meta):
            try:
                with open(cache_meta, 'r') as f:
                    cached_meta = json.load(f)
                cached_start = pd.to_datetime(cached_meta.get('earliest_date'))
                cached_end = pd.to_datetime(cached_meta.get('latest_date'))
                
                if cached_start is not None and cached_end is not None:
                    if req_end <= cached_end + pd.Timedelta(days=5) and req_start <= cached_end:
                        df = pd.read_csv(cache_csv, index_col='date', parse_dates=True)
                        actual_start = max(req_start, cached_start)
                        actual_end = min(req_end, cached_end)
                        df_sliced = df.loc[actual_start:actual_end]
                        
                        if len(df_sliced) >= 2:
                            actual_name = cached_meta.get('symphony_name', name)
                            df_sliced.attrs['metadata'] = {
                                'id': sym_id,
                                'earliest_date': df_sliced.index.min(),
                                'latest_date': df_sliced.index.max()
                            }
                            display_name = actual_name
                            suffix = 2
                            while display_name in symphony_data:
                                display_name = f"{actual_name} ({sym_id[:6]}" + (f"_{suffix})" if suffix > 2 else ")")
                                suffix += 1
                                
                            symphony_data[display_name] = df_sliced
                            use_cache = True
            except Exception:
                use_cache = False

        # Fetch new data if cache is missing or needs live update
        if not use_cache:
            try:
                alloc_df, actual_name, tickers = fetch_composer_symphony(url, start_date, end_date)
                returns, dates = calculate_symphony_returns(alloc_df, tickers)
                
                if len(dates) < 2:
                    raise ValueError("Insufficient trading days returned")
                    
                df = pd.DataFrame({'returns': returns}, index=dates)
                df.index.name = 'date'
                
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
                
                display_name = actual_name
                suffix = 2
                while display_name in symphony_data:
                    display_name = f"{actual_name} ({sym_id[:6]}" + (f"_{suffix})" if suffix > 2 else ")")
                    suffix += 1
                    
                symphony_data[display_name] = df
                time.sleep(0.3)  # Rate limit throttle between live queries
                
            except Exception as e:
                fallback_success = False
                if cached_meta and os.path.exists(cache_csv):
                    try:
                        df = pd.read_csv(cache_csv, index_col='date', parse_dates=True)
                        if len(df) >= 2:
                            actual_name = cached_meta.get('symphony_name', name)
                            df.attrs['metadata'] = {
                                'id': sym_id,
                                'earliest_date': df.index.min(),
                                'latest_date': df.index.max()
                            }
                            display_name = actual_name
                            suffix = 2
                            while display_name in symphony_data:
                                display_name = f"{actual_name} ({sym_id[:6]}" + (f"_{suffix})" if suffix > 2 else ")")
                                suffix += 1
                            symphony_data[display_name] = df
                            fallback_success = True
                    except Exception:
                        pass
                        
                if not fallback_success:
                    err_msg = str(e)
                    if "429" in err_msg:
                        err_msg = "Rate limit reached (HTTP 429). Please retry after a brief pause."
                    elif "400" in err_msg or "404" in err_msg:
                        err_msg = f"Composer API error ({err_msg}). Strategy may be private or invalid."
                    failed_symphonies[sym_id] = err_msg

    return symphony_data, failed_symphonies
