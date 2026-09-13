# Multi-Symphony Portfolio Monte Carlo with Composer API Integration
# Fetches data directly from Composer URLs - no CSV export needed!
# Extension of the multi_symphony_portfolio_monte_carlo.py framework

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple
import json
import os

# Import the core simulator
from multi_symphony_portfolio_monte_carlo import PortfolioMonteCarloSimulator, _dual_output, dual_print

def parse_urls_from_text(text: str) -> Dict[str, str]:
    """
    Parse Composer symphony URLs from pasted text.
    Handles full URLs, short URLs, or just symphony IDs.
    Skips comment lines (starting with #) and blank lines.
    
    Args:
        text: Multi-line text containing URLs or IDs
    
    Returns:
        Dict mapping auto-generated names to URLs
    """
    import re
    
    symphony_urls = {}
    
    # Pattern to match Composer URLs or IDs
    # Matches:
    # - Full URLs: https://app.composer.trade/symphony/IxUYGLhjD2rF1Xi2GEmI/details
    # - Short URLs: composer.trade/symphony/IxUYGLhjD2rF1Xi2GEmI
    # - Just IDs: IxUYGLhjD2rF1Xi2GEmI (12+ alphanumeric characters)
    
    url_pattern = r'(?:https?://)?(?:app\.)?composer\.trade/symphony/([A-Za-z0-9_-]+)'
    id_pattern = r'\b([A-Za-z0-9_-]{12,})\b'
    
    lines = text.split('\n')
    found_ids = set()
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Try to match full or partial URLs first
        url_match = re.search(url_pattern, line)
        if url_match:
            symphony_id = url_match.group(1)
            if symphony_id not in found_ids:
                found_ids.add(symphony_id)
                url = f"https://app.composer.trade/symphony/{symphony_id}/details"
                name = f"Symphony_{len(symphony_urls) + 1}"
                symphony_urls[name] = url
                continue
        
        # If no URL match, try to match standalone ID
        id_match = re.search(id_pattern, line)
        if id_match:
            symphony_id = id_match.group(1)
            # Avoid matching things that are clearly not symphony IDs
            if symphony_id not in found_ids and len(symphony_id) >= 12:
                found_ids.add(symphony_id)
                url = f"https://app.composer.trade/symphony/{symphony_id}/details"
                name = f"Symphony_{len(symphony_urls) + 1}"
                symphony_urls[name] = url
    
    return symphony_urls


def convert_trading_date(date_int):
    """Convert trading date integer to datetime object."""
    date_1 = datetime.strptime("01/01/1970", "%m/%d/%Y")
    dt = date_1 + timedelta(days=int(date_int))
    return dt

class YahooFinanceAPI:
    """Fetches historical price data using the yfinance package."""
    
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
            print("Successfully initialized yfinance package")
        except ImportError:
            print("yfinance package is not installed. Please install it with: pip install yfinance")
            raise ImportError("yfinance package is required")
        
        self.ticker_map = {'BRK/B': 'BRK-B'}
        self.use_batch_download = True
        self.batch_size = 5
    
    def fetch_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """Fetch historical price data for multiple symbols."""
        print(f"Fetching historical data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        mapped_symbols = {}
        for symbol in symbols:
            yahoo_symbol = self.ticker_map.get(symbol, symbol)
            mapped_symbols[yahoo_symbol] = symbol
        
        return self._individual_download(mapped_symbols, start_date, end_date)
    
    def _individual_download(self, mapped_symbols: Dict[str, str], start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """Download data for each symbol individually."""
        import time
        price_data = {}
        
        for yahoo_symbol, original_symbol in mapped_symbols.items():
            print(f"Fetching data for {original_symbol}")
            
            try:
                ticker_obj = self.yf.Ticker(yahoo_symbol)
                data = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
                
                if data.empty:
                    print(f"No data returned for {original_symbol}")
                    continue
                
                if 'Close' in data.columns:
                    series = data['Close'].copy()
                    series = series.dropna()
                    series = series.astype(np.float32)
                    
                    if series.index.tz is not None:
                        series.index = series.index.tz_convert('America/New_York')
                        series.index = series.index.tz_localize(None)
                    
                    series = series[~series.index.duplicated(keep='last')]
                    
                    if not series.empty:
                        series.name = original_symbol
                        price_data[original_symbol] = series
                        print(f"Successfully retrieved {original_symbol}: {len(series)} points")
                
            except Exception as e:
                print(f"Error fetching data for {original_symbol}: {str(e)}")
            
            time.sleep(0.5)  # Be nice to Yahoo Finance
        
        return price_data


def fetch_composer_symphony(symphony_url: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Fetch backtest data from Composer API.
    
    Args:
        symphony_url: Composer symphony URL (e.g., https://app.composer.trade/symphony/IxUYGLhjD2rF1Xi2GEmI/details)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        Tuple of (allocations_df, symphony_name, tickers)
    """
    # Extract symphony ID from URL
    if symphony_url.endswith('/details'):
        symphony_id = symphony_url.split('/')[-2]
    else:
        symphony_id = symphony_url.split('/')[-1]
    
    print(f"Fetching symphony {symphony_id} from Composer API...")
    
    # Prepare API request
    payload = {
        "capital": 100000,
        "apply_reg_fee": True,
        "apply_taf_fee": True,
        "backtest_version": "v2",
        "slippage_percent": 0.0005,
        "start_date": start_date,
        "end_date": end_date,
    }
    
    url = f"https://backtest-api.composer.trade/api/v2/public/symphonies/{symphony_id}/backtest"
    
    # Make API request
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    data = response.json()
    symphony_name = data['legend'][symphony_id]['name']
    
    print(f"  Symphony name: {symphony_name}")
    
    # Extract holdings and allocations
    current_holdings = data["last_market_days_holdings"]  # Current allocations
    tickers = list(current_holdings.keys())
    
    print(f"  Current holdings: {len(current_holdings)} assets")
    
    allocations = data["tdvm_weights"]
    date_range = pd.date_range(start=start_date, end=end_date)
    allocations_df = pd.DataFrame(0.0, index=date_range, columns=tickers)
    
    # Populate allocations
    for ticker in allocations:
        for date_int in allocations[ticker]:
            trading_date = convert_trading_date(date_int)
            percent = allocations[ticker][date_int]
            allocations_df.at[trading_date, ticker] = percent
    
    # Convert current holdings to dict of {ticker: percentage}
    current_holdings_dict = {}
    for ticker, value in current_holdings.items():
        if value > 0.0001:  # Only include non-zero holdings
            current_holdings_dict[ticker] = value / 100.0  # Convert from percentage to decimal
    
    return allocations_df, symphony_name, tickers, current_holdings_dict


def calculate_symphony_returns(allocations_df: pd.DataFrame, tickers: List[str]) -> Tuple[pd.Series, pd.DatetimeIndex]:
    """
    Calculate daily portfolio returns from allocations and price data.
    
    Args:
        allocations_df: DataFrame with allocation percentages
        tickers: List of ticker symbols
    
    Returns:
        Tuple of (daily_returns, dates)
    """
    # Find first valid allocation
    first_valid_index = allocations_df[(abs(allocations_df) > 0.000001).any(axis=1)].first_valid_index()
    
    # Clean up allocations
    allocations_df = allocations_df.loc[(allocations_df != 0).any(axis=1)] * 100.0
    
    # Add $USD column if not present
    if '$USD' not in allocations_df.columns:
        allocations_df['$USD'] = 0
    
    # Normalize dates
    allocations_df.index = pd.to_datetime(allocations_df.index).normalize()
    
    # Extract unique tickers (excluding cash)
    unique_tickers = {ticker for ticker in tickers if ticker != '$USD'}
    
    # Fetch historical prices
    start_date = allocations_df.index.min() - timedelta(days=10)
    end_date = allocations_df.index.max() + timedelta(days=10)
    
    print(f"Fetching price data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    yahoo_api = YahooFinanceAPI()
    prices_data = yahoo_api.fetch_historical_data(
        list(unique_tickers),
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    # Create price DataFrame
    prices = pd.DataFrame({ticker: prices_data[ticker] for ticker in prices_data})
    prices.index = pd.to_datetime(prices.index).normalize()
    prices['$USD'] = 1.0
    
    # Fill missing tickers
    for ticker in tickers:
        if ticker not in prices.columns and ticker != '$USD':
            print(f"Warning: Price data for {ticker} not found. Setting to NaN.")
            prices[ticker] = np.nan
    
    # Forward fill and backfill
    prices = prices.ffill().bfill().fillna(1.0)
    prices = prices[tickers]
    
    # Sort both DataFrames
    allocations_df.sort_index(inplace=True)
    prices.sort_index(inplace=True)
    
    print(f"Have {len(allocations_df.index)} allocation dates")
    print(f"Have {len(prices.index)} price dates")
    
    # Calculate daily returns
    daily_returns = []
    dates = []
    
    prev_value = None
    prev_alloc = None
    
    for date in allocations_df.index:
        date_str = date.strftime('%Y-%m-%d')
        
        # Get current allocation
        current_alloc = allocations_df.loc[date]
        
        # Get prices for this date and previous date
        if date not in prices.index:
            continue
        
        current_prices = prices.loc[date]
        
        if prev_value is not None and prev_alloc is not None:
            # Calculate return based on price changes and allocations
            prev_date = dates[-1]
            if prev_date in prices.index:
                prev_prices = prices.loc[prev_date]
                
                # Calculate weighted return
                daily_return = 0.0
                for ticker in tickers:
                    if prev_alloc[ticker] > 0.0001 and ticker in current_prices and ticker in prev_prices:
                        if prev_prices[ticker] > 0:
                            price_change = (current_prices[ticker] - prev_prices[ticker]) / prev_prices[ticker]
                            daily_return += (prev_alloc[ticker] / 100.0) * price_change
                
                daily_returns.append(daily_return)
                dates.append(date)
        else:
            # First day - no return
            daily_returns.append(0.0)
            dates.append(date)
        
        prev_alloc = current_alloc.copy()
        prev_value = 100000  # Just a placeholder
    
    return pd.Series(daily_returns, index=dates), dates


def load_multiple_symphonies_from_urls(symphony_urls: Dict[str, str], 
                                      start_date: str = None, 
                                      end_date: str = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    """
    Load multiple symphonies from Composer URLs.
    
    Args:
        symphony_urls: Dict mapping symphony names to Composer URLs
        start_date: Start date in YYYY-MM-DD format (None = use earliest available)
        end_date: End date in YYYY-MM-DD format (None = use today)
    
    Returns:
        Tuple of (symphony_data, current_holdings)
        - symphony_data: Dict mapping symphony names to DataFrames with 'date' and 'returns' columns
        - current_holdings: Dict mapping symphony names to current asset allocations
    """
    if start_date is None:
        start_date = '2015-01-01'  # Default to 10 years ago
    
    if end_date is None:
        end_date = date.today().strftime('%Y-%m-%d')
    
    print(f"\n{'='*80}")
    print(f"FETCHING {len(symphony_urls)} SYMPHONIES FROM COMPOSER")
    print(f"{'='*80}")
    print(f"Date range: {start_date} to {end_date}\n")
    
    symphony_data = {}
    current_holdings = {}
    symphony_metadata = {}  # Track IDs and date ranges
    
    for symphony_name, url in symphony_urls.items():
        print(f"\n{'-'*80}")
        print(f"Loading: {symphony_name}")
        print(f"URL: {url}")
        print(f"{'-'*80}")
        
        # Extract symphony ID from URL
        import re
        match = re.search(r'/symphony/([a-zA-Z0-9_-]+)', url)
        symphony_id = match.group(1) if match else 'unknown'
        
        try:
            # Fetch allocations from Composer
            allocations_df, actual_name, tickers, holdings_dict = fetch_composer_symphony(url, start_date, end_date)
            
            # Use actual name from Composer if we just provided a key
            if symphony_name.startswith('Symphony_'):
                symphony_name = actual_name
            
            # Calculate returns
            returns, dates = calculate_symphony_returns(allocations_df, tickers)
            
            # Create DataFrame
            symphony_df = pd.DataFrame({
                'date': dates,
                'returns': returns
            })
            
            symphony_data[symphony_name] = symphony_df
            current_holdings[symphony_name] = holdings_dict
            
            # Store metadata
            symphony_metadata[symphony_name] = {
                'id': symphony_id,
                'earliest_date': dates[0],
                'latest_date': dates[-1],
                'total_days': len(dates)
            }
            
            print(f"✓ Successfully loaded {symphony_name}")
            print(f"  ID: {symphony_id}")
            print(f"  Data: {len(symphony_df)} days from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
            print(f"  Current holdings: {len(holdings_dict)} assets")
            
        except Exception as e:
            print(f"✗ Error loading {symphony_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Successfully loaded {len(symphony_data)} out of {len(symphony_urls)} symphonies")
    print(f"{'='*80}\n")
    
    # Attach metadata to symphony_data for later use
    for name in symphony_data.keys():
        symphony_data[name].attrs['metadata'] = symphony_metadata.get(name, {})
    
    return symphony_data, current_holdings


def interactive_symphony_loader():
    """
    Interactive script to load multiple symphonies from Composer URLs.
    """
    print("="*80)
    print("MULTI-SYMPHONY PORTFOLIO MONTE CARLO - COMPOSER URL LOADER")
    print("="*80)
    
    # Ask for input mode
    print("\nHow would you like to enter symphony URLs?")
    print("1. Paste all URLs at once (fastest)")
    print("2. Enter one at a time (with custom names)")
    
    mode = input("\nEnter choice (1 or 2, default: 1): ").strip() or "1"
    
    symphony_urls = {}
    
    if mode == "1":
        # Batch paste mode
        print("\n" + "="*80)
        print("BATCH PASTE MODE")
        print("="*80)
        print("\nPaste all your Composer URLs below (one per line).")
        print("You can paste full URLs or just the symphony IDs.")
        print("Example URLs:")
        print("  https://app.composer.trade/symphony/IxUYGLhjD2rF1Xi2GEmI/details")
        print("  https://app.composer.trade/symphony/AbCdEfGhIjKl/details")
        print("Or just IDs:")
        print("  IxUYGLhjD2rF1Xi2GEmI")
        print("  AbCdEfGhIjKl")
        print("\nWhen done, press Enter on an empty line:\n")
        
        urls_text = []
        while True:
            line = input().strip()
            if not line:
                break
            urls_text.append(line)
        
        if not urls_text:
            print("No URLs entered, exiting...")
            return None
        
        # Parse URLs from text
        print(f"\nParsing {len(urls_text)} lines...")
        symphony_urls = parse_urls_from_text('\n'.join(urls_text))
        
        if not symphony_urls:
            print("No valid URLs found in pasted text")
            return None
        
        print(f"\nFound {len(symphony_urls)} valid Composer URLs:")
        for i, (name, url) in enumerate(symphony_urls.items(), 1):
            print(f"  {i}. {name}")
        
        # Ask if user wants to customize names
        customize = input("\nCustomize symphony names? (y/n, default: n): ").lower() == 'y'
        
        if customize:
            print("\nEnter custom names (press Enter to keep auto-detected name):")
            updated_urls = {}
            for name, url in symphony_urls.items():
                custom_name = input(f"  {name} -> ").strip()
                if custom_name:
                    updated_urls[custom_name] = url
                else:
                    updated_urls[name] = url
            symphony_urls = updated_urls
    
    else:
        # One-by-one mode (original behavior)
        while True:
            try:
                n_symphonies = int(input("\nHow many symphonies do you want to include? (2-15): "))
                if 2 <= n_symphonies <= 15:
                    break
                print("Please enter a number between 2 and 15")
            except ValueError:
                print("Please enter a valid number")
        
        print(f"\nEnter Composer URL for each symphony:")
        print("Example: https://app.composer.trade/symphony/IxUYGLhjD2rF1Xi2GEmI/details")
        print("(You can also just paste the symphony ID: IxUYGLhjD2rF1Xi2GEmI)\n")
        
        for i in range(n_symphonies):
            while True:
                url = input(f"Symphony {i+1} URL: ").strip()
                
                if not url:
                    print("URL cannot be empty, please try again")
                    continue
                
                # Allow just the ID to be pasted
                if not url.startswith('http'):
                    url = f"https://app.composer.trade/symphony/{url}/details"
                
                # Optional: provide custom name
                custom_name = input(f"  Custom name for this symphony (blank to auto-detect): ").strip()
                
                if custom_name:
                    symphony_urls[custom_name] = url
                else:
                    symphony_urls[f"Symphony_{i+1}"] = url
                
                break
    
    # Get date range
    print("\nDate range for backtests:")
    today = date.today().strftime('%Y-%m-%d')
    start_date = input(f"Start date (YYYY-MM-DD, blank for 2015-01-01): ") or '2015-01-01'
    end_date = input(f"End date (YYYY-MM-DD, blank for today {today}): ") or today
    
    # Load all symphonies
    symphony_data, current_holdings = load_multiple_symphonies_from_urls(symphony_urls, start_date, end_date)
    
    if len(symphony_data) < 2:
        print("\nError: Need at least 2 successfully loaded symphonies to continue")
        return None, None
    
    return symphony_data, current_holdings


def main():
    """Main function for interactive multi-symphony portfolio analysis."""
    
    # Enable dual output capture for text summary
    _dual_output.enable()
    
    # Add header to summary
    dual_print("="*80)
    dual_print("MULTI-SYMPHONY PORTFOLIO MONTE CARLO SIMULATION")
    dual_print("="*80)
    dual_print(f"Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    dual_print("="*80)
    dual_print()
    
    # Load symphonies from Composer URLs
    result = interactive_symphony_loader()
    
    if result is None or result[0] is None:
        return
    
    symphony_data, current_holdings = result
    
    # Ask if user wants correlation-only analysis
    print("\n" + "="*80)
    print("ANALYSIS MODE")
    print("="*80)
    print("\nWhat would you like to do?")
    print("1. Full Monte Carlo analysis (correlation + concentration + simulation)")
    print("2. Quick correlation analysis only (skip Monte Carlo)")
    
    mode_choice = input("\nEnter choice (1 or 2, default: 1): ").strip() or "1"
    
    if mode_choice == "2":
        # Correlation-only mode
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        symphony_names = list(symphony_data.keys())
        n_symphonies = len(symphony_names)
        
        # Align all symphonies to common date range
        all_dates = symphony_data[symphony_names[0]]['date'].values
        for name in symphony_names[1:]:
            all_dates = np.intersect1d(all_dates, symphony_data[name]['date'].values)
        
        dual_print(f"\nCommon date range: {len(all_dates)} days")
        dual_print(f"From {pd.to_datetime(all_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(all_dates[-1]).strftime('%Y-%m-%d')}")
        
        # Build returns matrix
        returns_matrix = np.zeros((len(all_dates), n_symphonies))
        for i, name in enumerate(symphony_names):
            df = symphony_data[name]
            df = df[df['date'].isin(all_dates)].sort_values('date')
            returns_matrix[:, i] = df['returns'].values
        
        # Calculate correlation
        correlation_matrix = np.corrcoef(returns_matrix.T)
        
        # Calculate drawdown correlation
        dual_print("\n" + "="*80)
        dual_print("CALCULATING DRAWDOWN CORRELATION")
        dual_print("="*80)
        
        # Calculate cumulative returns for each symphony
        cumulative_returns = np.zeros_like(returns_matrix)
        for i in range(n_symphonies):
            cumulative_returns[:, i] = np.cumprod(1 + returns_matrix[:, i]) - 1
        
        # Calculate drawdown series
        drawdown_matrix = np.zeros_like(returns_matrix)
        for i in range(n_symphonies):
            cumulative = 1 + cumulative_returns[:, i]
            running_max = np.maximum.accumulate(cumulative)
            drawdown_matrix[:, i] = (cumulative - running_max) / running_max
        
        # Calculate drawdown correlation matrix
        drawdown_correlation_matrix = np.corrcoef(drawdown_matrix.T)
        dual_print(f"Drawdown correlation matrix calculated")
        
        # Calculate mean correlation for each symphony (excluding self)
        mean_correlations = []
        mean_dd_correlations = []
        for i in range(n_symphonies):
            # Returns correlation
            correlations = np.concatenate([correlation_matrix[i, :i], correlation_matrix[i, i+1:]])
            mean_correlations.append(np.mean(correlations))
            
            # Drawdown correlation
            dd_correlations = np.concatenate([drawdown_correlation_matrix[i, :i], drawdown_correlation_matrix[i, i+1:]])
            mean_dd_correlations.append(np.mean(dd_correlations))
        
        # Print individual symphony date ranges
        dual_print("\n" + "="*80)
        dual_print("INDIVIDUAL SYMPHONY DATE RANGES")
        dual_print("="*80)
        for name in symphony_names:
            metadata = symphony_data[name].attrs.get('metadata', {})
            symphony_id = metadata.get('id', 'unknown')
            earliest = metadata.get('earliest_date')
            latest = metadata.get('latest_date')
            total_days = metadata.get('total_days', 0)
            
            if earliest and latest:
                dual_print(f"{name[:50]:<50} ID: {symphony_id:<20} {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')} ({total_days:,} days)")
            else:
                dual_print(f"{name[:50]:<50} ID: {symphony_id:<20} Date range unavailable")
        
        # Print common date range
        dual_print("\n" + "="*80)
        dual_print("CORRELATION ANALYSIS DATE RANGE")
        dual_print("="*80)
        dual_print(f"Common overlapping period: {len(all_dates):,} days")
        dual_print(f"From: {pd.to_datetime(all_dates[0]).strftime('%Y-%m-%d')}")
        dual_print(f"To:   {pd.to_datetime(all_dates[-1]).strftime('%Y-%m-%d')}")
        
        # Print correlation matrix with mean column
        dual_print("\n" + "="*80)
        dual_print("SYMPHONY CORRELATION MATRIX (Returns)")
        dual_print("="*80)
        
        # Header row
        header = f"{'Symphony':<40} | " + " ".join([f"{name[:8]:>8}" for name in symphony_names]) + " |   Mean"
        dual_print(header)
        dual_print("-" * len(header))
        
        # Data rows
        for i, name in enumerate(symphony_names):
            row_data = " ".join([f"{correlation_matrix[i, j]:>8.3f}" for j in range(n_symphonies)])
            row = f"{name[:40]:<40} | {row_data} | {mean_correlations[i]:>7.3f}"
            dual_print(row)
        
        dual_print("="*80)
        
        # Calculate stats
        upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        
        dual_print("\n" + "="*80)
        dual_print("CORRELATION STATISTICS")
        dual_print("="*80)
        dual_print(f"Number of symphony pairs: {len(upper_triangle)}")
        dual_print(f"Average correlation: {np.mean(upper_triangle):.3f}")
        dual_print(f"Median correlation: {np.median(upper_triangle):.3f}")
        dual_print(f"Min correlation: {np.min(upper_triangle):.3f}")
        dual_print(f"Max correlation: {np.max(upper_triangle):.3f}")
        dual_print(f"Std deviation: {np.std(upper_triangle):.3f}")
        
        # Find highest correlations
        dual_print("\n" + "-"*80)
        dual_print("HIGHEST CORRELATIONS (Top 5 pairs)")
        dual_print("-"*80)
        
        pairs = []
        for i in range(n_symphonies):
            for j in range(i+1, n_symphonies):
                pairs.append((symphony_names[i], symphony_names[j], correlation_matrix[i, j]))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i, (name1, name2, corr) in enumerate(pairs[:5], 1):
            dual_print(f"{i}. {name1[:35]:<35} ↔ {name2[:35]:<35} : {corr:>6.3f}")
        
        # Find lowest correlations
        dual_print("\n" + "-"*80)
        dual_print("LOWEST CORRELATIONS (Top 5 pairs - Best Diversifiers)")
        dual_print("-"*80)
        
        for i, (name1, name2, corr) in enumerate(pairs[-5:][::-1], 1):
            dual_print(f"{i}. {name1[:35]:<35} ↔ {name2[:35]:<35} : {corr:>6.3f}")
        
        dual_print("="*80)
        
        # Display drawdown correlation matrix
        dual_print("\n" + "="*80)
        dual_print("SYMPHONY CORRELATION MATRIX (Drawdowns)")
        dual_print("="*80)
        
        # Header row
        header = f"{'Symphony':<40} | " + " ".join([f"{name[:8]:>8}" for name in symphony_names]) + " |   Mean"
        dual_print(header)
        dual_print("-" * len(header))
        
        # Data rows
        for i, name in enumerate(symphony_names):
            row_data = " ".join([f"{drawdown_correlation_matrix[i, j]:>8.3f}" for j in range(n_symphonies)])
            row = f"{name[:40]:<40} | {row_data} | {mean_dd_correlations[i]:>7.3f}"
            dual_print(row)
        
        dual_print("="*80)
        
        # Drawdown correlation stats
        dd_upper_triangle = drawdown_correlation_matrix[np.triu_indices_from(drawdown_correlation_matrix, k=1)]
        
        dual_print("\n" + "="*80)
        dual_print("CORRELATION STATISTICS (Drawdowns)")
        dual_print("="*80)
        dual_print(f"Number of symphony pairs: {len(dd_upper_triangle)}")
        dual_print(f"Average correlation: {np.mean(dd_upper_triangle):.3f}")
        dual_print(f"Median correlation: {np.median(dd_upper_triangle):.3f}")
        dual_print(f"Min correlation: {np.min(dd_upper_triangle):.3f}")
        dual_print(f"Max correlation: {np.max(dd_upper_triangle):.3f}")
        dual_print(f"Std deviation: {np.std(dd_upper_triangle):.3f}")
        
        # Find highest drawdown correlations
        dual_print("\n" + "-"*80)
        dual_print("HIGHEST CORRELATIONS (Top 5 pairs - Drawdowns)")
        dual_print("-"*80)
        
        dd_pairs = []
        for i in range(n_symphonies):
            for j in range(i+1, n_symphonies):
                dd_pairs.append((symphony_names[i], symphony_names[j], drawdown_correlation_matrix[i, j]))
        
        dd_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i, (name1, name2, corr) in enumerate(dd_pairs[:5], 1):
            dual_print(f"{i}. {name1[:35]:<35} ↔ {name2[:35]:<35} : {corr:>6.3f}")
        
        # Find lowest drawdown correlations
        dual_print("\n" + "-"*80)
        dual_print("LOWEST CORRELATIONS (Top 5 pairs - Best Diversifiers - Drawdowns)")
        dual_print("-"*80)
        
        for i, (name1, name2, corr) in enumerate(dd_pairs[-5:][::-1], 1):
            dual_print(f"{i}. {name1[:35]:<35} ↔ {name2[:35]:<35} : {corr:>6.3f}")
        
        dual_print("="*80)
        
        # Compare returns vs drawdown correlations
        dual_print("\n" + "="*80)
        dual_print("RETURNS vs DRAWDOWN CORRELATION COMPARISON")
        dual_print("="*80)
        dual_print(f"{'Symphony':<50} | {'Returns':>10} | {'Drawdown':>10} | {'Difference':>10}")
        dual_print("-" * 100)
        for i, name in enumerate(symphony_names):
            diff = mean_dd_correlations[i] - mean_correlations[i]
            indicator = "⚠️" if diff > 0.2 else "  "
            dual_print(f"{name[:50]:<50} | {mean_correlations[i]:>10.3f} | {mean_dd_correlations[i]:>10.3f} | {diff:>+10.3f} {indicator}")
        dual_print("="*80)
        dual_print("\n⚠️ = Drawdown correlation significantly higher than returns correlation (>0.2 difference)")
        dual_print("This means the symphony correlates more during market stress than in normal conditions.")
        
        # Export results
        output_dir = input("\nOutput directory (blank for './correlation_results'): ").strip() or './correlation_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save text summary
        text_file = os.path.join(output_dir, "correlation_analysis.txt")
        _dual_output.save_to_file(text_file)
        dual_print(f"\n✓ Correlation analysis saved to: {text_file}")
        
        # Save CSV with enhanced metadata and both correlation matrices
        import pandas as pd
        
        # Create DataFrame with returns correlation matrix
        df_returns = pd.DataFrame(
            correlation_matrix,
            index=symphony_names,
            columns=symphony_names
        )
        
        # Add mean correlation column
        df_returns['Mean_Returns_Corr'] = mean_correlations
        
        # Create DataFrame with drawdown correlation matrix
        df_drawdown = pd.DataFrame(
            drawdown_correlation_matrix,
            index=symphony_names,
            columns=symphony_names
        )
        
        # Add mean drawdown correlation column
        df_drawdown['Mean_Drawdown_Corr'] = mean_dd_correlations
        
        # Add symphony metadata columns
        symphony_ids = []
        earliest_dates = []
        
        for name in symphony_names:
            metadata = symphony_data[name].attrs.get('metadata', {})
            symphony_ids.append(metadata.get('id', 'unknown'))
            earliest = metadata.get('earliest_date')
            earliest_dates.append(earliest.strftime('%Y-%m-%d') if earliest else 'N/A')
        
        df_returns.insert(0, 'Symphony_ID', symphony_ids)
        df_returns.insert(1, 'Earliest_Date_Available', earliest_dates)
        
        df_drawdown.insert(0, 'Symphony_ID', symphony_ids)
        df_drawdown.insert(1, 'Earliest_Date_Available', earliest_dates)
        
        # Create summary comparison DataFrame
        df_summary = pd.DataFrame({
            'Symphony_ID': symphony_ids,
            'Earliest_Date_Available': earliest_dates,
            'Symphony_Name': symphony_names,
            'Mean_Returns_Corr': mean_correlations,
            'Mean_Drawdown_Corr': mean_dd_correlations,
            'Difference': [dd - ret for dd, ret in zip(mean_dd_correlations, mean_correlations)]
        })
        
        # Add analysis info rows
        analysis_info = pd.DataFrame([{
            'Symphony_ID': 'ANALYSIS_INFO',
            'Earliest_Date_Available': 'Common_Range',
            **{name: pd.to_datetime(all_dates[0]).strftime('%Y-%m-%d') for name in symphony_names},
            'Mean_Returns_Corr': f"From: {pd.to_datetime(all_dates[0]).strftime('%Y-%m-%d')}"
        }])
        analysis_info2 = pd.DataFrame([{
            'Symphony_ID': 'ANALYSIS_INFO',
            'Earliest_Date_Available': 'Common_Range',
            **{name: pd.to_datetime(all_dates[-1]).strftime('%Y-%m-%d') for name in symphony_names},
            'Mean_Returns_Corr': f"To: {pd.to_datetime(all_dates[-1]).strftime('%Y-%m-%d')}"
        }])
        analysis_info3 = pd.DataFrame([{
            'Symphony_ID': 'ANALYSIS_INFO',
            'Earliest_Date_Available': 'Days',
            **{name: len(all_dates) for name in symphony_names},
            'Mean_Returns_Corr': f"{len(all_dates)} days"
        }])
        
        df_returns = pd.concat([df_returns, analysis_info, analysis_info2, analysis_info3], ignore_index=False)
        df_drawdown = pd.concat([df_drawdown, analysis_info, analysis_info2, analysis_info3], ignore_index=False)
        
        # Export to Excel with multiple sheets
        excel_file = os.path.join(output_dir, "correlation_matrix.xlsx")
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            df_returns.to_excel(writer, sheet_name='Returns_Correlation')
            df_drawdown.to_excel(writer, sheet_name='Drawdown_Correlation')
        
        dual_print(f"✓ Correlation matrices Excel saved to: {excel_file}")
        dual_print(f"  Sheets: Summary, Returns_Correlation, Drawdown_Correlation")
        
        # Also save summary as CSV
        csv_file = os.path.join(output_dir, "correlation_matrix.csv")
        df_summary.to_csv(csv_file, index=False)
        dual_print(f"✓ Summary CSV saved to: {csv_file}")
        
        # Save returns correlation heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=[name[:20] for name in symphony_names],
            yticklabels=[name[:20] for name in symphony_names],
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )
        ax.set_title('Symphony Returns Correlations', fontsize=16, fontweight='bold', pad=20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        
        heatmap_file = os.path.join(output_dir, "correlation_heatmap_returns.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        dual_print(f"✓ Returns correlation heatmap saved to: {heatmap_file}")
        
        # Save drawdown correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            drawdown_correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=[name[:20] for name in symphony_names],
            yticklabels=[name[:20] for name in symphony_names],
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )
        ax.set_title('Symphony Drawdown Correlations', fontsize=16, fontweight='bold', pad=20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        
        dd_heatmap_file = os.path.join(output_dir, "correlation_heatmap_drawdowns.png")
        plt.savefig(dd_heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        dual_print(f"✓ Drawdown correlation heatmap saved to: {dd_heatmap_file}")
        
        dual_print("\n" + "="*80)
        dual_print("CORRELATION ANALYSIS COMPLETE!")
        dual_print("="*80)
        
        return  # Exit without running Monte Carlo
    
    # Continue with full Monte Carlo analysis
    # Initialize simulator
    print("\n" + "="*80)
    print("INITIALIZING PORTFOLIO SIMULATOR")
    print("="*80)
    
    initial_capital = float(input("\nInitial portfolio capital (default: 100000): ") or "100000")
    
    simulator = PortfolioMonteCarloSimulator(
        symphony_data=symphony_data,
        initial_capital=initial_capital
    )
    
    # Choose weighting method
    print("\n" + "="*80)
    print("PORTFOLIO WEIGHTING")
    print("="*80)
    print("\nChoose weighting method:")
    print("1. Equal weight (1/N for each symphony)")
    print("2. Risk parity (inverse volatility)")
    print("3. Sharpe ratio weighted")
    print("4. Sortino ratio weighted")
    print("5. Custom weights")
    
    weight_choice = input("\nEnter choice (1-5, default: 1): ") or "1"
    
    weight_methods = {'1': 'equal', '2': 'risk_parity', '3': 'max_sharpe', '4': 'max_sortino', '5': 'custom'}
    method = weight_methods.get(weight_choice, 'equal')
    
    if method == 'custom':
        print("\nEnter custom weights for each symphony (must sum to 1.0):")
        custom_weights = {}
        for name in simulator.symphony_names:
            while True:
                try:
                    weight = float(input(f"{name}: "))
                    if 0 <= weight <= 1:
                        custom_weights[name] = weight
                        break
                    print("Weight must be between 0 and 1")
                except ValueError:
                    print("Please enter a valid number")
        
        total = sum(custom_weights.values())
        if abs(total - 1.0) > 0.01:
            print(f"\nWarning: Weights sum to {total:.4f}, normalizing to 1.0")
            custom_weights = {k: v/total for k, v in custom_weights.items()}
        
        simulator.set_weights(weights=custom_weights, method='custom')
    else:
        simulator.set_weights(method=method)
    
    # Set current holdings for concentration risk analysis
    if current_holdings:
        simulator.set_current_holdings(current_holdings)
    
    # Monte Carlo parameters
    print("\n" + "="*80)
    print("MONTE CARLO SIMULATION PARAMETERS")
    print("="*80)
    
    n_sims = int(input("\nNumber of simulations (default: 10000): ") or "10000")
    n_days = int(input("Days to simulate forward (default: 252 = 1 year): ") or "252")
    
    rebal_input = input("Rebalancing frequency in days (blank for buy-and-hold): ")
    rebal_freq = int(rebal_input) if rebal_input else None
    
    # Run simulation
    results = simulator.run_monte_carlo(
        n_simulations=n_sims,
        n_days_forward=n_days,
        rebalance_frequency=rebal_freq
    )
    
    # Generate outputs
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80)
    
    output_dir = input("\nOutput directory (default: ./portfolio_results): ") or "./portfolio_results"
    
    simulator.plot_results(output_dir)
    simulator.export_results(output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nFiles created:")
    print(f"  - Portfolio Simulation - Summary.txt (complete text output)")
    print(f"  - portfolio_monte_carlo_analysis.png (comprehensive dashboard)")
    print(f"  - portfolio_fan_chart_detailed.png (detailed projection chart)")
    print(f"  - monte_carlo_results.csv (detailed simulation data)")
    print(f"  - summary_statistics.json (portfolio statistics)")


if __name__ == "__main__":
    main()
