# Multi-Symphony Portfolio Monte Carlo Simulator
# Extended from single-symphony framework for portfolio-level analysis
# Handles 9-11 Composer symphonies with correlation modeling
# prairie@Investor's Collaborative - Extended by Claude

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.linalg import cholesky
from datetime import datetime, timedelta
import os
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple
import json
import sys

# Set random seed for reproducibility
np.random.seed(42)


class DualOutput:
    """Captures output to both console and a string buffer for saving to file."""
    
    def __init__(self):
        self.buffer = []
        self.enabled = False
    
    def enable(self):
        """Start capturing output."""
        self.enabled = True
        self.buffer = []
    
    def write(self, text):
        """Write text to both console and buffer."""
        print(text, end='')
        if self.enabled:
            self.buffer.append(text)
    
    def get_summary(self):
        """Get the captured output as a single string."""
        return ''.join(self.buffer)
    
    def save_to_file(self, filepath):
        """Save captured output to a text file."""
        with open(filepath, 'w') as f:
            f.write(self.get_summary())


# Global dual output instance
_dual_output = DualOutput()


def dual_print(*args, **kwargs):
    """Print to both console and capture buffer."""
    # Convert all arguments to strings and join them
    text = ' '.join(str(arg) for arg in args)
    end = kwargs.get('end', '\n')
    text += end
    print(text, end='')  # Print to console
    if _dual_output.enabled:
        _dual_output.buffer.append(text)


class PortfolioMonteCarloSimulator:
    """
    Monte Carlo simulator for a portfolio of Composer symphonies.
    Handles correlation modeling, position sizing, and portfolio-level metrics.
    """
    
    def __init__(self, symphony_data: Dict[str, pd.DataFrame], initial_capital: float = 100000):
        """
        Initialize the portfolio simulator.
        
        Args:
            symphony_data: Dict mapping symphony names to DataFrames with 'date' and 'returns' columns
            initial_capital: Starting portfolio value
        """
        self.symphony_data = symphony_data
        self.initial_capital = initial_capital
        self.symphony_names = list(symphony_data.keys())
        self.n_symphonies = len(self.symphony_names)
        
        # Validate input
        if self.n_symphonies < 2:
            raise ValueError("Need at least 2 symphonies for portfolio analysis")
        
        # Enable dual output capture for saving to summary file
        _dual_output.enable()
        
        dual_print(f"\nInitialized portfolio with {self.n_symphonies} symphonies:")
        for name in self.symphony_names:
            n_days = len(symphony_data[name])
            start_date = symphony_data[name]['date'].iloc[0]
            end_date = symphony_data[name]['date'].iloc[-1]
            self._dual_print(f"  - {name}: {n_days} days ({start_date} to {end_date})")
        
        # Align all data to common date range
        self._align_data()
        
        # Calculate statistics
        self._calculate_statistics()
    
    def _dual_print(self, text: str = ""):
        """Print to console and capture to text buffer."""
        dual_print(text)
    
    def _align_data(self):
        """Align all symphony data to common date range."""
        # Find common date range
        all_dates = [set(df['date']) for df in self.symphony_data.values()]
        common_dates = set.intersection(*all_dates)
        
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates found across symphonies")
        
        common_dates = sorted(list(common_dates))
        
        self._dual_print(f"\nAligned to common date range: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} days)")
        
        # Filter each symphony to common dates and sort
        aligned_data = {}
        for name, df in self.symphony_data.items():
            aligned_df = df[df['date'].isin(common_dates)].copy()
            aligned_df = aligned_df.sort_values('date').reset_index(drop=True)
            aligned_data[name] = aligned_df
        
        self.aligned_data = aligned_data
        self.dates = aligned_data[self.symphony_names[0]]['date'].values
        self.n_days = len(self.dates)
        
        # Create returns matrix (n_days x n_symphonies)
        self.returns_matrix = np.column_stack([
            aligned_data[name]['returns'].values for name in self.symphony_names
        ])
    
    def _calculate_statistics(self):
        """Calculate statistical parameters for each symphony and the correlation matrix."""
        self.stats = {}
        
        for i, name in enumerate(self.symphony_names):
            returns = self.returns_matrix[:, i]
            
            # Basic statistics
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            sharpe = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            
            # Drawdown analysis
            cumulative = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            self.stats[name] = {
                'mean_daily': mean_return,
                'std_daily': std_return,
                'mean_annual': mean_return * 252,
                'std_annual': std_return * np.sqrt(252),
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'total_return': cumulative[-1] - 1
            }
        
        # Calculate correlation matrix
        self.correlation_matrix = np.corrcoef(self.returns_matrix.T)
        
        # Print summary
        self._dual_print("\n" + "="*80)
        self._dual_print("SYMPHONY STATISTICS")
        self._dual_print("="*80)
        self._dual_print(f"{'Symphony':<40} {'Annual %':<12} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<8}")
        self._dual_print("-"*80)
        for name in self.symphony_names:
            s = self.stats[name]
            self._dual_print(f"{name:<40} {s['mean_annual']*100:>10.2f}% {s['std_annual']*100:>10.2f}% "
                  f"{s['sharpe']:>7.2f} {s['max_drawdown']*100:>7.1f}%")
        
        self._dual_print("\n" + "="*80)
        self._dual_print("CORRELATION MATRIX")
        self._dual_print("="*80)
        self._print_correlation_matrix()
    
    def _print_correlation_matrix(self):
        """Pretty print the correlation matrix."""
        # Abbreviate names for display
        abbrev_names = [name[:20] for name in self.symphony_names]
        
        # Print header
        header = f"{'':25}"
        for name in abbrev_names:
            header += f"{name[:8]:>10}"
        self._dual_print(header)
        
        # Print rows
        for i, name in enumerate(abbrev_names):
            row = f"{name:<25}"
            for j in range(self.n_symphonies):
                corr = self.correlation_matrix[i, j]
                # Color code correlations
                if i == j:
                    row += f"{'1.00':>10}"
                else:
                    row += f"{corr:>10.3f}"
            self._dual_print(row)
    
    def set_weights(self, weights: Dict[str, float] = None, method: str = 'equal'):
        """
        Set portfolio weights for each symphony.
        
        Args:
            weights: Dict mapping symphony names to weights (must sum to 1.0)
            method: 'equal', 'risk_parity', 'max_sharpe', or 'custom'
        """
        if method == 'custom' and weights is None:
            raise ValueError("Must provide weights dict when using 'custom' method")
        
        if method == 'equal':
            weight = 1.0 / self.n_symphonies
            self.weights = {name: weight for name in self.symphony_names}
        
        elif method == 'risk_parity':
            # Weight inversely by volatility
            vols = np.array([self.stats[name]['std_annual'] for name in self.symphony_names])
            inv_vols = 1.0 / vols
            weights_array = inv_vols / inv_vols.sum()
            self.weights = {name: weights_array[i] for i, name in enumerate(self.symphony_names)}
        
        elif method == 'max_sharpe':
            # Weight by Sharpe ratio
            sharpes = np.array([max(self.stats[name]['sharpe'], 0) for name in self.symphony_names])
            if sharpes.sum() > 0:
                weights_array = sharpes / sharpes.sum()
            else:
                weights_array = np.ones(self.n_symphonies) / self.n_symphonies
            self.weights = {name: weights_array[i] for i, name in enumerate(self.symphony_names)}
        
        elif method == 'custom':
            # Validate weights
            if set(weights.keys()) != set(self.symphony_names):
                raise ValueError("Weights must include all symphony names")
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"Weights must sum to 1.0, got {total}")
            self.weights = weights
        
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        # Convert to array for calculations
        self.weights_array = np.array([self.weights[name] for name in self.symphony_names])
        
        # Calculate portfolio statistics with current weights
        self._calculate_portfolio_stats()
        
        self._dual_print(f"\n{'='*60}")
        self._dual_print(f"PORTFOLIO WEIGHTS ({method})")
        self._dual_print(f"{'='*60}")
        for name in self.symphony_names:
            self._dual_print(f"{name:<40} {self.weights[name]*100:>6.2f}%")
        self._dual_print(f"\n Portfolio Expected Annual Return: {self.portfolio_expected_return*100:.2f}%")
        self._dual_print(f" Portfolio Annual Volatility: {self.portfolio_volatility*100:.2f}%")
        self._dual_print(f" Portfolio Sharpe Ratio: {self.portfolio_sharpe:.2f}")
    
    def _calculate_portfolio_stats(self):
        """Calculate expected portfolio-level statistics."""
        # Portfolio expected return (weighted average)
        self.portfolio_expected_return = np.sum(
            self.weights_array * np.array([self.stats[name]['mean_annual'] for name in self.symphony_names])
        )
        
        # Portfolio variance (accounting for correlations)
        annual_stds = np.array([self.stats[name]['std_annual'] for name in self.symphony_names])
        
        # Variance = w^T * Cov * w
        # Cov = std * corr * std^T (element-wise for each pair)
        cov_matrix = np.outer(annual_stds, annual_stds) * self.correlation_matrix
        portfolio_variance = self.weights_array @ cov_matrix @ self.weights_array
        
        self.portfolio_volatility = np.sqrt(portfolio_variance)
        self.portfolio_sharpe = self.portfolio_expected_return / self.portfolio_volatility if self.portfolio_volatility > 0 else 0
    
    def set_current_holdings(self, holdings_by_symphony: Dict[str, Dict[str, float]]):
        """
        Set current holdings for concentration risk analysis.
        
        Args:
            holdings_by_symphony: Dict mapping symphony names to dicts of {ticker: allocation_pct}
                                 Example: {'Symphony_1': {'SPY': 0.40, 'QQQ': 0.30, 'TLT': 0.30}}
        """
        self.current_holdings = holdings_by_symphony
        self._analyze_concentration_risk()
    
    def _analyze_concentration_risk(self):
        """Analyze concentration risk from current holdings across all symphonies."""
        if not hasattr(self, 'current_holdings') or not self.current_holdings:
            self.concentration_analysis = None
            return
        
        # Aggregate holdings across portfolio
        total_exposure = {}
        
        for i, symphony_name in enumerate(self.symphony_names):
            if symphony_name not in self.current_holdings:
                continue
            
            symphony_weight = self.weights_array[i]
            holdings = self.current_holdings[symphony_name]
            
            for ticker, allocation in holdings.items():
                if ticker == '$USD':  # Skip cash
                    continue
                total_exposure[ticker] = total_exposure.get(ticker, 0.0) + (allocation * symphony_weight)
        
        # Sort by exposure
        sorted_exposure = sorted(total_exposure.items(), key=lambda x: x[1], reverse=True)
        
        # Store results
        self.concentration_analysis = {
            'total_exposure': dict(sorted_exposure),
            'top_10': sorted_exposure[:10],
            'concentrated_assets': [(ticker, exp) for ticker, exp in sorted_exposure if exp > 0.15],  # >15% threshold
            'total_assets': len(total_exposure)
        }
        
        # Print concentration report
        self._print_concentration_report()
    
    def _print_concentration_report(self):
        """Print concentration risk analysis."""
        if not self.concentration_analysis:
            return
        
        self._dual_print("\n" + "="*80)
        self._dual_print("CONCENTRATION RISK ANALYSIS")
        self._dual_print("="*80)
        self._dual_print(f"Total unique assets across portfolio: {self.concentration_analysis['total_assets']}")
        
        # Flag high concentrations
        if self.concentration_analysis['concentrated_assets']:
            self._dual_print("\n⚠️  HIGH CONCENTRATION ALERTS (>15% of portfolio):")
            for ticker, exposure in self.concentration_analysis['concentrated_assets']:
                self._dual_print(f"  {ticker}: {exposure*100:.1f}%")
        else:
            self._dual_print("\n✓ No individual asset exceeds 15% concentration threshold")
        
        # Top 10 exposures
        self._dual_print("\nTop 10 Asset Exposures:")
        self._dual_print(f"{'Asset':<10} {'Exposure':<12} {'Bar':<40}")
        self._dual_print("-" * 65)
        
        for ticker, exposure in self.concentration_analysis['top_10']:
            bar_length = int(exposure * 100)  # Scale to 100 chars max
            bar = '█' * bar_length
            self._dual_print(f"{ticker:<10} {exposure*100:>10.1f}% {bar}")
        
        # Summary statistics
        top_5_exposure = sum([exp for _, exp in self.concentration_analysis['top_10'][:5]])
        self._dual_print(f"\nTop 5 assets represent: {top_5_exposure*100:.1f}% of portfolio")
        
        # Which symphonies hold what
        self._dual_print("\nHoldings by Symphony:")
        for symphony_name in self.symphony_names:
            if symphony_name not in self.current_holdings:
                continue
            holdings = self.current_holdings[symphony_name]
            non_cash = {k: v for k, v in holdings.items() if k != '$USD'}
            if non_cash:
                top_3 = sorted(non_cash.items(), key=lambda x: x[1], reverse=True)[:3]
                holdings_str = ", ".join([f"{t}: {v*100:.0f}%" for t, v in top_3])
                self._dual_print(f"  {symphony_name[:40]:<40} → {holdings_str}")
    
    def run_monte_carlo(self, 
                       n_simulations: int = 10000,
                       n_days_forward: int = 252,
                       rebalance_frequency: int = None) -> pd.DataFrame:
        """
        Run correlated Monte Carlo simulations for the portfolio.
        
        Args:
            n_simulations: Number of simulation paths
            n_days_forward: Number of days to simulate forward
            rebalance_frequency: Days between rebalancing (None = buy and hold)
        
        Returns:
            DataFrame with simulation results
        """
        self._dual_print(f"\n{'='*60}")
        self._dual_print(f"RUNNING MONTE CARLO SIMULATION")
        self._dual_print(f"{'='*60}")
        self._dual_print(f"Simulations: {n_simulations:,}")
        self._dual_print(f"Days forward: {n_days_forward}")
        self._dual_print(f"Rebalancing: {'Every ' + str(rebalance_frequency) + ' days' if rebalance_frequency else 'Buy and hold'}")
        
        # Get mean and std for each symphony
        means = np.array([self.stats[name]['mean_daily'] for name in self.symphony_names])
        stds = np.array([self.stats[name]['std_daily'] for name in self.symphony_names])
        
        # Generate correlated random returns using Cholesky decomposition
        self._dual_print("\nGenerating correlated random returns...")
        
        # Cholesky decomposition of correlation matrix
        try:
            L = cholesky(self.correlation_matrix, lower=True)
        except np.linalg.LinAlgError:
            self._dual_print("Warning: Correlation matrix not positive definite, using nearest positive definite matrix")
            L = self._nearest_positive_definite_cholesky(self.correlation_matrix)
        
        # Initialize arrays to store results
        portfolio_values = np.zeros((n_simulations, n_days_forward + 1))
        portfolio_values[:, 0] = self.initial_capital
        
        symphony_values = np.zeros((n_simulations, n_days_forward + 1, self.n_symphonies))
        for i in range(self.n_symphonies):
            symphony_values[:, 0, i] = self.initial_capital * self.weights_array[i]
        
        # Run simulations
        for sim in range(n_simulations):
            if (sim + 1) % 1000 == 0:
                self._dual_print(f"  Completed {sim + 1:,} / {n_simulations:,} simulations...")
            
            for day in range(n_days_forward):
                # Generate correlated normal random variables
                z = np.random.standard_normal(self.n_symphonies)
                correlated_z = L @ z
                
                # Generate returns for each symphony
                returns = means + stds * correlated_z
                
                # Update symphony values
                symphony_values[sim, day + 1, :] = symphony_values[sim, day, :] * (1 + returns)
                
                # Check if rebalancing day
                if rebalance_frequency and (day + 1) % rebalance_frequency == 0:
                    # Rebalance to target weights
                    total_value = symphony_values[sim, day + 1, :].sum()
                    symphony_values[sim, day + 1, :] = total_value * self.weights_array
                
                # Calculate portfolio value
                portfolio_values[sim, day + 1] = symphony_values[sim, day + 1, :].sum()
        
        self._dual_print(f"Completed all {n_simulations:,} simulations!\n")
        
        # Store results
        self.portfolio_values = portfolio_values
        self.symphony_values = symphony_values
        self.n_simulations = n_simulations
        self.n_days_forward = n_days_forward
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = (portfolio_values[:, -1] - self.initial_capital) / self.initial_capital
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'simulation': range(n_simulations),
            'final_value': portfolio_values[:, -1],
            'total_return': portfolio_returns,
            'annual_return': (1 + portfolio_returns) ** (252 / n_days_forward) - 1
        })
        
        # Calculate max drawdown for each simulation
        max_drawdowns = []
        for sim in range(n_simulations):
            cumulative = portfolio_values[sim, :]
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdowns.append(np.min(drawdown))
        
        results_df['max_drawdown'] = max_drawdowns
        
        self.results_df = results_df
        
        # Print summary statistics
        self._print_monte_carlo_summary()
        
        return results_df
    
    def _nearest_positive_definite_cholesky(self, A):
        """Find nearest positive definite matrix and return its Cholesky decomposition."""
        # Symmetrize
        A_sym = (A + A.T) / 2
        
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(A_sym)
        
        # Clip negative eigenvalues to small positive value
        eigvals[eigvals < 1e-10] = 1e-10
        
        # Reconstruct matrix
        A_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Return Cholesky decomposition
        return cholesky(A_pd, lower=True)
    
    def _print_monte_carlo_summary(self):
        """Print summary statistics from Monte Carlo simulation."""
        self._dual_print(f"{'='*60}")
        self._dual_print(f"MONTE CARLO RESULTS SUMMARY")
        self._dual_print(f"{'='*60}")
        
        returns = self.results_df['total_return']
        annual_returns = self.results_df['annual_return']
        drawdowns = self.results_df['max_drawdown']
        
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        
        self._dual_print("\nTotal Return Distribution:")
        self._dual_print(f"{'Percentile':<12} {'Total Return':<15} {'Annualized':<15}")
        self._dual_print("-" * 45)
        for p in percentiles:
            total_ret = np.percentile(returns, p)
            annual_ret = np.percentile(annual_returns, p)
            self._dual_print(f"{p}th{'':<9} {total_ret*100:>13.2f}% {annual_ret*100:>13.2f}%")
        
        self._dual_print(f"\nMean: {returns.mean()*100:>27.2f}% {annual_returns.mean()*100:>13.2f}%")
        self._dual_print(f"Std Dev: {returns.std()*100:>23.2f}% {annual_returns.std()*100:>13.2f}%")
        
        self._dual_print("\nMaximum Drawdown Distribution:")
        self._dual_print(f"{'Percentile':<12} {'Max Drawdown':<15}")
        self._dual_print("-" * 30)
        for p in percentiles:
            dd = np.percentile(drawdowns, p)
            self._dual_print(f"{p}th{'':<9} {dd*100:>13.2f}%")
        
        self._dual_print(f"\nMean: {drawdowns.mean()*100:>27.2f}%")
        
        # Probability of positive returns
        prob_positive = (returns > 0).sum() / len(returns) * 100
        self._dual_print(f"\nProbability of positive return: {prob_positive:.1f}%")
        
        # Probability of beating various benchmarks (if desired)
        benchmarks = [0.05, 0.10, 0.15]  # 5%, 10%, 15% annual
        for benchmark in benchmarks:
            total_benchmark = (1 + benchmark) ** (self.n_days_forward / 252) - 1
            prob_beat = (returns > total_benchmark).sum() / len(returns) * 100
            self._dual_print(f"Probability of beating {benchmark*100:.0f}% annual: {prob_beat:.1f}%")
    
    def plot_results(self, output_dir: str, percentiles: List[int] = [10, 25, 50, 75, 90]):
        """
        Generate comprehensive visualization of Monte Carlo results.
        
        Args:
            output_dir: Directory to save plots
            percentiles: Which percentiles to show in fan chart
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self._dual_print(f"\n{'='*60}")
        self._dual_print(f"GENERATING VISUALIZATIONS")
        self._dual_print(f"{'='*60}")
        
        # Create comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Fan chart showing portfolio value percentiles over time
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_fan_chart(ax1, percentiles)
        
        # 2. Distribution of final returns
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_return_distribution(ax2)
        
        # 3. Distribution of max drawdowns
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_drawdown_distribution(ax3)
        
        # 4. Scatter: Return vs Drawdown
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_return_vs_drawdown(ax4)
        
        # 5. Symphony contribution analysis (average across simulations)
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_symphony_contributions(ax5)
        
        # 6. Correlation heatmap
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_correlation_heatmap(ax6)
        
        # 7. Risk decomposition
        ax7 = fig.add_subplot(gs[2, 2])
        
        # If we have concentration analysis, show it; otherwise show risk decomposition
        if hasattr(self, 'concentration_analysis') and self.concentration_analysis:
            self._plot_concentration_risk(ax7)
        else:
            self._plot_risk_decomposition(ax7)
        
        plt.suptitle(f"Portfolio Monte Carlo Analysis - {self.n_simulations:,} Simulations", 
                    fontsize=16, fontweight='bold')
        
        # Save figure
        filepath = os.path.join(output_dir, "portfolio_monte_carlo_analysis.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        dual_print(f"Saved comprehensive analysis to: {filepath}")
        plt.close()
        
        # Create separate detailed fan chart
        self._create_detailed_fan_chart(output_dir, percentiles)
    
    def _plot_fan_chart(self, ax, percentiles):
        """Plot fan chart of portfolio value over time."""
        days = np.arange(self.n_days_forward + 1)
        
        # Calculate percentiles at each day
        percentile_values = {}
        for p in percentiles:
            percentile_values[p] = np.percentile(self.portfolio_values, p, axis=0)
        
        # Plot median
        ax.plot(days, percentile_values[50], 'b-', linewidth=2, label='Median (50th)')
        
        # Plot percentile bands
        colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(percentiles)//2))
        for i, (p_low, p_high) in enumerate([(10, 90), (25, 75)]):
            ax.fill_between(days, percentile_values[p_low], percentile_values[p_high],
                          alpha=0.3, color=colors[i], 
                          label=f'{p_low}th-{p_high}th percentile')
        
        # Plot initial capital line
        ax.axhline(y=self.initial_capital, color='k', linestyle='--', 
                  alpha=0.5, label='Initial Capital')
        
        ax.set_xlabel('Days Forward')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Portfolio Value Projection (Fan Chart)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    def _plot_return_distribution(self, ax):
        """Plot histogram of final returns."""
        returns = self.results_df['annual_return'] * 100
        
        ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(returns.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {returns.median():.1f}%')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Annualized Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Annualized Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_drawdown_distribution(self, ax):
        """Plot histogram of maximum drawdowns."""
        drawdowns = self.results_df['max_drawdown'] * 100
        
        ax.hist(drawdowns, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax.axvline(drawdowns.median(), color='darkred', linestyle='--', linewidth=2, 
                  label=f'Median: {drawdowns.median():.1f}%')
        
        ax.set_xlabel('Maximum Drawdown (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Maximum Drawdowns')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_return_vs_drawdown(self, ax):
        """Scatter plot of returns vs drawdowns."""
        returns = self.results_df['annual_return'] * 100
        drawdowns = self.results_df['max_drawdown'] * 100
        
        ax.scatter(drawdowns, returns, alpha=0.3, s=10)
        ax.set_xlabel('Maximum Drawdown (%)')
        ax.set_ylabel('Annualized Return (%)')
        ax.set_title('Return vs Risk Tradeoff')
        ax.grid(True, alpha=0.3)
        
        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    def _plot_symphony_contributions(self, ax):
        """Plot average contribution of each symphony to portfolio value."""
        # Calculate average final value contribution for each symphony
        final_values = self.symphony_values[:, -1, :]
        avg_contributions = final_values.mean(axis=0)
        
        # Create bar chart
        x_pos = np.arange(self.n_symphonies)
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_symphonies))
        
        ax.bar(x_pos, avg_contributions, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Symphony')
        ax.set_ylabel('Average Final Value ($)')
        ax.set_title('Average Symphony Contributions')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name[:15] for name in self.symphony_names], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    def _plot_correlation_heatmap(self, ax):
        """Plot correlation matrix heatmap."""
        sns.heatmap(self.correlation_matrix, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdBu_r',
                   center=0,
                   vmin=-1, 
                   vmax=1,
                   square=True,
                   xticklabels=[name[:15] for name in self.symphony_names],
                   yticklabels=[name[:15] for name in self.symphony_names],
                   cbar_kws={'label': 'Correlation'},
                   ax=ax)
        
        ax.set_title('Symphony Return Correlations')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
    
    def _plot_risk_decomposition(self, ax):
        """Plot risk contribution by symphony."""
        # Calculate marginal contribution to risk for each symphony
        # MCR = (weight * cov_matrix * weights) / portfolio_volatility
        
        annual_stds = np.array([self.stats[name]['std_annual'] for name in self.symphony_names])
        cov_matrix = np.outer(annual_stds, annual_stds) * self.correlation_matrix
        
        marginal_contrib = (cov_matrix @ self.weights_array) / self.portfolio_volatility
        risk_contrib = self.weights_array * marginal_contrib
        risk_contrib_pct = risk_contrib / risk_contrib.sum() * 100
        
        # Create bar chart
        x_pos = np.arange(self.n_symphonies)
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, self.n_symphonies))
        
        ax.bar(x_pos, risk_contrib_pct, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Symphony')
        ax.set_ylabel('Risk Contribution (%)')
        ax.set_title('Portfolio Risk Decomposition')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name[:15] for name in self.symphony_names], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add target line (equal risk contribution)
        target = 100 / self.n_symphonies
        ax.axhline(y=target, color='black', linestyle='--', alpha=0.5, 
                  label=f'Equal risk: {target:.1f}%')
        ax.legend()
    
    def _plot_concentration_risk(self, ax):
        """Plot concentration risk from current holdings."""
        if not self.concentration_analysis:
            ax.text(0.5, 0.5, 'No concentration data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get top 10 exposures
        top_exposures = self.concentration_analysis['top_10']
        
        if not top_exposures:
            ax.text(0.5, 0.5, 'No asset holdings data', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        tickers = [t for t, _ in top_exposures]
        exposures = [e * 100 for _, e in top_exposures]  # Convert to percentage
        
        # Color code by concentration level
        colors = []
        for exp in exposures:
            if exp > 20:
                colors.append('#d62728')  # Red for high concentration (>20%)
            elif exp > 15:
                colors.append('#ff7f0e')  # Orange for moderate (15-20%)
            elif exp > 10:
                colors.append('#ffbb33')  # Yellow for elevated (10-15%)
            else:
                colors.append('#2ca02c')  # Green for low (<10%)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(tickers))
        ax.barh(y_pos, exposures, color=colors, alpha=0.7, edgecolor='black')
        
        # Add concentration threshold lines
        ax.axvline(x=15, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='15% Alert')
        ax.axvline(x=20, color='red', linestyle='--', alpha=0.5, linewidth=1, label='20% High Risk')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tickers)
        ax.set_xlabel('Portfolio Exposure (%)')
        ax.set_title('Top 10 Asset Concentration Risk', fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels on bars
        for i, (ticker, exp) in enumerate(zip(tickers, exposures)):
            ax.text(exp + 0.5, i, f'{exp:.1f}%', va='center', fontsize=8)
    
    def _create_detailed_fan_chart(self, output_dir, percentiles):
        """Create a detailed standalone fan chart."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        days = np.arange(self.n_days_forward + 1)
        
        # Calculate percentiles at each day
        percentile_values = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percentile_values[p] = np.percentile(self.portfolio_values, p, axis=0)
        
        # Plot median
        ax.plot(days, percentile_values[50], 'b-', linewidth=3, label='Median (50th)', zorder=10)
        
        # Plot percentile bands with gradient
        bands = [(1, 99), (5, 95), (10, 90), (25, 75)]
        colors = plt.cm.Blues(np.linspace(0.2, 0.6, len(bands)))
        
        for i, (p_low, p_high) in enumerate(bands):
            ax.fill_between(days, percentile_values[p_low], percentile_values[p_high],
                          alpha=0.25, color=colors[i], 
                          label=f'{p_low}th-{p_high}th percentile')
        
        # Plot some extreme paths for context
        n_sample_paths = 50
        sample_indices = np.random.choice(self.n_simulations, n_sample_paths, replace=False)
        for idx in sample_indices:
            ax.plot(days, self.portfolio_values[idx, :], 'gray', alpha=0.05, linewidth=0.5)
        
        # Plot initial capital line
        ax.axhline(y=self.initial_capital, color='k', linestyle='--', 
                  linewidth=2, alpha=0.7, label='Initial Capital')
        
        ax.set_xlabel('Days Forward', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title(f'Portfolio Monte Carlo Projection - {self.n_simulations:,} Simulations', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Save
        filepath = os.path.join(output_dir, "portfolio_fan_chart_detailed.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        dual_print(f"Saved detailed fan chart to: {filepath}")
        plt.close()
    
    def export_results(self, output_dir: str, filename: str = "monte_carlo_results.csv"):
        """Export detailed results to CSV and text summary."""
        filepath = os.path.join(output_dir, filename)
        self.results_df.to_csv(filepath, index=False)
        dual_print(f"\nExported results to: {filepath}")
        
        # Also export summary statistics
        summary_filepath = os.path.join(output_dir, "summary_statistics.json")
        summary = {
            'portfolio': {
                'expected_annual_return': float(self.portfolio_expected_return),
                'annual_volatility': float(self.portfolio_volatility),
                'sharpe_ratio': float(self.portfolio_sharpe)
            },
            'symphonies': {},
            'weights': self.weights,
            'simulation_parameters': {
                'n_simulations': self.n_simulations,
                'n_days_forward': self.n_days_forward,
                'initial_capital': self.initial_capital
            }
        }
        
        for name in self.symphony_names:
            summary['symphonies'][name] = {
                'mean_annual': float(self.stats[name]['mean_annual']),
                'std_annual': float(self.stats[name]['std_annual']),
                'sharpe': float(self.stats[name]['sharpe']),
                'max_drawdown': float(self.stats[name]['max_drawdown'])
            }
        
        # Add concentration analysis if available
        if hasattr(self, 'concentration_analysis') and self.concentration_analysis:
            summary['concentration_risk'] = {
                'total_assets': self.concentration_analysis['total_assets'],
                'top_10_exposures': [
                    {'ticker': ticker, 'exposure_pct': float(exp * 100)}
                    for ticker, exp in self.concentration_analysis['top_10']
                ],
                'high_concentration_alerts': [
                    {'ticker': ticker, 'exposure_pct': float(exp * 100)}
                    for ticker, exp in self.concentration_analysis['concentrated_assets']
                ]
            }
        
        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        dual_print(f"Exported summary to: {summary_filepath}")
        
        # Export captured text output to summary file
        text_summary_filepath = os.path.join(output_dir, "Portfolio Simulation - Summary.txt")
        _dual_output.save_to_file(text_summary_filepath)
        dual_print(f"Exported text summary to: {text_summary_filepath}")


def load_composer_data(symphony_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load Composer backtest data from CSV files.
    
    Args:
        symphony_files: Dict mapping symphony names to CSV file paths
    
    Returns:
        Dict mapping symphony names to DataFrames with 'date' and 'returns' columns
    """
    symphony_data = {}
    
    for name, filepath in symphony_files.items():
        dual_print(f"Loading {name} from {filepath}...")
        
        # Try to read the CSV
        try:
            df = pd.read_csv(filepath)
            
            # Detect date and returns columns (flexible column naming)
            date_col = None
            returns_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'date' in col_lower or 'time' in col_lower:
                    date_col = col
                elif 'return' in col_lower or 'pct' in col_lower or 'change' in col_lower:
                    returns_col = col
            
            if date_col is None or returns_col is None:
                dual_print(f"Warning: Could not auto-detect columns for {name}")
                dual_print(f"Available columns: {df.columns.tolist()}")
                continue
            
            # Create standardized DataFrame
            clean_df = pd.DataFrame({
                'date': pd.to_datetime(df[date_col]),
                'returns': df[returns_col].astype(float)
            })
            
            # Remove any NaN values
            clean_df = clean_df.dropna()
            
            symphony_data[name] = clean_df
            dual_print(f"  Loaded {len(clean_df)} days of data")
            
        except Exception as e:
            dual_print(f"Error loading {name}: {e}")
            continue
    
    return symphony_data


def main():
    """Main function to run multi-symphony portfolio Monte Carlo simulation."""
    
    dual_print("="*80)
    dual_print("MULTI-SYMPHONY PORTFOLIO MONTE CARLO SIMULATOR")
    dual_print("="*80)
    
    # Example: Load symphony data
    # You would replace these with your actual Composer export files
    symphony_files = {
        'Symphony_1': '/path/to/symphony1_backtest.csv',
        'Symphony_2': '/path/to/symphony2_backtest.csv',
        'Symphony_3': '/path/to/symphony3_backtest.csv',
        # Add more symphonies...
    }
    
    dual_print("\nThis script requires CSV files with Composer backtest data.")
    dual_print("Each file should have columns for 'date' and 'returns' (or similar).\n")
    
    # Load data
    symphony_data = load_composer_data(symphony_files)
    
    if len(symphony_data) < 2:
        dual_print("\nError: Need at least 2 symphonies to run portfolio analysis.")
        dual_print("Please update symphony_files dict with your Composer backtest CSV files.")
        return
    
    # Create simulator
    simulator = PortfolioMonteCarloSimulator(
        symphony_data=symphony_data,
        initial_capital=100000
    )
    
    # Set portfolio weights
    dual_print("\nChoose weighting method:")
    dual_print("1. Equal weight")
    dual_print("2. Risk parity (inverse volatility)")
    dual_print("3. Sharpe ratio weighted")
    dual_print("4. Custom weights")
    
    weight_choice = input("Enter choice (1-4, default: 1): ") or "1"
    
    weight_methods = {'1': 'equal', '2': 'risk_parity', '3': 'max_sharpe', '4': 'custom'}
    method = weight_methods.get(weight_choice, 'equal')
    
    if method == 'custom':
        dual_print("\nEnter custom weights for each symphony (must sum to 1.0):")
        custom_weights = {}
        for name in simulator.symphony_names:
            weight = float(input(f"{name}: "))
            custom_weights[name] = weight
        simulator.set_weights(weights=custom_weights, method='custom')
    else:
        simulator.set_weights(method=method)
    
    # Run Monte Carlo simulation
    n_sims = int(input("\nNumber of simulations (default: 10000): ") or "10000")
    n_days = int(input("Days to simulate forward (default: 252): ") or "252")
    
    rebal_input = input("Rebalancing frequency in days (blank for buy-and-hold): ")
    rebal_freq = int(rebal_input) if rebal_input else None
    
    results = simulator.run_monte_carlo(
        n_simulations=n_sims,
        n_days_forward=n_days,
        rebalance_frequency=rebal_freq
    )
    
    # Generate visualizations
    output_dir = input("\nOutput directory for results (default: ./portfolio_results): ") or "./portfolio_results"
    simulator.plot_results(output_dir)
    simulator.export_results(output_dir)
    
    dual_print("\n" + "="*80)
    dual_print("ANALYSIS COMPLETE!")
    dual_print("="*80)
    dual_print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
