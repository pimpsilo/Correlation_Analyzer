#!/usr/bin/env python3
"""
Symphony Correlation Analyzer
Quickly analyze correlations between Composer symphonies without running Monte Carlo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import sys

# Import functions from composer_url_loader
from composer_url_loader import (
    parse_urls_from_text,
    load_multiple_symphonies_from_urls
)


def print_correlation_matrix(correlation_matrix, symphony_names, output_file=None):
    """Print correlation matrix in a nice formatted table."""
    
    # Prepare output
    output = []
    
    output.append("\n" + "="*100)
    output.append("SYMPHONY CORRELATION MATRIX (Returns)")
    output.append("="*100)
    
    # Header row
    header = f"{'Symphony':<40} | " + " ".join([f"{name[:8]:>8}" for name in symphony_names])
    output.append(header)
    output.append("-" * len(header))
    
    # Data rows
    for i, name in enumerate(symphony_names):
        row_data = " ".join([f"{correlation_matrix[i, j]:>8.3f}" for j in range(len(symphony_names))])
        row = f"{name[:40]:<40} | {row_data}"
        output.append(row)
    
    output.append("="*100)
    
    # Print to console
    for line in output:
        print(line)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output))
        print(f"\n✓ Correlation matrix saved to: {output_file}")
    
    return '\n'.join(output)


def export_correlation_csv(correlation_matrix, symphony_names, output_file):
    """Export correlation matrix as CSV."""
    df = pd.DataFrame(
        correlation_matrix,
        index=symphony_names,
        columns=symphony_names
    )
    df.to_csv(output_file)
    print(f"✓ Correlation matrix CSV saved to: {output_file}")


def plot_correlation_heatmap(correlation_matrix, symphony_names, output_file):
    """Create correlation heatmap visualization."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
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
    
    ax.set_title('Symphony Return Correlations', fontsize=16, fontweight='bold', pad=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Correlation heatmap saved to: {output_file}")
    plt.close()


def calculate_correlation_stats(correlation_matrix, symphony_names):
    """Calculate and display correlation statistics."""
    
    # Get upper triangle (excluding diagonal)
    upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
    
    print("\n" + "="*100)
    print("CORRELATION STATISTICS")
    print("="*100)
    print(f"Number of symphony pairs: {len(upper_triangle)}")
    print(f"Average correlation: {np.mean(upper_triangle):.3f}")
    print(f"Median correlation: {np.median(upper_triangle):.3f}")
    print(f"Min correlation: {np.min(upper_triangle):.3f}")
    print(f"Max correlation: {np.max(upper_triangle):.3f}")
    print(f"Std deviation: {np.std(upper_triangle):.3f}")
    
    # Find highest correlations
    print("\n" + "-"*100)
    print("HIGHEST CORRELATIONS (Top 5 pairs)")
    print("-"*100)
    
    pairs = []
    for i in range(len(symphony_names)):
        for j in range(i+1, len(symphony_names)):
            pairs.append((symphony_names[i], symphony_names[j], correlation_matrix[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, (name1, name2, corr) in enumerate(pairs[:5], 1):
        print(f"{i}. {name1[:35]:<35} ↔ {name2[:35]:<35} : {corr:>6.3f}")
    
    # Find lowest correlations
    print("\n" + "-"*100)
    print("LOWEST CORRELATIONS (Top 5 pairs - Best Diversifiers)")
    print("-"*100)
    
    for i, (name1, name2, corr) in enumerate(pairs[-5:][::-1], 1):
        print(f"{i}. {name1[:35]:<35} ↔ {name2[:35]:<35} : {corr:>6.3f}")
    
    print("="*100)


def main():
    """Main function for interactive correlation analysis."""
    
    print("\n" + "="*100)
    print("SYMPHONY CORRELATION ANALYZER")
    print("Quick correlation analysis without Monte Carlo simulation")
    print("="*100)
    
    # Load symphonies
    print("\nHow would you like to enter symphony URLs?")
    print("1. Paste all URLs at once (fastest)")
    print("2. Enter one at a time")
    
    choice = input("\nEnter choice (1 or 2, default: 1): ").strip() or "1"
    
    symphony_urls = {}
    
    if choice == "1":
        print("\nPaste all your Composer URLs below (one per line).")
        print("You can paste full URLs, short URLs, or just symphony IDs.")
        print("Lines starting with # are treated as comments and ignored.")
        print("\nExample URLs:")
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
            return
        
        symphony_urls = parse_urls_from_text('\n'.join(urls_text))
        
        if not symphony_urls:
            print("No valid URLs found in pasted text")
            return
        
        print(f"\nFound {len(symphony_urls)} valid Composer URLs")
    
    else:
        # One-by-one mode
        while True:
            try:
                n_symphonies = int(input("\nHow many symphonies? (2-15): "))
                if 2 <= n_symphonies <= 15:
                    break
                print("Please enter a number between 2 and 15")
            except ValueError:
                print("Please enter a valid number")
        
        print(f"\nEnter Composer URL for each symphony:")
        
        for i in range(n_symphonies):
            while True:
                url = input(f"Symphony {i+1} URL: ").strip()
                if url:
                    if not url.startswith('http'):
                        url = f"https://app.composer.trade/symphony/{url}/details"
                    symphony_urls[f"Symphony_{i+1}"] = url
                    break
                print("URL cannot be empty")
    
    # Get date range
    print("\nDate range for correlation analysis:")
    today = date.today().strftime('%Y-%m-%d')
    start_date = input(f"Start date (YYYY-MM-DD, blank for 2015-01-01): ") or '2015-01-01'
    end_date = input(f"End date (YYYY-MM-DD, blank for today {today}): ") or today
    
    # Load symphonies
    symphony_data, current_holdings = load_multiple_symphonies_from_urls(
        symphony_urls, 
        start_date, 
        end_date
    )
    
    if len(symphony_data) < 2:
        print("\nError: Need at least 2 successfully loaded symphonies")
        return
    
    # Calculate correlation matrix
    print("\n" + "="*100)
    print("CALCULATING CORRELATION MATRIX")
    print("="*100)
    
    symphony_names = list(symphony_data.keys())
    n_symphonies = len(symphony_names)
    
    # Align all symphonies to common date range
    all_dates = symphony_data[symphony_names[0]]['date'].values
    for name in symphony_names[1:]:
        all_dates = np.intersect1d(all_dates, symphony_data[name]['date'].values)
    
    print(f"Common date range: {len(all_dates)} days")
    
    # Build returns matrix
    returns_matrix = np.zeros((len(all_dates), n_symphonies))
    for i, name in enumerate(symphony_names):
        df = symphony_data[name]
        df = df[df['date'].isin(all_dates)].sort_values('date')
        returns_matrix[:, i] = df['returns'].values
    
    # Calculate correlation
    correlation_matrix = np.corrcoef(returns_matrix.T)
    
    # Display results
    print_correlation_matrix(correlation_matrix, symphony_names)
    calculate_correlation_stats(correlation_matrix, symphony_names)
    
    # Ask about exports
    print("\n" + "="*100)
    print("EXPORT OPTIONS")
    print("="*100)
    
    export_choice = input("\nExport results? (y/n, default: y): ").lower() or 'y'
    
    if export_choice == 'y':
        output_dir = input("Output directory (blank for current directory): ").strip() or '.'
        
        # Export text file
        text_file = f"{output_dir}/correlation_matrix.txt"
        print_correlation_matrix(correlation_matrix, symphony_names, text_file)
        
        # Export CSV
        csv_file = f"{output_dir}/correlation_matrix.csv"
        export_correlation_csv(correlation_matrix, symphony_names, csv_file)
        
        # Export heatmap
        heatmap_file = f"{output_dir}/correlation_heatmap.png"
        plot_correlation_heatmap(correlation_matrix, symphony_names, heatmap_file)
        
        print(f"\n✓ All exports saved to: {output_dir}/")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
