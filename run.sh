#!/usr/bin/env bash

# Symphony Correlation Analyzer - Runner Script
# Supports Streamlit Web UI, CLI interactive analyzer, and Monte Carlo simulator.

set -e

# Resolve script and project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Resolve Python & Streamlit binaries
resolve_binaries() {
    if command -v streamlit >/dev/null 2>&1; then
        STREAMLIT_BIN="$(command -v streamlit)"
        PYTHON_BIN="$(command -v python3 || command -v python)"
    elif [ -x "/Users/matthewhope/miniforge3/bin/streamlit" ]; then
        STREAMLIT_BIN="/Users/matthewhope/miniforge3/bin/streamlit"
        PYTHON_BIN="/Users/matthewhope/miniforge3/bin/python3"
    elif [ -x "$HOME/miniforge3/bin/streamlit" ]; then
        STREAMLIT_BIN="$HOME/miniforge3/bin/streamlit"
        PYTHON_BIN="$HOME/miniforge3/bin/python3"
    else
        echo "❌ Error: 'streamlit' was not found in PATH or Miniforge."
        echo "Please install requirements: pip install -r requirements.txt"
        exit 1
    fi
}

show_help() {
    cat << EOF
🎼 Symphony Correlation Analyzer Runner

Usage:
  $(basename "$0") [command|options]

Commands:
  ui, web             Launch the Streamlit Web Application (default)
  cli                 Launch the interactive CLI Correlation Analyzer
  mc, monte-carlo     Launch the Multi-Symphony Monte Carlo Simulator
  help, -h, --help    Show this help message

Options:
  Any extra options are passed directly to Streamlit (when running UI)
  e.g. $(basename "$0") ui --server.port 8502

Examples:
  $(basename "$0")          # Starts the Web UI on http://localhost:8501
  $(basename "$0") cli      # Runs correlation_analyzer.py in terminal
  $(basename "$0") mc       # Runs multi_symphony_portfolio_monte_carlo.py
EOF
}

resolve_binaries

# Command dispatcher
MODE="${1:-ui}"

case "$MODE" in
    ui|web|--ui|--web)
        shift 2>/dev/null || true
        echo "🚀 Starting Streamlit Web App (app.py)..."
        exec "$STREAMLIT_BIN" run app.py "$@"
        ;;
    cli|--cli)
        shift 2>/dev/null || true
        echo "📊 Starting CLI Correlation Analyzer (correlation_analyzer.py)..."
        exec "$PYTHON_BIN" correlation_analyzer.py "$@"
        ;;
    mc|monte-carlo|--mc|--monte-carlo)
        shift 2>/dev/null || true
        echo "🎲 Starting Monte Carlo Simulator (multi_symphony_portfolio_monte_carlo.py)..."
        exec "$PYTHON_BIN" multi_symphony_portfolio_monte_carlo.py "$@"
        ;;
    -h|--help|help)
        show_help
        exit 0
        ;;
    *)
        # If the first argument starts with a dash (e.g. --server.port 8502), pass directly to streamlit
        if [[ "$MODE" == -* ]]; then
            echo "🚀 Starting Streamlit Web App (app.py) with options: $*"
            exec "$STREAMLIT_BIN" run app.py "$@"
        else
            echo "❌ Unknown command: $MODE"
            echo ""
            show_help
            exit 1
        fi
        ;;
esac
