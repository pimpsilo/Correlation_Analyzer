# 🎼 Symphony & Asset Correlation Analyzer & Clustering Engine

A Streamlit web application for analyzing, clustering, and selecting Composer.trade trading symphonies, individual stocks, ETFs, and mutual funds via hierarchical correlation trees (dendrograms) and performance metrics (Sharpe, Sortino, Calmar, Max Drawdown).

---

## 🚀 Key Features

1. **Multi-Asset Ingestion (Composer Symphonies, Stocks, ETFs, Mutual Funds)**:
   - Paste Composer symphony URLs, symphony IDs, or raw ticker symbols (one per line).
   - Direct support for **Stocks** (e.g. `GEV`, `GOOG`), **ETFs** (e.g. `QQQ`, `SPY`, `ZDIS`), and **Mutual Funds** (e.g. `SMPIX`, `FSELX`, `RMQHX`).
   - Built-in local cache in `data_storage/` prevents redundant API calls and rate-limiting.

2. **Hierarchical Clustering & Löwdin Orthogonalization**:
   - **Löwdin Multi-Factor Risk Profile Clustering**: Clusters assets by Euclidean distance between democratic, order-invariant orthogonal risk vectors $\mathbf{C}^{1/2}$.
   - **Classic Pearson Distance ($1 - r$)**: Retained for standard correlation comparisons.
   - **Strategy Uniqueness Index**: Computes the diagonal $(C^{1/2})_{ii} \in [0, 1]$ measuring individual asset autonomy and pure alpha vs. shared factor entanglement.

3. **Multi-Objective Candidate Selection**:
   - Filter and pick cluster representatives by:
     - **Highest Uniqueness (Autonomy)**: Picks the purest diversifier from each cluster
     - **Highest Sharpe Ratio**
     - **Highest Sortino Ratio**
     - **Highest Calmar Ratio**
     - **Highest Annual Return**
     - **Lowest Maximum Drawdown**
     - **Manual Selection**

4. **Pairwise Correlation Heatmaps & Matrix Exports**:
   - Visualizes candidate inter-correlation matrix with Seaborn heatmaps.
   - Export cluster assignments and correlation matrices directly as CSV.

---

## 📱 Running the App

### 1. Run Locally (Desktop & Local Wi-Fi)

```bash
# Clone the repository (if not already local)
git clone https://github.com/pimpsilo/Correlation_Analyzer.git
cd Correlation_Analyzer

# Install dependencies
pip install -r requirements.txt

# Run with the runner script:
./run.sh            # Launches Streamlit Web UI (app.py)
./run.sh ab         # Launches Löwdin A/B Test comparative suite (app_lowdin_ab_test.py)
./run.sh cli        # Launches interactive CLI analyzer (correlation_analyzer.py)
./run.sh mc         # Launches Monte Carlo simulator (Löwdin or Cholesky)

# Or using the global terminal command:
correlation-analyzer       # Streamlit Web UI
correlation-analyzer cli   # CLI analyzer
correlation-analyzer-ui    # Quick zsh function
correlation-analyzer-cli   # Quick zsh function
```

- **On Desktop**: Open `http://localhost:8501`
- **On Phone / Tablet (Local Wi-Fi)**: Open `http://<your-mac-ip>:8501`

---

## ☁️ Deploying to Streamlit Community Cloud (Non-Local Access)

You can deploy this repository to **Streamlit Community Cloud** in 60 seconds for free global access on mobile or desktop:

1. Push this repository to GitHub under your account (e.g. `pimpsilo/Correlation_Analyzer`).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
3. Click **"New app"**.
4. Select:
   - **Repository**: `pimpsilo/Correlation_Analyzer`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**.
6. Access your app anywhere from mobile, tablet, or web!
