# 🎼 Symphony Correlation Analyzer & Clustering Engine

A Streamlit web application for analyzing, clustering, and selecting Composer.trade trading symphonies via hierarchical correlation trees (dendrograms) and performance metrics (Sharpe, Sortino, Calmar, Max Drawdown).

---

## 🚀 Key Features

1. **Composer API & Cache Ingestion**:
   - Paste Composer symphony URLs or IDs to fetch historical backtest allocations.
   - Built-in local cache in `data_storage/` prevents redundant API calls and rate-limiting.

2. **Hierarchical Clustering**:
   - Computes return correlation distance ($1 - r$) across synchronized trading days.
   - Generates interactive dendrograms to discover strategy clusters and eliminate redundant exposure.

3. **Multi-Objective Candidate Selection**:
   - Filter and pick cluster representatives by:
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

# Run Streamlit
streamlit run app.py
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
