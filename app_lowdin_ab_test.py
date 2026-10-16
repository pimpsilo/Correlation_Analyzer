import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from datetime import date, timedelta
import os

# Import backend engine functions
from symphony_engine import parse_urls_from_text, load_symphonies

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Löwdin Symmetric Orthogonalization vs Classic Correlation A-B Test",
    page_icon="⚖️",
    layout="wide"
)

# Custom Styling for A-B comparison
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .method-a {
        border-left: 4px solid #D81B60;
    }
    .method-b {
        border-left: 4px solid #004D40;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Correlation Analysis A/B Test: Classic vs Löwdin Symmetric Orthogonalization")
st.caption(
    "Empirical side-by-side comparison between **Method A: Sequential/Triangular (Pearson & Cholesky)** "
    "and **Method B: Löwdin Symmetric Orthogonalization (LSO / Minimum Torsion)**."
)

# --- MATHEMATICAL HELPERS ---
def compute_lowdin_matrices(corr_matrix: np.ndarray):
    """
    Computes Löwdin Symmetric Orthogonalization matrices:
    Given correlation matrix C = V * Lambda * V^T:
      1. C^(1/2) = V * Lambda^(1/2) * V^T  (Symmetric Square Root / Projection)
      2. C^(-1/2) = V * Lambda^(-1/2) * V^T (Symmetric Inverse Square Root / Decorrelator)
    """
    C_sym = (corr_matrix + corr_matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(C_sym)
    
    # Clip negative/zero eigenvalues for stability
    eigvals_clipped = np.maximum(eigvals, 1e-10)
    
    # Symmetric Square Root (C^(1/2))
    C_half = eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T
    C_half = (C_half + C_half.T) / 2.0
    
    # Inverse Square Root (C^(-1/2))
    C_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_clipped)) @ eigvecs.T
    C_inv_half = (C_inv_half + C_inv_half.T) / 2.0
    
    return C_half, C_inv_half, eigvals

def compute_cholesky_matrix(corr_matrix: np.ndarray):
    """Computes lower-triangular Cholesky factor L (C = L * L^T) with eigenvalue clipping fallback."""
    C_sym = (corr_matrix + corr_matrix.T) / 2.0
    try:
        L = np.linalg.cholesky(C_sym)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(C_sym)
        eigvals = np.maximum(eigvals, 1e-10)
        C_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(C_pd)
    return L

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. Input Data")
    
    # Preset options
    preset_choice = st.selectbox(
        "Choose Preset or Custom Input:",
        options=[
            "Sector ETFs (Diverse 6-Asset Mix: XLK, XLF, XLE, XLV, XLI, GLD)",
            "Broader Sector Mix (10 Sectors)",
            "Custom Tickers or Composer URLs"
        ],
        index=0
    )
    
    default_text = ""
    if preset_choice == "Sector ETFs (Diverse 6-Asset Mix: XLK, XLF, XLE, XLV, XLI, GLD)":
        default_text = "XLK\nXLF\nXLE\nXLV\nXLI\nGLD"
    elif preset_choice == "Broader Sector Mix (10 Sectors)":
        default_text = "XLK\nXLF\nXLE\nXLV\nXLI\nXLP\nXLU\nXLY\nXLB\nGLD"
    else:
        default_text = "XLK\nQQQ\nSPY\nGLD"

    urls_text = st.text_area("Tickers / Symphony URLs", value=default_text, height=180)
    
    EARLIEST_ALLOWED_DATE = date(2010, 1, 1)
    TODAY = date.today()
    start_date = st.date_input("Start date", value=date(2020, 1, 1), min_value=EARLIEST_ALLOWED_DATE, max_value=TODAY)
    end_date = st.date_input("End date", value=TODAY, min_value=EARLIEST_ALLOWED_DATE, max_value=TODAY)
    
    reverse_order_btn = st.checkbox("🔀 Reverse Input Order (Permutation Test)", value=False,
                                    help="Reverse the order of strategies to test order-invariance of Method A vs Method B.")
    
    load_btn = st.button("Load & Run A/B Analysis", type="primary")
    
    st.divider()
    st.subheader("Cluster Controls")
    target_clusters = st.slider("Target Number of Clusters", min_value=2, max_value=8, value=3)
    
    st.divider()
    st.caption("🔬 **Research Context**: Demonstrates Löwdin Symmetric Orthogonalization (LSO) vs Gram-Schmidt / Cholesky sequential methods as applied to portfolio returns and risk modeling.")

# Session State for loaded data
if 'ab_data' not in st.session_state:
    st.session_state.ab_data = None

if (load_btn or st.session_state.ab_data is None) and urls_text:
    urls = parse_urls_from_text(urls_text)
    if len(urls) >= 2:
        with st.spinner("Loading strategy returns..."):
            result = load_symphonies(urls, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if isinstance(result, tuple):
                sym_data, _ = result
            else:
                sym_data = result
            if len(sym_data) >= 2:
                st.session_state.ab_data = sym_data

if st.session_state.ab_data is None:
    st.info("👈 Please load strategies using the sidebar to begin the A/B test.")
    st.stop()

# --- PREPARE DATA ---
active_symphonies = dict(st.session_state.ab_data)
symphony_names = list(active_symphonies.keys())

# Handle reverse permutation test if toggled
if reverse_order_btn:
    symphony_names = symphony_names[::-1]

n_items = len(symphony_names)
if n_items < 2:
    st.error("Need at least 2 items for analysis.")
    st.stop()

# Align common date range
all_dates = active_symphonies[symphony_names[0]].index.values
for name in symphony_names[1:]:
    all_dates = np.intersect1d(all_dates, active_symphonies[name].index.values)

returns_matrix = np.zeros((len(all_dates), n_items))
for i, name in enumerate(symphony_names):
    df = active_symphonies[name]
    df_aligned = df[df.index.isin(all_dates)].sort_index()
    returns_matrix[:, i] = df_aligned['returns'].values

# Calculate Pearson Correlation Matrix
correlation_matrix = np.corrcoef(returns_matrix.T)

# Compute Cholesky (Method A)
L_cholesky = compute_cholesky_matrix(correlation_matrix)

# Compute Löwdin Symmetric Matrices (Method B)
C_half, C_inv_half, eigvals = compute_lowdin_matrices(correlation_matrix)

# Strategy Uniqueness Metric from Löwdin Diagonal
uniqueness_scores = np.diag(C_half)

# Short labels for plotting
short_labels = [name.replace("Ticker_", "")[:15] for name in symphony_names]

# --- TABS FOR A-B COMPARISONS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Grouping & Dendrograms (A vs B)",
    "🔀 2. Order-Invariance Test (Cholesky vs Löwdin)",
    "⚡ 3. Shock Propagation & Stress Attribution",
    "🎲 4. Correlated Monte Carlo Simulation"
])

# ==============================================================================
# TAB 1: GROUPING & DENDROGRAMS
# ==============================================================================
with tab1:
    st.subheader("Strategy Grouping & Cluster Tree Comparison")
    st.write(
        "Compare how strategies are clustered using **Method A (Bivariate Pearson Distance)** "
        "versus **Method B (Löwdin Multi-Factor Risk Profile Distance)**."
    )
    
    col_a, col_b = st.columns(2)
    
    # --- METHOD A ---
    with col_a:
        st.markdown("### 🔴 Method A: Classic Pearson Correlation")
        st.caption("Groups based solely on pairwise linear correlation distance: $d = 1 - r$.")
        
        # Distance & Hierarchical Clustering
        dist_A = np.clip(1 - correlation_matrix, 0, 2)
        dist_A = (dist_A + dist_A.T) / 2.0
        np.fill_diagonal(dist_A, 0)
        condensed_dist_A = ssd.squareform(dist_A)
        linkage_A = sch.linkage(condensed_dist_A, method='average')
        clusters_A = sch.fcluster(linkage_A, target_clusters, criterion='maxclust')
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sch.dendrogram(linkage_A, labels=short_labels, leaf_rotation=45, leaf_font_size=9, ax=ax)
        ax.set_ylabel("Pearson Distance (1 - r)")
        ax.set_title("Method A: Pearson Dendrogram")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("**Method A Cluster Groups:**")
        df_clust_A = pd.DataFrame({"Strategy": short_labels, "Cluster ID": clusters_A})
        st.dataframe(df_clust_A.sort_values(by="Cluster ID"), hide_index=True, use_container_width=True)

    # --- METHOD B ---
    with col_b:
        st.markdown("### 🟢 Method B: Löwdin Symmetric Profile Clustering")
        st.caption("Groups based on Euclidean distance between full Löwdin orthogonal factor profiles $\\mathbf{C}^{1/2}$.")
        
        # Distance across Löwdin Symmetric profiles
        dist_B = ssd.pdist(C_half, metric='euclidean')
        linkage_B = sch.linkage(dist_B, method='average')
        clusters_B = sch.fcluster(linkage_B, target_clusters, criterion='maxclust')
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sch.dendrogram(linkage_B, labels=short_labels, leaf_rotation=45, leaf_font_size=9, ax=ax)
        ax.set_ylabel("Löwdin Metric Distance")
        ax.set_title("Method B: Löwdin Symmetric Dendrogram")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("**Method B Cluster Groups:**")
        df_clust_B = pd.DataFrame({"Strategy": short_labels, "Cluster ID": clusters_B, "Löwdin Uniqueness": np.round(uniqueness_scores, 3)})
        st.dataframe(df_clust_B.sort_values(by="Cluster ID"), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("💡 Löwdin 'Strategy Uniqueness' Index (Diagonal of $\\mathbf{C}^{1/2}$)")
    st.write(
        "In quantum mechanics and Löwdin orthogonalization, $(C^{1/2})_{ii}$ measures the **self-overlap / uniqueness** of each vector. "
        "In finance, a score near **1.0** indicates a completely autonomous asset (pure alpha / true diversifier), "
        "while lower values indicate heavy entanglement / collinearity with other portfolio strategies."
    )
    
    u_df = pd.DataFrame({"Strategy": short_labels, "Uniqueness Score": uniqueness_scores}).sort_values(by="Uniqueness Score", ascending=False)
    fig_u, ax_u = plt.subplots(figsize=(8, 3))
    sns.barplot(data=u_df, x="Strategy", y="Uniqueness Score", palette="viridis", ax=ax_u)
    ax_u.set_ylim(0, 1.05)
    ax_u.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_u)

# ==============================================================================
# TAB 2: ORDER INVARIANCE TEST
# ==============================================================================
with tab2:
    st.subheader("🔀 The Permutation Test: Order-Dependence vs Order-Invariance")
    st.write(
        "A central finding of the Löwdin paper is that **sequential orthogonalization methods (like Cholesky or Gram-Schmidt) "
        "depend arbitrarily on the input sequence**, treating the first vector as privileged. "
        "**Löwdin Symmetric Orthogonalization treats all strategies symmetrically on an equal footing.**"
    )
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("#### 🔴 Cholesky Factor $\\mathbf{L}$ (Sequential / Triangular)")
        st.caption("Notice how Row 0 has only 1 entry `[L_00, 0, 0...]`! The first asset is arbitrarily treated as the master factor.")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(L_cholesky, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=short_labels, yticklabels=short_labels, ax=ax)
        ax.set_title("Cholesky L (Lower Triangular)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_t2:
        st.markdown("#### 🟢 Löwdin Factor $\\mathbf{C}^{1/2}$ (Symmetric / Democratic)")
        st.caption("Notice that $T_{ij} = T_{ji}$! No asset is arbitrarily privileged or zeroed out.")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(C_half, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=short_labels, yticklabels=short_labels, ax=ax)
        ax.set_title("Löwdin C^(1/2) (Symmetric Matrix Root)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    st.info(
        "💡 **Test this yourself!** Toggle the checkbox **'🔀 Reverse Input Order'** in the sidebar. "
        "Notice that Cholesky's rows and numerical values change completely (the new first asset takes the master slot), "
        "whereas Löwdin's matrix simply swaps rows and columns without changing the intrinsic pairwise relationships!"
    )

# ==============================================================================
# TAB 3: SHOCK PROPAGATION & STRESS ATTRIBUTION
# ==============================================================================
with tab3:
    st.subheader("⚡ Shock Propagation: How Stress Leaks Across the Portfolio")
    st.write(
        "Suppose one of your symphonies suffers a **-3.0 standard deviation crash shock** ($z_k = -3$). "
        "How does that shock propagate to other assets in the portfolio?"
    )
    
    selected_shock_asset = st.selectbox("Select Strategy to Shock:", options=short_labels, index=len(short_labels)-1)
    shock_idx = short_labels.index(selected_shock_asset)
    shock_magnitude = st.slider("Shock Magnitude (Standard Deviations)", min_value=-5.0, max_value=5.0, value=-3.0, step=0.5)
    
    # Vector of independent standard shocks
    z = np.zeros(n_items)
    z[shock_idx] = shock_magnitude
    
    # Propagation under Cholesky: x = L * z
    shock_cholesky = L_cholesky @ z
    
    # Propagation under Löwdin: x = C^(1/2) * z
    shock_lowdin = C_half @ z
    
    shock_comp_df = pd.DataFrame({
        "Strategy": short_labels,
        "Method A: Cholesky Shock": shock_cholesky,
        "Method B: Löwdin Shock": shock_lowdin
    })
    
    fig, ax = plt.subplots(figsize=(10, 4))
    x_axis = np.arange(len(short_labels))
    width = 0.35
    
    ax.bar(x_axis - width/2, shock_cholesky, width, label="Method A (Cholesky)", color="#D81B60")
    ax.bar(x_axis + width/2, shock_lowdin, width, label="Method B (Löwdin)", color="#004D40")
    
    ax.set_ylabel("Induced Return Shock (σ)")
    ax.set_title(f"Portfolio Shock Propagation when '{selected_shock_asset}' is Shocked by {shock_magnitude}σ")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    > [!IMPORTANT]
    > **Why this matters**:
    > * Under **Cholesky (Method A)**, any strategy ordered *before* the shocked strategy experiences **0.0 shock**! 
    >   If the last asset in your list crashes, Cholesky pretends no other asset feels anything. That is an artificial mathematical artifact of triangular decomposition.
    > * Under **Löwdin (Method B)**, the crash shock propagates **symmetrically and reciprocally** to every other asset based on their true mutual correlation.
    """)

# ==============================================================================
# TAB 4: CORRELATED MONTE CARLO SIMULATION
# ==============================================================================
with tab4:
    st.subheader("🎲 Monte Carlo Simulation: Cholesky vs Löwdin")
    st.write(
        "Run forward Monte Carlo paths using both decomposition methods to compare aggregate portfolio return distributions."
    )
    
    n_sims = st.slider("Number of Simulations", min_value=500, max_value=5000, value=1500, step=500)
    n_days = st.slider("Simulation Horizon (Trading Days)", min_value=63, max_value=504, value=252, step=63)
    
    if st.button("Run Simulation Comparison", type="primary"):
        with st.spinner("Simulating correlated trajectories..."):
            np.random.seed(42)
            
            # Equal weight portfolio
            weights = np.ones(n_items) / n_items
            means = returns_matrix.mean(axis=0)
            stds = returns_matrix.std(axis=0)
            
            # Draw standard normal random shocks (sims, days, items)
            raw_z = np.random.standard_normal((n_sims, n_days, n_items))
            
            # Correlated shocks:
            # Method A: L @ z
            corr_z_A = np.einsum('ij,ndj->ndi', L_cholesky, raw_z)
            returns_A = means + stds * corr_z_A
            port_ret_A = np.einsum('ndi,i->nd', returns_A, weights)
            cum_A = np.cumprod(1 + port_ret_A, axis=1) - 1.0
            final_returns_A = cum_A[:, -1]
            
            # Method B: C^(1/2) @ z
            corr_z_B = np.einsum('ij,ndj->ndi', C_half, raw_z)
            returns_B = means + stds * corr_z_B
            port_ret_B = np.einsum('ndi,i->nd', returns_B, weights)
            cum_B = np.cumprod(1 + port_ret_B, axis=1) - 1.0
            final_returns_B = cum_B[:, -1]
            
            # Display stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔴 Method A (Cholesky Paths)")
                st.metric("Median Return", f"{np.median(final_returns_A)*100:.2f}%")
                st.metric("5th Percentile (Downside)", f"{np.percentile(final_returns_A, 5)*100:.2f}%")
                st.metric("95th Percentile (Upside)", f"{np.percentile(final_returns_A, 95)*100:.2f}%")
                
            with col2:
                st.markdown("#### 🟢 Method B (Löwdin Paths)")
                st.metric("Median Return", f"{np.median(final_returns_B)*100:.2f}%")
                st.metric("5th Percentile (Downside)", f"{np.percentile(final_returns_B, 5)*100:.2f}%")
                st.metric("95th Percentile (Upside)", f"{np.percentile(final_returns_B, 95)*100:.2f}%")
                
            # Density / Histogram plot
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.kdeplot(final_returns_A * 100, label="Method A (Cholesky)", color="#D81B60", fill=True, alpha=0.3, ax=ax)
            sns.kdeplot(final_returns_B * 100, label="Method B (Löwdin)", color="#004D40", fill=True, alpha=0.3, ax=ax)
            ax.set_xlabel("Total Horizon Return (%)")
            ax.set_ylabel("Density")
            ax.set_title(f"Simulated Portfolio Cumulative Return Distribution ({n_sims} Paths, {n_days} Days)")
            ax.legend()
            st.pyplot(fig)
            
            st.caption(
                "💡 **Conclusion**: In unbounded Gaussian sampling, the aggregate portfolio distribution converges to the same theoretical distribution. "
                "However, Löwdin guarantees that individual asset paths retain order-invariance, symmetric shock propagation, and minimal distortion to the original asset identities."
            )
