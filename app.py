import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from datetime import date

# Import backend engine functions
from symphony_engine import parse_urls_from_text, load_symphonies

st.set_page_config(page_title="Composer Symphony Clustering", layout="wide")
st.title("Symphony Hierarchical Clustering & Selection")

# --- 1. INPUTS ---
with st.sidebar:
    st.header("1. Input Data")
    st.write("Batch load symphonies. The local cache prevents redundant API calls.")
    urls_text = st.text_area("Paste Composer URLs or IDs (one per line)", height=250)
    start_date = st.date_input("Start date", value=date(2020, 1, 1))
    end_date = st.date_input("End date", value=date.today())
    
    load_btn = st.button("Load Symphonies", type="primary")

if 'symphony_data' not in st.session_state:
    st.session_state.symphony_data = None

if load_btn and urls_text:
    urls = parse_urls_from_text(urls_text)
    if len(urls) < 2:
        st.error("Enter a minimum of 2 valid symphonies.")
    else:
        with st.spinner(f"Processing {len(urls)} symphonies..."):
            symphony_data = load_symphonies(urls, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if len(symphony_data) >= 2:
                st.session_state.symphony_data = symphony_data
                st.success(f"Successfully loaded {len(symphony_data)} symphonies.")
            else:
                st.error("Failed to load sufficient data.")

# --- 2. EXECUTION & CLUSTERING ---
if st.session_state.symphony_data:
    symphony_names = list(st.session_state.symphony_data.keys())
    n_symphonies = len(symphony_names)
    
    st.header("2. Dendrogram & Cluster Groupings")
    
    # Align dates across all datasets
    all_dates = st.session_state.symphony_data[symphony_names[0]].index.values
    for name in symphony_names[1:]:
        all_dates = np.intersect1d(all_dates, st.session_state.symphony_data[name].index.values)
        
    returns_matrix = np.zeros((len(all_dates), n_symphonies))
    
    # Calculate performance metrics for selection tie-breakers
    metrics = []
    
    for i, name in enumerate(symphony_names):
        df = st.session_state.symphony_data[name]
        meta = df.attrs['metadata']
        df_aligned = df[df.index.isin(all_dates)].sort_index()
        r = df_aligned['returns'].values
        returns_matrix[:, i] = r
        
        # Calculate Sharpe and Calmar
        annual_ret = r.mean() * 252
        annual_vol = r.std() * np.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
        
        cum_ret = (1 + r).cumprod()
        running_max = np.maximum.accumulate(cum_ret)
        drawdowns = (cum_ret - running_max) / running_max
        max_dd = drawdowns.min()
        
        calmar = annual_ret / abs(max_dd) if max_dd < 0 else np.nan
        
        metrics.append({
            "Symphony": name,
            "Composer ID": meta.get('id', 'N/A'),
            "Earliest Start Date": pd.to_datetime(meta.get('earliest_date')).strftime('%Y-%m-%d') if meta.get('earliest_date') else 'N/A',
            "Ann. Return": annual_ret,
            "Max Drawdown": max_dd,
            "Sharpe Ratio": sharpe,
            "Calmar Ratio": calmar
        })
        
    metrics_df = pd.DataFrame(metrics).set_index("Symphony")

    # Generate Distance Matrix for Clustering (Distance = 1 - Correlation)
    correlation_matrix = np.corrcoef(returns_matrix.T)
    distance_matrix = np.clip(1 - correlation_matrix, 0, 2)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    np.fill_diagonal(distance_matrix, 0)
    condensed_dist = ssd.squareform(distance_matrix)
    
    # Compute the linkage matrix using Average linkage
    linkage_matrix = sch.linkage(condensed_dist, method='average')
    
    # --- 3. VISUALIZATION ---
    c1, c2 = st.columns([3, 1])
    
    with c2:
        st.subheader("Cluster Controls")
        target_clusters = st.slider("Number of Target Clusters", min_value=2, max_value=min(50, n_symphonies), value=min(10, n_symphonies))
        
        # Assign cluster IDs based on the target number
        cluster_labels = sch.fcluster(linkage_matrix, target_clusters, criterion='maxclust')
        
        st.metric("Total Algorithms Analyzed", n_symphonies)
        st.metric("Overlapping Trading Days", len(all_dates))

    with c1:
        st.subheader("Hierarchical Tree (Dendrogram)")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        dendro = sch.dendrogram(
            linkage_matrix,
            labels=symphony_names,
            leaf_rotation=90,
            leaf_font_size=10,
            ax=ax,
            color_threshold=linkage_matrix[-(target_clusters-1), 2] if target_clusters > 1 else 0
        )
        
        ax.set_ylabel("Correlation Distance (1 - r)")
        ax.set_title("Algorithm Similarity Groupings")
        st.pyplot(fig)

    # --- 4. CLUSTER EVALUATION ---
    st.divider()
    st.subheader("Cluster Representatives Evaluation")
    st.write("Review the algorithms assigned to each branch.")
    
    # Combine clusters with metrics
    cluster_df = pd.DataFrame({
        "Symphony": symphony_names,
        "Cluster ID": cluster_labels
    }).set_index("Symphony")
    
    analysis_df = cluster_df.join(metrics_df).reset_index()
    analysis_df = analysis_df.sort_values(by=["Cluster ID", "Sharpe Ratio"], ascending=[True, False])
    
    st.dataframe(
        analysis_df.style.background_gradient(subset=['Sharpe Ratio', 'Calmar Ratio'], cmap='viridis')
                       .format({"Ann. Return": "{:.2%}", "Max Drawdown": "{:.2%}", "Sharpe Ratio": "{:.2f}", "Calmar Ratio": "{:.2f}"}),
        hide_index=True,
        use_container_width=True,
        height=600
    )
    
    st.download_button(
        label="Download Cluster Data as CSV",
        data=analysis_df.to_csv(index=False).encode('utf-8'),
        file_name="symphony_clusters.csv",
        mime="text/csv"
    )

    # --- 5. CANDIDATE SELECTION & CORRELATION ---
    st.divider()
    st.header("5. Candidate Selection & Correlation")
    st.write("Select one representative from each cluster to build your final portfolio.")
    
    selection_method = st.radio(
        "Selection Criteria:",
        options=["Highest Sharpe", "Highest Calmar", "Highest Ann. Return", "Lowest Max Drawdown", "Manual Selection"],
        horizontal=True
    )
    
    selected_symphonies = []
    
    if selection_method != "Manual Selection":
        for cluster_id, group in analysis_df.groupby("Cluster ID"):
            if selection_method == "Highest Sharpe":
                best_idx = group["Sharpe Ratio"].idxmax()
            elif selection_method == "Highest Calmar":
                best_idx = group["Calmar Ratio"].idxmax()
            elif selection_method == "Highest Ann. Return":
                best_idx = group["Ann. Return"].idxmax()
            elif selection_method == "Lowest Max Drawdown":
                best_idx = group["Max Drawdown"].idxmax()
            
            selected_symphonies.append(group.loc[best_idx, "Symphony"])
            
        st.write("**Automatically Selected Candidates:**")
        selected_df = analysis_df[analysis_df["Symphony"].isin(selected_symphonies)]
        st.dataframe(selected_df, hide_index=True)
        
    else:
        st.write("**Manual Selection:**")
        cluster_ids = sorted(analysis_df["Cluster ID"].unique())
        cols = st.columns(3)
        for idx, c_id in enumerate(cluster_ids):
            cluster_group = analysis_df[analysis_df["Cluster ID"] == c_id]
            options = cluster_group["Symphony"].tolist()
            with cols[idx % 3]:
                chosen = st.selectbox(f"Cluster {c_id}", options=options, key=f"select_{c_id}")
                selected_symphonies.append(chosen)

    if len(selected_symphonies) > 1:
        st.subheader("Candidate Correlation Analysis")
        
        cand_returns = np.zeros((len(all_dates), len(selected_symphonies)))
        for i, name in enumerate(selected_symphonies):
            df = st.session_state.symphony_data[name]
            df_aligned = df[df.index.isin(all_dates)].sort_index()
            cand_returns[:, i] = df_aligned['returns'].values
            
        cand_corr_matrix = np.corrcoef(cand_returns.T)
        cand_corr_df = pd.DataFrame(cand_corr_matrix, index=selected_symphonies, columns=selected_symphonies)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cand_corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True,
                xticklabels=[n[:20] for n in selected_symphonies],
                yticklabels=[n[:20] for n in selected_symphonies], ax=ax
            )
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
        with c2:
            upper = cand_corr_matrix[np.triu_indices_from(cand_corr_matrix, k=1)]
            st.metric("Average Correlation", f"{np.mean(upper):.3f}")
            st.metric("Max Correlation", f"{np.max(upper):.3f}")
            st.metric("Min Correlation", f"{np.min(upper):.3f}")
            
        st.write("**Raw Candidate Correlation Matrix**")
        st.dataframe(cand_corr_df.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1), use_container_width=True)
        
        st.download_button(
            label="Download Candidate Matrix CSV",
            data=cand_corr_df.to_csv().encode('utf-8'),
            file_name="candidate_correlation_matrix.csv",
            mime="text/csv"
        )
