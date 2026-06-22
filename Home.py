import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Ghost Job Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.session_state["sidebar_state"] = "expanded"
st.markdown("""
    <style>
    /* Force sidebar always visible, hide the collapse button */
    [data-testid="collapsedControl"] { display: none; }
    [data-testid="stSidebar"] {
        min-width: 220px !important;
        max-width: 220px !important;
        transform: none !important;
        visibility: visible !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebarNav"] { display: none; }
    .block-container {
        padding-top: 1.2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .nav-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #888;
        margin: 1rem 0 0.4rem 10px;
    }
    .section-header {
        font-size: 0.8rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.6rem;
        margin-top: 1.4rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:1.5rem;">
            <span style="font-size:1.1rem;">🔍</span>
            <span style="font-size:0.95rem;font-weight:600;">Ghost Job Detector</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    st.page_link("Home.py", label="📊  Overview", icon=None)
    st.page_link("pages/1_Detector.py", label="🔍  Detector", icon=None)

    st.markdown('<div class="nav-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="font-size:0.78rem;color:#888;padding:0 10px;line-height:1.6;">
            LinkedIn Job Postings<br>
            Source: Kaggle<br>
            Rows: 118,644<br>
            Modeled: 24,125<br>
            Date range: Mar–Apr 2024
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">Model</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="font-size:0.78rem;color:#888;padding:0 10px;line-height:1.6;">
            Algorithm: Isolation Forest<br>
            Contamination: 5%<br>
            Estimators: 200<br>
            Validated: DBSCAN + SHAP
        </div>
    """, unsafe_allow_html=True)

# ---- Load data ----
try:
    df = pd.read_csv('data/ghost_job_results.csv')
except FileNotFoundError:
    st.error("data/ghost_job_results.csv not found. Add it to the data/ folder.")
    st.stop()

# ---- Page title ----
st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <span style="font-size:1.1rem;font-weight:600;">Overview</span>
        <span style="font-size:0.8rem;color:#888;margin-left:10px;">
        LinkedIn Job Postings · Apr 2024</span>
    </div>
""", unsafe_allow_html=True)

# ---- KPI row ----
total = len(df)
views = int(df['views'].sum())
applies = int(df['applies'].fillna(0).sum())
ghost_count = int(df['is_ghost_candidate'].sum()) if 'is_ghost_candidate' in df.columns else 0
ghost_pct = ghost_count / total * 100 if total else 0
avg_ratio = df['apply_view_ratio'].mean() if 'apply_view_ratio' in df.columns else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total postings", f"{total:,}")
k2.metric("Total views", f"{views:,}")
k3.metric("Total applies", f"{applies:,}")
k4.metric("Avg apply/view ratio", f"{avg_ratio:.1%}")
k5.metric("Flagged ghost-like", f"{ghost_count:,}", f"{ghost_pct:.1f}% of postings")

# ---- Charts ----
st.markdown('<div class="section-header">Distribution</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Apply/view ratio distribution**")
    ratio_data = df['apply_view_ratio'].dropna().clip(upper=0.8)
    hist_vals, hist_edges = np.histogram(ratio_data, bins=25)
    hist_df = pd.DataFrame({
        'ratio': hist_edges[:-1].round(3),
        'count': hist_vals
    })
    st.bar_chart(hist_df.set_index('ratio'), height=220)

with c2:
    st.markdown("**Top 10 companies by posting count**")
    top_co = df['company_name'].value_counts().head(10).reset_index()
    top_co.columns = ['company', 'postings']
    st.bar_chart(top_co.set_index('company'), height=220)

# ---- Tables ----
st.markdown('<div class="section-header">Ghost-job candidates</div>',
            unsafe_allow_html=True)
t1, t2 = st.columns(2)

with t1:
    st.markdown("**Top flagged postings by anomaly score**")
    if 'is_ghost_candidate' in df.columns:
        ghost_df = (df[df['is_ghost_candidate'] == 1]
                    [['company_name', 'title', 'views', 'applies',
                      'repost_count', 'anomaly_score']]
                    .sort_values('anomaly_score')
                    .head(8)
                    .reset_index(drop=True))
        ghost_df.columns = ['Company', 'Title', 'Views', 'Applies', 'Reposts', 'Score']
        st.dataframe(ghost_df, hide_index=True, use_container_width=True, height=260)

with t2:
    st.markdown("**Most reposted job signatures**")
    if 'repost_count' in df.columns:
        repost_df = (df[['company_name', 'title', 'repost_count',
                         'views', 'apply_view_ratio']]
                     .drop_duplicates(subset=['company_name', 'title'])
                     .sort_values('repost_count', ascending=False)
                     .head(8)
                     .reset_index(drop=True))
        repost_df.columns = ['Company', 'Title', 'Reposts', 'Views', 'Ratio']
        repost_df['Ratio'] = repost_df['Ratio'].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        st.dataframe(repost_df, hide_index=True, use_container_width=True, height=260)

# ---- Experience level breakdown ----
st.markdown('<div class="section-header">Breakdown by experience level</div>',
            unsafe_allow_html=True)
if 'experience_level' in df.columns:
    exp_df = (df.groupby('experience_level')
              .agg(
                  postings=('job_id', 'count'),
                  avg_ratio=('apply_view_ratio', 'mean'),
                  ghost_count=('is_ghost_candidate', 'sum')
              )
              .reset_index()
              .sort_values('postings', ascending=False))
    exp_df.columns = ['Experience level', 'Postings', 'Avg ratio', 'Ghost-flagged']
    exp_df['Avg ratio'] = exp_df['Avg ratio'].apply(lambda x: f"{x:.1%}")
    st.dataframe(exp_df, hide_index=True, use_container_width=True)