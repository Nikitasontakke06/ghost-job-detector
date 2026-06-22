import streamlit as st
import joblib
import numpy as np
import pandas as pd
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

# ---- Load model ----
model = joblib.load('models/ghost_job_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# ---- Page title ----
st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <span style="font-size:1.1rem;font-weight:600;">Detector</span>
        <span style="font-size:0.8rem;color:#888;margin-left:10px;">
        Check a single job posting for ghost-job patterns</span>
    </div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.4], gap="large")

# ---- Left: inputs ----
with left:
    st.markdown('<div class="section-header">Posting inputs</div>',
                unsafe_allow_html=True)
    views = st.number_input("Views", min_value=1, value=500)
    applies = st.number_input("Applies", min_value=0, value=0)
    repost_count = st.number_input("Times reposted", min_value=1, value=1)
    skill_count = st.number_input("Required skills", min_value=0, value=3)
    salary_disclosed = st.toggle("Salary disclosed")
    st.write("")
    check = st.button("Run detection", type="primary", use_container_width=True)

# ---- Right: results ----
with right:
    st.markdown('<div class="section-header">Result</div>',
                unsafe_allow_html=True)

    if check:
        ratio = applies / views if views > 0 else 0
        log_views = np.log1p(views)
        X = np.array([[ratio, log_views, repost_count,
                       skill_count, int(salary_disclosed)]])
        X_scaled = scaler.transform(X)
        flag = model.predict(X_scaled)[0]
        score = model.decision_function(X_scaled)[0]
        is_ghost = flag == -1 and ratio < 0.12

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ratio", f"{ratio:.1%}")
        m2.metric("Score", f"{score:.3f}")
        m3.metric("Reposts", repost_count)
        m4.metric("Flag", "⚠ Ghost" if is_ghost else "✓ Normal")

        st.write("")
        if is_ghost:
            st.error("⚠️ **Ghost-job pattern detected** — High visibility with low "
                     "conversion and repeated reposting matches low-effort or "
                     "pipeline-building listings.")
        else:
            st.success("✅ **Normal posting** — Engagement pattern is consistent "
                       "with typical, actively-hiring listings.")

        st.markdown('<div class="section-header">Feature breakdown</div>',
                    unsafe_allow_html=True)
        feature_df = pd.DataFrame({
            "Feature": ["Apply/view ratio", "Log(views)", "Repost count",
                        "Skill count", "Salary disclosed"],
            "Value": [f"{ratio:.4f}", f"{log_views:.3f}",
                      str(repost_count), str(skill_count),
                      "Yes" if salary_disclosed else "No"],
            "Interpretation": [
                "Low = fewer people applied relative to views",
                "Higher = more visible posting",
                "High = same job posted repeatedly",
                "Number of skills required",
                "Whether salary was shown in the posting"
            ]
        })
        st.dataframe(feature_df, hide_index=True, use_container_width=True)

    else:
        st.info("Enter posting details on the left and click **Run detection**.")

    st.markdown('<div class="section-header">How it works</div>',
                unsafe_allow_html=True)
    st.markdown("""
        <div style="font-size:0.78rem;color:#888;line-height:1.7;">
        Uses an <strong>Isolation Forest</strong> model trained on 24,000+
        high-traffic LinkedIn postings. No ground-truth labels exist for ghost jobs,
        so the model flags postings whose engagement pattern deviates from typical
        behavior — high views, low conversion, repeated reposting.
        Cross-validated with DBSCAN and explained using SHAP.
        </div>
    """, unsafe_allow_html=True)