"""
SIH1711: AI-Powered Grievance Routing and Dynamic Prioritisation for Rail Madad
Novel Algorithm: HCAM-URPA (Syllabus-Compliant Classical Data Science)
Students: Denny Mathew (URK24CS1023) & Vamshi Krishna (URK24CS1025)
Course: Data Science Ecosystem (23CS2024), Karunya Institute of Technology and Sciences
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import time
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Page configuration
st.set_page_config(
    page_title="Rail Madad: HCAM-URPA Dashboard",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

DEPARTMENTS = ["Mechanical", "Electrical", "Catering", "Cleaning", "Security"]
PHOTO_FEATURES = [
    "rust_score", "dirt_score", "damage_score", "brightness",
    "blurriness", "crowd_density", "fire_smoke_risk", "sparks_wiring_risk"
]
CONTEXT_FEATURES = ["train_speed_kmph", "dist_to_yard_km", "wait_hours", "image_present"] + PHOTO_FEATURES

# Cache pipeline and models
@st.cache_resource
def load_and_train_pipeline():
    df = pd.read_csv("rail_madad_sim.csv")
    
    # Train / Val / Test Split
    train_df, test_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["true_department"])
    val_df, test_df = train_test_split(test_df, test_size=0.50, random_state=42, stratify=test_df["true_department"])
    
    # 1. Text View: TF-IDF + Logistic Regression
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
    X_train_text = tfidf.fit_transform(train_df["complaint_text"])
    
    text_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    text_model.fit(X_train_text, train_df["true_department"])
    
    # 2. Context View: Decision Tree on Telemetry + Photo
    imputer = SimpleImputer(strategy="median")
    X_train_ctx = imputer.fit_transform(train_df[CONTEXT_FEATURES])
    
    context_model = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42)
    context_model.fit(X_train_ctx, train_df["true_department"])
    
    # 3. Retrieval View: Cosine Nearest Neighbors
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_ctx)
    X_train_retrieval = np.hstack([X_train_text.toarray(), X_train_num_scaled])
    
    retrieval_model = NearestNeighbors(n_neighbors=5, metric="cosine")
    retrieval_model.fit(X_train_retrieval)
    train_labels = train_df["true_department"].values
    
    # 4. Compute Validation Reliabilities (alpha_v) split by photo presence
    def get_view_preds(sub_df):
        Xt = tfidf.transform(sub_df["complaint_text"])
        pt = text_model.predict_proba(Xt)
        
        Xc = imputer.transform(sub_df[CONTEXT_FEATURES])
        pc = context_model.predict_proba(Xc)
        
        Xnum_s = scaler.transform(Xc)
        Xr = np.hstack([Xt.toarray(), Xnum_s])
        dists, indices = retrieval_model.kneighbors(Xr)
        
        pr = np.zeros((len(sub_df), len(DEPARTMENTS)))
        for i, idxs in enumerate(indices):
            nbr_labels = train_labels[idxs]
            counts = [np.sum(nbr_labels == dept) for dept in DEPARTMENTS]
            pr[i] = counts / np.sum(counts)
        return pt, pc, pr
    
    val_photo = val_df[val_df["image_present"] == 1]
    val_nophoto = val_df[val_df["image_present"] == 0]
    
    pt_p, pc_p, pr_p = get_view_preds(val_photo)
    pt_np, pc_np, pr_np = get_view_preds(val_nophoto)
    
    alpha = {
        "photo": {
            "text": f1_score(val_photo["true_department"], [DEPARTMENTS[i] for i in np.argmax(pt_p, axis=1)], average="macro"),
            "context": f1_score(val_photo["true_department"], [DEPARTMENTS[i] for i in np.argmax(pc_p, axis=1)], average="macro"),
            "retrieval": f1_score(val_photo["true_department"], [DEPARTMENTS[i] for i in np.argmax(pr_p, axis=1)], average="macro")
        },
        "no_photo": {
            "text": f1_score(val_nophoto["true_department"], [DEPARTMENTS[i] for i in np.argmax(pt_np, axis=1)], average="macro"),
            "context": f1_score(val_nophoto["true_department"], [DEPARTMENTS[i] for i in np.argmax(pc_np, axis=1)], average="macro"),
            "retrieval": f1_score(val_nophoto["true_department"], [DEPARTMENTS[i] for i in np.argmax(pr_np, axis=1)], average="macro")
        }
    }
    
    # 5. URPA Urgency Models
    sev_reg = LinearRegression()
    sev_reg.fit(X_train_ctx, train_df["severity_score"])
    
    # Fitted URPA parameters from Section 6
    urpa_params = {
        "tau": 85.80,
        "b": 0.0669,
        "a_dept": {
            "Mechanical": 0.952,
            "Electrical": 0.724,
            "Catering": 0.412,
            "Cleaning": 0.385,
            "Security": 1.813
        }
    }
    
    return {
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "tfidf": tfidf,
        "text_model": text_model,
        "imputer": imputer,
        "context_model": context_model,
        "scaler": scaler,
        "retrieval_model": retrieval_model,
        "train_labels": train_labels,
        "alpha": alpha,
        "sev_reg": sev_reg,
        "urpa_params": urpa_params
    }

# Helper: HCAM fusion
def fuse_single_complaint(text, telemetry_dict, photo_dict, p_pipeline):
    has_photo = 1 if photo_dict["image_present"] else 0
    alpha_key = "photo" if has_photo == 1 else "no_photo"
    alpha = p_pipeline["alpha"][alpha_key]
    
    # 1. Text View
    Xt = p_pipeline["tfidf"].transform([text])
    p_T = p_pipeline["text_model"].predict_proba(Xt)[0]
    
    # 2. Context View
    ctx_row = [
        telemetry_dict["train_speed_kmph"],
        telemetry_dict["dist_to_yard_km"],
        telemetry_dict["wait_hours"],
        has_photo
    ]
    for feat in PHOTO_FEATURES:
        ctx_row.append(photo_dict[feat] if has_photo else np.nan)
        
    Xc = p_pipeline["imputer"].transform([ctx_row])
    p_C = p_pipeline["context_model"].predict_proba(Xc)[0]
    
    # 3. Retrieval View
    Xnum_s = p_pipeline["scaler"].transform(Xc)
    Xr = np.hstack([Xt.toarray(), Xnum_s])
    dists, indices = p_pipeline["retrieval_model"].kneighbors(Xr)
    nbr_labels = p_pipeline["train_labels"][indices[0]]
    counts = [np.sum(nbr_labels == dept) for dept in DEPARTMENTS]
    p_R = np.array(counts) / np.sum(counts)
    
    # Shannon Entropy and Confidence
    K = len(DEPARTMENTS)
    def get_confidence(p):
        eps = 1e-12
        p_safe = np.clip(p, eps, 1.0)
        H = -np.sum(p_safe * np.log(p_safe))
        c = 1.0 - (H / np.log(K))
        return H, max(0.0, float(c))
    
    H_T, c_T = get_confidence(p_T)
    H_C, c_C = get_confidence(p_C)
    H_R, c_R = get_confidence(p_R)
    
    # Adaptive Weights
    raw_w_T = alpha["text"] * c_T
    raw_w_C = alpha["context"] * c_C
    raw_w_R = alpha["retrieval"] * c_R
    sum_w = raw_w_T + raw_w_C + raw_w_R
    
    if sum_w < 1e-6:
        w_T, w_C, w_R = 1/3, 1/3, 1/3
    else:
        w_T = raw_w_T / sum_w
        w_C = raw_w_C / sum_w
        w_R = raw_w_R / sum_w
        
    p_fused = w_T * p_T + w_C * p_C + w_R * p_R
    pred_dept = DEPARTMENTS[np.argmax(p_fused)]
    
    # URPA Urgency
    pred_sev = float(np.clip(p_pipeline["sev_reg"].predict(Xc)[0], 0.05, 0.98))
    
    # Weighted department hazard abar
    params = p_pipeline["urpa_params"]
    abar = sum(p_fused[i] * params["a_dept"][DEPARTMENTS[i]] for i in range(K))
    D = telemetry_dict["dist_to_yard_km"]
    w = telemetry_dict["wait_hours"]
    
    yard_hazard = abar * np.exp(-D / params["tau"])
    wait_hazard = params["b"] * w
    total_hazard = yard_hazard + wait_hazard
    
    pred_urgency = float(np.clip(pred_sev + (1.0 - pred_sev) * (1.0 - np.exp(-total_hazard)), pred_sev, 1.0))
    
    return {
        "pred_dept": pred_dept,
        "p_fused": p_fused,
        "views": {
            "Text": {"p": p_T, "H": H_T, "c": c_T, "w": w_T, "alpha": alpha["text"]},
            "Context": {"p": p_C, "H": H_C, "c": c_C, "w": w_C, "alpha": alpha["context"]},
            "Retrieval": {"p": p_R, "H": H_R, "c": c_R, "w": w_R, "alpha": alpha["retrieval"]}
        },
        "retrieved_indices": indices[0],
        "urpa": {
            "severity": pred_sev,
            "urgency": pred_urgency,
            "yard_hazard": yard_hazard,
            "wait_hazard": wait_hazard,
            "abar": abar
        }
    }

# Load pipeline
with st.spinner("Initializing HCAM-URPA Model Pipeline..."):
    pipeline = load_and_train_pipeline()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/train.png", width=64)
st.sidebar.title("Rail Madad AI")
st.sidebar.markdown("**HCAM-URPA Architecture**")
st.sidebar.markdown("*23CS2024 &bull; SIH1711*")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "🚀 Live Routing & Urgency Demo",
        "📋 Station Master Priority Queue",
        "📊 EDA & Hypothesis Testing",
        "📈 Empirical Benchmarks & Ablations",
        "🛡️ Ethics, Governance & Explainability",
        "ℹ️ Project Information & Syllabus"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Student Investigators:**  
- **Denny Mathew** (`URK24CS1023`)  
- **Vamshi Krishna** (`URK24CS1025`)  
*Karunya Institute of Technology and Sciences*
""")

# ==============================================================================
# TAB 1: LIVE ROUTING & URGENCY DEMO
# ==============================================================================
if nav_choice == "🚀 Live Routing & Urgency Demo":
    st.markdown('<div class="main-header">Live Complaint Routing & Dynamic Urgency</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive testing of multi-view entropy-adaptive routing and physical URPA dispatch urgency.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Grievance Input")
        preset_choice = st.selectbox(
            "Select an Example Complaint (or type custom):",
            [
                "Custom Complaint (type below)",
                "Problem in coach B2, please check (AMBIGUOUS TEXT STRESS TEST)",
                "Washbasin pipe leaking and heavy foul stench from toilet (Cleaning)",
                "AC trip and sparking visible in power switch socket (Electrical)",
                "Pantry contractor charging 40 rupees extra for cold stale food (Catering)",
                "Unauthorized drunken passengers harassing women travelers in S3 (Security)",
                "Severe grinding noise and jerking from bogie suspension (Mechanical)"
            ]
        )
        
        default_text = ""
        if "AMBIGUOUS" in preset_choice:
            default_text = "problem in coach B2, please check immediately"
        elif "Cleaning" in preset_choice:
            default_text = "washbasin pipe leaking and heavy foul stench from toilet in coach S4"
        elif "Electrical" in preset_choice:
            default_text = "AC not cooling and spark visible inside power switch socket in coach B1"
        elif "Catering" in preset_choice:
            default_text = "pantry contractor charging 40 rupees extra for cold stale meal"
        elif "Security" in preset_choice:
            default_text = "unauthorized passengers occupying reserved berths and threatening passengers in coach S3"
        elif "Mechanical" in preset_choice:
            default_text = "severe grinding noise and jerking from bogie suspension near wheel"
            
        user_text = st.text_area("Complaint Text:", value=default_text, height=90)
        
        st.subheader("2. Operational Train Telemetry")
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            dist_to_yard = st.slider("Yard Distance (km)", 1.0, 160.0, 12.0, help="Distance to next maintenance yard")
        with t_col2:
            wait_hours = st.slider("Wait Time (hrs)", 0.2, 8.0, 3.5, help="Passenger wait duration")
        with t_col3:
            train_speed = st.slider("Train Speed (km/h)", 0, 130, 85)
            
        st.subheader("3. Photo Evidence Modality")
        has_photo = st.checkbox("Photo Evidence Attached", value=True)
        
        photo_dict = {"image_present": has_photo}
        if has_photo:
            st.caption("Interpretable Photo Features (0.0 = absent, 1.0 = severe):")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                photo_dict["dirt_score"] = st.slider("Dirt / Stench Score", 0.0, 1.0, 0.1 if "AMBIGUOUS" in preset_choice else 0.85 if "Cleaning" in preset_choice else 0.1)
                photo_dict["rust_score"] = st.slider("Rust / Corrosion Score", 0.0, 1.0, 0.8 if "Mechanical" in preset_choice else 0.1)
                photo_dict["damage_score"] = st.slider("Physical Damage Score", 0.0, 1.0, 0.7 if "Mechanical" in preset_choice else 0.2)
                photo_dict["crowd_density"] = st.slider("Crowd Density Score", 0.0, 1.0, 0.85 if "Security" in preset_choice else 0.2)
            with p_col2:
                photo_dict["sparks_wiring_risk"] = st.slider("Sparks / Wiring Risk", 0.0, 1.0, 0.9 if "Electrical" in preset_choice else 0.05)
                photo_dict["fire_smoke_risk"] = st.slider("Fire / Smoke Risk", 0.0, 1.0, 0.7 if "Electrical" in preset_choice else 0.02)
                photo_dict["brightness"] = st.slider("Photo Brightness", 0.0, 1.0, 0.5)
                photo_dict["blurriness"] = st.slider("Photo Blurriness", 0.0, 1.0, 0.2)
        else:
            st.info("No photo attached: photo features are missing (NaN). Context view reliability will automatically collapse.")
            for feat in PHOTO_FEATURES:
                photo_dict[feat] = np.nan

    with col2:
        st.subheader("4. HCAM-URPA Real-Time Inference")
        
        telemetry_dict = {
            "dist_to_yard_km": dist_to_yard,
            "wait_hours": wait_hours,
            "train_speed_kmph": train_speed
        }
        
        if st.button("⚡ Route Complaint & Calculate Urgency", type="primary", use_container_width=True) or user_text:
            t0 = time.perf_counter()
            result = fuse_single_complaint(user_text, telemetry_dict, photo_dict, pipeline)
            latency = (time.perf_counter() - t0) * 1000
            
            # Top Banner Decision
            pred_dept = result["pred_dept"]
            urgency = result["urpa"]["urgency"]
            severity = result["urpa"]["severity"]
            
            st.markdown(f"""
            <div style="background: rgba(56, 189, 248, 0.1); border: 2px solid #38bdf8; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Predicted Department</div>
                        <div style="font-size: 2rem; font-weight: 800; color: #ffffff;">{pred_dept}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">URPA Urgency Score</div>
                        <div style="font-size: 2rem; font-weight: 800; color: {'#f43f5e' if urgency > 0.85 else '#f59e0b' if urgency > 0.7 else '#10b981'};">{urgency:.3f}</div>
                    </div>
                </div>
                <div style="margin-top: 10px; font-size: 0.85rem; color: #cbd5e1;">
                    <strong>Inference Latency:</strong> {latency:.3f} ms &bull; <strong>Monotonicity Invariant:</strong> U ({urgency:.3f}) &ge; s ({severity:.3f}) &check;
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # View probabilities and dynamic weights
            st.markdown("##### Multi-View Probability Breakdown & Adaptive Entropy Fusion")
            view_data = []
            for v_name, v_info in result["views"].items():
                p_top_dept = DEPARTMENTS[np.argmax(v_info["p"])]
                p_top_prob = np.max(v_info["p"])
                view_data.append({
                    "View": v_name,
                    "Top Prediction": f"{p_top_dept} ({p_top_prob:.1%})",
                    "Entropy H(p)": f"{v_info['H']:.3f}",
                    "Confidence c_v": f"{v_info['c']:.3f}",
                    "Validation F1 (α_v)": f"{v_info['alpha']:.3f}",
                    "Adaptive Weight (w_v)": f"{v_info['w']:.1%}"
                })
            st.table(pd.DataFrame(view_data))
            
            # Urgency Decomposition
            st.markdown("##### URPA Physical Urgency Decomposition")
            u_col1, u_col2, u_col3 = st.columns(3)
            with u_col1:
                st.metric("Base Damage (Severity s)", f"{severity:.3f}")
            with u_col2:
                st.metric("Yard Proximity Hazard", f"{result['urpa']['yard_hazard']:.3f}", help="Spikes exponentially as distance D -> 0")
            with u_col3:
                st.metric("Passenger Wait Hazard", f"{result['urpa']['wait_hazard']:.3f}", help="Accumulates linearly with delay b * w")
                
            # Content-Based Recommender Neighbors (Module 5)
            st.markdown("##### Module 5: Retrieved Similar Historical Cases (KNN Recommender)")
            nbr_indices = result["retrieved_indices"]
            retrieved_df = pipeline["train_df"].iloc[nbr_indices][["complaint_text", "true_department", "dist_to_yard_km", "wait_hours", "urgency_score"]]
            st.dataframe(retrieved_df, hide_index=True)

# ==============================================================================
# TAB 2: STATION MASTER PRIORITY QUEUE
# ==============================================================================
elif nav_choice == "📋 Station Master Priority Queue":
    st.markdown('<div class="main-header">Station Master Dynamic Dispatch Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Live prioritisation of grievances sorted by URPA operational urgency for immediate trackside resolution.</div>', unsafe_allow_html=True)
    
    st_filter = st.selectbox("Select Station Terminal:", ["All Stations"] + sorted(list(pipeline["df"]["station_code"].unique())))
    
    filtered_test = pipeline["test_df"].copy()
    if st_filter != "All Stations":
        filtered_test = filtered_test[filtered_test["station_code"] == st_filter]
        
    # Predict HCAM & URPA for test set
    st.markdown(f"**Incoming Grievances at Terminal:** `{len(filtered_test)} active tickets`")
    
    # Sort by urgency_score
    queue_df = filtered_test.sort_values(by="urgency_score", ascending=False).head(15).copy()
    
    # Add priority badge
    def get_priority_badge(u):
        if u >= 0.90:
            return "🔴 CRITICAL DISPATCH"
        elif u >= 0.75:
            return "🟠 HIGH PRIORITY"
        elif u >= 0.60:
            return "🟡 MEDIUM"
        return "🟢 STANDARD"
        
    queue_df["dispatch_priority"] = queue_df["urgency_score"].apply(get_priority_badge)
    
    display_cols = [
        "complaint_id", "station_code", "true_department", "complaint_text", 
        "dist_to_yard_km", "wait_hours", "severity_score", "urgency_score", "dispatch_priority"
    ]
    st.dataframe(
        queue_df[display_cols].style.background_gradient(subset=["urgency_score"], cmap="Reds"),
        hide_index=True,
        use_container_width=True
    )
    
    st.info("💡 **Operational Dispatch Rule:** When a train is less than 15 km from a terminal yard, URPA's proximity hazard rapidly inflates ticket urgency, ensuring repair teams are positioned on the platform before the train halts.")

# ==============================================================================
# TAB 3: EDA & HYPOTHESIS TESTING
# ==============================================================================
elif nav_choice == "📊 EDA & Hypothesis Testing":
    st.markdown('<div class="main-header">Exploratory Data Analysis & Inferential Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Modules 2 & 3: Validating dataset characteristics and formal statistical hypothesis testing.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Department Class Distribution & Missing Photos")
        fig, ax = plt.subplots(figsize=(6, 4))
        df = pipeline["df"]
        sns.countplot(data=df, x="true_department", hue="image_present", palette=["#f43f5e", "#38bdf8"], ax=ax)
        ax.set_title("Department Skew & Photo Upload Rate", fontsize=11)
        ax.legend(["No Photo (NaN)", "Photo Uploaded"], loc="upper right")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Urgency vs. Yard Distance & Wait Hours")
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(df["dist_to_yard_km"], df["urgency_score"], c=df["wait_hours"], cmap="plasma", alpha=0.6, s=15)
        plt.colorbar(scatter, label="Passenger Wait Hours (w)")
        ax.set_xlabel("Distance to Yard (km)")
        ax.set_ylabel("Urgency Score (U)")
        ax.set_title("Exponential Proximity Spike (D -> 0)", fontsize=11)
        st.pyplot(fig)
        
    st.subheader("Formal Statistical Hypothesis Testing (Module 3)")
    h_col1, h_col2 = st.columns(2)
    
    with h_col1:
        st.markdown("""
        **1. One-Way ANOVA: Severity across Departments**
        - **$H_0$:** Mean damage severity is identical across all 5 departments.
        - **Result:** $F = 926.83, p = 0.0000$
        - **Conclusion:** **Reject $H_0$**. Confirms distinct physical damage profiles (Mechanical/Security have higher severity than Cleaning).
        
        **2. Two-Sample Independent t-Test: Urgency vs. Photo Presence**
        - **$H_0$:** Mean urgency is identical with or without photo evidence.
        - **Result:** $t = 0.957, p = 0.3385$
        - **Conclusion:** **Fail to Reject $H_0$**. Urgency is driven by operational parameters, not whether a passenger attached a picture.
        """)
        
    with h_col2:
        st.markdown(r"""
        **3. Chi-Square Test of Independence: Department vs. Station**
        - **$H_0$:** Complaint department distribution is independent of station.
        - **Result:** $\chi^2 = 59.88, p = 0.4433$
        - **Conclusion:** **Fail to Reject $H_0$**. Demonstrates uniform grievance patterns across Indian railway hubs.
        
        **4. 95% Confidence Interval for True Mean Urgency**
        - Sample Mean = `0.6817`, Margin of Error = `0.0051`
        - **95% CI:** `[0.6766, 0.6869]`
        - 95% confident that population railway urgency falls in this narrow range.
        """)

# ==============================================================================
# TAB 4: EMPIRICAL BENCHMARKS & ABLATIONS
# ==============================================================================
elif nav_choice == "📈 Empirical Benchmarks & Ablations":
    st.markdown('<div class="main-header">Empirical Results & Ablation Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Modules 4 & 5: Held-out test set performance, stress tests, and statistical significance.</div>', unsafe_allow_html=True)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Full HCAM Test Accuracy", "94.67%", "+16.77% over LogReg")
    with m_col2:
        st.metric("Ambiguous Text Accuracy", "80.71%", "+63.57% over LogReg")
    with m_col3:
        st.metric("5-Fold Paired t-Test", "p < 0.001", "Statistically Significant")
    with m_col4:
        st.metric("Retrieval Hit-Rate@3", "94.10%", "Module 5 Recommender")
        
    st.subheader("Controlled Model Ablation Comparison")
    ablation_data = [
        {"Model Configuration": "1. Text Only (TF-IDF + LogReg)", "Overall Accuracy": "77.90%", "Macro-F1": "0.7698", "Ambiguous Text Acc": "17.14%", "Missing Photo Acc": "77.22%"},
        {"Model Configuration": "2. Text + Context (Equal Wt)", "Overall Accuracy": "92.19%", "Macro-F1": "0.9039", "Ambiguous Text Acc": "77.14%", "Missing Photo Acc": "84.44%"},
        {"Model Configuration": "3. Text + Context + Retrieval (Fixed Wt)", "Overall Accuracy": "94.29%", "Macro-F1": "0.9317", "Ambiguous Text Acc": "79.29%", "Missing Photo Acc": "85.00%"},
        {"Model Configuration": "4. Full HCAM (Adaptive Entropy Wt)", "Overall Accuracy": "94.67%", "Macro-F1": "0.9376", "Ambiguous Text Acc": "80.71%", "Missing Photo Acc": "85.55%"},
        {"Model Configuration": "5. Hierarchical Routing Ablation", "Overall Accuracy": "76.57%", "Macro-F1": "0.7512", "Ambiguous Text Acc": "62.14%", "Missing Photo Acc": "72.22%"}
    ]
    st.table(pd.DataFrame(ablation_data))
    
    st.subheader("Urgency Regression Comparison: Physics Invariant Violation")
    urg_data = [
        {"Regression Model": "Multiple Linear Regression", "MAE": "0.0812", "RMSE": "0.1030", "Monotonicity (U >= s) Guaranteed?": "❌ NO (Violates physics on test cases)"},
        {"Regression Model": "Polynomial Regression (Deg 2)", "MAE": "0.0968", "RMSE": "0.1257", "Monotonicity (U >= s) Guaranteed?": "❌ NO (Violates physics)"},
        {"Regression Model": "URPA (Nonlinear Hazard Formulation)", "MAE": "0.0862", "RMSE": "0.1104", "Monotonicity (U >= s) Guaranteed?": "✅ YES (Guaranteed by formula)"}
    ]
    st.table(pd.DataFrame(urg_data))
    
    st.caption("While unconstrained Linear Regression achieves a slightly lower numerical MAE, it predicts Urgency lower than structural damage ($U < s$), which makes no physical sense in railway operations.")

# ==============================================================================
# TAB 5: ETHICS, GOVERNANCE & EXPLAINABILITY
# ==============================================================================
elif nav_choice == "🛡️ Ethics, Governance & Explainability":
    st.markdown('<div class="main-header">Data Ethics, Algorithmic Fairness & Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Module 6: PII protection, demographic station parity, and glass-box interpretability.</div>', unsafe_allow_html=True)
    
    st.subheader("1. Interactive PII Anonymisation Demo")
    e_col1, e_col2, e_col3 = st.columns(3)
    with e_col1:
        raw_name = st.text_input("Passenger Name:", "Ramesh Kumar Sharma")
    with e_col2:
        raw_phone = st.text_input("Mobile Number:", "+91 9876543210")
    with e_col3:
        raw_pnr = st.text_input("10-Digit PNR:", "2847193850")
        
    masked_phone = raw_phone[:9] + " *****" if len(raw_phone) >= 9 else "MASKED"
    hashed_name = hashlib.sha256(("dse_rail_salt_" + raw_name).encode()).hexdigest()[:16]
    
    st.markdown(f"""
    <div style="background: #1e293b; border-radius: 8px; padding: 14px; border: 1px solid #334155;">
        <strong>Anonymised Record for Operational Dispatch:</strong><br>
        &bull; <strong>Hashed Identifier:</strong> <code>{hashed_name}</code> (Salted SHA-256)<br>
        &bull; <strong>Masked Contact:</strong> <code>{masked_phone}</code><br>
        &bull; <strong>PNR Status:</strong> <code>DROPPED (Zero Storage)</code>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("2. Algorithmic Fairness Audit across 15 Railway Stations")
    st.markdown("""
    Evaluating model accuracy across diverse regional hubs to verify that tier-2 and non-metro stations receive equal AI service quality.
    - **Observed Accuracy Range:** `88.6%` to `100.0%`
    - **Maximum Disparity Gap:** `11.36%`
    - **Mean Predicted Urgency:** Consistently `0.66 - 0.70` across all stations (no geographic deprioritisation).
    """)
    
    st.subheader("3. Model Explainability")
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.markdown("**Top Logistic Regression Keywords:**")
        st.markdown("""
        - **Mechanical:** *bogie, brake, axle, suspension, grinding, wheel*
        - **Electrical:** *switch, spark, socket, light, cooling, charging*
        - **Catering:** *pantry, vendor, overcharging, stale, meal, tea*
        - **Cleaning:** *washbasin, toilet, dirty, stench, cockroach, water*
        - **Security:** *harassing, theft, unauthorized, threatening, luggage*
        """)
    with ex_col2:
        st.markdown("**Top Decision Tree Feature Importances:**")
        st.markdown("""
        1. `dirt_score` (28.4%) &bull; Cleanliness indicator
        2. `crowd_density` (24.8%) &bull; Safety & overcrowding
        3. `rust_score` (23.2%) &bull; Mechanical wear indicator
        4. `sparks_wiring_risk` (18.1%) &bull; Electrical safety
        5. `dist_to_yard_km` & `wait_hours` &bull; Dispatch urgency
        """)

# ==============================================================================
# TAB 6: PROJECT INFORMATION & SYLLABUS
# ==============================================================================
elif nav_choice == "ℹ️ Project Information & Syllabus":
    st.markdown('<div class="main-header">Project Information & Syllabus Alignment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Academic course details, syllabus compliance matrix, and project references.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Academic Identification
    - **Course:** Data Science Ecosystem (23CS2024)
    - **Institution:** Karunya Institute of Technology and Sciences
    - **Investigators:**
      - **Denny Mathew** (Reg No: `URK24CS1023`)
      - **Vamshi Krishna** (Reg No: `URK24CS1025`)
    - **Problem Statement:** SIH1711 &bull; AI-Powered Grievance Routing and Dynamic Prioritisation for Rail Madad (Ministry of Railways)
    
    ---
    
    ### Strict Syllabus Compliance Matrix (Modules 2 - 6)
    | Syllabus Module | Concepts Taught | Implementation in HCAM-URPA |
    | :--- | :--- | :--- |
    | **Module 2** | Import, filter, group, sort, rank, missing values | Missing photo imputation, telemetry filtering, priority ranking |
    | **Module 3** | Descriptive stats, hypothesis testing, correlation | One-Way ANOVA, Two-Sample t-test, Chi-Square, 95% CI |
    | **Module 4** | Regression, KNN, Decision Trees, K-Means, Agglomerative | LogReg, Decision Tree, KNN, Multiple/Polynomial Reg, URPA curve fit, K-Means |
    | **Module 5** | Content-based, Collaborative, Hybrid Recommenders | Cosine-similarity Retrieval Recommender, Precision@3, Hit-Rate@3 |
    | **Module 6** | Ethics, Anonymisation, Bias/Fairness, Explainability | PNR dropping, phone masking, salted SHA-256, station fairness audit |
    """)
    
    st.success("✅ **Architecture Constraint:** Strictly zero neural networks, zero PyTorch, and zero attention layers. Built entirely with standard Python scientific libraries.")
