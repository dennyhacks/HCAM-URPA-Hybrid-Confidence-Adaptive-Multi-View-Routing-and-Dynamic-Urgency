# HCAM-URPA: AI-Powered Grievance Routing & Dynamic Prioritisation for Rail Madad

> **Smart India Hackathon (SIH1711) — Ministry of Railways**
> Course: Data Science Ecosystem (23CS2024) | Karunya Institute of Technology and Sciences

## Authors

| Name | Registration No. |
|:---|:---|
| **Denny Mathew** | URK24CS1023 |
| **Vamshi Krishna** | URK24CS1025 |

## What Is HCAM-URPA?

A novel, syllabus-compliant algorithm for automated railway complaint routing and urgency scoring. Built using **only classical Data Science** (no neural networks, no PyTorch).

**HCAM** (Hybrid Confidence-Adaptive Multi-View Routing) fuses three independent classifiers — Text (TF-IDF + Logistic Regression), Context (Decision Tree on telemetry + photo features), and Retrieval (Cosine-Similarity KNN Recommender) — using Shannon entropy confidence weighting. When a view is uncertain, its weight automatically drops.

**URPA** (Urgency Propagation Algorithm) predicts operational dispatch urgency using a physics-inspired hazard equation fitted via `scipy.optimize.curve_fit`, mathematically guaranteeing that urgency can never fall below structural severity (U ≥ s).

## Key Results

| Metric | Value |
|:---|:---|
| HCAM Routing Accuracy (Test Set) | **94.67%** |
| Macro-F1 | **0.9376** |
| Accuracy on Ambiguous Text | **80.71%** (vs 17.14% for text-only) |
| URPA Urgency MAE | **0.0862** (with U ≥ s guaranteed) |
| Paired t-Test (HCAM vs LogReg) | t = 19.05, **p < 0.001** |
| Retrieval Hit-Rate@3 | **94.10%** |
| CPU Inference Latency | **0.018 ms / complaint** |

## Repository Structure

```
├── hcam_urpa_dse.ipynb       # Main Jupyter Notebook (10 sections, Modules 2-6)
├── rail_madad_sim.csv        # Simulated dataset (3,500 complaints, 23 columns)
├── rail_data.py              # Dataset generator & EDA script
├── app.py                    # Interactive Streamlit dashboard
├── presentation.html         # Full-screen HTML slide deck (7 slides)
├── requirements.txt          # Python dependencies
├── confusion_matrix.png      # HCAM normalised confusion matrix
├── eda_department_dist.png   # Department distribution & photo rates
└── eda_urgency_vs_dist_wait.png  # Urgency vs yard distance scatter
```

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook hcam_urpa_dse.ipynb

# 4. Or launch the Streamlit dashboard
streamlit run app.py
```

**Google Colab:** Upload `hcam_urpa_dse.ipynb` and `rail_madad_sim.csv`, then click **Runtime → Run all**.

## Syllabus Coverage

| Module | Topics Used |
|:---|:---|
| **Module 2** | Data import, filtering, grouping, sorting, ranking, missing value imputation |
| **Module 3** | ANOVA, t-test, Chi-Square, 95% CI, Shannon entropy, descriptive statistics |
| **Module 4** | Logistic Regression, Decision Trees, KNN, K-Means, Agglomerative Clustering, Linear/Polynomial Regression, curve_fit |
| **Module 5** | Content-based cosine-similarity recommender, Precision@3, Hit-Rate@3 |
| **Module 6** | PII anonymisation (SHA-256 hashing, phone masking), station fairness audit, model explainability |

## Tech Stack

Python 3.12 · pandas · numpy · scikit-learn · scipy · matplotlib · seaborn · Streamlit

---

*Strictly zero neural networks. Strictly zero deep learning. Classical Data Science only.*
