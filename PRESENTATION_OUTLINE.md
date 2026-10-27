# Presentation Deck Outline: HCAM-URPA (SIH1711)
**Course:** Data Science Ecosystem (23CS2024) | **Syllabus:** Modules 2 to 6
**Students:** Denny Mathew (URK24CS1023) & Vamshi Krishna (URK24CS1025)

---

### SLIDE 1: Title & Team Details (Title Page)
**Title:** AI-Powered Grievance Routing and Dynamic Prioritisation for Rail Madad
**Subtitle:** A Novel Multi-View Hybrid Architecture (HCAM-URPA) Using Classical Data Science
- **Student Investigators:**
  - **URK24CS1023 — Denny Mathew**
  - **URK24CS1025 — Vamshi Krishna**
- **Department:** Department of Computer Science & Engineering
- **Institution:** Karunya Institute of Technology and Sciences
- **Course:** Data Science Ecosystem (23CS2024)
- **National Initiative:** Smart India Hackathon (SIH1711 - Ministry of Railways)

---

### SLIDE 2: Problem Statement & Real-World Challenges (Module 2 & 3)
**Title:** Rail Madad Operations: Four Practical Data Science Bottlenecks
- **Context:** Rail Madad processes thousands of grievances daily across 5 departments: Mechanical, Electrical, Catering, Cleaning, and Security.
- **The Core Challenges:**
  1. *Multi-Modal Nature:* Grievances contain unstructured text, train operational telemetry (speed, yard distance, wait hours), and optional photos.
  2. *The Missing Modality Problem:* **35.5% of complaints lack photos**. Naive models crash or inject zero-vector noise.
  3. *Class Imbalance:* Mechanical (~33%) and Electrical (~24%) drastically outnumber minority safety alarms (Security ~10%).
  4. *Severity vs. Urgency Gap:* Physical damage (Severity) is static, but operational urgency is dynamic based on train movement toward a maintenance depot.
- **Figure to Display:** `eda_department_dist.png`
  *(Shows the 3:1 department imbalance and differing photo attachment rates across departments)*

---

### SLIDE 3: What Is the New Algorithm? (HCAM-URPA Formulation)
**Title:** The Novel Algorithm: HCAM-URPA (Syllabus-Compliant Innovation)
- **Why Standard Approaches Fail:**
  - *Text Classifiers (TF-IDF alone):* Collapse to **17.14% accuracy** when passengers write vague complaints (*"Problem in coach B2, please check"*).
  - *Naive Concatenation:* Assumes all features are always present, failing when photos are absent.
  - *Standard Linear Regressors:* Can predict Urgency lower than Severity ($U < s$), which violates physical reality.
- **The Two Novel Pillars of HCAM-URPA:**
  1. **HCAM (Hybrid Confidence-Adaptive Multi-view routing):**
     - Three views: Text View ($p_T$, LogReg), Context View ($p_C$, Decision Tree on telemetry + 8 photo features), and Retrieval View ($p_R$, Content-Based Recommender).
     - Fuses probability vectors using Shannon entropy confidence: $w_v \propto \alpha_v \cdot (1 - H(p_v)/\ln 5)$.
     - Dynamically collapses Context View weight when photos are missing ($\alpha_C^{\text{photo}} = 0.959 \to \alpha_C^{\text{no\_photo}} = 0.052$).
  2. **URPA (Urgency Propagation Algorithm):**
     - Physics-inspired differential hazard equation fitted via `scipy.optimize.curve_fit`:
       $$\text{hazard} = \bar{a} \cdot \exp\left(-\frac{D}{\tau}\right) + b \cdot w, \quad U = s + (1 - s) \cdot (1 - \exp(-\text{hazard}))$$
     - Mathematically guarantees operational monotonicity ($U \ge s$) and recovers interpretable physical parameters ($a_k, \tau, b$).

---

### SLIDE 4: Statistical Inference & Hypothesis Testing (Module 3)
**Title:** Grounding the Problem with Formal Statistical Testing
- **Test 1: One-Way ANOVA on Severity across Departments:**
  - $H_0$: Mean severity is identical across all 5 departments.
  - Result: $F = 926.83, p = 0.0000 \implies$ **Reject $H_0$**. Proves distinct physical damage profiles.
- **Test 2: Two-Sample t-Test on Urgency by Photo Presence:**
  - $H_0$: Urgency is identical with vs without photo attachment.
  - Result: $t = 0.957, p = 0.3385 \implies$ **Fail to Reject $H_0$**. Proves urgency is driven by operational yard proximity and wait hours, not photo upload alone.
- **Test 3: Chi-Square Test of Independence (Department vs Station):**
  - Result: $\chi^2 = 59.88, p = 0.4433 \implies$ **Fail to Reject $H_0$**. Proves complaint departments are uniformly distributed across Indian railway stations.
- **Test 4: 95% Confidence Interval for Mean Urgency:**
  - True population mean urgency lies in $[0.6766, 0.6869]$.
- **Figure to Display:** `eda_urgency_vs_dist_wait.png`
  *(Shows exponential spike in urgency near maintenance depots and wait-time amplification)*

---

### SLIDE 5: Empirical Results & Ablation Analysis (Modules 4 & 5)
**Title:** Model Evaluation on Held-Out Test Set ($N = 525$)
- **Controlled Ablation Breakdown:**
  - *1. Text Only (LogReg):* 77.90% Accuracy | 0.7698 Macro-F1 (**drops to 17.14% on ambiguous text!**)
  - *2. Text + Context (Equal Wt):* 92.19% Accuracy | 0.9039 Macro-F1
  - *3. Text + Context + Retrieval (Fixed Wt):* 94.29% Accuracy | 0.9317 Macro-F1
  - *4. Full HCAM (Adaptive Wt):* **94.67% Accuracy | 0.9376 Macro-F1 | 80.71% on ambiguous text!**
  - *5. Hierarchical Routing Ablation:* 76.57% Accuracy (cascading error in stage-1 grouping).
- **Statistical Significance (5-Fold Paired t-Test):**
  - HCAM (91.17% mean CV accuracy) vs LogReg (79.49%): $t = 19.049, p = 4.47 \times 10^{-5}$ (**$p < 0.001$, statistically significant**).
- **Recommender Evaluation (Module 5):**
  - Retrieval View achieves **Precision@3 = 0.8489** and **Hit-Rate@3 = 94.10%**.
- **Figure to Display:** Normalized Confusion Matrix from Section 5 of Notebook.

---

### SLIDE 6: URPA Urgency Prediction & Station Priority Queue (Module 4)
**Title:** Decoupling Structural Damage from Real-Time Operational Urgency
- **Regression Benchmark on Urgency Prediction:**
  - *Multiple Linear Regression:* MAE = 0.0812 (BUT violates physics: predicts $U < s$ on multiple test cases).
  - *Polynomial Regression (Deg 2):* MAE = 0.0968.
  - *URPA Physical Formulation:* **MAE = 0.0862 | RMSE = 0.1104 | Monotonicity Guaranteed ($U \ge s$).**
- **Physical Parameter Recovery (`curve_fit`):**
  - Yard Proximity Decay $\tau$: True = 60 km $\to$ Fitted = **85.80 km**.
  - Wait Time Hazard Rate $b$: True = 0.08 hr⁻¹ $\to$ Fitted = **0.0669 hr⁻¹** (**only 16.3% error!**).
  - Department Risk: Security learned **1.813**, Mechanical learned **0.952**, capturing that safety and technical faults take highest priority.
- **Actionable Station Priority Queue:**
  - Dynamically ranks incoming tickets to surface top-10 urgent cases for immediate station dispatch.
- **Figure to Display:** Priority Queue Table from Section 6 of Notebook.

---

### SLIDE 7: Ethics, Explainability, Latency & Limitations (Module 6)
**Title:** Governance, Operational Explainability & Edge Latency
- **Ethics & PII Protection:**
  - PNR numbers dropped; mobile numbers masked (`+91 98765 *****`); passenger names hashed via salted SHA-256.
  - Fairness Audit: Station accuracy ranges between 88.6% and 100.0% (low disparity gap of 11.4%).
- **Explainability & Transparency:**
  - Logistic Regression top keywords (e.g. "washbasin", "overflowing" $\to$ Cleaning).
  - Decision Tree feature importances (`dirt_score` 28.4%, `crowd_density` 24.8%, `rust_score` 23.2%).
  - Step-by-step diagnostic walkthrough showing view probabilities, entropy confidence, and weights for any individual ticket.
- **Edge Deployment & Complexity:**
  - Single-sample prediction latency: **0.018 ms / complaint** (**>55,000 complaints / second on CPU**).
  - Deployable on a $35 Raspberry Pi in any railway station master's office.
- **Limitations Acknowledged:** Simulated vocabulary dialect limits; URPA wait hazard $b$ is coupled with the linear severity regressor.
