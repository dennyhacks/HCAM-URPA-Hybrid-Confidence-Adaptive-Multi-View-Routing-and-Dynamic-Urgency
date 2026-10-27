# Slide Presentation Deck: HCAM-URPA Multi-Modal Pipeline
**Course:** Data Science Ecosystem (23CS2024) | **Problem Statement:** SIH1711 (Ministry of Railways)

---

### SLIDE 1: Title & Team Details (Title Slide)
**Title:** AI-Powered Complaint Routing and Dynamic Prioritisation for Rail Madad
**Subtitle:** A Novel Multi-Modal Architecture (HCAM-URPA) for Smart Railway Operations

**Student Investigators:**
- **URK24CS1023 — Denny Mathew**
- **URK24CS1025 — Vamshi Krishna**

**Department:** Department of Computer Science & Engineering
**Institution:** Karunya Institute of Technology and Sciences
**Course:** Data Science Ecosystem (23CS2024)
**National Initiative:** Smart India Hackathon (SIH1711 - Ministry of Railways)

---

### SLIDE 2: Problem Statement & Real-World Challenges
**Title:** Rail Madad Grievance Management: Operational Bottlenecks
- **Context:** Rail Madad receives tens of thousands of passenger complaints daily across five key departments: Mechanical, Electrical, Catering, Cleaning, and Security.
- **The Core Data Science Challenges:**
  1. **Multi-Modal Data:** Grievances contain unstructured text, train operational telemetry (speed, yard distance, wait hours), and optional photos.
  2. **The Missing Modality Problem:** ~29% of complaints have no photo attachment. Standard deep learning models either crash or inject zero-vector noise.
  3. **Heavy Class Imbalance:** Technical complaints (Mechanical ~33%, Electrical ~25%) drastically outnumber minority safety emergencies (Security ~9%).
  4. **The Severity vs. Urgency Gap:** A broken window latch has fixed structural damage (Severity), but its operational urgency changes dynamically as the train approaches a depot or passenger wait time increases.
- **Figure to Display:** `eda_department_dist.png`
  *(Shows the 3:1 departmental class imbalance and differing photo attachment rates across classes)*

---

### SLIDE 3: What Is the New Algorithm? (HCAM-URPA Innovation)
**Title:** The Novel Algorithm: HCAM-URPA
- **Why Existing Baselines Fail:**
  - *Standard Text Classifiers (TF-IDF, BERT):* Completely fail when passenger text is vague (e.g. *"Problem in coach B2, please check"*).
  - *Naive Concatenation (Concat):* Treat all inputs equally without masking, corrupting representations when photos are missing.
  - *Black-Box Regressors:* Can predict Urgency lower than Severity ($U < s$), violating physical operational logic.
- **The Three Novel Pillars of HCAM-URPA:**
  1. **Hierarchical Cross-Attention Mechanism (HCAM):** Operational metadata queries Text + Vision tokens. Includes a boolean key-padding mask and learnable temperature softmax gate that dynamically collapses vision weight to $0.0$ when photos are absent.
  2. **Hierarchical Routing Head:** Administrative 2-level probability tree ($P(\text{dept}) = P(\text{group}) \times P(\text{dept} \mid \text{group})$) that prevents majority technical complaints from suppressing minority security alarms.
  3. **Dynamic Urgency Propagation (URPA):** A differentiable forward layer enforcing a physics-inspired hazard law:
     $$\text{hazard} = \bar{a} \cdot \exp\left(-\frac{D}{\tau}\right) + b \cdot w$$
     $$U = s + (1 - s) \cdot \left(1 - \exp(-\text{hazard})\right)$$
     Guarantees physical monotonicity ($U \ge s$) while extracting interpretable risk parameters ($a_k, \tau, b$).
- **Figure to Display:** Architecture Schematic / `training_curves.png`
  *(Highlights the tripartite fusion and joint convergence across routing and urgency heads)*

---

### SLIDE 4: Cross-Attention Fusion & Missing Modality Handling
**Title:** HCAM Fusion Gate: Robustness to Missing Visual Data
- **Cross-Attention Formulation:**
  - Query: $Q = h_{\text{metadata}} \in \mathbb{R}^{B \times 1 \times 64}$ (operational context guides attention).
  - Key & Value: $K, V = [h_{\text{text}}, h_{\text{vision}}] \in \mathbb{R}^{B \times 2 \times 64}$.
  - Context vector: $c = \text{MHA}(Q, K, V) \in \mathbb{R}^{64}$.
- **Dual-Level Missing Modality Defense:**
  - *Attention Level:* `key_padding_mask` sets vision position to $-\infty$, preventing attention to empty vectors.
  - *Gating Level:* Pre-softmax vision logit is suppressed to $-10^9 \implies g_v \to 0$.
  - Fused representation: $z = \text{LayerNorm}(g_t h_t + g_v h_v + g_m h_m + c)$ where $g_t + g_m = 1$.
- **Explainability:** Softmax gate weights ($g_t, g_v, g_m$) offer railway officers a transparent breakdown of which modality influenced the routing decision.
- **Figure to Display:** `gate_weights.png`
  *(Shows learned convex modality weights across departments, confirming adaptive weighting)*

---

### SLIDE 5: URPA Physics Layer & Monotonicity ($U \ge s$)
**Title:** Grounding Urgency in Railway Physics: URPA Formulation
- **The Physical Dynamics of a Moving Train:**
  - **Yard Proximity ($D$):** As the train approaches a maintenance depot ($D \to 0$), the window of opportunity to repair the rake is closing $\implies \exp(-D/\tau) \to 1$.
  - **Passenger Wait Time ($w$):** Grievance urgency compounds linearly with delay ($b \cdot w$).
  - **Department Risk Prior ($\bar{a}$):** Dynamic hazard expectation weighted by predicted department probabilities: $\bar{a} = \sum_{k=1}^5 P(k) a_k$.
- **Mathematical Invariant ($U \ge s$):**
  - Because $D \ge 0, w \ge 0$, and softplus-constrained parameters $a_k, \tau, b > 0$, calculated $\text{hazard} > 0$.
  - Therefore, $1 - \exp(-\text{hazard}) \in (0, 1)$, ensuring $U \ge s$ strictly under all operational states.
- **Parameter Recovery:** Learns physical values: $a_{\text{Mechanical}} \approx 0.61$, $a_{\text{Security}} \approx 0.75$, $\tau \approx 44$ km (vs true 60 km).
- **Figure to Display:** `eda_urgency_vs_dist_wait.png`
  *(Visualizes the exponential spike in urgency near maintenance depots and wait-time amplification)*

---

### SLIDE 6: Experimental Evaluation & Ablation Analysis
**Title:** Empirical Validation on Held-Out Test Set ($N = 525$)
- **Controlled Comparison (With 25.9% Ambiguous Grievances):**
  - **TF-IDF + Logistic Regression:** **0.7845 Macro-F1** (fails when complaint text lacks department keywords).
  - **Neural Text-Only (MLP):** **0.7678 Macro-F1** (similar drop on ambiguous text).
  - **Neural Concat Baseline:** **0.9138 Macro-F1** | 0.0839 Urgency MAE.
  - **Full HCAM-URPA:** **0.9136 Macro-F1** | 0.1045 Urgency MAE.
- **Key Empirical Takeaway:**
  - Multi-modal fusion provides a **+12.9% Macro-F1 gain** over text-only models by using visual cues to resolve ambiguous texts.
  - Full HCAM-URPA matches naive concatenation on routing, while offering full physical interpretability and monotonicity guarantees.
- **Multi-Seed Robustness (Seeds 42, 7, 123):**
  - Full HCAM-URPA achieves **$0.9077 \pm 0.0084$ Macro-F1** and **$0.1039 \pm 0.0011$ Urgency MAE**, demonstrating tight convergence across random weight initializations.
- **Figure to Display:** `confusion_matrix.png`
  *(Normalized confusion matrix showing diagonal classification accuracy across all 5 departments)*

---

### SLIDE 7: Edge Deployment, Limitations & Future Roadmap
**Title:** Computational Complexity, Limitations & Real-Data Roadmap
- **Ultra-Lightweight Edge Benchmarking (CPU):**
  - Trainable Parameters: **44,563** (< 180 KB memory footprint).
  - Single-Sample Latency (Batch=1): **0.151 ms / complaint** (p95: 0.166 ms).
  - Batched Throughput: **103,582 complaints / second** on CPU.
  - Deployable directly on low-cost station-level hardware (e.g. Raspberry Pi) without GPU requirements.
- **Limitations Acknowledged:**
  1. *Synthetic Vocabulary:* Lacks real-world phonetic Hindi-English code-mixing (*"AC thik se nahi chal raha"*).
  2. *Joint Parameter Recovery:* Severity residual errors create slight covariance on wait hazard $b$ (learned 0.23 vs true 0.08).
  3. *Concat vs URPA MAE:* Unconstrained Concat fits continuous curves with slightly lower MAE (0.084 vs 0.104), but cannot guarantee $U \ge s$ or extract physical parameters.
- **Future Roadmap with Real Data:**
  - Integrate multilingual IndicBERT / IndicTrans for 12+ Indian regional languages.
  - Connect live GPS feeds from Indian Railways' National Train Enquiry System (NTES) API.
