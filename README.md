# HCAM-URPA — Hybrid Confidence-Adaptive Multi-View Routing with Dynamic Urgency Propagation

> An AI-driven complaint routing and prioritisation system built for Indian Railways' **Rail Madad** platform (SIH1711).

---

## What this project is about

Every day, Rail Madad receives thousands of passenger grievances — broken coach windows, dead charging sockets, food quality issues, security threats. These complaints arrive as messy text descriptions, sometimes with photos attached, sometimes without, along with real-time train telemetry like speed, distance to the nearest maintenance yard, and how long the passenger has been waiting.

The challenge is straightforward but tricky: **route each complaint to the right department** (Mechanical, Electrical, Catering, Cleaning, or Security) and **figure out how urgently it actually needs attention** — not just how severe the damage is, but how urgent action becomes given the train's current operational state.

That's what HCAM-URPA does. It fuses text, image, and operational metadata through a cross-attention gating mechanism, routes complaints hierarchically (mirroring how Indian Railways actually structures its divisions), and computes a physics-informed urgency score that respects a simple real-world constraint: urgency should never drop below intrinsic severity.

---

## Why not just use a standard classifier?

A few reasons came up during development:

- **Missing photos are the norm, not the exception.** About 35% of complaints have no image attached. Most multi-modal models either crash or quietly inject noise from a zero vector when a modality is absent. HCAM's gating layer explicitly masks out the vision channel when there's no photo, so it genuinely ignores what isn't there.

- **Text alone is ambiguous.** A complaint saying *"coach is very dirty and smelly"* could be Catering or Cleaning. When a photo is available, it resolves the ambiguity — and the model learns to lean on vision more heavily for exactly these departments (the Cleaning department's vision gate weight is 0.69, the highest of any department).

- **Severity isn't the same as urgency.** A cracked berth frame (high severity) on a train 5 km from a maintenance yard is less operationally urgent than a moderately damaged coach 150 km from any depot with a passenger who's been waiting 4 hours. The URPA layer captures this through a physics-inspired formula that accounts for department-specific risk, yard proximity decay, and wait-time hazard rate — all learned end-to-end through backpropagation.

---

## Architecture at a glance

```
                    ┌──────────────────────────────────────────┐
                    │         Raw Complaint Input               │
                    │  Text + Image (optional) + Metadata       │
                    └──────────┬──────────┬──────────┬──────────┘
                               │          │          │
                    ┌──────────▼──┐ ┌─────▼────┐ ┌──▼──────────┐
                    │ TF-IDF → SVD│ │ 8 Photo  │ │ Speed, Dist,│
                    │   64-dim    │ │ Features │ │ Wait Hours  │
                    │  Text Proj  │ │ 128→64   │ │   3→64      │
                    └──────────┬──┘ └─────┬────┘ └──┬──────────┘
                               │          │         │
                    ┌──────────▼──────────▼─────────▼──────────┐
                    │    Cross-Attention Fusion Gate (HCAM)      │
                    │  Metadata queries Text & Vision tokens     │
                    │  Softmax gating → convex combination       │
                    │  (Vision masked if no photo attached)      │
                    └──────────────────┬───────────────────────-─┘
                                       │
                          ┌────────────▼────────────┐
                          │  Hierarchical Routing    │
                          │  Group → Department      │
                          │  Technical | Housekeep   │
                          │  | Safety               │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Severity Head (MSE)     │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  URPA Layer (Urgency)    │
                          │  U = s + (1-s)(1-e^-h)   │
                          │  h = a_k · f(dist, wait) │
                          └─────────────────────────┘
```

---

## Results

Evaluated on a held-out test set of 525 complaints:

| Model | Accuracy | Macro-F1 | Severity MAE | Urgency MAE |
|---|---|---|---|---|
| TF-IDF + Logistic Regression (text-only baseline) | 79.8% | 0.784 | — | — |
| Neural Concat (text + metadata, no attention) | 93.1% | 0.914 | 0.115 | 0.084 |
| HCAM without URPA | 93.1% | 0.915 | 0.117 | 0.088 |
| **Full HCAM-URPA** | **93.1%** | **0.914** | **0.126** | **0.105** |

A few things worth noting:

- Multi-modal models blow past the text-only baseline by ~13 percentage points in accuracy. The image features genuinely help disambiguate department assignment.
- The unconstrained Neural Concat model gets slightly lower urgency MAE (0.084 vs 0.105) because it has free parameters to fit any curve. But it can't guarantee that urgency ≥ severity — it's a black box. HCAM-URPA trades a tiny bit of numerical error for a physically meaningful, interpretable urgency score that always respects monotonicity.
- Multi-seed evaluation (seeds 42, 7, 123) gives Macro-F1 of **0.908 ± 0.008**, confirming the model is stable across initialisations.

### Per-Department Breakdown

| Department | Precision | Recall | F1 |
|---|---|---|---|
| Mechanical | 0.964 | 0.931 | 0.947 |
| Electrical | 0.977 | 0.969 | 0.973 |
| Catering | 0.781 | 0.914 | 0.842 |
| Cleaning | 0.980 | 0.942 | 0.960 |
| Security | 0.854 | 0.837 | 0.845 |

### Confusion Matrix

<p align="center">
  <img src="confusion_matrix.png" alt="Normalized confusion matrix across 5 departments" width="500"/>
</p>

### Training Curves

<p align="center">
  <img src="training_curves.png" alt="Loss convergence, routing accuracy, and urgency MAE over 25 epochs" width="800"/>
</p>

### Learned Modality Gate Weights

<p align="center">
  <img src="gate_weights.png" alt="How much each department relies on text vs vision vs metadata" width="700"/>
</p>

The gate weight plot is particularly interesting — notice how the Cleaning department leans heavily on vision (0.69), which makes sense because distinguishing "dirty coach" complaints from "bad food" complaints is much easier from a photo than from text alone.

---

## Computational Profile

The whole model sits at **44,563 trainable parameters** and runs comfortably on CPU:

- Single-complaint latency: **~0.15 ms** (p95: 0.17 ms)
- Batched throughput (batch=64): **~103K complaints/second**

This is lightweight enough to deploy on edge servers at railway stations without needing any GPU hardware.

---

## Repository Structure

```
.
├── hcam_urpa.py             # Core PyTorch model, training loop, evaluation, and ablation suite
├── rail_data.py             # Data synthesis and EDA (generates rail_madad_sim.csv)
├── app.py                   # Streamlit dashboard for interactive demo
├── hcam_urpa_dse.ipynb      # Full Jupyter notebook walkthrough (end-to-end)
├── build_and_run_notebook.py # Script to programmatically build and execute the notebook
├── rail_madad_sim.csv       # Simulated dataset (3,500 complaints)
├── requirements.txt         # Python dependencies
├── RESULTS.md               # Detailed experimental results and ablation tables
├── LEARNING_GUIDE.md        # Concept glossary mapping techniques to course syllabus
├── PRESENTATION_OUTLINE.md  # Slide deck structure and talking points
├── PRESENTATION_SCRIPT.md   # Detailed presentation narration script
├── PRESENTATION_SLIDES.md   # Condensed slide content
├── VIVA_PREP.md             # Anticipated viva questions and answers
├── presentation.html        # Rendered HTML presentation
├── confusion_matrix.png     # Normalized confusion matrix visualisation
├── training_curves.png      # Loss and metric progression over epochs
├── gate_weights.png         # Learned modality fusion weights per department
├── eda_department_dist.png  # Department distribution from EDA
└── eda_urgency_vs_dist_wait.png  # Urgency heatmap vs distance and wait time
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/dennyhacks/HCAM-URPA-Hybrid-Confidence-Adaptive-Multi-View-Routing-and-Dynamic-Urgency.git
cd HCAM-URPA-Hybrid-Confidence-Adaptive-Multi-View-Routing-and-Dynamic-Urgency

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
# Generate the synthetic dataset and run EDA
python rail_data.py

# Train the model and run the complete ablation study
python hcam_urpa.py

# Launch the interactive Streamlit dashboard
streamlit run app.py
```

### Jupyter Notebook

The notebook `hcam_urpa_dse.ipynb` contains the entire pipeline as a self-contained walkthrough — data generation, preprocessing, model training, evaluation, and visualisation — all in one place.

---

## The URPA Formula

At its core, the urgency propagation follows:

```
U = s + (1 − s) · (1 − exp(−h))
```

where:
- `s` is the predicted severity (structural damage score, 0 to 1)
- `h` is the operational hazard, computed as `a_k · exp(−dist / τ) + b · wait`
- `a_k` is a learned department-specific risk coefficient
- `τ` is the yard proximity decay constant (~60 km ground truth)
- `b` is the wait-time hazard rate

This guarantees `U ≥ s` by construction — urgency can only go up from severity, never down. The parameters `a_k`, `τ`, and `b` are all learned end-to-end through backpropagation with softplus constraints to keep them positive.

---

## Known Limitations

1. **Synthetic data.** The dataset is simulated with realistic templates and ~25% deliberately ambiguous complaints, but it doesn't capture the full messiness of real grievances — Hindi-English code-mixing, spelling errors, multi-department overlap, etc.

2. **Wait-time parameter recovery.** The learned wait hazard rate `b` shows significant relative error (~185%) compared to ground truth. This happens because `b` and the severity head are jointly optimised, so small offsets in predicted severity let the optimiser compensate by adjusting `b`.

3. **Unconstrained models can fit urgency slightly better numerically.** The Neural Concat baseline achieves 0.084 urgency MAE vs HCAM-URPA's 0.105. The trade-off is interpretability and guaranteed monotonicity — which matters more in a safety-critical railway operations context.

---

## Authors

- **Denny Mathew** — URK24CS1023
- **Vamshi Krishna** — URK24CS1025

Department of Computer Science & Engineering, Karunya Institute of Technology and Sciences

Course: Data Science Ecosystem (23CS2024)  
Problem Statement: SIH1711 — AI-Powered Complaint Routing and Prioritisation for Rail Madad (Ministry of Railways)

---

## License

This project was built as an academic submission. Feel free to use it as a reference or build on top of it.
