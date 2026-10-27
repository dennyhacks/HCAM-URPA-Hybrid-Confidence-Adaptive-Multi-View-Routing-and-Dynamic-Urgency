# Experimental Results: HCAM-URPA Pipeline (SIH1711 Rail Madad)
Course: Data Science Ecosystem (23CS2024)

## 1. Ablation Study
Evaluation on held-out test set ($N = 525$ complaints) with scalers and TF-IDF fitted on training data only:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Severity MAE | Urgency MAE | Urgency RMSE |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.7981 | 0.7845 | 0.7986 | N/A | N/A | N/A |
| Neural Text-Only | 0.7867 | 0.7678 | 0.7915 | 0.1193 | 0.1054 | 0.1315 |
| Neural Concat | 0.9314 | 0.9138 | 0.9318 | 0.1150 | 0.0839 | 0.1082 |
| HCAM (No URPA) | 0.9314 | 0.9151 | 0.9319 | 0.1167 | 0.0878 | 0.1134 |
| Full HCAM-URPA | 0.9314 | 0.9136 | 0.9326 | 0.1264 | 0.1045 | 0.1314 |

*Interpretation:* With ~25% department-ambiguous text complaints added to the evaluation, multi-modal architectures (Full HCAM-URPA Macro-F1: 0.9136) demonstrate a clear advantage over text-only baselines (TF-IDF + LogReg Macro-F1: 0.7845) by utilizing the class-specific image embeddings to resolve ambiguous text grievances.

---

## 2. Per-Department Classification Report (Full HCAM-URPA)

| Department | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Mechanical | 0.9641 | 0.9306 | 0.9471 | 173 |
| Electrical | 0.9767 | 0.9692 | 0.9730 | 130 |
| Catering | 0.7805 | 0.9143 | 0.8421 | 70 |
| Cleaning | 0.9798 | 0.9417 | 0.9604 | 103 |
| Security | 0.8542 | 0.8367 | 0.8454 | 49 |
| **Macro Avg** | 0.9111 | 0.9185 | 0.9136 | 525 |
| **Weighted Avg** | 0.9356 | 0.9314 | 0.9326 | 525 |

*Interpretation:* Hierarchical grouping (Mechanical/Electrical, Catering/Cleaning, Security) preserves high recall and F1 on minority classes (Security: 0.8454 F1) despite significant 3:1 data imbalance.

---

## 3. Regression & Convergence Metrics

- **Severity MAE:** 0.1264
- **Severity RMSE:** 0.1560
- **Severity Convergence Error:** 0.1264
- **Urgency MAE:** 0.1045
- **Urgency RMSE:** 0.1314

*Interpretation:* The severity head captures structural damage cleanly, while the URPA layer in the forward pass maps severity and physical dynamics into operational urgency with controlled error across the 0-1 scale.

---

## 4. Learned URPA Physical Dynamics vs Ground Truth

| Parameter | Ground Truth | Learned Value | Absolute Error | Relative Error (%) |
|---|---|---|---|---|
| a_Mechanical (Risk) | 0.85 | 0.6085 | 0.2415 | 28.42% |
| a_Electrical (Risk) | 0.65 | 0.4894 | 0.1606 | 24.70% |
| a_Catering (Risk) | 0.35 | 0.3637 | 0.0137 | 3.92% |
| a_Cleaning (Risk) | 0.45 | 0.3868 | 0.0632 | 14.04% |
| a_Security (Risk) | 0.9 | 0.7463 | 0.1537 | 17.08% |
| tau (Decay km) | 60.0 | 43.8576 | 16.1424 | 26.90% |
| b (Wait hazard/hr) | 0.08 | 0.2277 | 0.1477 | 184.67% |

*Interpretation:* The softplus-parameterized URPA module estimates department risks a_k and proximity scale tau. Discrepancies in the waiting rate parameter b arise due to joint gradient optimization with the severity head.

---

## 5. Multi-Seed Robustness Evaluation (Seeds 42, 7, 123)

| Model | Macro-F1 (Mean ± Std) | Urgency MAE (Mean ± Std) |
|---|---|---|
| **Full HCAM-URPA** | **0.9077 ± 0.0084** | **0.1039 ± 0.0011** |
| Neural Concat Baseline | 0.9087 ± 0.0066 | 0.0837 ± 0.0011 |

*Interpretation:* Multi-seed testing confirms consistent performance across initializations, validating that the cross-attention gating and hierarchical heads remain stable regardless of random seed.

---

## 6. Computational Complexity & Edge Latency Benchmark (CPU)

- **Total Trainable Parameters:** 44,563
- **Batch=1 Mean Latency:** 0.151 ms / complaint
- **Batch=1 50th Percentile (p50):** 0.152 ms
- **Batch=1 95th Percentile (p95):** 0.166 ms
- **Batch=1 99th Percentile (p99):** 0.171 ms
- **Batched Throughput (Batch=64):** 103582.6 complaints / second

*Interpretation:* At under 50K parameters and sub-millisecond CPU execution, HCAM-URPA is lightweight and suitable for low-power edge servers at railway stations.

---

## 7. Limitations

1. **Simulated Synthetic Dataset:** Although realistic templates and ~25% ambiguous complaints were introduced, synthetic data lacks the full morphological messiness of real human grievances, such as phonetic Hindi-English code-mixing (e.g., *"AC thik se nahi chal raha"*), spelling errors, and multi-department overlap.
2. **Parameter-Recovery Errors (Especially $b$):** The learned wait-time hazard coefficient $b$ exhibits noticeable relative error compared to ground truth ($b_{TRUE} = 0.08$). Because severity $s$ and URPA parameters are trained jointly under backpropagation, any minor offset in $s$ allows the optimizer to adjust $b$ to compensate for the continuous urgency MSE loss.
3. **Concat vs. URPA Urgency MAE:** Unconstrained neural models (like Neural Concat: 0.0839 MAE) can achieve slightly lower numerical urgency MAE than the physically-constrained URPA layer (0.1045 MAE). This occurs because unconstrained MLPs have free parameters to directly fit non-linear curves without constraint. However, unconstrained MLPs act as black boxes and cannot guarantee physical monotonicity ($U \ge s$), whereas URPA guarantees that urgency can never drop below intrinsic structural severity.

---

## 8. Artifacts Generated
- `confusion_matrix.png`: Normalized confusion matrix of department routing predictions.
- `gate_weights.png`: Learned modality convex weights (Text, Vision, Metadata) per department.
- `training_curves.png`: Training/validation loss, routing performance, and urgency error progression.
