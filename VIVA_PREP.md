# VIVA PREP: 25 High-Yield Examiner Questions & Answers
**Course:** Data Science Ecosystem (23CS2024) | **Project:** SIH1711 (Rail Madad HCAM-URPA)
**Investigators:** Denny Mathew (URK24CS1023) & Vamshi Krishna (URK24CS1025)

---

### Q1: What is novel about HCAM-URPA?
**Simple Answer:** Most AI systems treat complaints as either simple text classification or flat regression. HCAM-URPA introduces two novel, syllabus-compliant ideas:
1. **HCAM (Hybrid Confidence-Adaptive Multi-view routing):** Instead of naive concatenation, it builds 3 independent views (Text, Context, and Retrieval Recommender) and dynamically blends their probability distributions using Shannon entropy confidence ($c_v = 1 - H(p)/\ln K$).
2. **URPA (Urgency Propagation Algorithm):** Instead of black-box regression, it implements a physical hazard differential equation ($U = s + (1-s)(1-e^{-\text{hazard}})$) that guarantees Urgency can never drop below structural Severity ($U \ge s$).

---

### Q2: Why did you not use Deep Learning, PyTorch, or Neural Networks?
**Simple Answer:** In production railway edge servers (like station control room PCs), deep learning models require heavy GPU hardware, suffer from hallucinations, and act as unexplainable black boxes. Our project demonstrates that using classical Data Science techniques from Modules 2 through 6 (Logistic Regression, Decision Trees, KNN, and Recommenders) achieves **94.67% accuracy** with sub-millisecond CPU latency (0.018 ms), full explainability, and zero GPU cost.

---

### Q3: What are the three views in HCAM and what model powers each?
**Simple Answer:**
1. **Text View ($p_T$):** TF-IDF n-grams processed through multinomial Logistic Regression.
2. **Context View ($p_C$):** Train telemetry (speed, yard distance, wait hours) and 8 interpretable photo features processed through a CART Decision Tree.
3. **Retrieval View ($p_R$):** A Content-Based Recommender using Cosine-Similarity Nearest Neighbors over historical complaints.

---

### Q4: Why use adaptive weights instead of simple equal weights (1/3 each)?
**Simple Answer:** Equal weighting is naive. If a passenger writes a vague text like *"Problem in coach B2, please help"*, the Text View's probability is nearly flat (high entropy, low confidence). Simultaneously, if a photo of broken glass is attached, the Context View's Decision Tree is 95% confident. Adaptive weighting dynamically shifts weight $w_v$ to the most confident, reliable view for that specific complaint.

---

### Q5: How do you handle complaints where the passenger did not upload a photo?
**Simple Answer:** In our dataset, 35.5% of complaints lack photos. We handle this at two levels:
1. **Imputation:** Missing photo features are imputed using the training set median so the Decision Tree does not crash.
2. **Split Reliability ($\alpha_v$):** We compute the validation F1-score of each view separately for photo-present and photo-absent complaints. When a photo is missing, the Context View's reliability $\alpha_C$ drops dramatically from **0.9591** to **0.0523**. As a result, its adaptive weight $w_C$ naturally collapses toward zero without manual intervention!

---

### Q6: What does the confidence metric $c_v = 1 - H(p_v)/\ln(K)$ mean mathematically?
**Simple Answer:** $H(p_v)$ is the Shannon entropy of the 5-class probability vector.
- If the model is completely unsure ($[0.2, 0.2, 0.2, 0.2, 0.2]$), entropy is $\ln(5) \approx 1.609$, so $c_v = 1 - 1.609/1.609 = 0.0$ (Zero Confidence).
- If the model is 100% sure ($[1.0, 0, 0, 0, 0]$), entropy is $0.0$, so $c_v = 1 - 0 = 1.0$ (100% Confidence).

---

### Q7: Explain the URPA Urgency formula in plain English.
**Simple Answer:**
$$\text{hazard} = \bar{a} \cdot \exp\left(-\frac{D}{\tau}\right) + b \cdot w$$
$$U = s + (1 - s) \cdot (1 - \exp(-\text{hazard}))$$
- $s$: Base structural damage (from Multiple Linear Regression).
- $D$: Distance to the maintenance yard (km). As the train gets closer to the yard ($D \to 0$), $\exp(-D/\tau)$ spikes to 1, because the window of opportunity to repair the train before departure is closing fast!
- $w$: Passenger waiting time (hours). Urgency compounds linearly with delay ($b \cdot w$).
- $U$: Final urgency score between 0.0 and 1.0.

---

### Q8: Why does the URPA formula guarantee that Urgency $\ge$ Severity ($U \ge s$)?
**Simple Answer:**
Since distance $D \ge 0$, wait time $w \ge 0$, and fitted parameters $a_k, \tau, b > 0$, the physical hazard is strictly positive ($\text{hazard} > 0$).
Therefore, $e^{-\text{hazard}} < 1$, which makes $(1 - e^{-\text{hazard}})$ a positive fraction between 0 and 1.
Because $(1 - s)$ is also positive, the entire second term is added to $s$:
$$U = s + \text{positive term} \implies U \ge s$$
This mathematically guarantees that operational urgency can never be lower than the physical damage.

---

### Q9: How did you fit the URPA parameters without a neural network?
**Simple Answer:** We used `scipy.optimize.curve_fit` (Non-Linear Least Squares), which is a standard Module 4 optimization technique. It iteratively updates the 7 physical parameters ($a_{\text{Mech}}..a_{\text{Sec}}, \tau, b$) with lower and upper bounds to minimize the residual sum of squares against true training urgency labels.

---

### Q10: How did the fitted URPA parameters compare to ground truth?
**Simple Answer:**
- Maintenance Yard Decay $\tau$: True = 60 km, Fitted = **85.80 km** (reasonable physical scale).
- Waiting Hazard Rate $b$: True = 0.08 / hr, Fitted = **0.0669 / hr** (only 16.3% relative error!).
- Department Risk Weights $a_k$: Mechanical learned **0.952** (True 0.85), Security learned **1.813** (True 0.90), capturing that mechanical and safety issues have the highest operational urgency.

---

### Q11: How does your model handle class imbalance (33% Mechanical vs 9% Security)?
**Simple Answer:** In Module 4 classification, we used `class_weight='balanced'`. This computes weights inversely proportional to class frequencies:
$$w_k = \frac{N}{K \cdot N_k}$$
This heavily penalizes mistakes on minority Security complaints, preventing the classifier from simply guessing the majority Mechanical class.

---

### Q12: What does the p-value in your hypothesis tests mean?
**Simple Answer:** The p-value is the probability of observing our sample results purely by random chance if the Null Hypothesis ($H_0$) were true.
- In our ANOVA test on severity across departments, $p = 0.0000 < 0.05$. We reject $H_0$, concluding that departments have statistically significantly different severity profiles.
- In our paired t-test comparing HCAM against Logistic Regression across 5 CV folds, $p = 4.47 \times 10^{-5} < 0.001$, proving HCAM's performance gain is statistically significant and not a fluke.

---

### Q13: What did the Chi-Square test between Station Code and Department reveal?
**Simple Answer:** $p = 0.2842 > 0.05$. We fail to reject $H_0$. This proves that complaint departments are uniformly and independently distributed across Indian railway stations, showing no geographic bias in complaint types.

---

### Q14: How does the Retrieval View work as a Recommender System (Module 5)?
**Simple Answer:** It acts as a **Content-Based Recommender**. It projects a new complaint into a combined feature space (Text SVD 32-d + Scaled Context features) and calculates Cosine Similarity against all historical training complaints. The top 5 nearest cases recommend their historical department outcomes.

---

### Q15: How did you evaluate the Recommender System?
**Simple Answer:** Using standard Module 5 metrics:
1. **Precision@3:** Achieved **0.8489** (84.9% of top-3 recommended historical cases matched the true department).
2. **Hit-Rate@3:** Achieved **0.9410** (94.1% of queries had at least one correct department match in the top 3 recommendations).

---

### Q16: How did HCAM perform when complaint texts were ambiguous?
**Simple Answer:** When 25% of complaint texts were vague (e.g. *"Problem in coach S4, please check"*), text-only Logistic Regression collapsed to **17.14% accuracy** (worse than random guessing). In contrast, Full HCAM achieved **80.71% accuracy** on those same vague complaints by leveraging the photo features in the Context View!

---

### Q17: What was the result of the Hierarchical Routing ablation?
**Simple Answer:** The Hierarchical model achieved **76.57% accuracy**, losing to flat HCAM (94.67%). The error cascaded: when the stage-1 group classifier made a mistake between Technical and Amenities, the stage-2 classifier had no chance of recovering. Flat multi-view fusion proved superior.

---

### Q18: What is the difference between Multiple Linear Regression and Polynomial Regression in Section 6?
**Simple Answer:** Multiple Linear Regression fits a straight hyperplane: $U = \beta_0 + \sum \beta_i X_i$. Degree-2 Polynomial Regression creates interaction terms ($s^2, D^2, w^2, s \cdot D, D \cdot w$). While Linear Regression achieved lower MAE (0.0812 vs 0.0862 for URPA), it predicted $U < s$ on several samples, violating physical reality. URPA is the only model that guarantees $U \ge s$.

---

### Q19: What did the K-Means clustering analysis in Section 7 show?
**Simple Answer:** The Elbow curve flattened around $k=4$ to $k=5$, and the silhouette score peaked around 0.12. Cross-tabulating clusters with true labels showed that technical complaints (Mechanical and Electrical) naturally clustered together based on vocabulary ("bogie", "socket", "wiring"), while Cleaning complaints formed a separate tight cluster ("toilet", "stench").

---

### Q20: How did you address Data Ethics and Anonymisation (Module 6)?
**Simple Answer:**
1. **PNR Dropping:** Removed raw 10-digit PNR numbers entirely from test records.
2. **Phone Masking:** Redacted middle digits (`+91 98765 *****`).
3. **Salted Hashing:** Converted passenger names into one-way cryptographic SHA-256 hashes using a secret salt string, preventing passenger re-identification while maintaining a unique ID.

---

### Q21: Is your algorithm biased against certain railway stations?
**Simple Answer:** In Section 9, we performed a Fairness Audit across all 15 railway station codes. The accuracy ranged between **88.6% and 100.0%** (disparity gap of 11.4%). Mean predicted urgency was consistent across stations (0.66 to 0.70), demonstrating that the system does not artificially de-prioritize tier-2 stations like Gorakhpur (GKP) or Patna (PNBE).

---

### Q22: How is HCAM explainable to a railway station superintendent?
**Simple Answer:**
1. **Global Explainability:** Logistic Regression coefficients show the top words triggering each department (e.g. "washbasin" $\to$ Cleaning, "pantry" $\to$ Catering).
2. **Local Instance Explainability:** For any individual ticket, we can show the exact view probabilities ($p_T, p_C, p_R$), the entropy values, the dynamic weights ($w_T, w_C, w_R$), and the top-3 historical cases retrieved.

---

### Q23: What is the single-complaint prediction latency and throughput?
**Simple Answer:** On standard CPU, Full HCAM-URPA processes a single complaint in **0.018 milliseconds** (equivalent to **>55,000 complaints per second**). It can run on a $35 Raspberry Pi in a station master's office.

---

### Q24: What are the main limitations of this project?
**Simple Answer:**
1. **Simulated Data:** Real complaints contain phonetic Hinglish ("AC kharab hai", "chori ho gayi"), voice recordings, and slang.
2. **Parameter Recovery Coupling:** In URPA, waiting parameter $b$ has a minor offset because it is fitted alongside predicted severity $s$.
3. **Simulated Photo Features:** We used 8 numerical feature scores rather than raw camera pixels.

---

### Q25: If given real Rail Madad data tomorrow, how would you extend this pipeline?
**Simple Answer:**
1. Use an open-source multilingual tokenizer (like IndicBERT) to extract text features in 12+ Indian languages.
2. Extract the 8 photo features using a lightweight pre-trained MobileNetV3 image feature extractor.
3. Keep the HCAM adaptive entropy fusion and URPA curve-fit layer exactly as they are, as the mathematical formulation is completely domain-agnostic.
