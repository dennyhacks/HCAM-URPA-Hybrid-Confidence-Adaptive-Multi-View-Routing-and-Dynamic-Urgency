# 🎤 CELL-BY-CELL PRESENTATION SCRIPT
**How to present `hcam_urpa_dse.ipynb` to your professor in Google Colab**

> **Before you start:** Open Google Colab with the notebook loaded. Make sure `rail_madad_sim.csv` is uploaded. Click **Runtime → Run all** so all outputs are already visible. Then scroll back to the top. You will walk through the cells one by one while reading the script below.

---

## Cell 0 — Title (Markdown)
**What is on screen:** The title of the project and your names.

**Say this:**
> "Good morning Sir/Ma'am. I am Denny Mathew, registration number URK24CS1023, and this project is in collaboration with Vamshi Krishna, URK24CS1025. Our project tackles Smart India Hackathon problem SIH1711: AI-Powered Complaint Routing and Prioritisation for Rail Madad, the Ministry of Railways grievance portal. We built a novel algorithm called HCAM-URPA using only techniques from our DSE syllabus — Modules 2 through 6. No neural networks, no PyTorch, no deep learning."

---

## Cell 1 — Section 1 Header (Markdown)
**What is on screen:** Section 1 title — Data Import, Filtering, Grouping, Sorting, Ranking & Imputation.

**Say this:**
> "Section 1 covers Module 2 of our syllabus: data wrangling. We load our simulated dataset of 3,500 railway complaints."

---

## Cell 2 — Imports and Data Loading (Code)
**What is on screen:** All the import statements and `pd.read_csv()`. The output shows the dataset shape and `df.info()`.

**Say this:**
> "Here we import standard Python libraries — pandas, numpy, scikit-learn, scipy, matplotlib, and seaborn. We load the CSV file which has 3,500 rows and 23 columns. The columns include complaint text, train telemetry like speed and distance to the maintenance yard, 8 interpretable photo features like rust score and dirt score, the target department, and the severity and urgency scores."

**If professor asks:** *"Why 23 columns?"*
> "We have: complaint_text, station_code, timestamp, train_speed, dist_to_yard, wait_hours, image_present, 8 photo features, true_department, severity_score, urgency_score, and 4 fake PII columns for the ethics section."

---

## Cell 3 — Missing Value Audit + Filtering + Grouping + Sorting + Ranking (Code)
**What is on screen:** Missing value count, filtering, group-by, sorting, and ranking outputs.

**Say this:**
> "First we audit missing values. About 35.5% of complaints don't have photos, so the 8 photo feature columns have around 1,244 NaN values each. This is intentional — in real life, many passengers don't attach photos. We then demonstrate Module 2 operations: filtering complaints by department, grouping by station to compute mean severity, sorting by urgency, and ranking complaints within each department. Finally, we impute the missing photo features using the training set median — this is important because we fit the imputer on training data only, so there is no data leakage."

**If professor asks:** *"Why median instead of mean?"*
> "Median is more robust to outliers. If one photo has an extreme rust score, it won't pull the imputed value away from the typical case."

---

## Cell 4 — Section 2 Header (Markdown)
**Say this:**
> "Section 2 is our Exploratory Data Analysis, also Module 2."

---

## Cell 5 — Descriptive Statistics (Code)
**What is on screen:** A table showing mean, std, min, max of severity, urgency, speed, etc. grouped by department.

**Say this:**
> "This table shows descriptive statistics per department. Notice that Security has the highest mean severity at 0.83, while Cleaning has the lowest at 0.35. This makes physical sense — a security threat like unauthorized passengers is more severe than a dirty toilet. The urgency scores are more uniform because urgency depends on distance and wait time, not just damage."

---

## Cell 6 — EDA Visualisation Grid (Code)
**What is on screen:** A 2×2 grid of plots — department distribution bar chart, urgency vs distance scatter, correlation heatmap, and severity box plot.

**Say this:**
> "These four visualisations show the key data characteristics. Top-left: the department distribution is imbalanced — Mechanical has roughly 33% of complaints while Security has only about 10%. Top-right: urgency clearly spikes when distance to yard is small — this is the exponential proximity effect that URPA models. Bottom-left: the correlation heatmap shows severity is most correlated with department-specific photo features. Bottom-right: box plots confirm each department has a distinct severity profile."

---

## Cell 7 — Section 3 Header (Markdown)
**Say this:**
> "Section 3 covers Module 3: Inferential Statistics and Hypothesis Testing."

---

## Cell 8 — Hypothesis Tests (Code)
**What is on screen:** Four formal statistical tests with printed results.

**Say this:**
> "We ran four formal hypothesis tests.

> **Test 1 — One-Way ANOVA on Severity across Departments.** The null hypothesis is that all 5 departments have the same mean severity. We got F = 926.83 with p = 0.0000, so we reject H₀. This confirms that departments have genuinely different damage profiles — it's not random noise.

> **Test 2 — Two-Sample t-Test: Urgency by Photo Presence.** We tested whether complaints with photos have higher urgency. We got t = 0.957, p = 0.3385, so we fail to reject H₀. This proves urgency is driven by operational factors like yard distance and passenger wait time, not by whether a photo was uploaded.

> **Test 3 — Chi-Square Test: Department vs Station.** We tested whether certain stations produce more complaints of a specific department. Chi-square = 59.88, p = 0.4433, fail to reject H₀. Complaint types are distributed uniformly across railway stations.

> **Test 4 — 95% Confidence Interval** for mean urgency: [0.6766, 0.6869]. We are 95% confident the true population mean urgency falls in this narrow range."

**If professor asks:** *"Why is the t-test not significant?"*
> "Because urgency is computed from yard distance and wait hours through the URPA physical formula. Whether a passenger uploads a photo changes the routing department, not the dynamic urgency of dispatching a repair crew."

---

## Cell 9 — Section 4 Header (Markdown)
**Say this:**
> "Section 4 is Module 4: Baseline Machine Learning Models."

---

## Cell 10 — Feature Preprocessing Pipeline (Code)
**What is on screen:** Stratified train/val/test split (70/15/15), TF-IDF fitting, imputation, and scaling.

**Say this:**
> "We split the data into 70% training, 15% validation, and 15% test using stratified sampling — meaning each split has the same proportion of all 5 departments. We then fit TF-IDF on training text only, fit the median imputer on training features only, and fit StandardScaler on training features only. Scalers and TF-IDF are fitted on training data only — this prevents data leakage."

---

## Cell 11 — Baseline Models with 5-Fold CV (Code)
**What is on screen:** Cross-validation results for LogReg, KNN, and Decision Tree.

**Say this:**
> "We evaluated three baseline classifiers using 5-fold cross-validation. Logistic Regression got 79.10% mean accuracy, KNN got 70.75%, and Decision Tree got 65.61%. These are our baselines. The key weakness is that LogReg, our best baseline, uses only text — when passengers write vague complaints like 'problem in coach B2, please check', it completely fails. That motivates our HCAM multi-view approach."

---

## Cell 12 — Section 5 Header (Markdown)
**Say this:**
> "Section 5 is the core of our project: the HCAM algorithm — Hybrid Confidence-Adaptive Multi-View Routing."

---

## Cell 13 — Content-Based Retrieval Recommender (Code)
**What is on screen:** Building the KNN recommender (Module 5) and computing validation alpha reliabilities.

**Say this:**
> "First we build View 3: a Content-Based Recommender using cosine-similarity nearest neighbours from Module 5. For each new complaint, we find the 5 most similar historical cases and look at what departments they belonged to. The probability distribution comes from counting the neighbour labels.

> Then we compute reliability scores, alpha, for each view. This is the key innovation. We evaluate each view's F1-score separately on validation complaints that have photos and those that don't. When photos are present, the Context View's alpha is 0.959 — very reliable. When photos are absent, alpha drops to just 0.052 — nearly zero. This means the system will automatically ignore the Context View when photos are missing, without any manual if-else logic."

**If professor asks:** *"Why not just use an if-else to check for photos?"*
> "Because hard-coded rules don't generalise. What if a photo is uploaded but is blurry and useless? The entropy-based confidence would still be high — the model is confused — so the weight naturally drops. It's a soft, data-driven mechanism."

---

## Cell 14 — Confidence-Adaptive Fusion Function (Code)
**What is on screen:** The NumPy implementation of Shannon entropy, confidence, and adaptive weighting.

**Say this:**
> "This is the mathematical core of HCAM. For each complaint, each view outputs a 5-class probability vector. We compute Shannon entropy H(p) which measures how spread out — how confused — the prediction is. If the model outputs equal probabilities for all 5 classes, entropy is maximum and confidence is zero. If it outputs 100% on one class, entropy is zero and confidence is 1.0.

> The normalised confidence is c = 1 - H(p) / ln(5). The final adaptive weight for each view is alpha times c, normalised so all weights sum to 1. The fused probability is the weighted sum of all three views."

---

## Cell 15 — Hierarchical Routing Head (Code)
**What is on screen:** The hierarchical two-stage routing ablation.

**Say this:**
> "This is an ablation experiment. We tested whether a hierarchical approach — first classifying into a broad group like Technical vs Service, then into the specific department — would work better. It achieved only 76.57% accuracy because errors in Stage 1 cascade down to Stage 2. This proves flat multi-view fusion is better for our problem."

---

## Cell 16 — Ablation Summary Table (Code)
**What is on screen:** A printed table comparing all 5 model configurations.

**Say this:**
> "This is our key results table. Let me walk through it:
> - Text Only: 77.90% overall, but drops to just 17.14% on ambiguous complaints.
> - Adding Context View with equal weights: jumps to 92.19%.
> - Adding Retrieval View with fixed weights: 94.29%.
> - Full HCAM with adaptive entropy weights: **94.67% overall and 80.71% on ambiguous text**.
> - Hierarchical Routing: 76.57%, proving it's worse than flat fusion.

> The improvement from 77.90% to 94.67% was validated with a 5-fold paired t-test: t = 19.05, p = 0.0000447 — well below the 0.05 significance threshold."

---

## Cell 17 — Confusion Matrix & Classification Report (Code)
**What is on screen:** A normalised confusion matrix heatmap and per-class precision/recall/F1.

**Say this:**
> "The confusion matrix shows that HCAM correctly classifies the vast majority of complaints in every department. The diagonal values are all above 0.90. The macro-F1 score is 0.9376, which means all classes are handled fairly, not just the majority ones."

---

## Cell 18 — Section 6 Header (Markdown)
**Say this:**
> "Section 6 introduces the second pillar of our algorithm: URPA — the Urgency Propagation Algorithm."

---

## Cell 19 — Severity Regression and URPA Curve Fit (Code)
**What is on screen:** Multiple Linear Regression for severity, then URPA nonlinear curve fit, comparison table.

**Say this:**
> "First, we predict the base structural damage score using Multiple Linear Regression. Then URPA models urgency using a physics-inspired formula: hazard equals a department-specific risk times the exponential decay of yard distance, plus a wait-time term. Urgency equals severity plus (1 minus severity) times (1 minus e to the negative hazard).

> The key insight: this formula **mathematically guarantees** that urgency is always greater than or equal to severity. A standard linear regression achieved MAE of 0.0812, slightly better numerically, but it predicted urgency LOWER than severity on several test cases — that violates physical reality. You can't have a repair be less urgent than the damage warrants. URPA's MAE is 0.0862 with guaranteed monotonicity."

**If professor asks:** *"Why accept a slightly higher MAE?"*
> "Because a model that predicts urgency lower than damage severity would tell a station master to delay repairing a broken coupling that's already causing sparks. The monotonicity guarantee means no safety-critical complaint ever gets deprioritised below its damage level."

---

## Cell 20 — Parameter Recovery (Code)
**What is on screen:** True vs fitted parameter comparison.

**Say this:**
> "Using scipy's curve_fit, we recovered the physical parameters. The yard proximity decay tau was fitted at 85.8 km versus the true 60 km. The wait-time hazard rate b was fitted at 0.0669 versus the true 0.08 — only 16.3% error. The department risk weights correctly ranked Security as highest at 1.813 and Cleaning as lowest at 0.385. So the model is recovering interpretable physical meaning, not just fitting numbers."

---

## Cell 21 — Section 7 Header (Markdown)
**Say this:**
> "Section 7 covers unsupervised learning from Module 4: K-Means and Agglomerative Clustering."

---

## Cell 22 — Clustering Analysis (Code)
**What is on screen:** Elbow plot, silhouette scores, and dendrogram.

**Say this:**
> "We applied K-Means with k from 2 to 7 on 32-dimensional SVD text features. The elbow curve flattens around k = 4 to 5. The silhouette score peaks around 0.12, which is expected since complaint text is naturally overlapping across departments. Cross-tabulating clusters with true labels shows that technical complaints — Mechanical and Electrical — group together, while Cleaning complaints form a distinct cluster based on vocabulary like 'toilet' and 'stench'. The Agglomerative Ward dendrogram visualises the hierarchical merging."

---

## Cell 23 — Section 8 Header (Markdown)
**Say this:**
> "Section 8 evaluates our Retrieval Recommender from Module 5."

---

## Cell 24 — Recommender Evaluation: Precision@3 and Hit-Rate@3 (Code)
**What is on screen:** Precision@3 and Hit-Rate@3 metrics.

**Say this:**
> "We evaluated the Content-Based Recommender using standard recommender metrics from Module 5. For each test complaint, we retrieve the 3 nearest historical cases using cosine similarity. Precision@3 is 0.8489 — meaning on average, 85% of the top-3 neighbours share the correct department. Hit-Rate@3 is 94.10% — meaning 94% of test complaints have at least one correct neighbour in the top 3."

---

## Cell 25 — Section 9 Header (Markdown)
**Say this:**
> "Section 9 covers Module 6: Ethics, Governance, Bias and Fairness, and Explainability."

---

## Cell 26 — PII Anonymisation (Code)
**What is on screen:** PNR dropping, phone masking, and salted SHA-256 hashing demo.

**Say this:**
> "For data ethics, we implement three anonymisation techniques. First, raw PNR numbers are completely dropped. Second, mobile numbers are masked: plus-91 98765 star-star-star-star-star. Third, passenger names are hashed using salted SHA-256 — we add a secret salt before hashing, so the hash cannot be reversed back to the original name."

---

## Cell 27 — Algorithmic Fairness Audit (Code)
**What is on screen:** Per-station accuracy table and mean urgency across stations.

**Say this:**
> "We performed a fairness audit across all 15 railway stations. The classification accuracy ranges from 88.6% to 100%, with a maximum disparity gap of 11.36%. Mean predicted urgency is consistent at 0.66 to 0.70 across all stations. This confirms the system does not deprioritise tier-2 stations."

---

## Cell 28 — Explainability (Code)
**What is on screen:** Top LogReg keywords, Decision Tree importances, and a complaint walkthrough.

**Say this:**
> "For explainability, we show three layers. First, top Logistic Regression keywords per department — 'washbasin' points to Cleaning, 'bogie' points to Mechanical. Second, Decision Tree feature importances: dirt_score 28.4%, crowd_density 24.8%, rust_score 23.2%. Third, a step-by-step walkthrough of one complaint showing the exact probability vectors, entropy, weights, and final prediction. A station superintendent can understand exactly why the system made its decision."

---

## Cell 29 — Section 10 Header (Markdown)
**Say this:**
> "Finally, Section 10: our summary, latency benchmark, and limitations."

---

## Cell 30 — Latency Benchmark & Final Scorecard (Code)
**What is on screen:** CPU latency and final scorecard.

**Say this:**
> "The full HCAM-URPA pipeline processes a single complaint in 0.018 milliseconds — over 55,000 complaints per second on CPU. It can run on a 35-dollar Raspberry Pi in a station master's office. Final scorecard: 94.67% routing accuracy, 0.9376 macro-F1, URPA MAE 0.0862 with guaranteed monotonicity, paired t-test p less than 0.001."

---

## Cell 31 — Limitations & Accountability (Markdown)
**Say this:**
> "We honestly acknowledge three limitations. First, our data is simulated — real complaints contain Hinglish and voice recordings. Second, the URPA wait parameter has a 16% estimation error. Third, we used numerical photo features instead of raw images. If given real data, we'd use a multilingual tokenizer and a lightweight image feature extractor, keeping the HCAM fusion and URPA formula exactly the same. Thank you, and I welcome your questions."

---

## 🎯 5 Tips for a Great Presentation

1. **Don't read code line-by-line.** Scroll past the code quickly and focus on the **output** — tables, numbers, and plots.
2. **Point at the screen.** When you say "94.67%", physically point at that number on the screen.
3. **Pause after key results.** After saying an accuracy or p-value, pause 2 seconds to let the professor absorb it.
4. **If you get nervous,** just read the printed output from the notebook — we designed those print statements to be presentation-ready.
5. **If the professor asks something you don't know,** say: *"That's a great question, let me scroll to the relevant section"* and navigate to it.
