# DSE Learning Guide & Concept Glossary: HCAM-URPA
**Course:** Data Science Ecosystem (23CS2024) | **Project:** SIH1711 (Rail Madad)

This learning guide maps every technique used in our project directly to the course syllabus modules, explaining each with an intuitive one-line real-world analogy and its mathematical purpose.

---

## 1. Module 2: Data Wrangling, Cleaning & Preprocessing

### 1. Missing Value Imputation (Median)
- **One-Line Analogy:** If a student skips one homework question, we fill it with the typical (median) class score rather than giving a zero.
- **In Our Project:** When passengers do not attach a photo (`image_present == 0`), the 8 photo features (`rust_score`, `dirt_score`, etc.) are missing (`NaN`). We impute them using the median of training complaints so models do not crash.

### 2. Stratified Train-Validation-Test Split
- **One-Line Analogy:** Cutting a fruit cake so every slice gets an equal share of cherries, raisins, and nuts.
- **In Our Project:** Splitting 3,500 complaints into 70% Train (2,450), 15% Validation (525), and 15% Test (525) such that each split contains identical proportions of the 5 departments.

### 3. Feature Scaling (StandardScaler)
- **One-Line Analogy:** Converting train speed (0–120 km/h) and yard distance (0–160 km) into a standardized grading curve (z-scores with mean 0 and standard deviation 1).
- **In Our Project:** Ensures Euclidean distance in KNN and clustering is not dominated by large numerical ranges.

---

## 2. Module 3: Descriptive Statistics & Hypothesis Testing

### 4. Shannon Entropy $H(p)$
- **One-Line Analogy:** A coin flip where you have no idea what will happen has maximum surprise (high entropy); a two-headed coin has zero surprise (zero entropy).
- **In Our Project:** Quantifies the uncertainty of a model's predicted probability distribution:
  $$H(p) = -\sum_{k=1}^K p_k \ln(p_k)$$
  If a model outputs $[0.2, 0.2, 0.2, 0.2, 0.2]$, entropy is maximal ($\ln 5 \approx 1.609$), signaling complete confusion.

### 5. Normalized Model Confidence $c_v$
- **One-Line Analogy:** A certainty meter from 0% to 100% measuring how confident a doctor is in a medical diagnosis.
- **In Our Project:** $c_v = 1 - \frac{H(p_v)}{\ln(K)}$. High when the probability distribution is sharp and peaked on one department; zero when completely spread out.

### 6. One-Way ANOVA (Analysis of Variance)
- **One-Line Analogy:** Testing whether students in 5 different classroom sections have significantly different exam averages or just random noise.
- **In Our Project:** Tests $H_0$: Mean severity is identical across all 5 railway departments ($p < 0.05 \implies$ rejected $H_0$, proving departments have distinct severity profiles).

### 7. Two-Sample Independent t-Test
- **One-Line Analogy:** Comparing whether shoppers with coupons spend significantly more on average than shoppers without coupons.
- **In Our Project:** Tests whether complaints with photos have significantly higher urgency than complaints without photos.

### 8. Chi-Square ($\chi^2$) Test of Independence
- **One-Line Analogy:** Checking whether chocolate vs vanilla ice cream preferences depend on which city the shop is in.
- **In Our Project:** Tests whether department grievances (e.g. Mechanical vs Catering) are independent of railway stations.

### 9. 95% Confidence Interval
- **One-Line Analogy:** Casting a fishing net that has a 95% probability of catching the true, unseen population average.
- **In Our Project:** Computes the lower and upper bounds $[\bar{X} - t \cdot \text{SE}, \bar{X} + t \cdot \text{SE}]$ for the true average urgency across Indian Railways.

---

## 3. Module 4: Machine Learning, Regression & Clustering

### 10. TF-IDF (Term Frequency - Inverse Document Frequency)
- **One-Line Analogy:** Words that appear everywhere (like "coach" or "train") get muted, while rare words (like "cockroach" or "sparking") get a microphone.
- **In Our Project:** Transforms complaint text into numerical n-gram importance vectors.

### 11. Logistic Regression (Multinomial with Class Weighting)
- **One-Line Analogy:** A referee that draws linear decision boundaries between 5 teams while giving extra points to the smaller teams so they aren't ignored.
- **In Our Project:** Our baseline and Text View model, using `class_weight='balanced'` to prevent majority Mechanical complaints from overwhelming minority Security complaints.

### 12. Decision Tree Classifier (CART)
- **One-Line Analogy:** A flowchart of yes/no medical screening questions (e.g., "Is dirt score > 0.6? If yes, check crowd density").
- **In Our Project:** Our Context View model, mapping train telemetry and 8 interpretable photo features into department predictions.

### 13. K-Nearest Neighbors (KNN)
- **One-Line Analogy:** Asking your 5 closest neighbors on your street for advice.
- **In Our Project:** Finds historical complaints closest in feature space using Euclidean or Cosine distance.

### 14. Multiple Linear Regression & Polynomial Regression
- **One-Line Analogy:** Fitting a straight ramp (linear) versus a curved parabolic ramp (polynomial degree 2) through multi-dimensional data points.
- **In Our Project:** Baseline models for predicting continuous Severity and Urgency scores.

### 15. Non-Linear Curve Fitting (`scipy.optimize.curve_fit`)
- **One-Line Analogy:** Finding the exact stiffness and damping of a physical suspension spring so a formula matches test track data.
- **In Our Project:** Fits the 7 physical parameters ($a_{\text{Mech}}..a_{\text{Sec}}, \tau, b$) of the URPA differential equation using non-linear least squares.

### 16. K-Means Clustering (Elbow & Silhouette)
- **One-Line Analogy:** Dividing railway passengers into $k$ natural travel groups without knowing their ticket types, and finding the sweet spot where clusters are tight and separate.
- **In Our Project:** Discovers natural complaint groupings from SVD-reduced text vectors.

### 17. Agglomerative Hierarchical Clustering (Dendrogram)
- **One-Line Analogy:** Building a family tree from individual passengers upwards until everyone belongs to the Indian Railways family.
- **In Our Project:** Visualizes how complaints merge step-by-step using Ward's minimum variance linkage.

---

## 4. Module 5: Recommender Systems

### 18. Content-Based Recommender (Retrieval View)
- **One-Line Analogy:** Recommending a movie to a viewer because its genre, actors, and plot keywords closely match movies they liked in the past.
- **In Our Project:** Given a new complaint, retrieves the 5 most similar historical complaints using Cosine Similarity over text and operational features.

### 19. Precision@K and Hit-Rate@K
- **One-Line Analogy:** If Netflix recommends 3 movies: Precision@3 is how many of the 3 you loved; Hit-Rate@3 is whether at least 1 of them was a hit.
- **In Our Project:** Precision@3 measures what fraction of the top-3 retrieved historical complaints belong to the query's true department (achieved **84.9%**); Hit-Rate@3 measures if at least one was a match (achieved **94.1%**).

### 20. Late Fusion (Hybrid Multi-View)
- **One-Line Analogy:** A panel of 3 doctors (text specialist, context specialist, historical case specialist) independently giving a diagnosis, and averaging their votes based on who is most certain.
- **In Our Project:** Fusing $p_T$, $p_C$, and $p_R$ using dynamic confidence weights $w_v$.

---

## 5. Module 6: Data Ethics, Governance & Accountability

### 21. Data Anonymisation (Masking & Salted Hashing)
- **One-Line Analogy:** Putting a privacy curtain over a passenger's face while keeping their medical record readable.
- **In Our Project:** Dropping raw PNR numbers, masking mobile numbers (`+91 98765 *****`), and hashing passenger names with SHA-256 and a secret salt.

### 22. Algorithmic Fairness Audit
- **One-Line Analogy:** Making sure an automated grading system gives fair marks to students from both rural and urban schools.
- **In Our Project:** Checking that routing accuracy and priority queues do not discriminate across different railway station codes.

### 23. Model Explainability
- **One-Line Analogy:** Requiring a judge to write down their exact legal reasoning rather than simply issuing a verdict.
- **In Our Project:** Extracting top predictive keywords from Logistic Regression, feature importances from Decision Trees, and displaying a step-by-step weight calculation for individual complaints.

### 24. Paired t-Test on Cross-Validation Folds
- **One-Line Analogy:** Testing two runners on the exact same 5 obstacle courses to prove one is genuinely faster, not just lucky.
- **In Our Project:** Proves HCAM's 91.2% mean fold accuracy is statistically significantly better than Logistic Regression's 79.5% ($p = 4.47 \times 10^{-5} < 0.001$).
