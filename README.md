# movie-revenue-predictor
1. Problem framing: two complementary tasks

Regression (Revenue prediction) gives a continuous, actionable number that producers can use for ROI planning.

Classification (Hit/Flop) converts business logic into an easy yes/no decision (e.g., where success = revenue > budget * threshold), which is often more interpretable for stakeholders.

Having both tasks allows cross-validation of results: if both predict "high revenue" and "hit", confidence increases.

2. Target transform (log scale)

Movie revenues are heavily right-skewed (orders of magnitude). Training the regressor on log(revenue + 1) stabilizes variance, improves model convergence, and makes errors more meaningful.

At inference we invert predictions using expm1() to present human-readable whole-dollar revenues.

3. Handling high-cardinality categorical features (actors, director, companies)

Why not naive One-Hot?
One-Hot causes huge dimensionality (sparse thousands of columns) and overfitting when applied to actors / keywords / companies.

Chosen approach:

CountVectorizer / Count features for genres, keywords, and production_companies (limited max_features) to capture presence patterns without exploding dimensionality.

Global actor encoding (the deep improvement): instead of separate actor_1/2/3 encodings, build a global actor frequency and global actor mean-revenue mapping aggregated across all positions. This captures each actor’s overall influence, independent of the column position.

Director encoded by frequency + mean-revenue (directors have strong signal).

Frequency encoding captures how often a name appears (popularity / experience).

Target encoding captures historical average revenue associated with that entity (direct signal).

Missing / unseen entities: fallback to 0 (frequency) and global_mean_revenue (target) to avoid crashes and provide neutral assumptions.

4. Preventing data leakage

All encoding maps (freq and target dictionaries) must be built only on the training set and then saved. During inference or validation, we only use those saved maps (never recompute using test/validation labels).

Target encoding includes the risk of leakage if computed on full dataset — ensure per-fold target-encoding or smoothing if used inside CV/hyperparameter tuning (not required for a deployment-ready single-model demonstration but crucial for production-grade pipelines).

5. Feature composition & scaling

Tree-based models (XGBoost) are robust to feature scaling; we avoid unnecessary scaling for most features.

Numeric features (budget, actor targets) preserved as raw (budget often log-transformed during training pipeline).

We limit vectorizer sizes (e.g., top 200 keywords) to keep the feature matrix manageable.

6. Model choice

XGBoost chosen for its robustness, handling of heterogeneous features, speed, and interpretability (feature importance).

We built two separate XGBoost models:

XGBRegressor for revenue (trained on log_revenue).

XGBClassifier for success (binary).

Baselines (Linear Regression / Logistic Regression) were used for comparison.

7. Evaluation & interpretation

Regression reported on log-scale (RMSE) and R². Remember to invert for business reporting.

Classification uses accuracy, precision, recall, f1, and confusion matrix to show the trade-offs between false positives/negatives.

Explainability: feature importance (from XGBoost) helps identify which features (budget, actor_target, genres flags, director_target) drive predictions.
