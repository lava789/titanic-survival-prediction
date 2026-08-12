# Titanic Survival Prediction

Predicting passenger survival on the Titanic using Logistic Regression, Decision Tree, and Random Forest — with feature engineering, missing-value handling, and cross-validation.

# Dataset

The classic Titanic dataset (891 passengers, 12 original columns), including demographics (Age, Sex), ticket info (Fare, Pclass, Embarked), family info (SibSp, Parch), and the target column Survived.

# Approach

# Data Cleaning
- Dropped PassengerId (identifier, no predictive value), Ticket and Cabin (mostly missing/unusable as raw text)
- Extracted Title (Mr, Mrs, Miss, Master, Rare) from Name before dropping it — titles carry age/status signal linked to survival ("women and children first")
- Filled missing Age with the median (robust to outliers/skew) and missing Embarked with the mode (categorical)

# Feature Engineering
- One-hot encoded Sex, Embarked, and Title
- Explored feature relationships with a correlation heatmap before modeling

# Modeling
- Logistic Regression (with feature scaling)
- Decision Tree (tuned via max_depth to prevent overfitting)
- Random Forest (same depth-tuning approach applied)
- Evaluated all three with 5-fold cross-validation, not just a single train/test split

# Results

 Model / CV Accuracy (5-fold)

 Logistic Regression (scaled)  82.6% 
 Decision Tree (max_depth=4)   82.3% 
 Random Forest (max_depth=8)   82.7% 

# Key Findings

- An unrestricted decision tree overfit badly — 76% test accuracy despite near-perfect training accuracy. Limiting max_depth (tested 2-7) improved test accuracy by over 10 points, to 87.7%. The same issue and fix applied to Random Forest.
- Single train/test split accuracy (86-88%) was noticeably more optimistic than 5-fold CV accuracy (82-83%) — a reminder that one split can be misleading, and cross-validation gives a more honest estimate of real-world performance.
- All three tuned models perform within ~0.5% of each other once evaluated fairly — model choice mattered far less than feature engineering and controlling overfitting.
- Sex/Title_Mr and Pclass were the strongest predictors of survival, consistent with the historical "women and children first" pattern.

 # What I'd try next

- Feature interactions (e.g. Sex x Pclass)
- GridSearchCV for more systematic hyperparameter tuning
- Gradient boosting (XGBoost) for comparison

# Tools

Python, pandas, scikit-learn, matplotlib, seaborn
