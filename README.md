<div align="center">

# Hazardous Near-Earth Object Prediction

Machine Learning for Asteroid Hazard Classification

<img src="images/Neo.PNG" width="900"/>

<br>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## Overview

This project develops a machine learning classification system to predict whether a Near-Earth Object (NEO) is hazardous using NASA asteroid observation data.

Near-Earth Objects are celestial bodies whose orbits bring them close to Earth. Accurate classification of hazardous objects is critical for planetary defense, long-term space monitoring, and impact risk mitigation.

The core objective of this project was not only to build a high-accuracy model, but to:

- Identify physically interpretable predictors
- Compare multiple feature selection strategies
- Reduce dimensionality without sacrificing performance
- Demonstrate that simple, interpretable models can compete with complex pipelines

The final optimized model achieved **95.5% accuracy**.

More importantly, a simplified three-feature model achieved **93.6% accuracy**, reducing dimensionality by over 90% while maintaining strong predictive performance.

This highlights the power of intelligent feature selection over brute-force modeling.

---

## Problem Statement

Given observational asteroid data, can we accurately predict whether an object is classified as hazardous?

This is a binary classification problem where:

- 1 → Hazardous  
- 0 → Non-Hazardous  

The challenge includes:

- Class imbalance  
- High multicollinearity  
- Redundant physical measurements  
- Mixed-scale numerical variables  
- Scientific interpretability requirements  

The goal was to design a robust, interpretable, and efficient prediction pipeline.

---

## Dataset Summary

| Attribute | Value |
|-----------|--------|
Dataset Size | 4,687 Objects  
Target Variable | Hazardous (Binary)  
Initial Feature Count | 40  
Final Optimal Features | 3  
Class Imbalance | ~16% Hazardous  

The dataset includes physical and orbital characteristics such as:

- Absolute Magnitude  
- Estimated Diameter (multiple units)  
- Relative Velocity  
- Minimum Orbit Intersection Distance  
- Orbit Uncertainty  
- Eccentricity  
- Semi-Major Axis  

---

## Key Results

| Metric | Value |
|--------|--------|
Best Accuracy | 95.5%
Feature Reduction | 91%
Simplified Model Accuracy | 93.6%
Cross Validation | 5-Fold
Model Type | Logistic Regression

---

## Key Data Insights

During structured exploratory analysis, several meaningful patterns emerged:

### 1. Class Imbalance Exists

Only approximately 16% of objects are labeled hazardous.  
This required careful evaluation metrics beyond raw accuracy.

### 2. Orbit Distance is the Strongest Predictor

Minimum Orbit Intersection Distance consistently showed the strongest relationship with hazard classification.

Objects with smaller intersection distances pose greater theoretical impact risk.

### 3. Asteroid Size Matters

Absolute Magnitude acts as a proxy for size.  
Lower magnitude values (larger asteroids) are statistically more likely to be hazardous.

### 4. Orbit Uncertainty Adds Predictive Signal

Higher orbit uncertainty introduces monitoring complexity and shows measurable predictive influence.

### 5. High Multicollinearity Detected

Multiple diameter columns across units were perfectly correlated.

Removing redundant features:
- Reduced model variance
- Improved computational efficiency
- Preserved predictive power

---

## Core Discovery

Across all feature selection techniques, three variables consistently ranked highest:

| Feature | Why It Matters |
|----------|----------------|
Minimum Orbit Intersection Distance | Direct physical measure of Earth impact proximity  
Absolute Magnitude | Size indicator linked to destructive potential  
Orbit Uncertainty | Confidence level in trajectory estimation  

Using only these three features achieved:

93.6% Accuracy  
Massive dimensionality reduction  
High interpretability  

This demonstrates that domain-aligned features outperform large unfiltered feature sets.

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution visualization
- Correlation heatmap
- Skewness detection
- Class distribution analysis

### 2. Data Cleaning
- Removal of redundant diameter columns
- Handling multicollinearity using VIF
- Feature normalization using StandardScaler
- Verification of logical consistency

### 3. Feature Selection Techniques Compared

Multiple statistical and algorithmic approaches were tested:

- Point Biserial Correlation
- Mutual Information
- Sequential Feature Selection
- Lasso Regularization
- Particle Swarm Optimization
- Backward Elimination

Each method was evaluated for stability, interpretability, and performance impact.

### 4. Model Training

Primary model:
- Logistic Regression

Validation approach:
- 5-Fold Cross Validation

Why Logistic Regression?
- Interpretable coefficients
- Stable performance
- Suitable for scientific explanation
- Low overfitting risk

### 5. Evaluation Metrics

To ensure balanced evaluation:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

This was especially important due to class imbalance.

---

## Model Performance Comparison

| Method | Features Used | Accuracy |
|--------|--------------|----------|
PSO + Logistic Regression | 22 | 95.5%
Backward Elimination | 8 | 95.4%
Baseline Model | 35 | 94.8%
Lasso Selected Features | 3 | 93.6%
Mutual Information | 5 | 87.5%

Observation:

Adding more features does not always improve performance significantly.  
Smart selection can preserve nearly identical results with far fewer inputs.

---

## Example: Training the 3-Feature Model

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/nasa.csv")

X = df[[
    "Absolute Magnitude",
    "Orbit Uncertainity",
    "Minimum Orbit Intersection"
]]

y = df["Hazardous"].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

print(model.score(X_scaled, y))
```

Expected Output:

```
0.936
```

---

## Repository Structure

```
hazardous-neo-prediction/

data/
  nasa.csv

notebooks/
  Predicting_Hazardous_NEO.ipynb

documents/
  project_report.docx

images/
  Neo.PNG

README.md
requirements.txt
LICENSE
```

---

## Skills Demonstrated

- Machine Learning  
- Statistical Feature Selection  
- Dimensionality Reduction  
- Logistic Regression  
- Cross Validation  
- Multicollinearity Handling  
- Scientific Data Interpretation  
- Model Evaluation  
- Clean Technical Documentation  

---

## Why This Project Matters

This project shows that:

- Interpretable models can compete with complex pipelines  
- Feature engineering is more impactful than model stacking  
- Scientific datasets require careful dimensionality control  
- Clean statistical reasoning improves model trust  

The approach used here reflects real-world industry standards for:

- Risk modeling  
- Fraud detection  
- Financial scoring  
- Medical diagnostics  
- Scientific forecasting  

---

## Future Improvements

- Implement ensemble models (Random Forest, XGBoost)
- Compare tree-based feature importance
- Add ROC Curve visualization
- Perform hyperparameter tuning
- Deploy model as REST API
- Build interactive web dashboard
- Integrate real-time NASA data feed
- Add SHAP-based explainability

---

## Author

Murtaza Majid  

GitHub  
https://github.com/MurtazaMajid  

LinkedIn  
https://linkedin.com/in/murtaza-majid  

---

If you found this project valuable, consider starring the repository.
