import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load the data
file_name = "regression_insurance.csv"
data = pd.read_csv(file_name)
X = data.drop(columns=["charges"])
y = data["charges"].values.reshape(-1, 1)

# one hot encode the categorical columns, scale the numerical columns
categorical_cols = ["sex", "smoker", "region"]
numerical_cols = ["age", "bmi", "children"]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", StandardScaler(), numerical_cols),
    ],
    verbose_feature_names_out=False,
)
y_scaler = StandardScaler()

# Fit preprocessor on training data
X = preprocessor.fit_transform(X)
# Scale the target
y = y_scaler.fit_transform(y).flatten()
# Extract feature names to make results readable
feature_names = preprocessor.get_feature_names_out()
n_features = X.shape[1]

# 3. Build Bayesian Model
with pm.Model() as model:
    # --- Priors ---
    # We use weakly informative priors since the data is standardized.
    # Intercept (alpha): Centered at 0, wide sigma
    intercept = pm.Normal("intercept", mu=0, sigma=1)

    # Coefficients (betas): Centered at 0, wide sigma
    # We use a single vector shape for efficiency
    coefficients = pm.Normal("coefficients", mu=0, sigma=1, shape=n_features)

    # Noise (sigma)
    sigma = pm.Uniform("sigma", lower=0, upper=1)

    # --- Model Expectation ---
    # mu = alpha + beta * X
    mu = intercept + pm.math.dot(X, coefficients)

    # --- Likelihood ---
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # --- MCMC Sampling ---
    idata = pm.sample(2000, chains=4, step=pm.NUTS())

summary = az.summary(idata, var_names=["intercept", "coefficients", "sigma"])

print("\nPosterior Means for Coefficients:")
print("-" * 50)
print(f"{'Feature':<20} | {'Mean Est.':<10} | {'94% HDI':<15}")
print("-" * 50)

# Print Intercept
intercept_mean = summary.loc["intercept", "mean"]
print(
    f"{'Intercept':<20} | {intercept_mean:<10.4f} | [{summary.loc['intercept', 'hdi_3%']:.3f}, {summary.loc['intercept', 'hdi_97%']:.3f}]"
)

# 1. Get the scaling factor (Standard Deviation of original charges)
# y_scaler.scale_ stores the standard deviation (sigma)
y_std_dev = y_scaler.scale_[0]
y_mean = y_scaler.mean_[0]

print("\n" + "=" * 60)
print(f"{'Feature':<25} | {'Mean Est.':<12} | {'Real Dollar Impact':<20}")
print("=" * 60)

# 2. Iterate and Un-scale
for i, feature in enumerate(feature_names):
    row_name = f"coefficients[{i}]"
    scaled_mean = summary.loc[row_name, "mean"]
    # convert to dollars
    dollar_impact = scaled_mean * y_std_dev
    print(f"{feature:<25} | {scaled_mean:<12.3f} | ${dollar_impact:,.2f}")
print("-" * 60)
# The model's sigma is also scaled. We multiply it by y_std_dev to get the prediction error in dollars.
scaled_sigma = summary.loc["sigma", "mean"]
dollar_sigma = scaled_sigma * y_std_dev
print(f"{'Noise (Sigma)':<25} | {scaled_sigma:<12.3f} | ${dollar_sigma:,.2f}")
print("=" * 60)

