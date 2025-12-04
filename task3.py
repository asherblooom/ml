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
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    coefficients = pm.Normal("coefficients", mu=0, sigma=1, shape=n_features)
    sigma = pm.Uniform("sigma", lower=0, upper=1)
    mu = intercept + pm.math.dot(X, coefficients)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
    idata = pm.sample(2000, chains=4, step=pm.NUTS())

summary = az.summary(idata, var_names=["intercept", "coefficients", "sigma"])

print("Posterior Means for Coefficients:")

y_std_dev = y_scaler.scale_[0]
y_mean = y_scaler.mean_[0]

print("\n" + "=" * 40)
print(f"{'Feature':<25} | {'Mean Est.':<12}")
print("=" * 40)

# Print Intercept
intercept_mean = summary.loc["intercept", "mean"]
print(
    f"{'Intercept':<25} | {intercept_mean:<10.4f}"
)

for i, feature in enumerate(feature_names):
    row_name = f"coefficients[{i}]"
    scaled_mean = summary.loc[row_name, "mean"]
    print(f"{feature:<25} | {scaled_mean:<12.3f}")
print("-" * 40)
scaled_sigma = summary.loc["sigma", "mean"]
dollar_sigma = scaled_sigma * y_std_dev
print(f"{'Noise (Sigma)':<25} | {scaled_sigma:<12.3f}")
print("=" * 40)

