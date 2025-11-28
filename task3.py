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
X = data.drop(columns=['charges'])
y = data['charges'].values.reshape(-1, 1)

# one hot encode the categorical columns, scale the numerical columns
categorical_cols = ['sex', 'smoker', 'region']
numerical_cols = ['age', 'bmi', 'children']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ],
    verbose_feature_names_out=False
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
    intercept = pm.Normal('intercept', mu=0, sigma=1)
    
    # Coefficients (betas): Centered at 0, wide sigma
    # We use a single vector shape for efficiency
    coefficients = pm.Normal('coefficients', mu=0, sigma=1, shape=n_features)
    
    # Noise (sigma)
    sigma = pm.Uniform('sigma', lower = 0, upper=1)
    
    # --- Model Expectation ---
    # mu = alpha + beta * X
    mu = intercept + pm.math.dot(X, coefficients)
    
    # --- Likelihood ---
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    
    # --- MCMC Sampling ---
    idata = pm.sample(draws=2000, tune=1000, chains=2, random_seed=42)

summary = az.summary(idata, var_names=["intercept", "coefficients", "sigma"])

print("\nPosterior Means for Coefficients:")
print("-" * 50)
print(f"{'Feature':<20} | {'Mean Est.':<10} | {'94% HDI':<15}")
print("-" * 50)

# Print Intercept
intercept_mean = summary.loc['intercept', 'mean']
print(f"{'Intercept':<20} | {intercept_mean:<10.4f} | [{summary.loc['intercept', 'hdi_3%']:.3f}, {summary.loc['intercept', 'hdi_97%']:.3f}]")

# Print Coefficients
# The summary index for array variables looks like "Coefficients[0]", "Coefficients[1]"
for i, feature in enumerate(feature_names):
    row_name = f"coefficients[{i}]"
    mean_val = summary.loc[row_name, 'mean']
    lower_hdi = summary.loc[row_name, 'hdi_3%']
    upper_hdi = summary.loc[row_name, 'hdi_97%']
    print(f"{feature:<20} | {mean_val:<10.4f} | [{lower_hdi:.3f}, {upper_hdi:.3f}]")

print("-" * 50)
print(f"Noise (Sigma) Mean: {summary.loc['sigma', 'mean']:.4f}")
