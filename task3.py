import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Load Data
# Ensure 'regression_insurance.csv' is in the same folder
try:
    df = pd.read_csv("regression_insurance.csv")
except FileNotFoundError:
    print("Error: Please make sure 'regression_insurance.csv' is in the working directory.")
    exit()

# 2. Preprocessing
# We use drop_first=True to avoid multicollinearity, which helps MCMC converge faster.
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Separate predictors (X) and target (y)
X_data = df_encoded.drop(columns=['charges'])
y_data = df_encoded['charges']

# Standardize features and target
# Scaling is critical for NUTS sampler efficiency (avoids "funnel" problems)
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X_data)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_data.values.reshape(-1, 1)).flatten()

# Get feature names for reporting
feature_names = X_data.columns.tolist()
n_features = X_scaled.shape[1]

print(f"Running Bayesian Regression on {n_features} features...")

# 3. Build Bayesian Model
with pm.Model() as model:
    # --- Priors ---
    # We use weakly informative priors since the data is standardized.
    # Intercept (alpha): Centered at 0, wide sigma
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    
    # Coefficients (betas): Centered at 0, wide sigma
    # We use a single vector shape for efficiency
    betas = pm.Normal('betas', mu=0, sigma=10, shape=n_features)
    
    # Noise (sigma): HalfNormal ensures it's positive
    sigma = pm.HalfNormal('sigma', sigma=10)
    
    # --- Model Expectation ---
    # mu = alpha + beta * X
    mu = alpha + pm.math.dot(X_scaled, betas)
    
    # --- Likelihood ---
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_scaled)
    
    # --- MCMC Sampling ---
    print("Sampling... (This may take a minute)")
    trace = pm.sample(draws=2000, tune=1000, chains=2, return_inferencedata=True)

# 4. Results & Reporting
print("\n--- Posterior Means (Standardized Scale) ---")
summary = az.summary(trace, var_names=['alpha', 'betas', 'sigma'])

# Map generic names (betas[0]) back to actual feature names
# Create a dictionary mapping indices to names
name_mapping = {f'betas[{i}]': name for i, name in enumerate(feature_names)}
summary = summary.rename(index=name_mapping)

# Print the Mean column for coefficients
print(summary['mean'])

# 5. Plot Posterior Distributions
print("\nGenerating Forest Plot...")
az.plot_forest(trace, var_names=['betas'], combined=True, figsize=(10, 6))
# Override y-labels with actual feature names
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names[::-1])
plt.title("Posterior Distributions of Coefficients (94% HDI)")
plt.tight_layout()
plt.show()

print("\nDone! To interpret coefficients in dollars, multiply by y_std (~12,000).")