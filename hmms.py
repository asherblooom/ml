import pandas as pd
import pymc as pm
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

raw_data = pd.read_csv(pm.get_data("deaths_and_temps_england_wales.csv"))


print(f"Data columns found: {raw_data.columns.tolist()}")

# ---------------------------------------------------------
# 2. Preprocessing & Discretization
# ---------------------------------------------------------

# Discretise Deaths into Low, Medium, High
# used qcut for quantile-based discretisation (equal number of points per bin) to ensure balanced classes for training.
deaths_discrete = pd.qcut(raw_data['deaths'], q=3, labels=['Low', 'Medium', 'High'])

# Encode these labels to integers (0, 1, 2) for hmmlearn
le_deaths = LabelEncoder()
deaths_encoded = le_deaths.fit_transform(deaths_discrete)

# B. Discretize Temperature (Hidden States) into 3 bins
# The prompt says "the state is the temperature".
# Since we are training UNSUPERVISED on deaths, we will use this column
# later to check if the model actually learned these states.
# Note: Low Temp = Cold, High Temp = Hot.
temp_discrete = pd.qcut(raw_data['temp'], q=3, labels=['Cold', 'Mild', 'Hot'])
temp_encoded = LabelEncoder().fit_transform(temp_discrete)

# We use CategoricalHMM because our observations (Deaths) are discrete categories (0, 1, 2)
# n_components=3 corresponds to our hypothesis that there are 3 temperature states.
model = hmm.CategoricalHMM(n_components=3, n_iter=100, random_state=42, init_params='ste')

# Prepare data for hmmlearn
# Input must be shape (n_samples, 1)
X = deaths_encoded.reshape(1, -1)
lengths = [len(X)]
model.fit(X, lengths)

# Predict the hidden states (most likely path) given the observed deaths
hidden_states = model.predict(X)

plt.figure(figsize=(14, 8))
# Plot 1: The Actual Data (Temperature) vs The Inferred States
# Note: The model learns state 0, 1, 2. These might not align perfectly with
# Cold (0), Mild (1), Hot (2). We might need to permute them visually, 
# but raw plotting usually reveals the structure.
plt.subplot(2, 1, 1)
plt.plot(raw_data['temp'], label='Actual Temperature', color='gray', alpha=0.5)
plt.title("Actual Temperature Data")
plt.ylabel("Temperature")
plt.legend()

plt.subplot(2, 1, 2)
# Plot the Inferred States
plt.plot(hidden_states, label='HMM Inferred States (from Deaths)', color='blue', marker='o', linestyle='none', alpha = 0.5)
# Plot the "True" discretized temperature for comparison
plt.plot(temp_encoded, label='Actual Discretized Temp (Ground Truth)', color='red', marker='o', linestyle='none', alpha = 0.5)

plt.title("Comparison: HMM Inferred States vs Actual Temperature Categories")
plt.ylabel("State / Category (0, 1, 2)")
plt.xlabel("Time (Months)")
plt.yticks([0, 1, 2], ['State 0', 'State 1', 'State 2'])
plt.legend()

plt.tight_layout()
# plt.savefig('hmm_results.png')
plt.show()

# ---------------------------------------------------------
# 6. Interpret Model Parameters
# ---------------------------------------------------------
print("\n--- Model Parameters ---")
print("Transition Matrix (Probability of moving from State i to State j):")
print(model.transmat_.round(3))

print("\nEmission Matrix (Probability of observing Death Level k given State i):")
# Columns correspond to Death Levels: 0, 1, 2 (check le_deaths.classes_ for order)
df_emission = pd.DataFrame(model.emissionprob_, columns=le_deaths.classes_)
df_emission.index.name = "Hidden State"
print(df_emission.round(3))

print("\nInterpretation:")
print("Look at the Emission Matrix above.")
print("If a State has a high probability of 'High' deaths, that State likely corresponds to 'Cold' or 'Hot' temperature extremes (depending on the data correlation).")