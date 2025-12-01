
import pandas as pd
import pymc as pm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def train_supervised_hmm(states, observations, n_states, n_obs):
    """
    Trains an HMM using Supervised Learning (Maximum Likelihood Estimation).
    
    Args:
        states (array): Sequence of hidden state indices (integers 0 to n_states-1)
        observations (array): Sequence of observation indices (integers 0 to n_obs-1)
        n_states (int): Number of unique hidden states
        n_obs (int): Number of unique observation types
        
    Returns:
        pi (array): Initial state probabilities
        A (matrix): Transition matrix (State -> State)
        B (matrix): Emission matrix (State -> Observation)
    """
    
    # 1. Initialize Matrices
    # A: Transition Matrix [n_states x n_states]
    A = np.zeros((n_states, n_states))
    
    # B: Emission Matrix [n_states x n_obs]
    B = np.zeros((n_states, n_obs))
    
    # pi: Start probabilities [n_states]
    pi = np.zeros(n_states)

    # 2. Calculate Start Probabilities (pi)
    # We count the first state, or for a long time series, 
    # we can use the marginal distribution of states.
    # Here we simply count the occurrence of every state to get the stationary distribution
    unique, counts = np.unique(states, return_counts=True)
    total_counts = np.sum(counts)
    for u, c in zip(unique, counts):
        pi[u] = c / total_counts

    # 3. Calculate Transitions (A)
    # Count how many times state i is followed by state j
    for t in range(len(states) - 1):
        current_state = states[t]
        next_state = states[t+1]
        A[current_state, next_state] += 1
        
    # Normalize rows of A (so probabilities sum to 1)
    # We add a tiny epsilon to avoid division by zero if a state never transitions
    A = (A + 1e-8) / (A + 1e-8).sum(axis=1, keepdims=True)

    # 4. Calculate Emissions (B)
    # Count how many times we see observation o while in state i
    for t in range(len(states)):
        current_state = states[t]
        current_obs = observations[t]
        B[current_state, current_obs] += 1
        
    # Normalize rows of B
    B = (B + 1e-8) / (B + 1e-8).sum(axis=1, keepdims=True)

    return pi, A, B






df = pd.read_csv(pm.get_data("deaths_and_temps_england_wales.csv"))

# A. Discretize Temperature (Hidden States)
# Strategy: Use Quartiles (4 bins) to get equivalent data distribution
# Labels: 0 (Coldest), 1 (Cold), 2 (Warm), 3 (Hot)
n_states = 4
df['temp_disc'] = pd.qcut(df['temp'], q=n_states, labels=False)
temp_labels = ['Coldest', 'Cold', 'Warm', 'Hot']

# B. Discretize Deaths (Observations)
# Strategy: Use 3 bins as requested (Low, Medium, High)
# Labels: 0 (Low), 1 (Medium), 2 (High)
n_obs = 3
df['deaths_disc'] = pd.qcut(df['deaths'], q=n_obs, labels=False)
obs_labels = ['Low', 'Medium', 'High']

print("\n--- Discretization Thresholds ---")
print(f"Temperature was split into {n_states} bins (States).")
print(f"Deaths were split into {n_obs} bins (Observations).")

# 3. Train HMM "By Hand"
states_seq = df['temp_disc'].values
obs_seq = df['deaths_disc'].values

pi, A, B = train_supervised_hmm(states_seq, obs_seq, n_states, n_obs)

# 4. Display Results

# Plotting Transition Matrix
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(A, annot=True, cmap="Blues", fmt=".2f", 
            xticklabels=temp_labels, yticklabels=temp_labels)
plt.title("Transition Matrix (A)\nProbability of State(t+1) given State(t)")
plt.xlabel("Next Temperature State")
plt.ylabel("Current Temperature State")

# Plotting Emission Matrix
plt.subplot(1, 2, 2)
sns.heatmap(B, annot=True, cmap="Greens", fmt=".2f", 
            xticklabels=obs_labels, yticklabels=temp_labels)
plt.title("Emission Matrix (B)\nProbability of Death Level given Temperature")
plt.xlabel("Reported Deaths (Observation)")
plt.ylabel("Temperature State")

plt.tight_layout()
plt.show()

# Print raw matrices
print("\n--- Learned Parameters ---")

print("\n1. Initial State Probabilities (Pi):")
for i, p in enumerate(pi):
    print(f"   {temp_labels[i]}: {p:.4f}")

print("\n2. Transition Matrix (A) - Row Stochastic:")
print(np.round(A, 3))

print("\n3. Emission Matrix (B) - Row Stochastic:")
print(np.round(B, 3))

# Interpretation
print("\n--- Interpretation ---")
max_death_state = np.argmax(B[:, 2]) # State with highest prob of 'High' deaths
print(f"The temperature state most likely to produce 'High' deaths is: {temp_labels[max_death_state]}")

min_death_state = np.argmax(B[:, 0]) # State with highest prob of 'Low' deaths
print(f"The temperature state most likely to produce 'Low' deaths is: {temp_labels[min_death_state]}")