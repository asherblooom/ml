import pandas as pd
import pymc as pm
import numpy as np
import seaborn as sns
from hmmlearn.hmm import CategoricalHMM
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance


def train_supervised_hmm(states, observations, n_states, n_obs):
    # Initialize matrices
    A = np.zeros((n_states, n_states)) # transition matrix
    B = np.zeros((n_states, n_obs)) # emission matrix
    pi = np.zeros(n_states) # start probabilities

    # Calculate start probabilities
    # assume the starting state is just as likely as any random state.
    # and so, count the occurrence of every state to get the stationary distribution
    for s in states:
        pi[s] += 1
    pi = pi / np.sum(pi)

    # Calculate transitions
    # Count how many times state i is followed by state j
    for t in range(len(states) - 1):
        current_state = states[t]
        next_state = states[t+1]
        A[current_state, next_state] += 1
        
    # Normalise rows of A (so probabilities sum to 1)
    # add a tiny epsilon to avoid division by zero if a state never transitions
    A = (A + 1e-8) / (A + 1e-8).sum(axis=1, keepdims=True)

    # Calculate emissions
    # Count how many times we see observation o while in state i
    for t in range(len(states)):
        current_state = states[t]
        current_obs = observations[t]
        B[current_state, current_obs] += 1
        
    # Normalise rows of B
    B = (B + 1e-8) / (B + 1e-8).sum(axis=1, keepdims=True)

    return pi, A, B

df = pd.read_csv(pm.get_data("deaths_and_temps_england_wales.csv"))

temp_discrete = pd.qcut(df['temp'], q=3, labels=False).values
deaths_discrete = pd.qcut(df['deaths'], q=3, labels=False).values
deaths_labels = ['Low', 'Medium', 'High']

hmm1 = CategoricalHMM(n_components=3)
pi, A, B = train_supervised_hmm(temp_discrete, deaths_discrete, 3, 3)
hmm1.startprob_ = pi
hmm1.transmat_ = A
hmm1.emissionprob_ = B

hmm2 = CategoricalHMM(3)
hmm2.fit(deaths_discrete.reshape(1, -1))

# Generate samples
X_sample_1, Z_sample_1 = hmm1.sample(n_samples=len(deaths_discrete))
X_sample_2, Z_sample_2 = hmm2.sample(n_samples=len(deaths_discrete))

# Compare Log Likelihoods
print(f"HMM1 (Supervised) Score:   {hmm1.score(deaths_discrete.reshape(1, -1))}")
print(f"HMM2 (Unsupervised) Score: {hmm2.score(deaths_discrete.reshape(1, -1))}")
# Compare Wasserstein distances
w_dist1 = wasserstein_distance(deaths_discrete, X_sample_1.flatten())
w_dist2 = wasserstein_distance(deaths_discrete, X_sample_2.flatten())
print(f"HMM1 Wasserstein Distance: {w_dist1}")
print(f"HMM2 Wasserstein Distance: {w_dist2}")



# Plot samples of both hmms against real data
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Real Data
sns.countplot(x=deaths_discrete.flatten(), ax=ax[0], color='gray')
ax[0].set_title("Real Data (Deaths)")
ax[0].set_xticks([0,1,2])
ax[0].set_xticklabels(deaths_labels)

# HMM1 Samples
sns.countplot(x=X_sample_1.flatten(), ax=ax[1], color='skyblue')
ax[1].set_title("HMM1 Generated (Supervised)")
ax[1].set_xticks([0,1,2])
ax[1].set_xticklabels(deaths_labels)

# HMM2 Samples
sns.countplot(x=X_sample_2.flatten(), ax=ax[2], color='salmon')
ax[2].set_title("HMM2 Generated (Unsupervised)")
ax[2].set_xticks([0,1,2])
ax[2].set_xticklabels(deaths_labels)

plt.savefig("hmmSamples.png")
