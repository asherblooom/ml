
import pandas as pd
import pymc as pm
import numpy as np
import seaborn as sns
from hmmlearn.hmm import CategoricalHMM
from matplotlib import pyplot as plt


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

temp_discrete = pd.qcut(df['temp'], q=3, labels=False).values.reshape(-1, 1)
# temp_discrete = temp_discrete.reshape(1, -1)
# temp_labels = ['Cold', 'Warm', 'Hot']
deaths_discrete = pd.qcut(df['deaths'], q=3, labels=False).values.reshape(-1, 1)
# deaths_discrete = deaths_discrete.reshape(1, -1)
deaths_labels = ['Low', 'Medium', 'High']

hmm1 = CategoricalHMM(n_components=3)
pi, A, B = train_supervised_hmm(temp_discrete, deaths_discrete, 3, 3)
hmm1.startprob_ = pi
hmm1.transmat_ = A
hmm1.emissionprob_ = B

hmm2 = CategoricalHMM(3)
lengths = [len(deaths_discrete)] # we only have one sequence, so we just take the length of that
hmm2.fit(deaths_discrete, lengths)

X1_sampled, discard = hmm1.sample(n_samples=10000)
X2_sampled, discard = hmm2.sample(n_samples=10000)




# Plot distributions
data_orig = deaths_discrete.flatten()
data_hmm1 = X1_sampled.flatten()
data_hmm2 = X2_sampled.flatten()

categories = [0, 1, 2]
labels = ['Low', 'Medium', 'High']

# List comprehensions to calculate the probability of each category
probs_orig = [np.mean(data_orig == cat) for cat in categories]
probs_hmm1 = [np.mean(data_hmm1 == cat) for cat in categories]
probs_hmm2 = [np.mean(data_hmm2 == cat) for cat in categories]

x = np.arange(len(labels))  # Label locations (0, 1, 2)
width = 0.25                # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars with offsets so they appear side-by-side
# Shift Original to left, HMM1 to center, HMM2 to right
rects1 = ax.bar(x - width, data_orig, width, label='Original Data', color='#1f77b4')
rects2 = ax.bar(x, data_hmm1, width, label='HMM1 Sampled', color='#ff7f0e')
rects3 = ax.bar(x + width, data_hmm2, width, label='HMM2 Sampled', color='#2ca02c')

# 4. Formatting
ax.set_ylabel('Probability')
ax.set_title('Comparison of Distributions: Original vs HMM Samples')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()