import pickle
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        X = batch[b'data']
        y = np.array(batch[b'labels'])
        return X, y

def load_cifar10(data_dir):
    X_all = []
    y_all = []
    # 5 training batches
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X, y = load_batch(batch_path)
        X_all.append(X)
        y_all.append(y)
    X_train = np.concatenate(X_all)
    y_train = np.concatenate(y_all)
    # test batch
    X_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))
    return X_train/255, y_train, X_test/255, y_test

# Load the data
# You may need to specify ‘data_dir’ the directory of dataset folder
data_dir = "cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_cifar10(data_dir)

pca = PCA(n_components=200)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# Define Hyperparameters to tune
# We will tune:
# C: Regularization parameter (trade-off between correct classification and margin size)
# gamma: Kernel coefficient (defines how far the influence of a single training example reaches)
# kernel: We focus on 'rbf' (Radial Basis Function) as it generally performs best for image data.
# We use 'reciprocal' for C and gamma. This creates a log-uniform distribution.
# It allows the search to explore 0.01, 0.1, 1, 10, 100 with equal probability.
param_dist = {
    'C': reciprocal(0.1, 1000),      # Regularization: Searches between 0.1 and 1000
    'gamma': reciprocal(0.0001, 0.1),# Kernel Coefficient: Searches between 0.0001 and 0.1
    'kernel': ['rbf'],               # RBF is standard for images. 'poly' is extremely slow on this data size.
}

svc = SVC()

# Perform Cross-Validation
# n_jobs=-1 uses all available processor cores
random_search = RandomizedSearchCV(
    estimator=svc, 
    param_distributions=param_dist, 
    n_iter=20,          # Try 20 random combinations
    cv=3,               # 3-fold CV
    scoring='accuracy', 
    n_jobs=-1,          # Use all cores
    verbose=2,
)

random_search.fit(X_train_reduced, y_train)
print(f"\nBest Parameters found: {random_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {random_search.best_score_:.4f}")

best_svm = random_search.best_estimator_
# best_svm.fit(X_train_reduced, y_train)

# Evaluate on Test Set
y_pred = best_svm.predict(X_test_reduced)
test_acc = accuracy_score(y_test, y_pred)
print(f"Final Test Set Accuracy: {test_acc:.4f}")