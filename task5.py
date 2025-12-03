import pickle
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint


def load_batch(batch_path):
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        X = batch[b"data"]
        y = np.array(batch[b"labels"])
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
    return X_train / 255, y_train, X_test / 255, y_test


data_dir = "cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_cifar10(data_dir)

pca = PCA(n_components=200)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# Hyperparameters
param_dist = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": randint(2, 50),
    "min_samples_leaf": randint(1, 20),
    "max_features": ["sqrt", "log2"],
}

dt_clf = DecisionTreeClassifier()

# Perform Cross-Validation
# n_jobs=-1 uses all available processor cores
random_search = RandomizedSearchCV(
    estimator=dt_clf,
    param_distributions=param_dist,
    n_iter=100,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
)

random_search.fit(X_train_reduced, y_train)
print(f"Best Parameters found: {random_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {random_search.best_score_:.4f}")

best_clf = random_search.best_estimator_

# Evaluate on Test Set
y_pred = best_clf.predict(X_test_reduced)
test_acc = accuracy_score(y_test, y_pred)
print(f"Final Test Set Accuracy: {test_acc:.4f}")

