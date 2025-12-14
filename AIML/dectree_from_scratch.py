"""
Decision Tree from Scratch
===========================================================

Implementing a decision tree classifier from scratch.

Key implementations:
1. Gini impurity to measure of node purity
2. Information gain for reduction in impurity from a split
3. Recursive splitting to build tree top-down
4. Prediction bry traversing tree to leaf

Dataset: Iris classification (classic ML dataset)
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ============================================================
# %% LOAD DATA
# ============================================================
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data  # 4 features: sepal length/width, petal length/width
y = iris.target  # 3 classes: setosa, versicolor, virginica

# Use only first 2 features for easy visualization
X = X[:, :2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {iris.feature_names[:2]}")
print(f"Classes: {iris.target_names}")
print("-" * 60)


# ============================================================
# %% HELPER FUNCTIONS
# ============================================================


def gini_impurity(y):
    r"""
    Calculate Gini impurity for a set of labels.

    $$Gini = 1 - \sum p_i^2,$$ where $p_i$ is proportion of the
    $i$-th class

    Args:
        y: array of labels (e.g., [0, 1, 1, 0, 2])

    Returns:
        Gini impurity (float between 0 and 1)
        0 = pure (all same class)
        ~0.67 = maximum impurity for 3 classes

    Examples:
        gini_impurity([0, 0, 0]) = 0 (pure)
        gini_impurity([0, 1, 0, 1]) = 0.5 (50/50 split)
    """

    if len(y) == 0:
        return 0

    counts = np.bincount(y)
    proportions = counts / sum(counts)
    gini = 1 - sum(proportions**2)

    return gini


def split_data(X, y, feature_idx, threshold):
    """
    Split data based on a feature and threshold.

    Args:
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)
        feature_idx: which feature to split on (0 to n_features-1)
        threshold: split threshold

    Returns:
        left_X, left_y: samples where X[:, feature_idx] <= threshold
        right_X, right_y: samples where X[:, feature_idx] > threshold

    Example:
        If feature_idx=0 (sepal length) and threshold=5.5:
        - Left: all flowers with sepal length <= 5.5
        - Right: all flowers with sepal length > 5.5
    """

    mask = X[:, feature_idx] <= threshold

    left_X = X[mask]
    left_y = y[mask]
    right_X = X[~mask]
    right_y = y[~mask]

    return left_X, left_y, right_X, right_y


def information_gain(y, left_y, right_y):
    """
    Calculate information gain from a split.

    Information Gain = Gini(parent) - weighted_avg(Gini(children))

    Args:
        y: parent labels
        left_y: left child labels
        right_y: right child labels

    Returns:
        Information gain (float)
        Higher = better split
    """
    parent_gini = gini_impurity(y)

    n_total = len(y)
    n_left = len(left_y)
    n_right = len(right_y)

    if n_left == 0 or n_right == 0:
        return 0  # No split happened

    left_gini = gini_impurity(left_y)
    right_gini = gini_impurity(right_y)

    # Weighted average of child Gini impurities
    weighted_child_gini = (n_left / n_total) * left_gini + (
        n_right / n_total
    ) * right_gini

    gain = parent_gini - weighted_child_gini

    return gain


def find_best_split(X, y):
    """
    Find the best feature and threshold to split on.

    Strategy: Try all features and many thresholds, pick the one
    with highest information gain.

    Args:
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)

    Returns:
        best_feature: index of best feature to split on
        best_threshold: threshold value for split
        best_gain: information gain achieved
    """
    best_gain = -1
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]

    for feature_idx in range(n_features):
        # Get unique values of this feature as candidate thresholds
        # We'll try splitting at each unique value
        thresholds = np.unique(X[:, feature_idx])

        for threshold in thresholds:
            left_X, left_y, right_X, right_y = split_data(X, y, feature_idx, threshold)

            # Skip if split creates empty node
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            gain = information_gain(y, left_y, right_y)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


# ============================================================
# %% DECISION TREE NODE
# ============================================================


class TreeNode:
    """
    A node in the decision tree.

    Internal nodes have:
    - feature: which feature to split on
    - threshold: split threshold
    - left: left child (TreeNode)
    - right: right child (TreeNode)

    Leaf nodes have:
    - value: predicted class (most common class in node)
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes

    def is_leaf(self):
        return self.value is not None


# ============================================================
# %% DECISION TREE CLASSIFIER
# ============================================================


class DecisionTreeClassifier:
    """
    Decision tree classifier built from scratch.
    """

    def __init__(self, max_depth=10, min_samples_split=2):
        """
        Args:
            max_depth: maximum tree depth (prevents overfitting)
            min_samples_split: minimum samples required to split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """
        Build the decision tree.

        Uses recursive splitting until stopping criteria are met.
        """
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """
        Recursively build the tree.

        Stopping criteria:
        1. Reached max_depth
        2. Not enough samples to split (< min_samples_split)
        3. Node is pure (all same class)
        4. No information gain from splitting

        Args:
            X: feature matrix
            y: labels
            depth: current depth in tree

        Returns:
            TreeNode (internal node or leaf)
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_classes == 1
        ):
            leaf_value = np.bincount(y).argmax()
            return TreeNode(value=leaf_value)

        feature, threshold, gain = find_best_split(X, y)

        # If no gain OR no valid split found, make this a leaf
        if gain == 0 or feature is None:
            leaf_value = np.bincount(y).argmax()
            return TreeNode(value=leaf_value)

        left_X, left_y, right_X, right_y = split_data(X, y, feature, threshold)

        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)

        # Create internal node
        return TreeNode(
            feature=feature, threshold=threshold, left=left_child, right=right_child
        )

    def predict(self, X):
        """
        Predict class for samples.

        Args:
            X: feature matrix (n_samples, n_features)

        Returns:
            predictions: array of predicted classes
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse tree to find prediction for a single sample.

        Args:
            x: single sample (n_features,)
            node: current TreeNode

        Returns:
            predicted class
        """

        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def print_tree(self, node=None, depth=0):
        """
        Print tree structure (for debugging).
        """
        if node is None:
            node = self.root

        indent = "  " * depth
        if node.is_leaf():
            print(f"{indent}Leaf: class={node.value}")
        else:
            print(f"{indent}Split: feature_{node.feature} <= {node.threshold:.2f}")
            print(f"{indent}Left:")
            self.print_tree(node.left, depth + 1)
            print(f"{indent}Right:")
            self.print_tree(node.right, depth + 1)


# ============================================================
# %% TRAINING
# ============================================================

print("\nTraining decision tree...")
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
tree.fit(X_train, y_train)

print("\nTree structure:")
tree.print_tree()
print("-" * 60)


# ============================================================
# %% EVALUATION
# ============================================================

y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

train_accuracy = np.mean(y_train_pred == y_train)
test_accuracy = np.mean(y_test_pred == y_test)

print(f"\nTrain Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")
print(f"Accuracy Gap: {train_accuracy - test_accuracy:.3f}")

if train_accuracy - test_accuracy > 0.1:
    print("⚠️ Large gap suggests overfitting!")
else:
    print("✅ Good generalization!")

print("-" * 60)


# ============================================================
# %% VISUALIZATION
# ============================================================


def plot_decision_boundary(model, X, y, title):
    """Plot decision boundary for 2D data."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="black", s=50)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(title)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plot_decision_boundary(
    tree, X_train, y_train, f"Training Data\nAccuracy: {train_accuracy:.3f}"
)

plt.subplot(1, 3, 2)
plot_decision_boundary(
    tree, X_test, y_test, f"Test Data\nAccuracy: {test_accuracy:.3f}"
)

plt.subplot(1, 3, 3)
# Show overfitting by training a deep tree
deep_tree = DecisionTreeClassifier(max_depth=20, min_samples_split=2)
deep_tree.fit(X_train, y_train)
y_test_deep = deep_tree.predict(X_test)
test_acc_deep = np.mean(y_test_deep == y_test)
plot_decision_boundary(
    deep_tree,
    X_test,
    y_test,
    f"Deep Tree (max_depth=20)\nTest Accuracy: {test_acc_deep:.3f}",
)

plt.tight_layout()
plt.show()
