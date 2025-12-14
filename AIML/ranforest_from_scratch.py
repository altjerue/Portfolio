"""
Random Forest from Scratch
======================================================

Key concepts:
1. Bootstrap sampling: train each tree on random subset with replacement
2. Feature randomness: each split considers only sqrt(n_features) random features
3. Majority voting: aggregate predictions from all trees

This fixes the overfitting problem when using deep single decision trees.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from dectree_from_scratch import DecisionTreeClassifier

np.random.seed(42)


# ============================================================
# %% RANDOM FOREST IMPLEMENTATION
# ============================================================


class RandomForestClassifier:
    """
    Random Forest classifier built from scratch.

    Trains multiple decision trees on bootstrap samples and
    aggregates their predictions.
    """

    def __init__(
        self, n_trees=10, max_depth=10, min_samples_split=2, max_features=None
    ):
        """
        Args:
            n_trees: number of trees in the forest
            max_depth: maximum depth for each tree
            min_samples_split: minimum samples to split a node
            max_features: number of features to consider at each split
                         (None = use all features, 'sqrt' = sqrt(n_features))
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample of the data.

        Bootstrap sampling: randomly sample n_samples WITH REPLACEMENT.
        This means some samples may appear multiple times, others not at all.

        Args:
            X: feature matrix (n_samples, n_features)
            y: labels (n_samples,)

        Returns:
            X_sample, y_sample: bootstrap sample (same size as input)

        Example:
            Original indices: [0, 1, 2, 3, 4]
            Bootstrap sample might be: [1, 1, 3, 0, 4]
            (notice 1 appears twice, 2 doesn't appear)
        """
        n_samples = X.shape[0]

        # Generate random indices with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Return the samples at those indices
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        Train the random forest.

        Strategy:
        1. For each tree in the forest:
           a. Create bootstrap sample
           b. Train tree on that sample
           c. Store the tree
        """
        self.trees = []

        for i in range(self.n_trees):
            # Create bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # Create a decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )

            # Train the tree on bootstrap sample
            tree.fit(X_sample, y_sample)

            # Add tree to forest
            self.trees.append(tree)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"Trained tree {i + 1}/{self.n_trees}")

        return self

    def _predict_tree(self, tree, X):
        """
        Get predictions from a single tree.

        Args:
            tree: trained DecisionTreeClassifier
            X: samples to predict

        Returns:
            predictions from this tree
        """
        return tree.predict(X)

    def predict(self, X):
        """
        Predict using all trees and aggregate via majority voting.

        Strategy:
        1. Get predictions from each tree
        2. For each sample, count votes across all trees
        3. Return the most common vote (mode)

        Args:
            X: feature matrix (n_samples, n_features)

        Returns:
            predictions: array of predicted classes
        """

        tree_predictions = []

        # Collect predictions from all trees
        for tree in self.trees:
            # Get predictions from this tree
            preds = tree.predict(X)
            tree_predictions.append(preds)

        # Convert to numpy array: shape (n_trees, n_samples)
        tree_predictions = np.array(tree_predictions)

        final_predictions = []

        # Majority voting
        for i in range(X.shape[0]):
            # Get all tree predictions for sample i
            sample_votes = tree_predictions[:, i]

            # Find most common vote
            most_common = np.bincount(sample_votes).argmax()

            final_predictions.append(most_common)

        return np.array(final_predictions)


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


# ============================================================
# %% EXPERIMENT: Compare Single Tree vs Random Forest
# ============================================================


if __name__ == "__main__":
    # LOAD DATA (moved into guard so importing doesn't run the experiment)
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data[:, :2]  # 2 features for visualization
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("-" * 60)

    print("\n" + "=" * 60)
    print("EXPERIMENT: Single Tree vs Random Forest")
    print("=" * 60)

    # Train single decision tree (your baseline)
    print("\n1. Training single decision tree (max_depth=20)...")
    single_tree = DecisionTreeClassifier(max_depth=20)
    single_tree.fit(X_train, y_train)

    y_train_single = single_tree.predict(X_train)
    y_test_single = single_tree.predict(X_test)

    train_acc_single = np.mean(y_train_single == y_train)
    test_acc_single = np.mean(y_test_single == y_test)

    print(f"   Train Accuracy: {train_acc_single:.3f}")
    print(f"   Test Accuracy: {test_acc_single:.3f}")
    print(f"   Gap: {train_acc_single - test_acc_single:.3f}")

    # Train Random Forest with 30 trees
    print("\n2. Training Random Forest (30 trees, max_depth=10)...")
    rf1 = RandomForestClassifier(
        n_trees=30,
        max_depth=10,
        min_samples_split=5,
        max_features=None,
    )
    rf1.fit(X_train, y_train)

    y_train_rf1 = rf1.predict(X_train)
    y_test_rf1 = rf1.predict(X_test)

    train_acc_rf1 = np.mean(y_train_rf1 == y_train)
    test_acc_rf1 = np.mean(y_test_rf1 == y_test)

    print(f"   Train Accuracy: {train_acc_rf1:.3f}")
    print(f"   Test Accuracy: {test_acc_rf1:.3f}")
    print(f"   Gap: {train_acc_rf1 - test_acc_rf1:.3f}")

    # Train Random Forest with 100 trees and sqrt(n_)
    print("\n3. Training Random Forest (100 trees, max_depth=20)...")
    # Improved Random Forest with better hyperparameters
    rf2 = RandomForestClassifier(
        n_trees=100,
        max_depth=20,
        min_samples_split=4,
        max_features="sqrt",  # Only consider sqrt(n_features) at each split
    )
    rf2.fit(X_train, y_train)

    y_train_rf2 = rf2.predict(X_train)
    y_test_rf2 = rf2.predict(X_test)

    train_acc_rf2 = np.mean(y_train_rf2 == y_train)
    test_acc_rf2 = np.mean(y_test_rf2 == y_test)

    print(f"   Train Accuracy: {train_acc_rf2:.3f}")
    print(f"   Test Accuracy: {test_acc_rf2:.3f}")
    print(f"   Gap: {train_acc_rf2 - test_acc_rf2:.3f}")

    print("\n" + "-" * 60)
    print("COMPARISON:")
    print(f"Single Tree Test Acc: {test_acc_single:.3f}")
    print(f"Random Forest 30 Test Acc: {test_acc_rf1:.3f}")
    print(f"Improvement: {(test_acc_rf1 - test_acc_single):.3f}")
    print(f"Random Forest 100 Test Acc: {test_acc_rf2:.3f}")
    print(f"Improvement: {(test_acc_rf2 - test_acc_single):.3f}")

    # if test_acc_rf2 > test_acc_single:
    #     print("✅ Random Forest wins! Ensemble reduces overfitting.")
    # else:
    #     print("⚠️  Unexpected result - check implementation!")

    print("-" * 60)

    print("\nGenerating visualizations...")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plot_decision_boundary(
        single_tree,
        X_test,
        y_test,
        f"Single Tree (depth=20)\nTest Acc: {test_acc_single:.3f}",
    )

    plt.subplot(1, 3, 2)
    plot_decision_boundary(
        rf2, X_test, y_test, f"Random Forest (100 trees)\nTest Acc: {test_acc_rf2:.3f}"
    )

    plt.subplot(1, 3, 3)
    # Show how test accuracy improves with number of trees
    n_trees_list = [1, 5, 10, 20, 50, 100]
    test_accs = []

    for n in n_trees_list:
        rf_temp = RandomForestClassifier(
            n_trees=n, max_depth=20, min_samples_split=4, max_features="sqrt"
        )
        rf_temp.fit(X_train, y_train)
        acc = np.mean(rf_temp.predict(X_test) == y_test)
        test_accs.append(acc)

    plt.plot(n_trees_list, test_accs, "o-", linewidth=2, markersize=8)
    plt.axhline(
        y=test_acc_single,
        color="r",
        linestyle="--",
        label=f"Single Tree: {test_acc_single:.3f}",
    )
    plt.xlabel("Number of Trees")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Number of Trees")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
