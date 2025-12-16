"""
Gradient Boosting from Scratch
=========================================================

Key concepts:
1. Sequential training: each tree corrects previous mistakes
2. Residual fitting: train on errors, not original labels
3. Shrinkage: learning rate controls contribution of each tree
4. Additive model: final prediction = sum of all tree predictions

This is the foundation of XGBoost, LightGBM, CatBoost.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from dectree_from_scratch import DecisionTreeClassifier, DecisionTreeRegressor
from ranforest_from_scratch import RandomForestClassifier
from plot_helper import plot_decision_boundary

# ============================================================
# %% GRADIENT BOOSTING CLASSIFIER
# ============================================================


class GradientBoostingClassifier:
    """
    Gradient Boosting classifier for multi-class classification.

    Uses one-vs-all strategy: train separate boosting ensemble
    for each class.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Args:
            n_estimators: number of boosting rounds (trees)
            learning_rate: shrinkage parameter (0 < lr <= 1)
            max_depth: maximum depth of each tree (shallow trees!)
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []  # Will be list of lists: trees[class][tree_idx]
        self.init_predictions = None
        self.n_classes = None

    def _one_hot_encode(self, y):
        """
        Convert labels to one-hot encoding.

        Example: [0, 1, 2, 0] -> [[1,0,0], [0,1,0], [0,0,1], [1,0,0]]

        Args:
            y: class labels (n_samples,)

        Returns:
            one_hot: binary matrix (n_samples, n_classes)
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        one_hot = np.zeros((n_samples, n_classes))

        # Fill in one-hot encoding
        for i in range(n_samples):
            one_hot[i, y[i]] = 1.0

        return one_hot

    def _softmax(self, logits):
        """
        Convert raw scores (logits) to probabilities.

        Softmax: p_i = exp(logit_i) / sum(exp(logit_j))

        Args:
            logits: raw scores (n_samples, n_classes)

        Returns:
            probabilities: (n_samples, n_classes), rows sum to 1
        """
        # Numerical stability trick: subtract max
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)

        # Compute softmax
        exp_logits = np.exp(logits_stable)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        probs = exp_logits / sum_exp

        return probs

    def fit(self, X, y):
        """
        Train the gradient boosting model.

        Algorithm:
        1. Initialize predictions (all zeros or class frequencies)
        2. For each boosting round:
           a. Calculate residuals (true - predicted probabilities)
           b. For each class, train tree to predict residuals for that class
           c. Update predictions with new tree outputs (scaled by learning_rate)
        """
        n_samples = len(y)
        self.n_classes = len(np.unique(y))

        # Convert labels to one-hot
        y_one_hot = self._one_hot_encode(y)

        # Initialize predictions
        current_logits = np.zeros((n_samples, self.n_classes))

        # Store initial prediction (for later use in predict)
        self.init_predictions = np.zeros(self.n_classes)

        # Initialize tree storage: one list per class
        self.trees = [[] for _ in range(self.n_classes)]

        print(f"\nTraining Gradient Boosting with {self.n_estimators} rounds...")

        # Boosting loop
        for iteration in range(self.n_estimators):
            # Convert logits to probabilities
            probs = self._softmax(current_logits)

            # Calculate residuals (gradients)
            residuals = y_one_hot - probs

            # Train one tree per class
            for class_idx in range(self.n_classes):
                # Get residuals for this class
                class_residuals = residuals[:, class_idx]

                # Train tree to predict these residuals
                tree = DecisionTreeRegressor(max_depth=self.max_depth)

                # Fit tree on X and class_residuals
                tree.fit(X, class_residuals)

                # Store the tree
                self.trees[class_idx].append(tree)

                # Update predictions for this class
                predictions = tree.predict(X)
                current_logits[:, class_idx] += self.learning_rate * predictions

            # Print progress
            if (iteration + 1) % 20 == 0 or iteration == 0:
                # Calculate training accuracy
                train_preds = np.argmax(current_logits, axis=1)
                train_acc = np.mean(train_preds == y)
                print(
                    f"  Round {iteration + 1}/{self.n_estimators}, Train Acc: {train_acc:.3f}"
                )

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: feature matrix (n_samples, n_features)

        Returns:
            probabilities: (n_samples, n_classes)
        """
        n_samples = X.shape[0]

        # Start with initial predictions
        logits = np.tile(self.init_predictions, (n_samples, 1))

        # Add contributions from all trees
        for class_idx in range(self.n_classes):
            for tree in self.trees[class_idx]:
                # Get predictions from this tree
                predictions = tree.predict(X)

                # Add to logits (scaled by learning_rate)
                logits[:, class_idx] += self.learning_rate * predictions

        # Convert logits to probabilities
        probs = self._softmax(logits)

        return probs

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: feature matrix (n_samples, n_features)

        Returns:
            predictions: class labels (n_samples,)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


if __name__ == "__main__":
    # ============================================================
    # LOAD DATA
    # ============================================================
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data[:, :2]  # 2 features for visualization
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {iris.target_names}")
    print("-" * 60)

    # ============================================================
    # TRAINING
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING GRADIENT BOOSTING")
    print("=" * 60)

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

    gb.fit(X_train, y_train)

    # Evaluate
    y_train_pred = gb.predict(X_train)
    y_test_pred = gb.predict(X_test)

    train_acc = np.mean(y_train_pred == y_train)
    test_acc = np.mean(y_test_pred == y_test)

    print("\n" + "-" * 60)
    print("RESULTS:")
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Gap: {train_acc - test_acc:.3f}")
    print("-" * 60)

    # ============================================================
    # COMPARISON: Single Tree vs RF vs Gradient Boosting
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARISON: Single Tree vs RF vs Gradient Boosting")
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

    # Train Random Forest with 100 trees and sqrt(n_)
    print("\n2. Training Random Forest")
    rf = RandomForestClassifier(
        n_trees=100,
        max_depth=20,
        min_samples_split=4,
        max_features="sqrt",
    )
    rf.fit(X_train, y_train)

    y_train_rf = rf.predict(X_train)
    y_test_rf = rf.predict(X_test)

    train_acc_rf = np.mean(y_train_rf == y_train)
    test_acc_rf = np.mean(y_test_rf == y_test)

    print(f"   Train Accuracy: {train_acc_rf:.3f}")
    print(f"   Test Accuracy: {test_acc_rf:.3f}")
    print(f"   Gap: {train_acc_rf - test_acc_rf:.3f}")

    # Train Gradient Boosting with 100 trees
    print("\n3. Training Gradient Boosting")
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    gbc.fit(X_train, y_train)

    y_train_gbc = gbc.predict(X_train)
    y_test_gbc = gbc.predict(X_test)

    train_acc_gbc = np.mean(y_train_gbc == y_train)
    test_acc_gbc = np.mean(y_test_gbc == y_test)

    print(f"   Train Accuracy: {train_acc_gbc:.3f}")
    print(f"   Test Accuracy: {test_acc_gbc:.3f}")
    print(f"   Gap: {train_acc_gbc - test_acc_gbc:.3f}")

    print("\nTypical results on Iris (2 features):")
    print(f"Single Tree (depth=20):    {test_acc_single:.2f} test accuracy")
    print(f"Random Forest (100 trees): {test_acc_rf:.2f} test accuracy")
    print(f"Gradient Boosting:         {train_acc_gbc:.2f} test accuracy")

    # ============================================================
    # VISUALIZATION
    # ============================================================
    print("\nGenerating visualizations...")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plot_decision_boundary(
        gb,
        X_train,
        y_train,
        f"Training Data\nAcc: {train_acc:.3f}",
        feature_names=iris.feature_names[:2],
    )

    plt.subplot(1, 3, 2)
    plot_decision_boundary(
        gb,
        X_test,
        y_test,
        f"Test Data\nAcc: {test_acc:.3f}",
        feature_names=iris.feature_names[:2],
    )

    plt.subplot(1, 3, 3)
    # Show how accuracy improves with boosting rounds
    n_rounds = [1, 5, 10, 20, 50, 100]
    test_accs = []

    for n in n_rounds:
        gb_temp = GradientBoostingClassifier(
            n_estimators=n, learning_rate=0.1, max_depth=3
        )
        gb_temp.fit(X_train, y_train)
        acc = np.mean(gb_temp.predict(X_test) == y_test)
        test_accs.append(acc)

    plt.plot(n_rounds, test_accs, "o-", linewidth=2, markersize=8, color="green")
    plt.xlabel("Number of Boosting Rounds")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Boosting Rounds")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
