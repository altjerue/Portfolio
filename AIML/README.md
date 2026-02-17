# Machine Learning from First Principles

I implemented these algorithms from scratch to understand what XGBoost, scikit-learn, and PyTorch are actually doing under the hood. If you can build it, you can debug it, extend it, and explain it to anyone.

This directory contains clean Python implementations of core ML algorithms built without ML libraries, using only NumPy. Each implementation covers the full mathematical derivation, working code, empirical benchmarks, and visualizations.

---

## Why This Matters

Most data scientists can call `sklearn.ensemble.GradientBoostingClassifier`. Far fewer can explain why residual fitting converges, what the softmax gradient looks like, or why shallow trees are used as weak learners. These implementations demonstrate mathematical depth that goes beyond API familiarity.

---

## Implementations

### 1. Neural Network

**File:** [`neuralnet_from_scratch.py`](./neuralnet_from_scratch.py)

**Problem solved:** The XOR problem, which is not linearly separable and cannot be solved by a single-layer perceptron.

**Architecture:**
```
Input (2) -> Hidden Layer (4, tanh) -> Output (1)
```

**Key concepts implemented:**
- Forward propagation with tanh activation
- Mean Squared Error loss
- Backpropagation via chain rule
- Gradient descent weight updates

**Mathematical core:**

The forward pass computes:
```
z1 = W1 @ x + b1
h1 = tanh(z1)
z2 = W2 @ h1 + b2
```

Backpropagation uses the chain rule to compute gradients for each parameter. The derivative of tanh is particularly clean: `tanh'(z) = 1 - tanh(z)^2`, which is why we cache `h1` (the output of tanh, not the input) during the forward pass.

**Result:**
```
Input: [0 0], True: 0, Predicted: 0.0000
Input: [0 1], True: 1, Predicted: 1.0000
Input: [1 0], True: 1, Predicted: 1.0000
Input: [1 1], True: 0, Predicted: 0.0000
```
Loss converges to ~0 by epoch 1000.

---

### 2. Decision Tree (Classifier + Regressor)

**File:** [`dectree_from_scratch.py`](./dectree_from_scratch.py)

**Problem solved:** Multi-class classification on Iris dataset; regression target for use as weak learners in gradient boosting.

**Key concepts implemented:**
- Gini impurity for classification
- MSE impurity for regression
- Information gain calculation
- Recursive tree building with stopping criteria
- Tree traversal for prediction

**Mathematical core:**

Gini impurity measures node purity:
```
Gini = 1 - sum(p_i^2)
```
where `p_i` is the proportion of class i. A pure node (all same class) has Gini = 0.

Information gain drives split selection:
```
Gain = Gini(parent) - [n_left/n * Gini(left) + n_right/n * Gini(right)]
```

The `find_best_split` function tries all features and all unique threshold values, selecting the split maximizing information gain. Feature subsampling (`max_features='sqrt'`) is supported for use in Random Forest.

**Result (max_depth=5, 2 features):**
```
Train Accuracy: 0.848
Test Accuracy:  0.778
Gap:            0.070  (good generalization)
```

**Key design decision:** Both a `DecisionTreeClassifier` (leaf = majority class) and `DecisionTreeRegressor` (leaf = mean value) are implemented, since gradient boosting requires regression trees even for classification tasks.

---

### 3. Random Forest

**File:** [`ranforest_from_scratch.py`](./ranforest_from_scratch.py)

**Problem solved:** Reducing overfitting of a single deep decision tree through ensemble averaging.

**Key concepts implemented:**
- Bootstrap sampling (with replacement)
- Feature randomness at each split (sqrt subsampling)
- Majority voting for aggregation
- Comparison with single tree baseline

**Mathematical intuition:**

A single deep tree memorizes training data (high variance). Random Forest reduces variance by averaging many uncorrelated trees. The two sources of randomness, bootstrap sampling and feature subsampling, ensure trees are decorrelated even when trained on similar data.

Variance reduction follows:
```
Var(mean of n trees) = rho * sigma^2 + (1-rho)/n * sigma^2
```
where `rho` is the correlation between trees. Lower correlation = more variance reduction.

**Benchmark results (100 trees, Iris, 2 features):**
```
Single Tree (depth=20):      Test Acc: 0.689  Gap: 0.273
Random Forest (30 trees):    Test Acc: 0.711  Gap: 0.203
Random Forest (100 trees):   Test Acc: 0.756  Gap: 0.187
```
The ensemble reduces both overfitting (gap) and test error simultaneously.

---

### 4. Gradient Boosting

**File:** [`gradboost_from_scratch.py`](./gradboost_from_scratch.py)

**Problem solved:** Multi-class classification using sequential ensemble of regression trees.

**Key concepts implemented:**
- One-vs-all multi-class strategy
- Softmax for probability conversion
- Residual fitting (pseudo-gradients)
- Shrinkage (learning rate)
- Additive model updates
- Comparison with single tree and random forest

**Mathematical core:**

Gradient boosting performs functional gradient descent. At each iteration:

1. Convert current logits to probabilities via softmax:
```
p_i = exp(logit_i) / sum(exp(logit_j))
```

2. Compute residuals (negative gradient of cross-entropy loss):
```
residuals = y_one_hot - probabilities
```

3. For each class, fit a regression tree to its residuals

4. Update logits with shrinkage:
```
logits[:, k] += learning_rate * tree_k.predict(X)
```

This is exactly what XGBoost and LightGBM do, with additional optimizations (second-order gradients, histogram binning, etc.).

**Why shallow trees?** Deep trees overfit residuals. Shallow trees (max_depth=3-5) are "weak learners" that capture the most important structure without memorizing noise. Many weak learners sum to a strong learner.

**Benchmark results (100 rounds, Iris, all 4 features):**
```
Single Tree (depth=20):      Test Acc: 0.956  Gap: 0.044
Random Forest (100 trees):   Test Acc: 1.000  Gap: 0.000
Gradient Boosting (lr=0.1):  Test Acc: 1.000  Gap: 0.000
```

Training progression shows gradient boosting converging systematically:
```
Round  1/100,  Train Acc: 0.971
Round 20/100,  Train Acc: 0.981
Round 40/100,  Train Acc: 0.990
Round 60/100,  Train Acc: 1.000
```

---

## Key Design Patterns

**Shared infrastructure:** All tree-based models share the same `TreeNode`, `find_best_split`, `split_data`, and `information_gain` functions from `dectree_from_scratch.py`. This mirrors how production libraries like scikit-learn structure their code.

**Composability:** `GradientBoostingClassifier` imports `DecisionTreeRegressor` directly, demonstrating that gradient boosting is not a monolithic algorithm but an ensemble framework that accepts any weak learner.

**Both classifier and regressor for Decision Tree:** This is a deliberate design choice. Gradient boosting for classification uses regression trees internally (fitting residuals, which are continuous), not classification trees.

---

## Running the Code

```bash
# Decision Tree
python dectree_from_scratch.py

# Random Forest
python ranforest_from_scratch.py

# Gradient Boosting
python gradboost_from_scratch.py

# Neural Network
python neuralnet_from_scratch.py
```

**Dependencies:** `numpy`, `matplotlib`, `scikit-learn` (for datasets and train/test split only)

---

## What's Next

- [ ] XGBoost extensions: second-order gradients, regularization terms
- [ ] K-Means and DBSCAN clustering from scratch
- [ ] Logistic Regression with L1/L2 regularization
- [ ] Cross-validation and hyperparameter tuning framework
- [ ] PyTorch CNN for image classification
