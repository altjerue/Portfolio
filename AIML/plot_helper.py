import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, title, feature_names=None):
    """Plot decision boundary for 2D data."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="black", s=50)

    # Determine feature names
    if feature_names is None:
        if hasattr(model, "feature_names"):
            feature_names = model.feature_names[:2]
        else:
            feature_names = ["x1", "x2"]

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
