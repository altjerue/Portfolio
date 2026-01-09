import numpy as np


class MatrixFactorization:
    """
    Approximate ratings matrix R as R ≈ U @ V.T

    Where:
    - U: user embeddings (n_users, k)
    - V: movie embeddings (n_movies, k)
    - k: latent factors (e.g., k=20)
    """

    def __init__(self, n_users, n_movies, k=20, learning_rate=0.01, reg=0.01):
        # Initialize U and V with small random values
        self.U = np.random.random((n_users, k))
        self.V = np.random.random((n_movies, k))
        self.k = k
        self.lr = learning_rate
        self.reg = reg

    def predict(self, user_id, movie_id):
        """
        Predict rating for user_id on movie_id.

        prediction = U[user_id] @ V[movie_id]
        """
        return self.U[user_id, :] @ self.V[movie_id, :]

    def train(self, ratings_df, epochs=100):
        """
        Train using gradient descent.

        For each rating (u, m, r):
        1. Compute prediction: r_pred = U[u] @ V[m]
        2. Compute error: e = r - r_pred
        3. Update: U[u] += lr * (e * V[m] - reg * U[u])
        4. Update: V[m] += lr * (e * U[u] - reg * V[m])
        """
        # Loop over epochs
        for epoch in range(epochs):
            for u, m, r in ratings_df[["user_id", "movie_id", "rating"]].values:
                r_pred = self.U[u, :] @ self.V[m, :]
                e = r - r_pred
                self.U[u, :] += self.lr * (e * self.V[m, :] - self.reg * self.U[u, :])
                self.V[m, :] += self.lr * (e * self.U[u, :] - self.reg * self.V[m, :])

            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                loss = self.compute_loss(ratings_df)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def compute_loss(self, ratings_df):
        """Compute RMSE on all ratings"""
        total_error = 0
        for u, m, r in ratings_df[["user_id", "movie_id", "rating"]].values:
            r_pred = self.predict(u, m)
            total_error += (r - r_pred) ** 2
        return np.sqrt(total_error / len(ratings_df))

    def recommend(self, user_id, seen_movies, n=5):
        """
        Recommend by computing predictions for all unseen movies.

        Args:
            user_id: User to recommend for
            seen_movies: Set of movie_ids user has already rated
            n: Number of recommendations

        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        # Get user's embedding vector
        user_vector = self.U[user_id, :]  # Shape: (k,)

        # Compute predictions for ALL movies at once
        all_predictions = user_vector @ self.V.T  # Shape: (n_movies,)

        # Create list of (movie_id, predicted_rating)
        movie_scores = []
        for movie_id in range(len(all_predictions)):
            if movie_id not in seen_movies:
                movie_scores.append((movie_id, all_predictions[movie_id]))

        # Sort by predicted rating (descending) and return top n
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        return movie_scores[:n]
