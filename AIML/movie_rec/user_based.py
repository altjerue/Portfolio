import numpy as np


class UserBasedRecommender:
    """
    Recommend movies based on similar users.

    Algorithm:
    1. Find users similar to target user (cosine similarity)
    2. Get movies they liked that target hasn't seen
    3. Rank by weighted average of similar users' ratings
    """

    def __init__(self, ratings_df):
        """
        Args:
            ratings_df: DataFrame with columns [user_id, movie_id, rating]
        """
        self.ratings = {}
        for u, m, r in ratings_df[["user_id", "movie_id", "rating"]].values:
            u_val = int(u)
            m_val = int(m)
            r_val = int(r)

            # Initialize user dict if doesn't exist
            if u_val not in self.ratings:
                self.ratings[u_val] = {}

            self.ratings[u_val][m_val] = r_val

    def compute_user_similarity(self, user1, user2):
        """
        Compute cosine similarity between two users.

        Cosine similarity = dot(u1, u2) / (||u1|| * ||u2||)

        Only consider movies both users have rated.

        Returns:
            similarity score between 0 and 1
        """
        # Implement cosine similarity
        u1_dict = self.ratings.get(user1, {})
        u2_dict = self.ratings.get(user2, {})

        # Find common movies
        common_movies = set(u1_dict.keys()) & set(u2_dict.keys())

        if len(common_movies) == 0:
            return 0.0  # No overlap

        # Compute dot product (only for common users)
        dot_product = 0
        u1_norm_sq = 0
        u2_norm_sq = 0

        for movie_id in common_movies:
            r1 = u1_dict[movie_id]
            r2 = u2_dict[movie_id]

            dot_product += r1 * r2
            u1_norm_sq += r1**2
            u2_norm_sq += r2**2

        u1_norm = np.sqrt(u1_norm_sq)
        u2_norm = np.sqrt(u2_norm_sq)

        if u1_norm == 0 or u2_norm == 0:
            return 0.0

        return dot_product / (u1_norm * u2_norm)

    def find_similar_users(self, target_user, k=10):
        """
        Find k most similar users to target_user.

        Returns:
            List of (user_id, similarity_score) tuples, sorted by similarity
        """
        # Compute similarity with all users
        similarities = []
        for u in self.ratings.keys():
            sim = self.compute_user_similarity(target_user, u)
            similarities.append((u, sim))

        similarities.sort(key=lambda t: (t[1], t[0]), reverse=True)

        return similarities[:k]

    def recommend(self, user_id, n=5):
        """
        Recommend n movies for user_id.

        Algorithm:
        1. Find similar users
        2. Get movies they rated highly that target hasn't seen
        3. Score = weighted average of similar users' ratings
        4. Return top n

        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        # Implement recommendation logic
        sim_users = self.find_similar_users(user_id)

        movie_sum = {}
        movie_weighted = {}

        # Get movies rated that user_id has not seen
        for _, sim in sim_users:
            if sim <= 0:
                continue
            for m, r in self.ratings[u].items():
                if m in self.ratings[user_id].keys():
                    continue
                movie_weighted[m] = movie_weighted.get(m, 0.0) + sim * r
                movie_sum[m] = movie_sum.get(m, 0.0) + sim

        recommend = []
        for m in movie_weighted.keys():
            if movie_sum[m] > 0.0:
                recommend.append((m, movie_weighted[m] / movie_sum[m]))

        recommend.sort(key=lambda t: (t[1], t[0]), reverse=True)

        return recommend[:n]
