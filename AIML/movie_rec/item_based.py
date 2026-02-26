import numpy as np


class ItemBasedRecommender:
    """
    Recommend movies based on similar movies.

    "Users who liked this movie also liked..."
    """

    def __init__(self, ratings_df):
        self.ratings = {}
        for u, m, r in ratings_df[["user_id", "movie_id", "rating"]].values:
            m_val = int(m)
            u_val = int(u)
            r_val = int(r)

            # Initialize movie dict if doesn't exist
            if m_val not in self.ratings:
                self.ratings[m_val] = {}

            self.ratings[m_val][u_val] = r_val

    def compute_movie_similarity(self, movie1, movie2):
        """
        Compute similarity between two movies.

        Based on users who rated both movies.
        """
        # Implement cosine similarity
        m1_dict = self.ratings.get(movie1, {})
        m2_dict = self.ratings.get(movie2, {})

        # Find common users (users who rated both movies)
        common_users = set(m1_dict.keys()) & set(m2_dict.keys())

        if len(common_users) == 0:
            return 0.0  # No overlap

        # Compute dot product (only for common users)
        dot_product = 0
        m1_norm_sq = 0
        m2_norm_sq = 0

        for user_id in common_users:
            r1 = m1_dict[user_id]
            r2 = m2_dict[user_id]

            dot_product += r1 * r2
            m1_norm_sq += r1**2
            m2_norm_sq += r2**2

        m1_norm = np.sqrt(m1_norm_sq)
        m2_norm = np.sqrt(m2_norm_sq)

        if m1_norm == 0 or m2_norm == 0:
            return 0.0

        return dot_product / (m1_norm * m2_norm)

    def recommend(self, user_id, n=5):
        """
        Recommend based on movies user already liked.

        Algorithm:
        1. Get movies user rated highly (4-5 stars)
        2. For each, find similar movies
        3. Aggregate scores from all similar movies
        4. Return top n that user hasn't seen
        """
        # Find movies this user has rated
        user_ratings = {}  # {movie_id: rating}

        for movie_id, users_dict in self.ratings.items():
            if user_id in users_dict:
                user_ratings[movie_id] = users_dict[user_id]

        # Get highly rated movies (4-5 stars)
        liked_movies = [m for m, r in user_ratings.items() if r >= 4]

        if len(liked_movies) == 0:
            return []  # User has not liked anything

        # For each liked movie, find similar movies
        candidate_scores = {}  # {movie_id: aggregated_score}

        for liked_movie in liked_movies:
            # Find all movies and compute similarity
            for candidate_movie in self.ratings.keys():
                # Skip if user already rated it
                if candidate_movie in user_ratings:
                    continue

                # Compute similarity
                sim = self.compute_movie_similarity(liked_movie, candidate_movie)

                # Aggregate score (weighted by similarity)
                if candidate_movie not in candidate_scores:
                    candidate_scores[candidate_movie] = 0

                # Weight by similarity and user's rating of liked_movie
                candidate_scores[candidate_movie] += sim * user_ratings[liked_movie]

        # Sort by score and return top n
        recommendations = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )

        return recommendations[:n]
