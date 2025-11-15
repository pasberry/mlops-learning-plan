"""Generate sample movie recommendation data."""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_recommendation_data(
    n_users=1000,
    n_movies=500,
    n_interactions=10000
):
    """Generate synthetic user-movie interaction data.

    Features:
    - User features: age, gender, favorite_genre, activity_level
    - Movie features: genre, year, duration, avg_rating
    - Interaction: rating (1-5 stars)

    We'll create:
    - users.parquet: User features
    - movies.parquet: Movie features
    - interactions.parquet: User-movie ratings
    """
    np.random.seed(42)

    # === User Features ===
    user_ids = np.arange(n_users)
    user_ages = np.random.randint(18, 70, n_users)
    user_genders = np.random.choice(['M', 'F'], n_users)

    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance']
    user_fav_genres = np.random.choice(genres, n_users)

    # Activity level (how often they watch)
    user_activity = np.random.choice(['low', 'medium', 'high'], n_users, p=[0.3, 0.5, 0.2])

    users_df = pd.DataFrame({
        'user_id': user_ids,
        'age': user_ages,
        'gender': user_genders,
        'favorite_genre': user_fav_genres,
        'activity_level': user_activity
    })

    # One-hot encode categorical features
    users_df = pd.get_dummies(users_df, columns=['gender', 'favorite_genre', 'activity_level'])

    # Normalize age
    users_df['age'] = (users_df['age'] - users_df['age'].mean()) / users_df['age'].std()

    # === Movie Features ===
    movie_ids = np.arange(n_movies)
    movie_genres = np.random.choice(genres, n_movies)
    movie_years = np.random.randint(1990, 2024, n_movies)
    movie_durations = np.random.randint(80, 180, n_movies)  # minutes
    movie_avg_ratings = np.random.uniform(2.0, 5.0, n_movies)

    movies_df = pd.DataFrame({
        'movie_id': movie_ids,
        'genre': movie_genres,
        'year': movie_years,
        'duration': movie_durations,
        'avg_rating': movie_avg_ratings
    })

    # One-hot encode genre
    movies_df = pd.get_dummies(movies_df, columns=['genre'])

    # Normalize numerical features
    for col in ['year', 'duration', 'avg_rating']:
        movies_df[col] = (movies_df[col] - movies_df[col].mean()) / movies_df[col].std()

    # === Interactions (User-Movie Ratings) ===
    # Generate interactions with some logic:
    # - Users tend to like movies in their favorite genre
    # - Higher activity users have more interactions
    # - Ratings influenced by movie avg_rating and genre match

    interaction_user_ids = np.random.choice(user_ids, n_interactions)
    interaction_movie_ids = np.random.choice(movie_ids, n_interactions)

    # Create ratings with some logic
    ratings = []
    for user_id, movie_id in zip(interaction_user_ids, interaction_movie_ids):
        user = users_df[users_df['user_id'] == user_id].iloc[0]
        movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]

        # Base rating from movie avg_rating (denormalize)
        base_rating = 3.5

        # Boost if genre matches user preference
        # (Check if user has favorite_genre_X and movie has genre_X both = 1)
        genre_match = 0
        for genre in genres:
            user_col = f'favorite_genre_{genre}'
            movie_col = f'genre_{genre}'
            if user_col in user and movie_col in movie:
                if user[user_col] == 1 and movie[movie_col] == 1:
                    genre_match = 1
                    break

        if genre_match:
            base_rating += np.random.uniform(0.5, 1.5)
        else:
            base_rating += np.random.uniform(-1.0, 0.5)

        # Add noise
        rating = base_rating + np.random.normal(0, 0.5)

        # Clip to 1-5
        rating = np.clip(rating, 1, 5)

        ratings.append(rating)

    interactions_df = pd.DataFrame({
        'user_id': interaction_user_ids,
        'movie_id': interaction_movie_ids,
        'rating': ratings
    })

    # Create binary label: 1 if rating >= 4 (liked), 0 otherwise
    interactions_df['label'] = (interactions_df['rating'] >= 4.0).astype(int)

    # Remove duplicates (keep last interaction)
    interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')

    print(f"Generated data:")
    print(f"  Users: {len(users_df)}")
    print(f"  Movies: {len(movies_df)}")
    print(f"  Interactions: {len(interactions_df)}")
    print(f"  Positive rate: {interactions_df['label'].mean():.3f}")

    return users_df, movies_df, interactions_df


def create_train_val_test_splits(users_df, movies_df, interactions_df, output_dir):
    """Create train/val/test splits for ranking.

    Strategy:
    - Split interactions by user (chronological would be better in production)
    - Keep all users and movies in features
    """
    n = len(interactions_df)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    # Shuffle
    interactions_df = interactions_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split
    train_interactions = interactions_df[:n_train]
    val_interactions = interactions_df[n_train:n_train + n_val]
    test_interactions = interactions_df[n_train + n_val:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save user and movie features (same for all splits)
    users_df.to_parquet(output_dir / 'users.parquet', index=False)
    movies_df.to_parquet(output_dir / 'movies.parquet', index=False)

    # Save interaction splits
    for split, split_df in [('train', train_interactions),
                            ('val', val_interactions),
                            ('test', test_interactions)]:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        split_df.to_parquet(split_dir / 'interactions.parquet', index=False)
        print(f"Saved {len(split_df)} interactions to {split_dir}/")

    print(f"\n✅ Data saved to {output_dir}/")
    print(f"   - users.parquet ({len(users_df)} users)")
    print(f"   - movies.parquet ({len(movies_df)} movies)")
    print(f"   - train/interactions.parquet ({len(train_interactions)} interactions)")
    print(f"   - val/interactions.parquet ({len(val_interactions)} interactions)")
    print(f"   - test/interactions.parquet ({len(test_interactions)} interactions)")


if __name__ == '__main__':
    # Generate data
    users_df, movies_df, interactions_df = generate_recommendation_data(
        n_users=1000,
        n_movies=500,
        n_interactions=10000
    )

    # Save splits
    create_train_val_test_splits(
        users_df,
        movies_df,
        interactions_df,
        output_dir='data/recommendations/v1'
    )
