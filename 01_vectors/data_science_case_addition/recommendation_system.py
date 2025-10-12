"""
Data Science & Machine Learning Case Study
==========================================

Vector Addition in Recommendation Systems
This example shows how vector addition is used in real-world ML applications.
"""

import numpy as np
import matplotlib.pyplot as plt

def create_user_profile():
    """Create a user profile with genre preferences."""
    print("=== User Profile Creation ===")
    
    # Define genre categories
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
    
    # User's genre preferences (0-1 scale, where 1 = loves, 0 = hates)
    user_preferences = np.array([0.8, 0.6, 0.9, 0.2, 0.7, 0.5])
    
    print(f"Genres: {genres}")
    print(f"User preferences: {user_preferences}")
    
    return genres, user_preferences

def create_movie_features():
    """Create movie feature vectors."""
    print("\n=== Movie Feature Vectors ===")
    
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
    
    # Movie 1: Action-Comedy
    movie1_features = np.array([0.9, 0.7, 0.3, 0.0, 0.2, 0.1])
    
    # Movie 2: Drama-Romance
    movie2_features = np.array([0.1, 0.2, 0.9, 0.0, 0.8, 0.0])
    
    # Movie 3: Horror-Sci-Fi
    movie3_features = np.array([0.3, 0.1, 0.2, 0.9, 0.1, 0.8])
    
    movies = {
        'The Avengers': movie1_features,
        'The Notebook': movie2_features,
        'Alien': movie3_features
    }
    
    print("Movie feature vectors:")
    for movie, features in movies.items():
        print(f"{movie}: {features}")
    
    return movies

def calculate_basic_recommendation(user_prefs, movie_features):
    """Calculate basic recommendation using vector addition."""
    print("\n=== Basic Recommendation Calculation ===")
    
    # TRUE vector addition - element-wise addition
    combined_vector = user_prefs + movie_features
    
    # Calculate recommendation score as sum of combined vector
    recommendation_score = np.sum(combined_vector)
    
    print(f"User preferences: {user_prefs}")
    print(f"Movie features: {movie_features}")
    print(f"Combined vector (addition): {combined_vector}")
    print(f"Recommendation score (sum): {recommendation_score:.3f}")
    
    return recommendation_score, combined_vector

def advanced_recommendation_system():
    """Advanced recommendation system with multiple factors."""
    print("\n=== Advanced Recommendation System ===")
    
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
    
    # User profile
    user_prefs = np.array([0.8, 0.6, 0.9, 0.2, 0.7, 0.5])
    
    # Movie features
    movie_features = np.array([0.1, 0.2, 0.9, 0.0, 0.8, 0.0])  # Drama-Romance movie
    
    # Additional factors (also represented as vectors)
    time_factor = np.array([0.3, 0.1, 0.8, 0.0, 0.6, 0.2])    # Evening viewing preferences
    device_factor = np.array([0.2, 0.4, 0.7, 0.1, 0.8, 0.3])  # Mobile device preferences
    social_factor = np.array([0.1, 0.5, 0.6, 0.0, 0.9, 0.2])  # Social viewing preferences
    
    # Weights for different factors
    weights = {
        'user_prefs': 0.5,      # 50% weight to user preferences
        'time_factor': 0.2,     # 20% weight to time of day
        'device_factor': 0.15,  # 15% weight to device type
        'social_factor': 0.15   # 15% weight to social context
    }
    
    print("Factors and weights:")
    for factor, weight in weights.items():
        print(f"{factor}: {weight}")
    
    # Calculate weighted combination using vector addition
    # First, add all factors together (vector addition)
    combined_factors = (weights['user_prefs'] * user_prefs +
                       weights['time_factor'] * time_factor +
                       weights['device_factor'] * device_factor +
                       weights['social_factor'] * social_factor)
    
    # Then calculate final score using dot product with movie features
    final_score = np.dot(combined_factors, movie_features)
    
    print(f"Combined factors vector: {combined_factors}")
    print(f"Movie features: {movie_features}")
    
    print(f"\nFinal recommendation score: {final_score:.3f}")
    
    return final_score

def visualize_recommendation_factors():
    """Visualize how different factors contribute to the recommendation."""
    print("\n=== Recommendation Factors Visualization ===")
    
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
    
    # User preferences
    user_prefs = np.array([0.8, 0.6, 0.9, 0.2, 0.7, 0.5])
    
    # Movie features
    movie_features = np.array([0.1, 0.2, 0.9, 0.0, 0.8, 0.0])
    
    # Calculate individual factor contributions using vector addition
    user_contribution = user_prefs + movie_features
    print(f"User contribution (vector addition): {user_contribution}")
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: User preferences
    ax1.bar(genres, user_prefs, color='blue', alpha=0.7)
    ax1.set_title('User Preferences')
    ax1.set_ylabel('Preference Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Movie features
    ax2.bar(genres, movie_features, color='red', alpha=0.7)
    ax2.set_title('Movie Features')
    ax2.set_ylabel('Feature Strength')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Combined contribution
    ax3.bar(genres, user_contribution, color='green', alpha=0.7)
    ax3.set_title('Combined Contribution (User + Movie)')
    ax3.set_ylabel('Contribution Score')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate final score
    final_score = np.sum(user_contribution)
    print(f"Final recommendation score: {final_score:.3f}")

def multiple_movies_comparison():
    """Compare recommendations for multiple movies."""
    print("\n=== Multiple Movies Comparison ===")
    
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
    user_prefs = np.array([0.8, 0.6, 0.9, 0.2, 0.7, 0.5])
    
    movies = {
        'The Avengers': np.array([0.9, 0.7, 0.3, 0.0, 0.2, 0.1]),
        'The Notebook': np.array([0.1, 0.2, 0.9, 0.0, 0.8, 0.0]),
        'Alien': np.array([0.3, 0.1, 0.2, 0.9, 0.1, 0.8]),
        'Inception': np.array([0.6, 0.2, 0.7, 0.1, 0.3, 0.9]),
        'Toy Story': np.array([0.2, 0.9, 0.4, 0.0, 0.1, 0.3])
    }
    
    print("Movie recommendations for user:")
    print(f"User preferences: {user_prefs}")
    print()
    
    recommendations = []
    for movie, features in movies.items():
        # Use vector addition to combine user preferences and movie features
        combined_vector = user_prefs + features
        score = np.sum(combined_vector)  # Sum of the combined vector
        recommendations.append((movie, score))
        print(f"{movie}: {score:.3f} (combined vector: {combined_vector})")
    
    # Sort by recommendation score
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nRanked recommendations:")
    for i, (movie, score) in enumerate(recommendations, 1):
        print(f"{i}. {movie}: {score:.3f}")

def real_world_scaling_example():
    """Show how this scales to real-world systems."""
    print("\n=== Real-World Scaling Example ===")
    
    print("In real recommendation systems:")
    print("1. User vectors have 1000s of dimensions (not just 6 genres)")
    print("2. Multiple factors are combined using weighted vector addition")
    print("3. Machine learning models learn optimal weights automatically")
    print("4. Vector operations are performed on millions of users and items")
    
    # Simulate a larger system
    n_users = 1000
    n_movies = 10000
    n_features = 100
    
    print(f"\nSimulated large system:")
    print(f"- {n_users:,} users")
    print(f"- {n_movies:,} movies")
    print(f"- {n_features} features per user/movie")
    
    # Create random user and movie vectors
    user_vector = np.random.rand(n_features)
    movie_vector = np.random.rand(n_features)
    
    # Calculate recommendation score using vector addition
    combined_vector = user_vector + movie_vector
    score = np.sum(combined_vector)
    print(f"Combined vector (first 5 elements): {combined_vector[:5]}")
    print(f"Recommendation score: {score:.3f}")
    
    print(f"\nThis same vector addition operation happens millions of times")
    print(f"per second in real recommendation systems!")

if __name__ == "__main__":
    print("Data Science & Machine Learning Case Study")
    print("Vector Addition in Recommendation Systems")
    print("=" * 50)
    
    # Run all examples
    genres, user_prefs = create_user_profile()
    movies = create_movie_features()
    
    # Basic recommendation
    for movie_name, movie_features in movies.items():
        print(f"\n--- {movie_name} ---")
        calculate_basic_recommendation(user_prefs, movie_features)
    
    # Advanced system
    advanced_recommendation_system()
    
    # Visualization
    visualize_recommendation_factors()
    
    # Multiple movies comparison
    multiple_movies_comparison()
    
    # Real-world scaling
    real_world_scaling_example()
    
    print("\n" + "=" * 50)
    print("Case Study Complete!")
    print("Key Takeaway: Vector addition is the foundation of modern ML systems!")
