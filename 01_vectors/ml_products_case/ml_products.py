"""
Machine Learning with Outer and Inner Products
=============================================

This module demonstrates how outer and inner products are used in real-world
machine learning applications including neural networks and recommendation systems.
"""

import numpy as np
import matplotlib.pyplot as plt

def neural_network_layer():
    """Demonstrate how neural networks use inner and outer products."""
    print("=== Neural Network Layer ===")
    
    # Simulate a simple neural network layer
    input_size = 4
    hidden_size = 3
    output_size = 2
    
    print(f"Network architecture:")
    print(f"Input layer: {input_size} neurons")
    print(f"Hidden layer: {hidden_size} neurons")
    print(f"Output layer: {output_size} neurons")
    
    # Input data (batch of 3 samples)
    input_data = np.array([
        [0.8, 0.6, 0.4, 0.2],  # Sample 1
        [0.5, 0.7, 0.3, 0.9],  # Sample 2
        [0.2, 0.1, 0.8, 0.6]   # Sample 3
    ])
    
    print(f"\nInput data (3 samples): {input_data}")
    
    # Weight matrices (using outer product for initialization)
    # Layer 1: Input to Hidden
    input_weights = np.random.randn(input_size, hidden_size) * 0.1
    print(f"\nInput to Hidden weights shape: {input_weights.shape}")
    print(f"Weight matrix: {input_weights}")
    
    # Layer 2: Hidden to Output
    hidden_weights = np.random.randn(hidden_size, output_size) * 0.1
    print(f"\nHidden to Output weights shape: {hidden_weights.shape}")
    print(f"Weight matrix: {hidden_weights}")
    
    # Forward pass using inner products (matrix multiplication)
    print(f"\n=== Forward Pass (Inner Products) ===")
    
    # Layer 1: Input to Hidden
    hidden_activations = np.dot(input_data, input_weights)
    print(f"Hidden activations: {hidden_activations}")
    
    # Apply activation function (ReLU)
    hidden_activations = np.maximum(0, hidden_activations)
    print(f"Hidden activations (after ReLU): {hidden_activations}")
    
    # Layer 2: Hidden to Output
    output_activations = np.dot(hidden_activations, hidden_weights)
    print(f"Output activations: {output_activations}")
    
    # Apply softmax activation
    exp_outputs = np.exp(output_activations)
    output_probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
    print(f"Output probabilities: {output_probabilities}")
    
    # Show the inner product calculation for one sample
    print(f"\n=== Inner Product Calculation (Sample 1) ===")
    sample_1 = input_data[0]
    print(f"Input sample 1: {sample_1}")
    print(f"First hidden neuron calculation:")
    print(f"  {sample_1[0]}*{input_weights[0,0]:.3f} + {sample_1[1]}*{input_weights[1,0]:.3f} + {sample_1[2]}*{input_weights[2,0]:.3f} + {sample_1[3]}*{input_weights[3,0]:.3f}")
    print(f"  = {np.dot(sample_1, input_weights[:, 0]):.3f}")
    print()
    
    return input_data, input_weights, hidden_weights, output_probabilities

def recommendation_system():
    """Demonstrate outer and inner products in recommendation systems."""
    print("=== Recommendation System ===")
    
    # User and item data
    n_users = 5
    n_items = 4
    n_features = 3
    
    print(f"System parameters:")
    print(f"Users: {n_users}")
    print(f"Items: {n_items}")
    print(f"Features: {n_features}")
    
    # User embeddings (learned representations)
    user_embeddings = np.random.randn(n_users, n_features) * 0.5
    print(f"\nUser embeddings: {user_embeddings}")
    
    # Item embeddings (learned representations)
    item_embeddings = np.random.randn(n_items, n_features) * 0.5
    print(f"\nItem embeddings: {item_embeddings}")
    
    # Method 1: Inner Product for Similarity
    print(f"\n=== Method 1: Inner Product (Dot Product) ===")
    print("Calculate user-item similarity using dot product")
    
    # Calculate similarity matrix using inner products
    similarity_matrix = np.dot(user_embeddings, item_embeddings.T)
    print(f"User-Item similarity matrix: {similarity_matrix}")
    
    # Find recommendations for user 0
    user_0_similarities = similarity_matrix[0, :]
    print(f"\nSimilarities for User 0: {user_0_similarities}")
    
    # Get top recommendations
    top_items = np.argsort(user_0_similarities)[::-1]
    print(f"Top recommendations for User 0: Items {top_items}")
    
    # Method 2: Outer Product for Feature Interactions
    print(f"\n=== Method 2: Outer Product for Feature Interactions ===")
    print("Create feature interaction matrices using outer product")
    
    # Create interaction matrix for user 0 and item 0
    user_0_features = user_embeddings[0]
    item_0_features = item_embeddings[0]
    
    print(f"User 0 features: {user_0_features}")
    print(f"Item 0 features: {item_0_features}")
    
    # Outer product creates interaction matrix
    interaction_matrix = np.outer(user_0_features, item_0_features)
    print(f"Feature interaction matrix: {interaction_matrix}")
    
    # Calculate interaction score (sum of all interactions)
    interaction_score = np.sum(interaction_matrix)
    print(f"Interaction score: {interaction_score}")
    
    # Compare with dot product
    dot_product_score = np.dot(user_0_features, item_0_features)
    print(f"Dot product score: {dot_product_score}")
    print(f"Are they equal? {np.isclose(interaction_score, dot_product_score)}")
    
    # Method 3: Matrix Factorization
    print(f"\n=== Method 3: Matrix Factorization ===")
    print("Decompose user-item matrix using outer products")
    
    # Create a user-item rating matrix
    rating_matrix = np.random.randint(1, 6, (n_users, n_items)).astype(float)
    print(f"Rating matrix: {rating_matrix}")
    
    # Factorize using SVD (Singular Value Decomposition)
    U, s, Vt = np.linalg.svd(rating_matrix, full_matrices=False)
    
    # Reconstruct using outer products
    reconstructed = np.zeros_like(rating_matrix, dtype=float)
    for i in range(len(s)):
        rank1_component = s[i] * np.outer(U[:, i], Vt[i, :])
        reconstructed += rank1_component
        print(f"Rank-1 component {i+1}: {rank1_component}")
    
    print(f"Reconstructed matrix: {reconstructed}")
    print(f"Reconstruction error: {np.mean((rating_matrix - reconstructed)**2):.3f}")
    print()

def feature_engineering():
    """Demonstrate feature engineering using outer and inner products."""
    print("=== Feature Engineering ===")
    
    # Sample data: User demographics and item features
    n_users = 3
    n_items = 4
    
    # User features: [age, income, education]
    user_features = np.array([
        [25, 50000, 16],  # User 1: 25 years, $50k, 16 years education
        [35, 80000, 18],  # User 2: 35 years, $80k, 18 years education
        [45, 120000, 20]  # User 3: 45 years, $120k, 20 years education
    ])
    
    # Item features: [price, rating, category]
    item_features = np.array([
        [100, 4.5, 1],    # Item 1: $100, 4.5 stars, category 1
        [200, 4.2, 2],    # Item 2: $200, 4.2 stars, category 2
        [150, 4.8, 1],    # Item 3: $150, 4.8 stars, category 1
        [300, 4.0, 3]     # Item 4: $300, 4.0 stars, category 3
    ])
    
    print(f"User features: {user_features}")
    print(f"Item features: {item_features}")
    
    # Normalize features for better comparison
    user_features_norm = user_features / np.max(user_features, axis=0)
    item_features_norm = item_features / np.max(item_features, axis=0)
    
    print(f"\nNormalized user features: {user_features_norm}")
    print(f"Normalized item features: {item_features_norm}")
    
    # Method 1: Inner Product for Similarity
    print(f"\n=== Method 1: Inner Product Similarity ===")
    similarities = np.dot(user_features_norm, item_features_norm.T)
    print(f"User-Item similarity matrix: {similarities}")
    
    # Method 2: Outer Product for Feature Interactions
    print(f"\n=== Method 2: Outer Product Feature Interactions ===")
    
    # Create interaction features for user 0 and item 0
    user_0 = user_features_norm[0]
    item_0 = item_features_norm[0]
    
    print(f"User 0 normalized features: {user_0}")
    print(f"Item 0 normalized features: {item_0}")
    
    # Outer product creates interaction matrix
    interaction_matrix = np.outer(user_0, item_0)
    print(f"Feature interaction matrix: {interaction_matrix}")
    
    # Extract meaningful interactions
    print(f"\nMeaningful interactions:")
    print(f"Age-Price interaction: {interaction_matrix[0, 0]:.3f}")
    print(f"Income-Rating interaction: {interaction_matrix[1, 1]:.3f}")
    print(f"Education-Category interaction: {interaction_matrix[2, 2]:.3f}")
    
    # Method 3: Polynomial Features
    print(f"\n=== Method 3: Polynomial Features ===")
    
    # Create polynomial features using outer product
    # For user 0, create all pairwise interactions
    user_0_poly = np.outer(user_0, user_0)
    print(f"User 0 polynomial features: {user_0_poly}")
    
    # Extract upper triangle (avoid duplicates)
    upper_triangle = np.triu(user_0_poly)
    polynomial_features = upper_triangle[upper_triangle != 0]
    print(f"Unique polynomial features: {polynomial_features}")
    print()

def attention_mechanism():
    """Demonstrate attention mechanism using inner and outer products."""
    print("=== Attention Mechanism ===")
    
    # Simulate a simple attention mechanism
    seq_length = 4
    hidden_size = 3
    
    # Input sequence (e.g., words in a sentence)
    input_sequence = np.random.randn(seq_length, hidden_size) * 0.5
    print(f"Input sequence: {input_sequence}")
    
    # Query, Key, Value matrices (learned parameters)
    W_q = np.random.randn(hidden_size, hidden_size) * 0.1
    W_k = np.random.randn(hidden_size, hidden_size) * 0.1
    W_v = np.random.randn(hidden_size, hidden_size) * 0.1
    
    print(f"\nQuery matrix W_q: {W_q}")
    print(f"Key matrix W_k: {W_k}")
    print(f"Value matrix W_v: {W_v}")
    
    # Step 1: Compute Query, Key, Value
    Q = np.dot(input_sequence, W_q)  # Inner product
    K = np.dot(input_sequence, W_k)  # Inner product
    V = np.dot(input_sequence, W_v)  # Inner product
    
    print(f"\nQuery Q: {Q}")
    print(f"Key K: {K}")
    print(f"Value V: {V}")
    
    # Step 2: Compute attention scores using inner product
    attention_scores = np.dot(Q, K.T)  # Inner product
    print(f"\nAttention scores: {attention_scores}")
    
    # Step 3: Apply softmax to get attention weights
    attention_weights = np.exp(attention_scores)
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
    print(f"Attention weights: {attention_weights}")
    
    # Step 4: Compute weighted sum using inner product
    output = np.dot(attention_weights, V)  # Inner product
    print(f"Attention output: {output}")
    
    # Visualize attention weights
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Attention Weights Matrix')
    plt.xticks(range(seq_length))
    plt.yticks(range(seq_length))
    
    # Add text annotations
    for i in range(seq_length):
        for j in range(seq_length):
            plt.text(j, i, f'{attention_weights[i, j]:.2f}',
                    ha="center", va="center", color="white", fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization shows attention weights between query and key positions")
    print()

def matrix_factorization():
    """Demonstrate matrix factorization using outer products."""
    print("=== Matrix Factorization ===")
    
    # Create a user-item rating matrix
    n_users = 4
    n_items = 5
    
    # Simulate rating matrix (0 means no rating)
    rating_matrix = np.array([
        [5, 3, 0, 1, 4],  # User 1 ratings
        [4, 0, 0, 1, 3],  # User 2 ratings
        [1, 1, 0, 5, 0],  # User 3 ratings
        [0, 0, 4, 4, 5]   # User 4 ratings
    ], dtype=float)
    
    print(f"Rating matrix: {rating_matrix}")
    
    # Factorize using SVD
    U, s, Vt = np.linalg.svd(rating_matrix, full_matrices=False)
    
    print(f"\nSVD decomposition:")
    print(f"U shape: {U.shape}")
    print(f"Singular values: {s}")
    print(f"Vt shape: {Vt.shape}")
    
    # Reconstruct using outer products
    print(f"\nReconstruction using outer products:")
    reconstructed = np.zeros_like(rating_matrix, dtype=float)
    
    for i in range(len(s)):
        # Each component is an outer product
        component = s[i] * np.outer(U[:, i], Vt[i, :])
        reconstructed += component
        print(f"Component {i+1}: {component}")
    
    print(f"\nReconstructed matrix: {reconstructed}")
    print(f"Original matrix: {rating_matrix}")
    print(f"Reconstruction error: {np.mean((rating_matrix - reconstructed)**2):.3f}")
    
    # Predict missing ratings
    print(f"\n=== Predicting Missing Ratings ===")
    missing_mask = (rating_matrix == 0)
    predicted_ratings = reconstructed.copy()
    predicted_ratings[missing_mask] = reconstructed[missing_mask]
    
    print(f"Predicted ratings: {predicted_ratings}")
    print(f"Missing ratings predictions: {predicted_ratings[missing_mask]}")
    print()

def visualization_comparison():
    """Visualize the difference between inner and outer products."""
    print("=== Visualization: Inner vs Outer Products ===")
    
    # Create sample vectors (same length for inner product)
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")
    
    # Inner product
    inner_product = np.dot(v1, v2)
    print(f"Inner product: {inner_product}")
    
    # For outer product demonstration, use different length vectors
    v1_outer = np.array([1, 2, 3])
    v2_outer = np.array([4, 5])
    
    # Outer product
    outer_product = np.outer(v1_outer, v2_outer)
    print(f"Outer product: {outer_product}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Inner product (scalar)
    ax1.bar(['Inner Product'], [inner_product], color='blue', alpha=0.7)
    ax1.set_title('Inner Product (Scalar)')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Outer product (matrix)
    im = ax2.imshow(outer_product, cmap='viridis', aspect='auto')
    ax2.set_title('Outer Product (Matrix)')
    ax2.set_xlabel('v2 components')
    ax2.set_ylabel('v1 components')
    ax2.set_xticks(range(len(v2_outer)))
    ax2.set_yticks(range(len(v1_outer)))
    ax2.set_xticklabels([f'v2[{i}]' for i in range(len(v2_outer))])
    ax2.set_yticklabels([f'v1[{i}]' for i in range(len(v1_outer))])
    
    # Add text annotations
    for i in range(len(v1_outer)):
        for j in range(len(v2_outer)):
            ax2.text(j, i, f'{outer_product[i, j]}',
                    ha="center", va="center", color="white", fontweight='bold')
    
    plt.colorbar(im, ax=ax2, label='Value')
    plt.tight_layout()
    plt.show()
    
    print("Visualization shows:")
    print("- Inner product: Single scalar value")
    print("- Outer product: Matrix with all combinations")
    print()

if __name__ == "__main__":
    print("Machine Learning with Outer and Inner Products\n")
    neural_network_layer()
    recommendation_system()
    feature_engineering()
    attention_mechanism()
    matrix_factorization()
    visualization_comparison()
