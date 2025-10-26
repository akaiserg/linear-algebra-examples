"""
Linear Algebra Examples - Outer Product of Vectors
=================================================

This module demonstrates outer product operations and their applications.
"""

import numpy as np
import matplotlib.pyplot as plt

def basic_outer_product():
    """Demonstrate basic outer product operations."""
    print("=== Basic Outer Product ===")
    
    # Create two vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5])
    
    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")
    
    # Calculate outer product
    outer_product = np.outer(v1, v2)
    print(f"\nOuter product v1 ⊗ v2:")
    print(outer_product)
    print(f"Shape: {outer_product.shape}")
    
    # Show the calculation step by step
    print(f"\nStep-by-step calculation:")
    print(f"v1 ⊗ v2 = v1 * v2^T")
    print(f"v1 = {v1}")
    print(f"v2^T = {v2.reshape(-1, 1)}")
    print(f"Result: {v1.reshape(-1, 1) @ v2.reshape(1, -1)}")
    
    # Alternative calculation using broadcasting
    print(f"\nAlternative calculation (broadcasting):")
    result_broadcast = v1[:, np.newaxis] * v2
    print(f"v1[:, np.newaxis] * v2 = {result_broadcast}")
    print(f"Are they equal? {np.array_equal(outer_product, result_broadcast)}")
    print()

def outer_product_properties():
    """Demonstrate properties of outer product."""
    print("=== Outer Product Properties ===")
    
    # Create vectors
    u = np.array([1, 2])
    v = np.array([3, 4])
    w = np.array([5, 6])
    
    print(f"u = {u}")
    print(f"v = {v}")
    print(f"w = {w}")
    
    # Property 1: Not commutative
    uv = np.outer(u, v)
    vu = np.outer(v, u)
    print(f"\nProperty 1: Not commutative")
    print(f"u ⊗ v = {uv}")
    print(f"v ⊗ u = {vu}")
    print(f"Are they equal? {np.array_equal(uv, vu)}")
    
    # Property 2: Distributive over addition
    u_plus_w = u + w
    left_side = np.outer(u_plus_w, v)
    right_side = np.outer(u, v) + np.outer(w, v)
    print(f"\nProperty 2: Distributive over addition")
    print(f"(u + w) ⊗ v = {left_side}")
    print(f"u ⊗ v + w ⊗ v = {right_side}")
    print(f"Are they equal? {np.array_equal(left_side, right_side)}")
    
    # Property 3: Scalar multiplication
    c = 2
    left_scalar = np.outer(c * u, v)
    right_scalar = c * np.outer(u, v)
    print(f"\nProperty 3: Scalar multiplication")
    print(f"(c*u) ⊗ v = {left_scalar}")
    print(f"c*(u ⊗ v) = {right_scalar}")
    print(f"Are they equal? {np.array_equal(left_scalar, right_scalar)}")
    
    # Property 4: Rank-1 matrix
    print(f"\nProperty 4: Rank-1 matrix")
    print(f"u ⊗ v has rank 1")
    rank = np.linalg.matrix_rank(uv)
    print(f"Rank of u ⊗ v: {rank}")
    print()

def outer_product_vs_inner_product():
    """Compare outer product with inner product (dot product)."""
    print("=== Outer Product vs Inner Product ===")
    
    # Create vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")
    
    # Inner product (dot product)
    inner_product = np.dot(v1, v2)
    print(f"\nInner product (dot product):")
    print(f"v1 · v2 = {inner_product}")
    print(f"Result type: scalar")
    print(f"Result shape: {inner_product.shape}")
    
    # Outer product
    outer_product = np.outer(v1, v2)
    print(f"\nOuter product:")
    print(f"v1 ⊗ v2 = {outer_product}")
    print(f"Result type: matrix")
    print(f"Result shape: {outer_product.shape}")
    
    # Show the relationship
    print(f"\nKey differences:")
    print(f"- Inner product: v1 · v2 = scalar (single number)")
    print(f"- Outer product: v1 ⊗ v2 = matrix (2D array)")
    print(f"- Inner product: combines vectors into a scalar")
    print(f"- Outer product: combines vectors into a matrix")
    print()

def outer_product_visualization():
    """Visualize outer product with 2D vectors."""
    print("=== Outer Product Visualization ===")
    
    # Create 2D vectors
    v1 = np.array([3, 2])
    v2 = np.array([2, 4])
    
    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")
    
    # Calculate outer product
    outer_product = np.outer(v1, v2)
    print(f"\nOuter product:")
    print(outer_product)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Original vectors
    ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.005, label=f'v1 = {v1}')
    ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.005, label=f'v2 = {v2}')
    
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(-1, 5)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Original Vectors')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot 2: Outer product matrix as heatmap
    im = ax2.imshow(outer_product, cmap='viridis', aspect='auto')
    ax2.set_title('Outer Product Matrix')
    ax2.set_xlabel('v2 components')
    ax2.set_ylabel('v1 components')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels([f'v2[{i}]' for i in range(len(v2))])
    ax2.set_yticklabels([f'v1[{i}]' for i in range(len(v1))])
    
    # Add text annotations
    for i in range(len(v1)):
        for j in range(len(v2)):
            text = ax2.text(j, i, f'{outer_product[i, j]}',
                           ha="center", va="center", color="white", fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Value')
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization shows:")
    print("- Original vectors in 2D space")
    print("- Outer product matrix as a heatmap")
    print("- Each cell shows v1[i] * v2[j]")
    print()

def practical_outer_product_examples():
    """Show practical examples of outer product."""
    print("=== Practical Examples ===")
    
    # Example 1: Image processing
    print("Example 1: Image Processing")
    # Simulate a simple 3x3 image
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Original image: {image}")
    
    # Create a filter using outer product
    filter_v1 = np.array([1, 0, -1])  # Vertical edge detection
    filter_v2 = np.array([1, 2, 1])   # Smoothing
    filter_kernel = np.outer(filter_v1, filter_v2)
    print(f"Filter kernel (outer product): {filter_kernel}")
    
    # Example 2: Probability distributions
    print("\nExample 2: Probability Distributions")
    # Two probability distributions
    p_x = np.array([0.3, 0.4, 0.3])  # P(X)
    p_y = np.array([0.2, 0.8])       # P(Y)
    
    # Joint distribution (assuming independence)
    joint_dist = np.outer(p_x, p_y)
    print(f"P(X): {p_x}")
    print(f"P(Y): {p_y}")
    print(f"Joint distribution P(X,Y): {joint_dist}")
    
    # Example 3: Feature combinations
    print("\nExample 3: Feature Combinations")
    # User features
    user_features = np.array([0.8, 0.6, 0.9])  # [age, income, education]
    item_features = np.array([0.7, 0.5])       # [price, rating]
    
    # Feature interaction matrix
    interaction_matrix = np.outer(user_features, item_features)
    print(f"User features: {user_features}")
    print(f"Item features: {item_features}")
    print(f"Feature interaction matrix: {interaction_matrix}")
    
    # Example 4: Weight matrices
    print("\nExample 4: Weight Matrices")
    # Input and output dimensions
    input_dim = 3
    output_dim = 2
    
    # Create weight matrix using outer product
    input_weights = np.array([0.5, 0.3, 0.8])
    output_weights = np.array([0.2, 0.7])
    weight_matrix = np.outer(input_weights, output_weights)
    print(f"Input weights: {input_weights}")
    print(f"Output weights: {output_weights}")
    print(f"Weight matrix: {weight_matrix}")
    print()

def outer_product_in_linear_algebra():
    """Show outer product in linear algebra context."""
    print("=== Outer Product in Linear Algebra ===")
    
    # Example 1: Rank-1 matrices
    print("Example 1: Rank-1 Matrices")
    u = np.array([1, 2, 3])
    v = np.array([4, 5])
    
    rank1_matrix = np.outer(u, v)
    print(f"u = {u}")
    print(f"v = {v}")
    print(f"Rank-1 matrix u ⊗ v: {rank1_matrix}")
    print(f"Rank: {np.linalg.matrix_rank(rank1_matrix)}")
    
    # Example 2: Matrix decomposition
    print("\nExample 2: Matrix Decomposition")
    # Any matrix can be written as sum of rank-1 matrices
    A = np.array([[1, 2], [3, 4], [5, 6]])
    print(f"Matrix A: {A}")
    
    # Decompose using SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"Singular values: {s}")
    
    # Reconstruct using outer products
    reconstructed = np.zeros_like(A)
    for i in range(len(s)):
        rank1_component = s[i] * np.outer(U[:, i], Vt[i, :])
        reconstructed += rank1_component
        print(f"Rank-1 component {i+1}: {rank1_component}")
    
    print(f"Reconstructed A: {reconstructed}")
    print(f"Are they equal? {np.allclose(A, reconstructed)}")
    
    # Example 3: Tensor product
    print("\nExample 3: Tensor Product")
    # Outer product is a special case of tensor product
    a = np.array([1, 2])
    b = np.array([3, 4, 5])
    
    tensor_product = np.outer(a, b)
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"Tensor product a ⊗ b: {tensor_product}")
    print(f"Shape: {tensor_product.shape}")
    print()

def outer_product_applications():
    """Show real-world applications of outer product."""
    print("=== Real-World Applications ===")
    
    # Application 1: Machine Learning
    print("Application 1: Machine Learning")
    # Feature interaction in neural networks
    input_features = np.array([0.8, 0.6, 0.4])
    hidden_weights = np.array([0.5, 0.3, 0.7])
    
    # Weight matrix for fully connected layer
    weight_matrix = np.outer(input_features, hidden_weights)
    print(f"Input features: {input_features}")
    print(f"Hidden weights: {hidden_weights}")
    print(f"Weight matrix: {weight_matrix}")
    
    # Application 2: Computer Graphics
    print("\nApplication 2: Computer Graphics")
    # Normal vector and light direction
    normal = np.array([0, 0, 1])      # Surface normal
    light_dir = np.array([1, 1, 1])   # Light direction
    
    # Reflection matrix
    reflection_matrix = np.outer(normal, light_dir)
    print(f"Normal vector: {normal}")
    print(f"Light direction: {light_dir}")
    print(f"Reflection matrix: {reflection_matrix}")
    
    # Application 3: Signal Processing
    print("\nApplication 3: Signal Processing")
    # Filter design
    time_samples = np.array([0, 1, 2, 3])
    frequency_components = np.array([1, 0.5, 0.25])
    
    # Filter bank
    filter_bank = np.outer(time_samples, frequency_components)
    print(f"Time samples: {time_samples}")
    print(f"Frequency components: {frequency_components}")
    print(f"Filter bank: {filter_bank}")
    
    # Application 4: Economics
    print("\nApplication 4: Economics")
    # Supply and demand analysis
    supply_factors = np.array([0.8, 0.6, 0.4])  # [labor, capital, technology]
    demand_factors = np.array([0.7, 0.5])       # [price, income]
    
    # Market interaction matrix
    market_matrix = np.outer(supply_factors, demand_factors)
    print(f"Supply factors: {supply_factors}")
    print(f"Demand factors: {demand_factors}")
    print(f"Market interaction matrix: {market_matrix}")
    print()

def outer_product_advanced():
    """Show advanced outer product concepts."""
    print("=== Advanced Outer Product Concepts ===")
    
    # Example 1: Multiple outer products
    print("Example 1: Multiple Outer Products")
    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])
    
    # First outer product
    ab = np.outer(a, b)
    print(f"a ⊗ b = {ab}")
    
    # Second outer product
    abc = np.outer(ab.flatten(), c)
    print(f"(a ⊗ b) ⊗ c = {abc}")
    print(f"Shape: {abc.shape}")
    
    # Example 2: Outer product with different data types
    print("\nExample 2: Different Data Types")
    # Integer vectors
    int_v1 = np.array([1, 2, 3], dtype=int)
    int_v2 = np.array([4, 5], dtype=int)
    int_outer = np.outer(int_v1, int_v2)
    print(f"Integer outer product: {int_outer}")
    print(f"Data type: {int_outer.dtype}")
    
    # Float vectors
    float_v1 = np.array([1.5, 2.5, 3.5], dtype=float)
    float_v2 = np.array([4.2, 5.8], dtype=float)
    float_outer = np.outer(float_v1, float_v2)
    print(f"Float outer product: {float_outer}")
    print(f"Data type: {float_outer.dtype}")
    
    # Example 3: Outer product with complex numbers
    print("\nExample 3: Complex Numbers")
    complex_v1 = np.array([1+2j, 3+4j])
    complex_v2 = np.array([5+6j, 7+8j])
    complex_outer = np.outer(complex_v1, complex_v2)
    print(f"Complex v1: {complex_v1}")
    print(f"Complex v2: {complex_v2}")
    print(f"Complex outer product: {complex_outer}")
    print()

if __name__ == "__main__":
    print("Linear Algebra - Outer Product of Vectors\n")
    basic_outer_product()
    outer_product_properties()
    outer_product_vs_inner_product()
    outer_product_visualization()
    practical_outer_product_examples()
    outer_product_in_linear_algebra()
    outer_product_applications()
    outer_product_advanced()
