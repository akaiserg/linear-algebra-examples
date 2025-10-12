import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def vector_creation_examples():
    """Demonstrate different ways to create vectors in NumPy."""
    print("=== Vector Creation Examples ===")
    
    # 1D arrays (vectors)
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    v3 = np.zeros(3)
    v4 = np.ones(3)
    v5 = np.arange(1, 4)  # [1, 2, 3]
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v3 (zeros) = {v3}")
    print(f"v4 (ones) = {v4}")
    print(f"v5 (arange) = {v5}")
    print(f"Shape of v1: {v1.shape}")
    print(f"Dimension of v1: {v1.ndim}")
    print()

def vector_operations():
    """Demonstrate basic vector operations."""
    print("=== Vector Operations ===")
    
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    # Addition
    v_sum = v1 + v2
    print(f"v1 + v2 = {v_sum}")
    
    # Subtraction
    v_diff = v2 - v1
    print(f"v2 - v1 = {v_diff}")
    
    # Scalar multiplication
    scalar = 3
    v_scaled = scalar * v1
    print(f"{scalar} * v1 = {v_scaled}")
    
    # Element-wise multiplication
    v_elementwise = v1 * v2
    print(f"v1 * v2 (element-wise) = {v_elementwise}")
    
    # Dot product
    dot_product = np.dot(v1, v2)
    print(f"v1 · v2 (dot product) = {dot_product}")
    
    # Alternative dot product syntax
    dot_product_alt = v1 @ v2
    print(f"v1 @ v2 (dot product) = {dot_product_alt}")
    print()

def vector_properties():
    """Calculate important vector properties."""
    print("=== Vector Properties ===")
    
    v = np.array([3, 4, 0])
    
    # Magnitude (norm)
    magnitude = np.linalg.norm(v)
    print(f"Vector v = {v}")
    print(f"Magnitude ||v|| = {magnitude}")
    
    # Unit vector (normalized)
    unit_v = v / magnitude
    print(f"Unit vector = {unit_v}")
    print(f"Magnitude of unit vector = {np.linalg.norm(unit_v)}")
    
    # Distance between two vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    distance = np.linalg.norm(v2 - v1)
    print(f"Distance between v1 and v2 = {distance}")
    print()

def vector_visualization():
    """Visualize 2D and 3D vectors."""
    print("=== Vector Visualization ===")
    
    # 2D vectors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2D plot
    v1_2d = np.array([2, 3])
    v2_2d = np.array([1, 1])
    v_sum_2d = v1_2d + v2_2d
    
    ax1.quiver(0, 0, v1_2d[0], v1_2d[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v1')
    ax1.quiver(0, 0, v2_2d[0], v2_2d[1], angles='xy', scale_units='xy', scale=1, color='red', label='v2')
    ax1.quiver(0, 0, v_sum_2d[0], v_sum_2d[1], angles='xy', scale_units='xy', scale=1, color='green', label='v1 + v2')
    
    ax1.set_xlim(-1, 4)
    ax1.set_ylim(-1, 5)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('2D Vector Addition')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # 3D plot
    ax2 = fig.add_subplot(122, projection='3d')
    v1_3d = np.array([2, 3, 1])
    v2_3d = np.array([1, 1, 2])
    
    ax2.quiver(0, 0, 0, v1_3d[0], v1_3d[1], v1_3d[2], color='blue', label='v1')
    ax2.quiver(0, 0, 0, v2_3d[0], v2_3d[1], v2_3d[2], color='red', label='v2')
    
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, 4)
    ax2.set_zlim(0, 3)
    ax2.legend()
    ax2.set_title('3D Vectors')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    plt.tight_layout()
    plt.show()

def cross_product_example():
    """Demonstrate cross product in 3D."""
    print("=== Cross Product Example ===")
    
    v1 = np.array([1, 0, 0])  # x-axis
    v2 = np.array([0, 1, 0])  # y-axis
    
    cross_product = np.cross(v1, v2)
    print(f"v1 = {v1} (x-axis)")
    print(f"v2 = {v2} (y-axis)")
    print(f"v1 × v2 = {cross_product} (should be z-axis)")
    
    # Verify orthogonality
    dot1 = np.dot(v1, cross_product)
    dot2 = np.dot(v2, cross_product)
    print(f"v1 · (v1 × v2) = {dot1} (should be 0)")
    print(f"v2 · (v1 × v2) = {dot2} (should be 0)")
    print()

if __name__ == "__main__":
    print("Linear Algebra - Vector Basics\n")
    vector_creation_examples()
    vector_operations()
    vector_properties()
    cross_product_example()
    vector_visualization()
