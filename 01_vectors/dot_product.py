
import numpy as np
import matplotlib.pyplot as plt

def dot_product_formula():
    """Demonstrate dot product using the mathematical formula."""
    print("=== Dot Product Formula ===")
    
    # Create two vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")
    
    # Manual calculation using formula: v1 · v2 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    manual_dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    print(f"\nManual calculation:")
    print(f"v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] = {v1[0]}*{v2[0]} + {v1[1]}*{v2[1]} + {v1[2]}*{v2[2]}")
    print(f"= {v1[0]*v2[0]} + {v1[1]*v2[1]} + {v1[2]*v2[2]} = {manual_dot}")
    
    # Using NumPy
    numpy_dot = np.dot(v1, v2)
    print(f"\nNumPy calculation:")
    print(f"np.dot(v1, v2) = {numpy_dot}")
    
    # Alternative NumPy syntax
    alternative_dot = v1 @ v2
    print(f"v1 @ v2 = {alternative_dot}")
    
    # Verify they're all equal
    print(f"\nAll methods give the same result: {manual_dot == numpy_dot == alternative_dot}")
    print()

def dot_product_properties():
    """Demonstrate properties of dot product."""
    print("=== Dot Product Properties ===")
    
    # Create vectors
    u = np.array([1, 2, 3])
    v = np.array([4, 5, 6])
    w = np.array([7, 8, 9])
    c = 2  # scalar
    
    print(f"u = {u}")
    print(f"v = {v}")
    print(f"w = {w}")
    print(f"c = {c}")
    
    # Commutative property: u · v = v · u
    dot_uv = np.dot(u, v)
    dot_vu = np.dot(v, u)
    print(f"\nCommutative property:")
    print(f"u · v = {dot_uv}")
    print(f"v · u = {dot_vu}")
    print(f"Are they equal? {dot_uv == dot_vu}")
    
    # Distributive property: u · (v + w) = u · v + u · w
    left_side = np.dot(u, v + w)
    right_side = np.dot(u, v) + np.dot(u, w)
    print(f"\nDistributive property:")
    print(f"u · (v + w) = {left_side}")
    print(f"u · v + u · w = {right_side}")
    print(f"Are they equal? {left_side == right_side}")
    
    # Scalar multiplication: (c*u) · v = c*(u · v)
    left_scalar = np.dot(c * u, v)
    right_scalar = c * np.dot(u, v)
    print(f"\nScalar multiplication:")
    print(f"(c*u) · v = {left_scalar}")
    print(f"c*(u · v) = {right_scalar}")
    print(f"Are they equal? {left_scalar == right_scalar}")
    
    # Dot product with zero vector
    zero = np.zeros(3)
    dot_with_zero = np.dot(u, zero)
    print(f"\nDot product with zero vector:")
    print(f"u · zero = {dot_with_zero}")
    print()

def dot_product_vs_vector_addition():
    """Compare dot product with vector addition."""
    print("=== Dot Product vs Vector Addition ===")
    
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")
    
    # Vector addition (element-wise)
    vector_sum = v1 + v2
    print(f"\nVector addition (element-wise):")
    print(f"v1 + v2 = {vector_sum}")
    
    # Dot product (scalar result)
    dot_product = np.dot(v1, v2)
    print(f"\nDot product (scalar result):")
    print(f"v1 · v2 = {dot_product}")
    
    print(f"\nKey differences:")
    print(f"- Vector addition: {v1.shape} + {v2.shape} = {vector_sum.shape}")
    print(f"- Dot product: {v1.shape} · {v2.shape} = scalar ({dot_product})")
    print(f"- Addition: element-wise operation")
    print(f"- Dot product: sum of element-wise products")
    print()


if __name__ == "__main__":
    print("Linear Algebra - Dot Product of Vectors\n")
    dot_product_formula()
    dot_product_properties()   
    dot_product_vs_vector_addition()
   
