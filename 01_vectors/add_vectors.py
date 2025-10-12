
import numpy as np
import matplotlib.pyplot as plt

# Create two vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"Vector 1: {v1}")
print(f"Vector 2: {v2}")

# Add the vectors (element-wise addition)
result = v1 + v2
print(f"v1 + v2 = {result}")

# Alternative way using np.add()
result2 = np.add(v1, v2)
print(f"np.add(v1, v2) = {result2}")

# Show that both methods give the same result
print(f"Are they equal? {np.array_equal(result, result2)}")

# Create a simple 2D visualization
def draw_vector_addition():
    """Draw a simple chart showing vector addition."""
    print("\n=== Vector Addition Visualization ===")
    
    # Use 2D vectors for easier visualization
    a = np.array([2, 3])  # First vector
    b = np.array([3, 1])  # Second vector
    c = a + b             # Sum vector
    
    print(f"Vector a = {a}")
    print(f"Vector b = {b}")
    print(f"Vector c = a + b = {c}")
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Draw vectors from origin
    plt.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.005, label=f'Vector a = {a}')
    
    plt.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.005, label=f'Vector b = {b}')
    
    # Draw the sum vector
    plt.quiver(0, 0, c[0], c[1], angles='xy', scale_units='xy', scale=1, 
               color='green', width=0.005, label=f'Sum c = a + b = {c}')
    
    # Draw vector b starting from the end of vector a (parallelogram rule)
    plt.quiver(a[0], a[1], b[0], b[1], angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.003, alpha=0.5, linestyle='--')
    
    # Draw vector a starting from the end of vector b
    plt.quiver(b[0], b[1], a[0], a[1], angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.003, alpha=0.5, linestyle='--')
    
    # Set plot limits and grid
    max_val = max(c[0], c[1]) + 1
    plt.xlim(-1, max_val)
    plt.ylim(-1, max_val)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vector Addition: a + b = c')
    plt.legend()
    
    # Add text annotations
    plt.text(a[0]/2, a[1]/2, 'a', fontsize=12, color='blue', weight='bold')
    plt.text(b[0]/2, b[1]/2, 'b', fontsize=12, color='red', weight='bold')
    plt.text(c[0]/2, c[1]/2, 'c', fontsize=12, color='green', weight='bold')
    
    plt.show()

# Call the visualization function
draw_vector_addition()
