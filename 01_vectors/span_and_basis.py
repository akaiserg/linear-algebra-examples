"""
Linear Algebra Examples - Span and Basis
=======================================

This module demonstrates the concepts of span and basis in linear algebra.
"""

import numpy as np
import matplotlib.pyplot as plt

def what_is_span():
    """Explain what span means with simple examples."""
    print("=== What is Span? ===")
    
    # Create two vectors
    v1 = np.array([1, 0])  # Unit vector along x-axis
    v2 = np.array([0, 1])  # Unit vector along y-axis
    
    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")
    
    print(f"\nSpan of {{v1, v2}} is the set of all vectors that can be written as:")
    print(f"c1 * v1 + c2 * v2")
    print(f"where c1 and c2 are any real numbers")
    
    # Show some examples
    print(f"\nExamples of vectors in the span:")
    
    # Example 1: c1=2, c2=3
    c1, c2 = 2, 3
    result1 = c1 * v1 + c2 * v2
    print(f"2 * v1 + 3 * v2 = 2 * {v1} + 3 * {v2} = {result1}")
    
    # Example 2: c1=-1, c2=2
    c1, c2 = -1, 2
    result2 = c1 * v1 + c2 * v2
    print(f"-1 * v1 + 2 * v2 = -1 * {v1} + 2 * {v2} = {result2}")
    
    # Example 3: c1=0, c2=1
    c1, c2 = 0, 1
    result3 = c1 * v1 + c2 * v2
    print(f"0 * v1 + 1 * v2 = 0 * {v1} + 1 * {v2} = {result3}")
    
    print(f"\nThe span of {{v1, v2}} is the ENTIRE 2D plane!")
    print(f"Any point (x, y) can be written as x * v1 + y * v2")
    print()

def what_is_basis():
    """Explain what a basis is with examples."""
    print("=== What is a Basis? ===")
    
    # Standard basis for 2D
    e1 = np.array([1, 0])  # Unit vector along x-axis
    e2 = np.array([0, 1])  # Unit vector along y-axis
    
    print(f"Standard basis vectors:")
    print(f"e1 = {e1}")
    print(f"e2 = {e2}")
    
    print(f"\nA basis is a set of vectors that:")
    print(f"1. Spans the entire space")
    print(f"2. Is linearly independent (no vector can be written as a combination of others)")
    
    # Show that any vector can be written as combination of basis vectors
    target_vector = np.array([3, 4])
    print(f"\nExample: Express {target_vector} as a combination of basis vectors")
    
    # For standard basis, coefficients are just the components
    c1, c2 = target_vector[0], target_vector[1]
    result = c1 * e1 + c2 * e2
    print(f"{target_vector} = {c1} * {e1} + {c2} * {e2} = {result}")
    
    # Show another basis
    print(f"\nAnother basis for 2D space:")
    b1 = np.array([1, 1])  # Vector at 45 degrees
    b2 = np.array([1, -1]) # Vector at -45 degrees
    
    print(f"b1 = {b1}")
    print(f"b2 = {b2}")
    
    # Express target vector in this new basis
    # We need to solve: c1 * b1 + c2 * b2 = target_vector
    # This gives us a system of equations:
    # c1 + c2 = 3
    # c1 - c2 = 4
    # Solving: c1 = 3.5, c2 = -0.5
    
    c1, c2 = 3.5, -0.5
    result = c1 * b1 + c2 * b2
    print(f"{target_vector} = {c1} * {b1} + {c2} * {b2} = {result}")
    print()

def span_visualization():
    """Visualize span of vectors in 2D."""
    print("=== Span Visualization ===")
    
    # Create different sets of vectors
    cases = [
        {
            'name': 'Standard Basis (spans entire 2D plane)',
            'vectors': [np.array([1, 0]), np.array([0, 1])],
            'colors': ['blue', 'red']
        },
        {
            'name': 'Two vectors at 45 degrees (spans entire 2D plane)',
            'vectors': [np.array([1, 1]), np.array([1, -1])],
            'colors': ['green', 'orange']
        },
        {
            'name': 'Two parallel vectors (spans only a line)',
            'vectors': [np.array([1, 0]), np.array([2, 0])],
            'colors': ['purple', 'brown']
        }
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, case in enumerate(cases):
        ax = axes[i]
        vectors = case['vectors']
        colors = case['colors']
        
        # Draw the vectors
        for j, (vec, color) in enumerate(zip(vectors, colors)):
            ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
                     color=color, width=0.005, label=f'v{j+1} = {vec}')
        
        # Draw some points in the span
        if i < 2:  # For cases that span the entire plane
            # Generate some random combinations
            for _ in range(20):
                c1 = np.random.uniform(-2, 2)
                c2 = np.random.uniform(-2, 2)
                point = c1 * vectors[0] + c2 * vectors[1]
                ax.plot(point[0], point[1], 'o', color='gray', alpha=0.3, markersize=3)
        else:  # For parallel vectors (spans only a line)
            # Generate points along the line
            for t in np.linspace(-3, 3, 20):
                point = t * vectors[0]  # All combinations are multiples of v1
                ax.plot(point[0], point[1], 'o', color='gray', alpha=0.3, markersize=3)
        
        # Set plot properties
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(case['name'])
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization shows:")
    print("- Blue/Red: Standard basis spans entire plane")
    print("- Green/Orange: 45-degree vectors span entire plane")
    print("- Purple/Brown: Parallel vectors span only a line")
    print()

def linear_independence():
    """Demonstrate linear independence and dependence."""
    print("=== Linear Independence ===")
    
    # Linearly independent vectors
    print("Linearly Independent Vectors:")
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"These are linearly independent because:")
    print(f"c1 * v1 + c2 * v2 = 0 only when c1 = c2 = 0")
    
    # Check: c1 * [1,0] + c2 * [0,1] = [c1, c2] = [0,0]
    # This gives us c1 = 0 and c2 = 0
    print(f"Solution: c1 = 0, c2 = 0 (trivial solution)")
    
    # Linearly dependent vectors
    print(f"\nLinearly Dependent Vectors:")
    v1 = np.array([1, 2])
    v2 = np.array([2, 4])  # v2 = 2 * v1
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"These are linearly dependent because:")
    print(f"v2 = 2 * v1 (one vector is a multiple of the other)")
    
    # Check: c1 * [1,2] + c2 * [2,4] = [c1+2*c2, 2*c1+4*c2] = [0,0]
    # This gives us: c1 + 2*c2 = 0 and 2*c1 + 4*c2 = 0
    # From first equation: c1 = -2*c2
    # Substituting: 2*(-2*c2) + 4*c2 = -4*c2 + 4*c2 = 0 (always true)
    # So c2 can be any value, and c1 = -2*c2
    print(f"Solution: c1 = -2*c2 (infinite solutions, not just c1=c2=0)")
    
    # Show the relationship
    print(f"\nVerification: 2 * v1 = 2 * {v1} = {2 * v1}")
    print(f"v2 = {v2}")
    print(f"Are they equal? {np.array_equal(2 * v1, v2)}")
    print()

def basis_examples():
    """Show different bases for the same space."""
    print("=== Different Bases for 2D Space ===")
    
    # Target vector to express in different bases
    target = np.array([3, 4])
    print(f"Target vector: {target}")
    
    # Basis 1: Standard basis
    print(f"\nBasis 1: Standard basis")
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    print(f"e1 = {e1}, e2 = {e2}")
    
    # Express target in standard basis
    c1, c2 = target[0], target[1]
    result1 = c1 * e1 + c2 * e2
    print(f"{target} = {c1} * {e1} + {c2} * {e2} = {result1}")
    
    # Basis 2: Rotated basis
    print(f"\nBasis 2: Rotated basis (45 degrees)")
    b1 = np.array([1, 1]) / np.sqrt(2)  # Normalized
    b2 = np.array([-1, 1]) / np.sqrt(2)  # Normalized
    print(f"b1 = {b1}")
    print(f"b2 = {b2}")
    
    # Express target in rotated basis
    # We need to solve: c1 * b1 + c2 * b2 = target
    # This gives us a system of equations
    # c1/sqrt(2) - c2/sqrt(2) = 3
    # c1/sqrt(2) + c2/sqrt(2) = 4
    # Solving: c1 = 7/sqrt(2), c2 = 1/sqrt(2)
    
    c1 = 7 / np.sqrt(2)
    c2 = 1 / np.sqrt(2)
    result2 = c1 * b1 + c2 * b2
    print(f"{target} = {c1:.3f} * {b1} + {c2:.3f} * {b2} = {result2}")
    
    # Basis 3: Non-orthogonal basis
    print(f"\nBasis 3: Non-orthogonal basis")
    b1 = np.array([1, 0])
    b2 = np.array([1, 1])
    print(f"b1 = {b1}")
    print(f"b2 = {b2}")
    
    # Express target in non-orthogonal basis
    # We need to solve: c1 * [1,0] + c2 * [1,1] = [3,4]
    # This gives us: c1 + c2 = 3 and c2 = 4
    # So: c2 = 4, c1 = 3 - 4 = -1
    
    c1, c2 = -1, 4
    result3 = c1 * b1 + c2 * b2
    print(f"{target} = {c1} * {b1} + {c2} * {b2} = {result3}")
    
    print(f"\nAll three representations give the same vector: {target}")
    print(f"Different bases, same result!")
    print()

def span_in_3d():
    """Show span examples in 3D space."""
    print("=== Span in 3D Space ===")
    
    # Two vectors in 3D
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    
    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")
    
    print(f"\nSpan of {{v1, v2}} in 3D:")
    print(f"All vectors of the form: c1 * v1 + c2 * v2")
    print(f"This spans the XY-plane (z = 0)")
    
    # Show some examples
    examples = [
        (1, 0, "v1"),
        (0, 1, "v2"),
        (2, 3, "2*v1 + 3*v2"),
        (-1, 2, "-v1 + 2*v2")
    ]
    
    print(f"\nExamples:")
    for c1, c2, desc in examples:
        result = c1 * v1 + c2 * v2
        print(f"{desc}: {c1} * {v1} + {c2} * {v2} = {result}")
    
    # What about three vectors?
    print(f"\nAdding a third vector:")
    v3 = np.array([0, 0, 1])
    print(f"v3: {v3}")
    
    print(f"\nSpan of {{v1, v2, v3}} in 3D:")
    print(f"All vectors of the form: c1 * v1 + c2 * v2 + c3 * v3")
    print(f"This spans the ENTIRE 3D space!")
    
    # Show example
    c1, c2, c3 = 2, 3, 4
    result = c1 * v1 + c2 * v2 + c3 * v3
    print(f"\nExample: {c1} * {v1} + {c2} * {v2} + {c3} * {v3} = {result}")
    print()

def basis_dimension():
    """Explain the relationship between basis and dimension."""
    print("=== Basis and Dimension ===")
    
    print("Key facts about basis and dimension:")
    print(f"1. The number of vectors in a basis is called the DIMENSION")
    print(f"2. All bases for the same space have the same number of vectors")
    print(f"3. The dimension of a space is the minimum number of vectors needed to span it")
    
    # Examples
    print(f"\nExamples:")
    print(f"- 2D plane: dimension = 2 (need 2 vectors for a basis)")
    print(f"- 3D space: dimension = 3 (need 3 vectors for a basis)")
    print(f"- Line: dimension = 1 (need 1 vector for a basis)")
    print(f"- Point: dimension = 0 (need 0 vectors for a basis)")
    
    # Show that you can't span 2D with 1 vector
    print(f"\nCan 1 vector span 2D space?")
    v = np.array([1, 0])
    print(f"Vector v: {v}")
    print(f"Span of {{v}}: all vectors c * v = c * {v}")
    print(f"This gives us vectors like: {v}, {2*v}, {-1*v}, etc.")
    print(f"All these vectors lie on the X-axis (a line)")
    print(f"Answer: NO! 1 vector can only span a line, not the entire 2D plane")
    
    # Show that you can't span 2D with 3 vectors (they're dependent)
    print(f"\nCan 3 vectors span 2D space?")
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    v3 = np.array([1, 1])
    print(f"Vectors: v1 = {v1}, v2 = {v2}, v3 = {v3}")
    print(f"Notice: v3 = v1 + v2 (v3 is a combination of v1 and v2)")
    print(f"Answer: YES, but they're not linearly independent")
    print(f"We only need 2 of them (like v1 and v2) to span 2D space")
    print()

if __name__ == "__main__":
    print("Linear Algebra - Span and Basis\n")
    what_is_span()
    what_is_basis()
    span_visualization()
    linear_independence()
    basis_examples()
    span_in_3d()
    basis_dimension()
