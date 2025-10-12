

import numpy as np
import matplotlib.pyplot as plt



# 1D vector (row vector)
v1 = np.array([1, 2, 3])
print(f"Original vector v1: {v1}")
print(f"Shape of v1: {v1.shape}")
print(f"Transpose of v1: {v1.T}")
print(f"Shape of v1.T: {v1.T.shape}")
print()

# 2D row vector
v2 = np.array([[1, 2, 3]])
print(f"Row vector v2: {v2}")
print(f"Shape of v2: {v2.shape}")
print(f"Transpose of v2: {v2.T}")
print(f"Shape of v2.T: {v2.T.shape}")
print()


# 2D column vector
v3 = np.array([[1], [2], [3]])
print(f"Column vector v3: {v3}")
print(f"Shape of v3: {v3.shape}")
print(f"Transpose of v3: {v3.T}")
print(f"Shape of v3.T: {v3.T.shape}")
print()


# Create a 2D vector
v = np.array([[1, 2, 3, 4]])
print(f"Original vector v: {v}")
print(f"Shape: {v.shape}")

# Transpose
v_T = v.T
print(f"Transpose v.T: {v_T}")
print(f"Shape: {v_T.shape}")

# Double transpose (should return to original)
v_TT = v_T.T
print(f"Double transpose (v.T).T: {v_TT}")
print(f"Shape: {v_TT.shape}")
print(f"Are they equal? {np.array_equal(v, v_TT)}")
print()
