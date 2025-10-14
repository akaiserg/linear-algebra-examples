import numpy as np
import matplotlib.pyplot as plt


# Create two vectors
v1 = np.array([5, 7, 9])
v2 = np.array([2, 3, 4])

print(f"Vector v1: {v1}")
print(f"Vector v2: {v2}")

# Subtract vectors
result = v1 - v2
print(f"v1 - v2 = {result}")

# Alternative way using np.subtract()
result2 = np.subtract(v1, v2)
print(f"np.subtract(v1, v2) = {result2}")

# Show that subtraction is not commutative
result3 = v2 - v1
print(f"v2 - v1 = {result3}")
print(f"Are v1-v2 and v2-v1 equal? {np.array_equal(result, result3)}")
print()


# Create vectors
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
w = np.array([7, 8, 9])

print(f"u = {u}")
print(f"v = {v}")
print(f"w = {w}")

# Associative property: (u - v) - w = u - (v + w)
left_side = (u - v) - w
right_side = u - (v + w)
print(f"\nAssociative property:")
print(f"(u - v) - w = {left_side}")
print(f"u - (v + w) = {right_side}")
print(f"Are they equal? {np.array_equal(left_side, right_side)}")

# Zero vector (additive identity)
zero = np.zeros(3)
print(f"\nZero vector: {zero}")
print(f"u - zero = {u - zero}")
print(f"Are they equal? {np.array_equal(u - zero, u)}")

# Subtracting a vector from itself
print(f"\nSubtracting vector from itself:")
print(f"u - u = {u - u}")
print(f"Is it zero? {np.array_equal(u - u, zero)}")

# Points in 2D space
point1 = np.array([1, 2])
point2 = np.array([4, 6])

# Calculate displacement vector
displacement = point2 - point1
print(f"Point 1: {point1}")
print(f"Point 2: {point2}")
print(f"Displacement vector: {displacement}")



def real_world_applications():
    """Show real-world applications of vector subtraction."""
    print("=== Real-World Applications ===")
    
    # Application 1: GPS Navigation
    print("Application 1: GPS Navigation")
    current_location = np.array([40.7128, -74.0060])  # New York
    destination = np.array([34.0522, -118.2437])      # Los Angeles
    direction_vector = destination - current_location
    print(f"Current location: {current_location}")
    print(f"Destination: {destination}")
    print(f"Direction vector: {direction_vector}")
    
    # Application 2: Stock Price Changes
    print("\nApplication 2: Stock Price Changes")
    yesterday_prices = np.array([100, 150, 200, 75])
    today_prices = np.array([105, 145, 210, 80])
    price_changes = today_prices - yesterday_prices
    print(f"Yesterday's prices: {yesterday_prices}")
    print(f"Today's prices: {today_prices}")
    print(f"Price changes: {price_changes}")
    
    # Application 3: Image Processing
    print("\nApplication 3: Image Processing")
    original_pixel = np.array([255, 128, 64])    # RGB values
    modified_pixel = np.array([200, 100, 50])    # Modified RGB values
    pixel_difference = original_pixel - modified_pixel
    print(f"Original pixel: {original_pixel}")
    print(f"Modified pixel: {modified_pixel}")
    print(f"Pixel difference: {pixel_difference}")
    print()

if __name__ == "__main__":    
    real_world_applications()
