"""
Image Processing with Vector Subtraction
=======================================

This module demonstrates how vector subtraction is used in image processing
for change detection, enhancement, and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt

def pixel_vector_subtraction():
    """Demonstrate basic pixel vector subtraction."""
    print("=== Pixel Vector Subtraction ===")
    
    # Original pixel (RGB values)
    original_pixel = np.array([255, 128, 64])    # Red-orange pixel
    modified_pixel = np.array([200, 100, 50])    # Darker version
    
    print(f"Original pixel (RGB): {original_pixel}")
    print(f"Modified pixel (RGB): {modified_pixel}")
    
    # Calculate difference using vector subtraction
    pixel_difference = original_pixel - modified_pixel
    print(f"Pixel difference: {pixel_difference}")
    
    # Interpret the difference
    print(f"Red channel change: {pixel_difference[0]} (darker by {pixel_difference[0]})")
    print(f"Green channel change: {pixel_difference[1]} (darker by {pixel_difference[1]})")
    print(f"Blue channel change: {pixel_difference[2]} (darker by {pixel_difference[2]})")
    print()

def image_change_detection():
    """Demonstrate change detection between two images."""
    print("=== Image Change Detection ===")
    
    # Simulate two images (small 3x3 pixel arrays)
    # Each pixel is represented as [R, G, B] values
    image1 = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],      # Red, Green, Blue
        [[255, 255, 0], [255, 0, 255], [0, 255, 255]], # Yellow, Magenta, Cyan
        [[128, 128, 128], [255, 255, 255], [0, 0, 0]]  # Gray, White, Black
    ])
    
    # Modified image (some pixels changed)
    image2 = np.array([
        [[200, 0, 0], [0, 255, 0], [0, 0, 255]],      # Red darker, others same
        [[255, 255, 0], [200, 0, 200], [0, 255, 255]], # Magenta darker
        [[128, 128, 128], [255, 255, 255], [50, 50, 50]] # Black lighter
    ])
    
    print("Image 1 (original):")
    print(image1)
    print("\nImage 2 (modified):")
    print(image2)
    
    # Calculate differences using vector subtraction
    differences = image1 - image2
    print("\nDifferences (Image1 - Image2):")
    print(differences)
    
    # Find which pixels changed
    changed_pixels = np.any(differences != 0, axis=2)  # Check if any RGB channel changed
    print(f"\nChanged pixels (True/False):")
    print(changed_pixels)
    
    # Count changes
    num_changes = np.sum(changed_pixels)
    print(f"Number of changed pixels: {num_changes}")
    print()

def visualize_pixel_changes():
    """Visualize pixel changes with charts."""
    print("=== Pixel Change Visualization ===")
    
    # Create sample data for visualization
    pixels_original = np.array([
        [255, 128, 64],   # Red-orange
        [100, 200, 150],  # Green
        [80, 120, 200],   # Blue
        [200, 200, 100],  # Yellow
        [150, 100, 200]   # Purple
    ])
    
    pixels_modified = np.array([
        [200, 100, 50],   # Darker red-orange
        [120, 180, 130],  # Slightly different green
        [80, 120, 200],   # Same blue
        [180, 180, 80],   # Darker yellow
        [150, 100, 200]   # Same purple
    ])
    
    # Calculate differences
    differences = pixels_original - pixels_modified
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Original pixels
    colors_original = pixels_original / 255.0  # Normalize to 0-1 for matplotlib
    ax1.bar(range(len(pixels_original)), [1]*len(pixels_original), color=colors_original)
    ax1.set_title('Original Pixels')
    ax1.set_xlabel('Pixel Index')
    ax1.set_ylabel('Intensity')
    ax1.set_ylim(0, 1.2)
    
    # Plot 2: Modified pixels
    colors_modified = pixels_modified / 255.0
    ax2.bar(range(len(pixels_modified)), [1]*len(pixels_modified), color=colors_modified)
    ax2.set_title('Modified Pixels')
    ax2.set_xlabel('Pixel Index')
    ax2.set_ylabel('Intensity')
    ax2.set_ylim(0, 1.2)
    
    # Plot 3: RGB channel differences
    x = np.arange(len(pixels_original))
    width = 0.25
    
    ax3.bar(x - width, differences[:, 0], width, label='Red', color='red', alpha=0.7)
    ax3.bar(x, differences[:, 1], width, label='Green', color='green', alpha=0.7)
    ax3.bar(x + width, differences[:, 2], width, label='Blue', color='blue', alpha=0.7)
    
    ax3.set_title('RGB Channel Differences')
    ax3.set_xlabel('Pixel Index')
    ax3.set_ylabel('Difference Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Total change magnitude
    total_changes = np.sum(np.abs(differences), axis=1)
    ax4.bar(range(len(total_changes)), total_changes, color='orange', alpha=0.7)
    ax4.set_title('Total Change Magnitude per Pixel')
    ax4.set_xlabel('Pixel Index')
    ax4.set_ylabel('Total Change')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization created showing:")
    print("- Original vs Modified pixels")
    print("- RGB channel differences")
    print("- Total change magnitude")
    print()

def image_enhancement():
    """Demonstrate image enhancement using vector subtraction."""
    print("=== Image Enhancement ===")
    
    # Simulate a noisy image
    original_pixel = np.array([200, 150, 100])
    noisy_pixel = np.array([210, 140, 110])  # Added noise
    
    print(f"Original pixel: {original_pixel}")
    print(f"Noisy pixel: {noisy_pixel}")
    
    # Calculate noise (difference)
    noise = noisy_pixel - original_pixel
    print(f"Detected noise: {noise}")
    
    # Remove noise by subtracting it
    enhanced_pixel = noisy_pixel - noise
    print(f"Enhanced pixel: {enhanced_pixel}")
    print(f"Are original and enhanced equal? {np.array_equal(original_pixel, enhanced_pixel)}")
    
    # Brightness adjustment
    brightness_adjustment = np.array([20, 20, 20])  # Make brighter
    brightened_pixel = original_pixel + brightness_adjustment
    print(f"\nBrightness adjustment: {brightness_adjustment}")
    print(f"Brightened pixel: {brightened_pixel}")
    
    # Contrast adjustment (simplified)
    contrast_factor = 1.2
    contrasted_pixel = (original_pixel - 128) * contrast_factor + 128
    contrasted_pixel = np.clip(contrasted_pixel, 0, 255).astype(int)
    print(f"Contrasted pixel: {contrasted_pixel}")
    print()

def edge_detection_simple():
    """Simple edge detection using vector subtraction."""
    print("=== Simple Edge Detection ===")
    
    # Create a simple 3x3 image with edges
    image = np.array([
        [[100, 100, 100], [100, 100, 100], [200, 200, 200]],  # Edge between 2nd and 3rd column
        [[100, 100, 100], [100, 100, 100], [200, 200, 200]],
        [[100, 100, 100], [100, 100, 100], [200, 200, 200]]
    ])
    
    print("Original image (3x3 pixels):")
    print(image)
    
    # Detect horizontal edges (compare rows)
    horizontal_edges = np.zeros((2, 3, 3))
    for i in range(2):
        horizontal_edges[i] = image[i+1] - image[i]  # Vector subtraction
    
    print("\nHorizontal edges (row differences):")
    print(horizontal_edges)
    
    # Detect vertical edges (compare columns)
    vertical_edges = np.zeros((3, 2, 3))
    for j in range(2):
        vertical_edges[:, j] = image[:, j+1] - image[:, j]  # Vector subtraction
    
    print("\nVertical edges (column differences):")
    print(vertical_edges)
    
    # Calculate edge strength
    horizontal_strength = np.sum(np.abs(horizontal_edges), axis=2)
    vertical_strength = np.sum(np.abs(vertical_edges), axis=2)
    
    print(f"\nHorizontal edge strength:")
    print(horizontal_strength)
    print(f"Vertical edge strength:")
    print(vertical_strength)
    print()

def color_correction():
    """Demonstrate color correction using vector subtraction."""
    print("=== Color Correction ===")
    
    # Simulate color cast (too much red)
    original_pixel = np.array([150, 150, 150])  # Neutral gray
    color_cast_pixel = np.array([180, 150, 150])  # Too much red
    
    print(f"Original pixel: {original_pixel}")
    print(f"Color cast pixel: {color_cast_pixel}")
    
    # Calculate color cast
    color_cast = color_cast_pixel - original_pixel
    print(f"Color cast: {color_cast}")
    
    # Correct by subtracting the cast
    corrected_pixel = color_cast_pixel - color_cast
    print(f"Corrected pixel: {corrected_pixel}")
    
    # Partial correction (subtract half the cast)
    partial_correction = color_cast_pixel - (color_cast * 0.5)
    print(f"Partially corrected pixel: {partial_correction}")
    print()

def image_comparison_metrics():
    """Calculate image comparison metrics using vector subtraction."""
    print("=== Image Comparison Metrics ===")
    
    # Two similar images
    image1 = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 0], [255, 0, 255], [0, 255, 255]]
    ])
    
    image2 = np.array([
        [[250, 5, 5], [5, 250, 5], [5, 5, 250]],  # Slightly different
        [[250, 250, 5], [250, 5, 250], [5, 250, 250]]
    ])
    
    print("Image 1:")
    print(image1)
    print("\nImage 2:")
    print(image2)
    
    # Calculate differences
    differences = image1 - image2
    print(f"\nDifferences:")
    print(differences)
    
    # Calculate metrics
    mean_squared_error = np.mean(differences ** 2)
    mean_absolute_error = np.mean(np.abs(differences))
    max_error = np.max(np.abs(differences))
    
    print(f"\nComparison Metrics:")
    print(f"Mean Squared Error: {mean_squared_error:.2f}")
    print(f"Mean Absolute Error: {mean_absolute_error:.2f}")
    print(f"Maximum Error: {max_error}")
    
    # Calculate similarity percentage
    total_pixels = image1.size
    similar_pixels = np.sum(np.abs(differences) < 10)  # Within 10 units
    similarity_percentage = (similar_pixels / total_pixels) * 100
    print(f"Similarity Percentage: {similarity_percentage:.1f}%")
    print()

if __name__ == "__main__":
    print("Image Processing with Vector Subtraction\n")
    pixel_vector_subtraction()
    image_change_detection()
    visualize_pixel_changes()
    image_enhancement()
    edge_detection_simple()
    color_correction()
    image_comparison_metrics()
