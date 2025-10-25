"""
PCA Data Analysis with Span and Basis
====================================

This module demonstrates how Principal Component Analysis (PCA) uses
span and basis concepts for data analysis and dimensionality reduction.
"""

import numpy as np
import matplotlib.pyplot as plt

def simple_2d_pca():
    """Demonstrate PCA on simple 2D data."""
    print("=== Simple 2D PCA Example ===")
    
    # Create sample 2D data (correlated)
    np.random.seed(42)
    n_points = 100
    
    # Generate data along a line with some noise
    x = np.random.normal(0, 2, n_points)
    y = 0.5 * x + np.random.normal(0, 0.5, n_points)  # y = 0.5x + noise
    
    data = np.column_stack([x, y])
    print(f"Data shape: {data.shape}")
    print(f"First 5 points: {data[:5]}")
    
    # Step 1: Center the data (subtract mean)
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    print(f"\nMean: {mean}")
    print(f"Centered data (first 5): {centered_data[:5]}")
    
    # Step 2: Calculate covariance matrix
    cov_matrix = np.cov(centered_data.T)
    print(f"\nCovariance matrix:")
    print(cov_matrix)
    
    # Step 3: Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors (Principal Components):")
    print(f"PC1: {eigenvectors[:, 0]}")
    print(f"PC2: {eigenvectors[:, 1]}")
    
    # Step 4: Transform data to new basis
    transformed_data = centered_data @ eigenvectors
    print(f"\nTransformed data (first 5): {transformed_data[:5]}")
    
    # Step 5: Calculate explained variance
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues / total_variance
    print(f"\nExplained variance:")
    print(f"PC1: {explained_variance[0]:.3f} ({explained_variance[0]*100:.1f}%)")
    print(f"PC2: {explained_variance[1]:.3f} ({explained_variance[1]*100:.1f}%)")
    
    # Visualize the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Original data with principal components
    ax1.scatter(centered_data[:, 0], centered_data[:, 1], alpha=0.6, color='blue')
    
    # Draw principal components
    scale = 3
    ax1.quiver(0, 0, eigenvectors[0, 0] * scale, eigenvectors[1, 0] * scale,
               angles='xy', scale_units='xy', scale=1, color='red', width=0.005,
               label=f'PC1 (var: {explained_variance[0]:.1%})')
    ax1.quiver(0, 0, eigenvectors[0, 1] * scale, eigenvectors[1, 1] * scale,
               angles='xy', scale_units='xy', scale=1, color='green', width=0.005,
               label=f'PC2 (var: {explained_variance[1]:.1%})')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Original Data with Principal Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot 2: Transformed data
    ax2.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.6, color='purple')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Data in New Basis (PC1, PC2)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualization shows:")
    print("- Original data with principal component directions")
    print("- Data transformed to new coordinate system")
    print("- PC1 captures most variance (main direction)")
    print()

def dimensionality_reduction():
    """Demonstrate dimensionality reduction with PCA."""
    print("=== Dimensionality Reduction ===")
    
    # Create 3D data
    np.random.seed(42)
    n_points = 100
    
    # Generate data in 3D space
    x = np.random.normal(0, 2, n_points)
    y = 0.5 * x + np.random.normal(0, 0.5, n_points)
    z = 0.3 * x + 0.2 * y + np.random.normal(0, 0.3, n_points)
    
    data_3d = np.column_stack([x, y, z])
    print(f"3D Data shape: {data_3d.shape}")
    
    # Center the data
    mean_3d = np.mean(data_3d, axis=0)
    centered_3d = data_3d - mean_3d
    
    # Calculate covariance matrix and eigenvectors
    cov_3d = np.cov(centered_3d.T)
    eigenvalues_3d, eigenvectors_3d = np.linalg.eig(cov_3d)
    
    # Sort by eigenvalue
    idx_3d = np.argsort(eigenvalues_3d)[::-1]
    eigenvalues_3d = eigenvalues_3d[idx_3d]
    eigenvectors_3d = eigenvectors_3d[:, idx_3d]
    
    print(f"Eigenvalues: {eigenvalues_3d}")
    
    # Calculate explained variance
    total_var_3d = np.sum(eigenvalues_3d)
    explained_var_3d = eigenvalues_3d / total_var_3d
    
    print(f"\nExplained variance:")
    for i, var in enumerate(explained_var_3d):
        print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    # Reduce to 2D (keep first 2 principal components)
    pc_2d = eigenvectors_3d[:, :2]  # First 2 principal components
    data_2d = centered_3d @ pc_2d
    
    print(f"\nReduced data shape: {data_2d.shape}")
    print(f"Variance retained: {np.sum(explained_var_3d[:2]):.1%}")
    
    # Visualize 3D to 2D reduction
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Original 3D data
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(centered_3d[:, 0], centered_3d[:, 1], centered_3d[:, 2], 
                alpha=0.6, color='blue')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original 3D Data')
    
    # Plot 2: Reduced 2D data
    ax2 = fig.add_subplot(132)
    ax2.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6, color='red')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Reduced to 2D (PC1, PC2)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Explained variance
    ax3 = fig.add_subplot(133)
    ax3.bar(range(1, 4), explained_var_3d, color='green', alpha=0.7)
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Explained Variance')
    ax3.set_title('Explained Variance by Component')
    ax3.set_xticks(range(1, 4))
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for i, var in enumerate(explained_var_3d):
        ax3.text(i+1, var + 0.01, f'{var*100:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization shows:")
    print("- Original 3D data")
    print("- Data reduced to 2D using first 2 principal components")
    print("- Variance explained by each component")
    print()

def customer_data_analysis():
    """Analyze customer data using PCA."""
    print("=== Customer Data Analysis ===")
    
    # Simulate customer data
    np.random.seed(42)
    n_customers = 200
    
    # Generate customer features
    age = np.random.normal(35, 10, n_customers)
    income = np.random.normal(50000, 15000, n_customers)
    spending = 0.3 * income + np.random.normal(0, 5000, n_customers)
    visits = np.random.poisson(10, n_customers)
    
    # Create correlation between features
    spending = 0.3 * income + 0.1 * age + np.random.normal(0, 3000, n_customers)
    visits = 0.2 * spending / 1000 + np.random.poisson(8, n_customers)
    
    customer_data = np.column_stack([age, income, spending, visits])
    
    print("Customer Data Features:")
    print("- Age (years)")
    print("- Income (dollars)")
    print("- Spending (dollars)")
    print("- Visits (number of visits)")
    
    print(f"\nData shape: {customer_data.shape}")
    print(f"First 5 customers: {customer_data[:5]}")
    
    # Center the data
    mean_customer = np.mean(customer_data, axis=0)
    centered_customer = customer_data - mean_customer
    
    # Calculate covariance matrix
    cov_customer = np.cov(centered_customer.T)
    print(f"\nCovariance matrix:")
    print(cov_customer)
    
    # Find principal components
    eigenvalues_customer, eigenvectors_customer = np.linalg.eig(cov_customer)
    idx_customer = np.argsort(eigenvalues_customer)[::-1]
    eigenvalues_customer = eigenvalues_customer[idx_customer]
    eigenvectors_customer = eigenvectors_customer[:, idx_customer]
    
    # Calculate explained variance
    total_var_customer = np.sum(eigenvalues_customer)
    explained_var_customer = eigenvalues_customer / total_var_customer
    
    print(f"\nPrincipal Components:")
    feature_names = ['Age', 'Income', 'Spending', 'Visits']
    for i in range(4):
        print(f"\nPC{i+1} (explains {explained_var_customer[i]*100:.1f}% of variance):")
        for j, feature in enumerate(feature_names):
            print(f"  {feature}: {eigenvectors_customer[j, i]:.3f}")
    
    # Reduce to 2D for visualization
    pc_2d_customer = eigenvectors_customer[:, :2]
    customer_2d = centered_customer @ pc_2d_customer
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Customer data in 2D
    ax1.scatter(customer_2d[:, 0], customer_2d[:, 1], alpha=0.6, color='blue')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Customers in 2D Space')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Explained variance
    ax2.bar(range(1, 5), explained_var_customer, color='green', alpha=0.7)
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance')
    ax2.set_title('Explained Variance by Component')
    ax2.set_xticks(range(1, 5))
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, var in enumerate(explained_var_customer):
        ax2.text(i+1, var + 0.01, f'{var*100:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nBusiness Insights:")
    print(f"- PC1 explains {explained_var_customer[0]*100:.1f}% of customer variation")
    print(f"- PC2 explains {explained_var_customer[1]*100:.1f}% of customer variation")
    print(f"- First 2 components explain {np.sum(explained_var_customer[:2])*100:.1f}% of total variance")
    print(f"- We can reduce 4D customer data to 2D with minimal information loss")
    print()

def image_compression_example():
    """Demonstrate PCA for image compression."""
    print("=== Image Compression with PCA ===")
    
    # Create a simple 8x8 image (simulating a small image)
    np.random.seed(42)
    image_size = 8
    
    # Create a simple pattern (diagonal stripes)
    image = np.zeros((image_size, image_size))
    for i in range(image_size):
        for j in range(image_size):
            if (i + j) % 2 == 0:
                image[i, j] = 255  # White
            else:
                image[i, j] = 0    # Black
    
    # Add some noise
    noise = np.random.normal(0, 10, (image_size, image_size))
    image = np.clip(image + noise, 0, 255)
    
    print(f"Original image shape: {image.shape}")
    print(f"Original image (first 4x4):")
    print(image[:4, :4])
    
    # Flatten image to 1D vector
    image_flat = image.flatten()
    print(f"Flattened image shape: {image_flat.shape}")
    
    # For PCA, we need multiple samples. Let's create variations
    n_samples = 50
    images = np.zeros((n_samples, image_size * image_size))
    
    # Create variations of the original image
    for i in range(n_samples):
        # Add different noise to create variations
        noise = np.random.normal(0, 5, (image_size, image_size))
        variation = image + noise
        variation = np.clip(variation, 0, 255)
        images[i] = variation.flatten()
    
    print(f"Image dataset shape: {images.shape}")
    
    # Center the data
    mean_image = np.mean(images, axis=0)
    centered_images = images - mean_image
    
    # Calculate covariance matrix
    cov_image = np.cov(centered_images.T)
    
    # Find principal components
    eigenvalues_image, eigenvectors_image = np.linalg.eig(cov_image)
    idx_image = np.argsort(eigenvalues_image)[::-1]
    eigenvalues_image = eigenvalues_image[idx_image]
    eigenvectors_image = eigenvectors_image[:, idx_image]
    
    # Calculate explained variance
    total_var_image = np.sum(eigenvalues_image)
    explained_var_image = eigenvalues_image / total_var_image
    
    print(f"\nExplained variance (first 10 components):")
    for i in range(min(10, len(explained_var_image))):
        print(f"PC{i+1}: {explained_var_image[i]*100:.1f}%")
    
    # Compress image using different numbers of components
    compression_ratios = [1, 2, 4, 8, 16]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot compressed images
    for i, n_components in enumerate(compression_ratios):
        if i + 1 < len(axes):
            # Use first n_components
            pc_compressed = eigenvectors_image[:, :n_components]
            
            # Transform and reconstruct
            transformed = centered_images[0] @ pc_compressed
            reconstructed = transformed @ pc_compressed.T + mean_image
            
            # Reshape back to image
            reconstructed_image = reconstructed.reshape(image_size, image_size)
            reconstructed_image = np.clip(reconstructed_image, 0, 255)
            
            axes[i + 1].imshow(reconstructed_image, cmap='gray')
            variance_retained = np.sum(explained_var_image[:n_components])
            axes[i + 1].set_title(f'{n_components} PCs\n({variance_retained*100:.1f}% variance)')
            axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nCompression Results:")
    for n_components in compression_ratios:
        variance_retained = np.sum(explained_var_image[:n_components])
        compression_ratio = (image_size * image_size) / n_components
        print(f"{n_components:2d} components: {variance_retained*100:5.1f}% variance, "
              f"{compression_ratio:4.1f}x compression")
    print()

def pca_vs_original_basis():
    """Compare PCA basis with original basis."""
    print("=== PCA Basis vs Original Basis ===")
    
    # Create 2D data
    np.random.seed(42)
    n_points = 100
    
    # Generate correlated data
    x = np.random.normal(0, 2, n_points)
    y = 0.7 * x + np.random.normal(0, 0.5, n_points)
    data = np.column_stack([x, y])
    
    # Center the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Find PCA basis
    cov_matrix = np.cov(centered_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Original basis (standard basis)
    original_basis = np.eye(2)  # [1,0] and [0,1]
    
    print("Original Basis (Standard Basis):")
    print(f"e1 = {original_basis[:, 0]}")
    print(f"e2 = {original_basis[:, 1]}")
    
    print(f"\nPCA Basis (Principal Components):")
    print(f"PC1 = {eigenvectors[:, 0]}")
    print(f"PC2 = {eigenvectors[:, 1]}")
    
    # Transform data to both bases
    data_original = centered_data @ original_basis  # Same as centered_data
    data_pca = centered_data @ eigenvectors
    
    # Calculate variance in each direction
    var_original_x = np.var(data_original[:, 0])
    var_original_y = np.var(data_original[:, 1])
    var_pca_1 = np.var(data_pca[:, 0])
    var_pca_2 = np.var(data_pca[:, 1])
    
    print(f"\nVariance in Original Basis:")
    print(f"X direction: {var_original_x:.3f}")
    print(f"Y direction: {var_original_y:.3f}")
    print(f"Total: {var_original_x + var_original_y:.3f}")
    
    print(f"\nVariance in PCA Basis:")
    print(f"PC1 direction: {var_pca_1:.3f}")
    print(f"PC2 direction: {var_pca_2:.3f}")
    print(f"Total: {var_pca_1 + var_pca_2:.3f}")
    
    # Visualize both representations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Original basis
    ax1.scatter(data_original[:, 0], data_original[:, 1], alpha=0.6, color='blue')
    ax1.set_xlabel('X (Original)')
    ax1.set_ylabel('Y (Original)')
    ax1.set_title('Data in Original Basis')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot 2: PCA basis
    ax2.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.6, color='red')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Data in PCA Basis')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nKey Insights:")
    print(f"- PCA basis maximizes variance in first direction")
    print(f"- Original basis has mixed variance in both directions")
    print(f"- PCA basis is more efficient for representing the data")
    print(f"- Same data, different coordinate systems!")
    print()

if __name__ == "__main__":
    print("PCA Data Analysis with Span and Basis\n")
    simple_2d_pca()
    dimensionality_reduction()
    customer_data_analysis()
    image_compression_example()
    pca_vs_original_basis()
