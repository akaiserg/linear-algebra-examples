# Machine Learning with Outer and Inner Products

## Problem Statement
Machine Learning algorithms heavily rely on both outer and inner products for:

- **Neural Networks**: Weight matrices and activations
- **Recommendation Systems**: User-item interactions
- **Feature Engineering**: Creating interaction features
- **Matrix Factorization**: Decomposing large matrices
- **Embedding Layers**: Converting categorical data to vectors

## Outer Product vs Inner Product in ML

### **Inner Product (Dot Product)**
- **Purpose**: Calculate similarity, compute activations, find projections
- **Result**: Scalar (single number)
- **Formula**: `v1 · v2 = v1[0]*v2[0] + v1[1]*v2[1] + ... + v1[n]*v2[n]`
- **ML Applications**: 
  - Neural network activations
  - Similarity calculations
  - Loss functions
  - Attention mechanisms

### **Outer Product**
- **Purpose**: Create interaction matrices, build weight matrices, feature combinations
- **Result**: Matrix (2D array)
- **Formula**: `v1 ⊗ v2 = v1 * v2^T`
- **ML Applications**:
  - Weight matrix initialization
  - Feature interaction matrices
  - Embedding combinations
  - Matrix factorization

## Key Concepts Demonstrated
- **Neural Network Layers**: How outer/inner products build neural networks
- **Recommendation Systems**: User-item interaction matrices
- **Feature Engineering**: Creating new features from existing ones
- **Matrix Factorization**: Breaking down large matrices
- **Embedding Layers**: Converting categories to vectors

## Learning Objectives
- Understand when to use outer vs inner products in ML
- Learn how neural networks use these operations
- Apply products in recommendation systems
- Practice feature engineering techniques
- Visualize ML operations with charts

## Files
- `ml_products.py`: Main implementation with examples
- `README.md`: This documentation

## Business Applications
- **Neural Networks**: Deep learning models
- **Recommendation Systems**: Netflix, Amazon, Spotify
- **Search Engines**: Google, Bing ranking
- **Computer Vision**: Image recognition
- **Natural Language Processing**: Text analysis
- **Fraud Detection**: Anomaly detection
- **Personalization**: Customized user experiences
