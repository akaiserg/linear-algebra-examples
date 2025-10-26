# Linear Algebra Examples

A comprehensive Python project for learning and practicing linear algebra concepts with hands-on examples and visualizations.

## Project Structure

```
linear-algebra-examples/
├── 01_vectors/                    # Vector operations and properties
│   ├── vector_creation.py         # Different ways to create vectors
│   ├── add_vectors.py             # Vector addition with visualization
│   ├── subtract_vectors.py        # Vector subtraction examples
│   ├── dot_product.py             # Dot product operations
│   ├── outer_product.py           # Outer product operations
│   ├── vector_transpose.py        # Vector transposition
│   ├── span_and_basis.py          # Span and basis concepts
│   ├── data_science_case_addition/    # Movie recommendation system
│   │   ├── README.md
│   │   └── recommendation_system.py
│   ├── customer_segmentation_case/    # Customer analysis with transpose
│   │   ├── README.md
│   │   └── customer_segmentation_simple.py
│   ├── image_processing_case/         # Image processing with subtraction
│   │   ├── README.md
│   │   └── image_processing.py
│   ├── document_similarity_case/      # Document similarity with dot product
│   │   ├── README.md
│   │   └── document_similarity.py
│   ├── pca_data_analysis_case/        # PCA analysis with span and basis
│   │   ├── README.md
│   │   └── pca_data_analysis.py
│   └── ml_products_case/              # ML with outer and inner products
│       ├── README.md
│       └── ml_products.py
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

## Getting Started

### Pre-configuration: Virtual Environment (Recommended)

Before installing dependencies, it's recommended to create a virtual environment to isolate project dependencies:

#### Option 1: Using `venv` (built into Python 3.3+)
```bash
# Navigate to project directory
cd linear-algebra-examples

# Create virtual environment
python -m venv linear_algebra_env

# Activate virtual environment
# On macOS/Linux:
source linear_algebra_env/bin/activate

# On Windows:
# linear_algebra_env\Scripts\activate
```

#### Option 2: Using `conda` (if you have Anaconda/Miniconda)
```bash
# Create environment with Python 3.9
conda create -n linear_algebra python=3.9

# Activate environment
conda activate linear_algebra
```

### Installation Steps

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python -c "import numpy, matplotlib, scipy; print('All packages installed successfully!')"
```

3. **Run examples:**
```bash
python 01_vectors/vector_basics.py
```

4. **Deactivate when done:**
```bash
deactivate
```

### Why Use Virtual Environments?
- **Isolation**: Keeps project dependencies separate from system Python
- **Reproducibility**: Ensures consistent package versions
- **Clean workspace**: Avoids conflicts between different projects
- **Easy cleanup**: Can delete the entire environment if needed

## Learning Path

### 1. Vectors (`01_vectors/`)

**What are Vectors?**
- Vectors represent direction and magnitude in space
- In NumPy, vectors are typically represented as 1D arrays
- Shape: `(n,)` where n is the number of components
- Dimension: `1` (1-dimensional array)

**Key Concepts:**
- **1D Arrays as Vectors**: `np.array([1, 2, 3])` creates a 3D vector
- **Shape vs Dimension**: 
  - Shape `(3,)` means "3 elements in 1 dimension"
  - Dimension `1` means "1-dimensional array (vector)"
- **Vector Types**:
  - 1D arrays `(n,)` - most common vector representation
  - Row vectors `(1, n)` - 2D arrays with 1 row
  - Column vectors `(n, 1)` - 2D arrays with 1 column

**Why Shape and Dimension Matter:**
- **Compatibility**: Vector operations require compatible shapes
- **Matrix Operations**: Matrix-vector multiplication needs matching dimensions
- **Error Prevention**: Shape mismatches are common sources of errors
- **Memory Efficiency**: 1D arrays use less memory than 2D arrays for the same data

**Examples Covered:**
- Vector creation and properties
- Basic operations (addition, subtraction, scalar multiplication)
- Dot product operations and applications
- Outer product operations and applications
- Vector transposition and applications
- Span and basis concepts
- Real-world applications and case studies

**Files:**
- `vector_creation.py` - Different ways to create vectors in NumPy
- `add_vectors.py` - Vector addition with geometric visualization
- `subtract_vectors.py` - Vector subtraction with practical examples
- `dot_product.py` - Dot product operations and geometric meaning
- `outer_product.py` - Outer product operations and applications
- `vector_transpose.py` - Vector transposition and applications
- `span_and_basis.py` - Span and basis concepts with examples

**Real-World Case Studies:**
- `data_science_case_addition/` - Movie recommendation system using vector addition
- `customer_segmentation_case/` - Customer analysis using vector transpose
- `image_processing_case/` - Image processing using vector subtraction
- `document_similarity_case/` - Document similarity using dot product
- `pca_data_analysis_case/` - PCA analysis using span and basis concepts
- `ml_products_case/` - Machine learning with outer and inner products

## Case Studies Overview

### 1. Movie Recommendation System (`data_science_case_addition/`)
**Learning Focus:** Vector Addition in Data Science
- **Problem:** Recommend movies based on user preferences
- **Solution:** Use vector addition to combine user preferences with movie features
- **Key Concepts:** Element-wise addition, weighted combinations, recommendation algorithms
- **Real Applications:** Netflix, Amazon, Spotify recommendation systems

### 2. Customer Segmentation (`customer_segmentation_case/`)
**Learning Focus:** Vector Transpose in Business Analytics
- **Problem:** Analyze customer behavior and segment customers
- **Solution:** Use transpose operations to analyze customer-category relationships
- **Key Concepts:** Matrix transpose, feature engineering, customer analytics
- **Real Applications:** Marketing, customer relationship management, business intelligence

### 3. Image Processing (`image_processing_case/`)
**Learning Focus:** Vector Subtraction in Computer Vision
- **Problem:** Detect changes, enhance images, and analyze differences
- **Solution:** Use vector subtraction for change detection and image enhancement
- **Key Concepts:** Pixel vectors, change detection, image enhancement, error analysis
- **Real Applications:** Security systems, medical imaging, quality control

### 4. Document Similarity (`document_similarity_case/`)
**Learning Focus:** Dot Product in Natural Language Processing
- **Problem:** Find similar documents and rank search results
- **Solution:** Use dot product to measure document similarity
- **Key Concepts:** Bag of words, TF-IDF, similarity scoring, search engines
- **Real Applications:** Google search, plagiarism detection, content management

### 5. PCA Data Analysis (`pca_data_analysis_case/`)
**Learning Focus:** Span and Basis in Machine Learning
- **Problem:** Reduce dimensionality and find important patterns in data
- **Solution:** Use PCA to find new basis vectors that maximize variance
- **Key Concepts:** Principal components, dimensionality reduction, variance analysis
- **Real Applications:** Data visualization, feature engineering, image compression

### 6. Machine Learning Products (`ml_products_case/`)
**Learning Focus:** Outer and Inner Products in ML
- **Problem:** Build neural networks, recommendation systems, and attention mechanisms
- **Solution:** Use outer products for matrices and inner products for activations
- **Key Concepts:** Neural networks, matrix factorization, attention mechanisms, feature engineering
- **Real Applications:** Deep learning, recommendation systems, transformers, collaborative filtering

## Quick Start Examples

### Run Basic Vector Operations
```bash
# Vector creation
python 01_vectors/vector_creation.py

# Vector addition with visualization
python 01_vectors/add_vectors.py

# Vector subtraction
python 01_vectors/subtract_vectors.py

# Dot product operations
python 01_vectors/dot_product.py

# Outer product operations
python 01_vectors/outer_product.py
```

### Run Real-World Case Studies
```bash
# Movie recommendation system
python 01_vectors/data_science_case_addition/recommendation_system.py

# Customer segmentation
python 01_vectors/customer_segmentation_case/customer_segmentation_simple.py

# Image processing
python 01_vectors/image_processing_case/image_processing.py

# Document similarity
python 01_vectors/document_similarity_case/document_similarity.py

# PCA data analysis
python 01_vectors/pca_data_analysis_case/pca_data_analysis.py

# Machine learning with outer and inner products
python 01_vectors/ml_products_case/ml_products.py
```