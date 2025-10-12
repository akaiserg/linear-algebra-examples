# Linear Algebra Examples

A comprehensive Python project for learning and practicing linear algebra concepts with hands-on examples and visualizations.

## Project Structure

```
linear-algebra-examples/
├── 01_vectors/           # Vector operations and properties
└── requirements.txt      # Python dependencies
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
- Dot product and cross product
- Vector magnitude and normalization
- 2D and 3D vector visualization

**Files:**
- `vector_basics.py` - Fundamental vector operations and properties