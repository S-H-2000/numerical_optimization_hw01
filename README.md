# Numerical Optimization Programming Assignment

A comprehensive implementation of unconstrained optimization algorithms including **Gradient Descent** and **Newton's Method** with visualization tools.

## 🎯 Project Overview

This project implements and compares two fundamental optimization algorithms:
- **Gradient Descent (GD)**: First-order optimization method
- **Newton's Method**: Second-order optimization method using Hessian information

The implementation includes:
- ✅ Multiple test functions (quadratic, Rosenbrock, exponential, linear)
- ✅ Backtracking line search with Wolfe conditions
- ✅ Comprehensive visualization (contour plots + convergence plots)
- ✅ Performance comparison and analysis

## 📁 Project Structure

```
numerical_optimization_hw01/
├── src/
│   ├── unconstrained_min.py    # Core optimization algorithms
│   └── utils.py                # Plotting utilities
├── tests/
│   ├── examples.py             # Test functions with gradients/Hessians
│   └── test_unconstrained_min.py # Test suite and evaluation
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🔧 Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Navigate to project directory:**
   ```bash
   cd numerical_optimization_hw01
   ```

## 🚀 Running the Tests

Execute all optimization tests and generate visualizations:

```bash
python -m tests.test_unconstrained_min
```

This will:
- Run both GD and Newton's method on all test functions
- Display optimization paths on contour plots
- Show convergence plots for function values
- Print detailed results for each test

## 📊 Test Functions

### 1. **Quadratic Circle** 
- `f(x) = x^T x` (Identity matrix)
- Circular contours, easy optimization

### 2. **Quadratic Ellipse**
- `f(x) = x^T Q x` with `Q = diag(1, 100)` 
- Elongated elliptical contours

### 3. **Quadratic Rotated**
- Rotated elliptical quadratic (45° rotation)
- Tests algorithm robustness to coordinate system

### 4. **Rosenbrock Function**
- `f(x,y) = 100(y - x²)² + (1 - x)²`
- Classic "banana" function, challenging for GD

### 5. **Linear Function**
- `f(x) = a^T x`
- Unbounded optimization (tests algorithm behavior)

### 6. **Triangle Exponential**
- `f(x₁,x₂) = exp(x₁+3x₂) + exp(x₁-3x₂) + exp(-x₁)`
- Non-quadratic function with interesting geometry

## 🎨 Visualization Features

- **Contour Plots**: Function contours with optimization paths overlaid
- **Convergence Plots**: Function value vs iteration (log scale)
- **Path Comparison**: Side-by-side comparison of GD vs Newton paths
- **Color-coded**: Different colors for each algorithm

## ⚙️ Algorithm Details

### Gradient Descent
- Direction: `-∇f(x)`
- Step size: Backtracking line search
- Suitable for: Large-scale problems, when Hessian is expensive

### Newton's Method  
- Direction: `-H⁻¹∇f(x)` (where H is Hessian)
- Step size: Backtracking line search
- Suitable for: Problems where Hessian is available, faster convergence

### Line Search
- **Backtracking** with Wolfe condition
- Ensures sufficient decrease: `f(x + αd) ≤ f(x) + c₁α∇f^T d`
- Parameters: `c₁ = 0.01`, backtrack factor `β = 0.5`

## 📈 Expected Results

- **Quadratic functions**: Newton converges in 1-2 iterations, GD takes more steps
- **Rosenbrock**: Newton typically outperforms GD significantly  
- **Linear**: Both methods should detect unbounded nature
- **Exponential**: Tests robustness on non-polynomial functions

## 🔬 Performance Metrics

For each test, the output includes:
- Final optimized point `x*`
- Final function value `f(x*)`
- Number of iterations required
- Success/failure status
- Visual comparison of optimization paths

## 🛠️ Customization

You can easily:
- Add new test functions in `tests/examples.py`
- Modify optimization parameters in `src/unconstrained_min.py`
- Adjust plotting ranges in `tests/test_unconstrained_min.py`
- Change convergence tolerances

## 📚 Dependencies

- **NumPy**: Numerical computations and linear algebra
- **Matplotlib**: Plotting and visualization  
- **SciPy**: (Optional) Additional numerical utilities

---

**Author**: Numerical Optimization Course  
**Purpose**: Programming Assignment - Unconstrained Optimization 