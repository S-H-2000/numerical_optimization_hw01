import numpy as np

### Creates different types shapes of functions to test the optimization algorithms

def quadratic_circle(x, hessian=False):
    """
    Simple function: f(x) = x^T Q x with Q = Identity matrix
    """
    # Convert input to numpy array to make sure we can do math with it
    x = np.array(x, dtype=float)
    # Create identity matrix (1's on diagonal, 0's elsewhere)
    Q = np.eye(len(x))
    
    # Calculate function value: x^T Q x
    f = x.T @ Q @ x
    # Calculate gradient: 2Qx
    grad = 2 * Q @ x
    
    # If we need the Hessian, return it along with f and grad
    if hessian:
        hess = 2 * Q
        return f, grad, hess
    else:
        return f, grad, None



def quadratic_ellipse(x, hessian=False):
    """
    Elliptical: f(x) = x^T Q x with Q = diag(1, 100)
    """
    # Convert input to numpy array
    x = np.array(x, dtype=float)
    # Create diagonal matrix with different values (1 and 100)
    Q = np.diag([1, 100])
    
    # Calculate function value: x^T Q x
    f = x.T @ Q @ x
    # Calculate gradient: 2Qx
    grad = 2 * Q @ x
    
    # Return Hessian if requested
    if hessian:
        hess = 2 * Q
        return f, grad, hess
    else:
        return f, grad, None



def quadratic_rotated(x, hessian=False):
    """
    Rotated ellipse: Q = R^T diag(100, 1) R where R is rotation matrix
    """
    # Convert input to numpy array
    x = np.array(x, dtype=float)
    # Create rotation matrix (rotates by 30 degrees)
    R = np.array([[np.sqrt(3)/2, -0.5],
                  [0.5,           np.sqrt(3)/2]])
    # Create diagonal matrix
    D = np.diag([100, 1])
    # Combine rotation and diagonal matrices
    Q = R.T @ D @ R
    
    # Calculate function value and gradient
    f = x.T @ Q @ x
    grad = 2 * Q @ x
    
    # Return Hessian if requested
    if hessian:
        hess = 2 * Q
        return f, grad, hess
    else:
        return f, grad, None



def rosenbrock(x, hessian=False):
    """
    Rosenbrock: f(x) = 100(y - x^2)^2 + (1 - x)^2
    """
    # Convert input to numpy array
    x = np.array(x, dtype=float)
    # Split into x1 and x2 for easier reading
    x1, x2 = x[0], x[1]
    
    # Calculate function value
    f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    
    # Calculate partial derivatives for gradient
    df_dx1 = -400 * x1 * (x2 - x1**2) - 2 * (1 - x1)
    df_dx2 = 200 * (x2 - x1**2)
    grad = np.array([df_dx1, df_dx2])
    
    # Calculate Hessian if requested
    if hessian:
        # Calculate second derivatives
        d2f_dx1x1 = 1200 * x1**2 - 400 * x2 + 2
        d2f_dx1x2 = -400 * x1
        d2f_dx2x1 = -400 * x1
        d2f_dx2x2 = 200
        # Put second derivatives into matrix
        hess = np.array([[d2f_dx1x1, d2f_dx1x2],
                         [d2f_dx2x1, d2f_dx2x2]])
        return f, grad, hess
    else:
        return f, grad, None

def linear_func(x, hessian=False):
    """
    Linear function f(x) = a^T x
    """
    # Convert input to numpy array
    x = np.array(x, dtype=float)
    # Define coefficients for linear function
    a = np.array([1, -2])
    
    # Calculate function value and gradient
    f = a.T @ x
    grad = a
    
    # Return zero Hessian if requested (linear function has no curvature)
    if hessian:
        hess = np.zeros((len(x), len(x)))
        return f, grad, hess
    else:
        return f, grad, None

def triangle_exp(x, hessian=False):
    """
    f(x1, x2) = exp(x1+3x2) + exp(x1-3x2) + exp(-x1)
    """
    # Convert input to numpy array
    x = np.array(x, dtype=float)
    # Split into x1 and x2 for easier reading
    x1, x2 = x[0], x[1]
    
    # Calculate each exponential term
    exp1 = np.exp(x1 + 3*x2 - 0.1)
    exp2 = np.exp(x1 - 3*x2 - 0.1)
    exp3 = np.exp(-x1 - 0.1)
    
    # Sum up the exponentials
    f = exp1 + exp2 + exp3
    
    # Calculate partial derivatives for gradient
    df_dx1 = exp1 + exp2 - exp3
    df_dx2 = 3*exp1 - 3*exp2
    grad = np.array([df_dx1, df_dx2])
    
    # Calculate Hessian if requested
    if hessian:
        # Calculate second derivatives
        d2f_dx1x1 = exp1 + exp2 + exp3
        d2f_dx1x2 = 3*exp1 - 3*exp2
        d2f_dx2x1 = 3*exp1 - 3*exp2
        d2f_dx2x2 = 9*exp1 + 9*exp2
        # Put second derivatives into matrix
        hess = np.array([[d2f_dx1x1, d2f_dx1x2],
                         [d2f_dx2x1, d2f_dx2x2]])
        return f, grad, hess
    else:
        return f, grad, None
