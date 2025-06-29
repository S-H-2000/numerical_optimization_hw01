import numpy as np

# HW1 Functions (unchanged from original)

def quadratic_circle(x, hessian=False):
    """f(x) = x^T x"""
    x = np.array(x, dtype=float)
    Q = np.eye(len(x))
    f = x.T @ Q @ x
    grad = 2 * Q @ x
    if hessian:
        return f, grad, 2 * Q
    return f, grad, None

def quadratic_ellipse(x, hessian=False):
    """f(x) = x^T Q x with Q = diag(1, 100)"""
    x = np.array(x, dtype=float)
    Q = np.diag([1, 100])
    f = x.T @ Q @ x
    grad = 2 * Q @ x
    if hessian:
        return f, grad, 2 * Q
    return f, grad, None

def quadratic_rotated(x, hessian=False):
    """Rotated ellipse with 30 degree rotation"""
    x = np.array(x, dtype=float)
    R = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    D = np.diag([100, 1])
    Q = R.T @ D @ R
    f = x.T @ Q @ x
    grad = 2 * Q @ x
    if hessian:
        return f, grad, 2 * Q
    return f, grad, None

def rosenbrock(x, hessian=False):
    """Rosenbrock function: 100(y - x^2)^2 + (1 - x)^2"""
    x = np.array(x, dtype=float)
    x1, x2 = x[0], x[1]
    f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    df_dx1 = -400 * x1 * (x2 - x1**2) - 2 * (1 - x1)
    df_dx2 = 200 * (x2 - x1**2)
    grad = np.array([df_dx1, df_dx2])
    if hessian:
        d2f_dx1x1 = 1200 * x1**2 - 400 * x2 + 2
        d2f_dx1x2 = -400 * x1
        d2f_dx2x2 = 200
        hess = np.array([[d2f_dx1x1, d2f_dx1x2], [d2f_dx1x2, d2f_dx2x2]])
        return f, grad, hess
    return f, grad, None

def linear_func(x, hessian=False):
    """Linear function f(x) = a^T x"""
    x = np.array(x, dtype=float)
    a = np.array([1, -2])
    f = a.T @ x
    grad = a
    if hessian:
        return f, grad, np.zeros((len(x), len(x)))
    return f, grad, None

def triangle_exp(x, hessian=False):
    """f(x1, x2) = exp(x1+3x2) + exp(x1-3x2) + exp(-x1)"""
    x = np.array(x, dtype=float)
    x1, x2 = x[0], x[1]
    exp1 = np.exp(x1 + 3*x2 - 0.1)
    exp2 = np.exp(x1 - 3*x2 - 0.1)
    exp3 = np.exp(-x1 - 0.1)
    f = exp1 + exp2 + exp3
    df_dx1 = exp1 + exp2 - exp3
    df_dx2 = 3*exp1 - 3*exp2
    grad = np.array([df_dx1, df_dx2])
    if hessian:
        d2f_dx1x1 = exp1 + exp2 + exp3
        d2f_dx1x2 = 3*exp1 - 3*exp2
        d2f_dx2x2 = 9*exp1 + 9*exp2
        hess = np.array([[d2f_dx1x1, d2f_dx1x2], [d2f_dx1x2, d2f_dx2x2]])
        return f, grad, hess
    return f, grad, None

# HW2 Constrained Problems

# QP Problem: min x² + y² + (z+1)²
# subject to: x + y + z = 1, x,y,z ≥ 0

def qp_objective(x, hessian=False):
    """Objective: x² + y² + (z+1)²"""
    x = np.array(x, dtype=float)
    f = x[0]**2 + x[1]**2 + (x[2] + 1)**2
    grad = np.array([2*x[0], 2*x[1], 2*(x[2] + 1)])
    if hessian:
        return f, grad, 2 * np.eye(3)
    return f, grad, None

def qp_ineq_x(x, hessian=False):
    """Constraint: x ≥ 0"""
    x = np.array(x, dtype=float)
    val = x[0]
    grad = np.array([1, 0, 0])
    if hessian:
        return val, grad, np.zeros((3, 3))
    return val, grad, None

def qp_ineq_y(x, hessian=False):
    """Constraint: y ≥ 0"""
    x = np.array(x, dtype=float)
    val = x[1]
    grad = np.array([0, 1, 0])
    if hessian:
        return val, grad, np.zeros((3, 3))
    return val, grad, None

def qp_ineq_z(x, hessian=False):
    """Constraint: z ≥ 0"""
    x = np.array(x, dtype=float)
    val = x[2]
    grad = np.array([0, 0, 1])
    if hessian:
        return val, grad, np.zeros((3, 3))
    return val, grad, None

# QP problem data
qp_ineq_constraints = [qp_ineq_x, qp_ineq_y, qp_ineq_z]
qp_eq_matrix = np.array([[1, 1, 1]])
qp_eq_rhs = np.array([1])

# LP Problem: max x + y
# subject to: y ≥ -x + 1, y ≤ 1, x ≤ 2, y ≥ 0

def lp_objective(x, hessian=False):
    """Objective: maximize x + y (minimize -(x + y))"""
    x = np.array(x, dtype=float)
    f = -(x[0] + x[1])
    grad = np.array([-1, -1])
    if hessian:
        return f, grad, np.zeros((2, 2))
    return f, grad, None

def lp_ineq_1(x, hessian=False):
    """Constraint: y ≥ -x + 1 → x + y - 1 ≥ 0"""
    x = np.array(x, dtype=float)
    val = x[0] + x[1] - 1
    grad = np.array([1, 1])
    if hessian:
        return val, grad, np.zeros((2, 2))
    return val, grad, None

def lp_ineq_2(x, hessian=False):
    """Constraint: y ≤ 1 → 1 - y ≥ 0"""
    x = np.array(x, dtype=float)
    val = 1 - x[1]
    grad = np.array([0, -1])
    if hessian:
        return val, grad, np.zeros((2, 2))
    return val, grad, None

def lp_ineq_3(x, hessian=False):
    """Constraint: x ≤ 2 → 2 - x ≥ 0"""
    x = np.array(x, dtype=float)
    val = 2 - x[0]
    grad = np.array([-1, 0])
    if hessian:
        return val, grad, np.zeros((2, 2))
    return val, grad, None

def lp_ineq_4(x, hessian=False):
    """Constraint: y ≥ 0"""
    x = np.array(x, dtype=float)
    val = x[1]
    grad = np.array([0, 1])
    if hessian:
        return val, grad, np.zeros((2, 2))
    return val, grad, None

# LP problem data
lp_ineq_constraints = [lp_ineq_1, lp_ineq_2, lp_ineq_3, lp_ineq_4]