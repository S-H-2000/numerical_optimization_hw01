import numpy as np
# Fix import for your HW1 unconstrained minimizer
import sys
import os
sys.path.append(os.path.dirname(__file__))
from unconstrained_min import minimize

def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    """
    Interior Point Method using Log-Barrier for constrained optimization.
    
    Args:
        func: objective function that returns (f, grad, hess)
        ineq_constraints: list of inequality constraint functions g_i(x) >= 0
        eq_constraints_mat: matrix A for equality constraints Ax = b (can be None)
        eq_constraints_rhs: vector b for equality constraints Ax = b (can be None)
        x0: initial interior point (must be strictly feasible)
        
    Returns:
        x_final: final solution point
        obj_values: list of objective values at each outer iteration
        central_path: list of points along the central path
    """
    
    # Interior point parameters (as specified by assignment)
    t = 1.0          # Initial barrier parameter
    mu = 10.0        # Multiplication factor
    tolerance = 1e-6 # Convergence tolerance
    max_outer_iter = 50
    
    # Storage for results
    obj_values = []
    central_path = []
    
    x = np.array(x0, dtype=float)
    
    print(f"Starting Interior Point Method with t={t}, mu={mu}")
    print(f"Initial point: {x}")
    
    for outer_iter in range(max_outer_iter):
        print(f"\n=== Outer Iteration {outer_iter + 1} (t={t:.2e}) ===")
        
        # Create barrier function for this iteration
        def barrier_function(x_inner, hessian=False):
            """
            Barrier function: t * f(x) + sum(-log(g_i(x)))
            """
            x_inner = np.array(x_inner, dtype=float)
            
            # Get objective function value/grad/hess
            f_val, f_grad, f_hess = func(x_inner, hessian=hessian)
            
            # Initialize barrier terms
            barrier_val = 0.0
            barrier_grad = np.zeros_like(x_inner)
            if hessian:
                barrier_hess = np.zeros((len(x_inner), len(x_inner)))
            else:
                barrier_hess = None
            
            # Add log-barrier terms for inequality constraints
            for i, constraint in enumerate(ineq_constraints):
                g_val, g_grad, g_hess = constraint(x_inner, hessian=hessian)
                
                # Check if constraint is violated
                if g_val <= 1e-10:
                    # Return large penalty to push back to interior
                    large_penalty = 1e10
                    penalty_grad = -1e8 * g_grad / (abs(g_val) + 1e-12)
                    if hessian:
                        penalty_hess = 1e6 * np.eye(len(x_inner))
                        return large_penalty, penalty_grad, penalty_hess
                    else:
                        return large_penalty, penalty_grad, None
                
                # Add log barrier: -log(g_i(x))
                barrier_val += -np.log(g_val)
                barrier_grad += -g_grad / g_val
                
                if hessian and g_hess is not None:
                    # Hessian of -log(g(x)) = g'g'^T / g^2 - g'' / g
                    barrier_hess += np.outer(g_grad, g_grad) / (g_val**2) - g_hess / g_val
            
            # Combine: t * f(x) + barrier_terms
            total_val = t * f_val + barrier_val
            total_grad = t * f_grad + barrier_grad
            
            if hessian:
                total_hess = t * f_hess + barrier_hess
                return total_val, total_grad, total_hess
            else:
                return total_val, total_grad, None
        
        # Handle equality constraints by projecting onto feasible manifold
        if eq_constraints_mat is not None and eq_constraints_rhs is not None:
            # Project current point to satisfy equality constraints
            x = project_to_equality_constraints(x, eq_constraints_mat, eq_constraints_rhs)
            
            # Use custom Newton method for equality constrained problems
            x_new = solve_equality_constrained_newton(
                barrier_function, eq_constraints_mat, eq_constraints_rhs, x
            )
        else:
            # No equality constraints - use regular unconstrained optimization
            x_new, _, success = minimize(
                barrier_function, x, method="newton", 
                obj_tol=1e-8, param_tol=1e-8, max_iter=100
            )
            
            if not success:
                print(f"Warning: Inner optimization did not converge at iteration {outer_iter + 1}")
        
        # Evaluate true objective at new point
        true_obj_val, _, _ = func(x_new, hessian=False)
        obj_values.append(true_obj_val)
        central_path.append(x_new.copy())
        
        print(f"New point: {x_new}")
        print(f"True objective value: {true_obj_val:.6e}")
        
        # Check convergence: m/t < tolerance where m = number of inequalities
        m = len(ineq_constraints)
        duality_gap = m / t
        print(f"Duality gap estimate: {duality_gap:.6e}")
        
        if duality_gap < tolerance:
            print(f"Converged! Duality gap {duality_gap:.6e} < {tolerance}")
            break
        
        # Update for next iteration
        t *= mu
        x = x_new
    
    return x, obj_values, central_path

def project_to_equality_constraints(x, A, b):
    """Project point x onto equality constraint manifold Ax = b"""
    if A is None or b is None:
        return x
    
    A = np.array(A)
    b = np.array(b)
    
    # Solve: x_proj = x + A^T * lambda where A * lambda = (b - A*x)
    residual = b - A @ x
    
    try:
        # lambda = (A * A^T)^{-1} * residual
        lambda_val = np.linalg.solve(A @ A.T, residual)
        correction = A.T @ lambda_val
        return x + correction
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if A*A^T is singular
        lambda_val = np.linalg.pinv(A @ A.T) @ residual
        correction = A.T @ lambda_val
        return x + correction

def solve_equality_constrained_newton(func, A, b, x0, max_iter=50, tol=1e-8):
    """Newton's method for equality constrained optimization using KKT system"""
    x = np.array(x0, dtype=float)
    
    for i in range(max_iter):
        # Get function derivatives
        f_val, f_grad, f_hess = func(x, hessian=True)
        
        # Project to feasible manifold
        x = project_to_equality_constraints(x, A, b)
        
        n = len(x)
        A_mat = np.array(A) if A is not None else np.zeros((0, n))
        b_vec = np.array(b) if b is not None else np.zeros(0)
        m = len(b_vec)
        
        if m > 0:
            # Solve KKT system: [H A^T; A 0] [dx; dlambda] = [-g; 0]
            KKT_matrix = np.block([
                [f_hess, A_mat.T],
                [A_mat, np.zeros((m, m))]
            ])
            rhs = np.concatenate([-f_grad, np.zeros(m)])
            
            try:
                solution = np.linalg.solve(KKT_matrix, rhs)
                dx = solution[:n]
            except np.linalg.LinAlgError:
                # Fallback: just use projected gradient
                dx = -f_grad
                # Project gradient to null space of A
                if A is not None:
                    dx = dx - A.T @ np.linalg.pinv(A @ A.T) @ (A @ dx)
        else:
            # No equality constraints - regular Newton step
            try:
                dx = np.linalg.solve(f_hess, -f_grad)
            except np.linalg.LinAlgError:
                dx = -f_grad
        
        # Line search with projection
        alpha = 1.0
        for _ in range(20):
            x_new = x + alpha * dx
            # Project back to feasible region
            x_new = project_to_equality_constraints(x_new, A, b)
            
            try:
                f_new, _, _ = func(x_new, hessian=False)
                if f_new < f_val:
                    x = x_new
                    break
            except:
                pass
            alpha *= 0.5
        else:
            # Line search failed, take projected gradient step
            x = x - 0.01 * f_grad
            x = project_to_equality_constraints(x, A, b)
        
        # Check convergence
        if np.linalg.norm(dx) < tol:
            break
    
    return x