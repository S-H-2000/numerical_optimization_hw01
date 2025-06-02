import numpy as np
# This list will keep track of all the points we visit during optimization
_optimization_history = []

def minimize(f_func, x0, method="gd", obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    """
    Performs unconstrained minimization using Gradient Descent or Newton's Method.
    
    Args:
        f_func: objective function that returns (f, grad, hess)
        x0: initial guess (numpy array)
        method: "gd" for gradient descent, "newton" for Newton's method
        obj_tol: tolerance for objective function change
        param_tol: tolerance for parameter change
        max_iter: maximum number of iterations
        
    Returns:
        x_final: final point
        f_final: final function value
        success_flag: True/False convergence flag
    """
    # Make sure we're working with floating point numbers
    x = np.array(x0, dtype=float)
    
    global _optimization_history
    # Start our history with the initial point
    _optimization_history = [x.copy()]
    
    # Get our starting point's function value
    f_prev, _, _ = f_func(x, hessian=False)
    
    # Main optimization loop - try up to max_iter times
    for i in range(max_iter):
        # Figure out if we need the Hessian (only for Newton's method)
        need_hessian = (method == "newton")
        # Get function value, gradient, and maybe Hessian at current point
        f, grad, hess = f_func(x, hessian=need_hessian)
        
        # Let's see how we're doing
        print(f"Iteration {i}: x = {x}, f(x) = {f:.6e}")
        
        # If gradient is tiny, we're probably at a minimum
        if np.linalg.norm(grad) < param_tol:
            return x, f, True
        
        # Choose our next direction based on the method
        if method == "gd":
            # For gradient descent, just go opposite to the gradient
            direction = -grad
        elif method == "newton":
            try:
                # For Newton, solve H * direction = -grad to get better direction
                direction = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                # If Hessian is bad, fall back to gradient descent
                direction = -grad
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Find how far to step in our chosen direction
        alpha = line_search(f_func, x, direction)
        
        # Take the step
        x_new = x + alpha * direction
        # Remember where we went
        _optimization_history.append(x_new.copy())
        
        # Check if we've converged
        f_new, _, _ = f_func(x_new, hessian=False)
        
        # If function value barely changed, we're done
        if abs(f_new - f_prev) < obj_tol:
            return x_new, f_new, True
        
        # If position barely changed, we're done
        if np.linalg.norm(x_new - x) < param_tol:
            return x_new, f_new, True
        
        # Get ready for next iteration
        x = x_new
        f_prev = f_new
    
    # If we get here, we hit max iterations without converging
    return x, f_prev, False

def get_optimization_history():
    """Returns the history from the last optimization run."""
    return _optimization_history.copy()

def line_search(f_func, x, direction, alpha=1.0, beta=0.5, c=0.01):
    """
    Performs backtracking line search satisfying the Wolfe condition.
    
    Args:
        f_func: objective function
        x: current position
        direction: search direction
        alpha: initial step size
        beta: backtracking factor (0 < beta < 1)
        c: Wolfe condition parameter (0 < c < 1)
        
    Returns:
        alpha: final step size
    """
    # Get starting point info.
    f0, grad0, _ = f_func(x, hessian=False)
    
    # How fast function decreases in our direction
    directional_deriv = grad0.T @ direction
    
    # If we're not going downhill, take tiny step
    if directional_deriv >= 0:
        return 1e-8
    
    # Try to find a good step size
    while alpha > 1e-10:  # Don't let step get too tiny
        # Try this step
        x_new = x + alpha * direction
        f_new, _, _ = f_func(x_new, hessian=False)
        
        # Check if this step is good enough
        if f_new <= f0 + c * alpha * directional_deriv:
            return alpha
        
        # If not good enough, try smaller step
        alpha *= beta
    
    return alpha
