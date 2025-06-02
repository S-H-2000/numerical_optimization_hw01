import matplotlib.pyplot as plt
import numpy as np

# Plot 2D contour lines of a function and optionally show the optimization path
def plot_contours(f_func, xlims, ylims, path=None, title="Contour Plot"):
    """
    Draws contour lines of a function and overlays the optimization path.
    
    Args:
        f_func: objective function
        xlims: [xmin, xmax] for plotting range
        ylims: [ymin, ymax] for plotting range
        path: list of points [(x1,y1), (x2,y2), ...] or None
        title: plot title
    """
    # Create a grid of points to evaluate the function
    x = np.linspace(xlims[0], xlims[1], 100)
    y = np.linspace(ylims[0], ylims[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate function value at each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j], _, _ = f_func(point, hessian=False)
    
    # Create a new figure with nice size
    plt.figure(figsize=(10, 8))
    
    # Draw contour lines and fill between them
    contour_levels = np.logspace(np.log10(np.min(Z) + 1e-10), np.log10(np.max(Z) + 1), 20)
    plt.contour(X, Y, Z, levels=contour_levels, colors='blue', alpha=0.6)
    plt.contourf(X, Y, Z, levels=contour_levels, alpha=0.3, cmap='viridis')
    
    # If we have an optimization path, plot it
    if path is not None and len(path) > 0:
        path = np.array(path)
        # Plot the path as red dots connected by lines
        plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=6, 
                label='Optimization Path', alpha=0.8)
        # Mark start point in green
        plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
        # Mark end point in red square
        plt.plot(path[-1, 0], path[-1, 1], 'rs', markersize=10, label='End')
    
    # Add labels and title
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.colorbar(label='Function Value')

# Plot how the objective value changes with each iteration
def plot_function_values(values, title="Function Value vs Iteration"):
    """
    Plots the function value at each iteration.
    
    Args:
        values: list or array of function values
        title: plot title
    """
    # Create a new figure
    plt.figure(figsize=(10, 6))
    # Plot values as blue dots connected by lines
    plt.plot(range(len(values)), values, 'b-o', linewidth=2, markersize=4)
    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    #plt.yscale('log')  # Log scale for better visualization of convergence
    plt.tight_layout()
