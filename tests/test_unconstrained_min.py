import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.unconstrained_min import minimize, get_optimization_history
from src.utils import plot_contours, plot_function_values
from tests import examples

class TestMinimizers(unittest.TestCase):

    def run_and_plot_optimization(self, func, func_name, x0, xlims, ylims, max_iter_gd=100, max_iter_newton=100):
        """
        Helper function to run both GD and Newton and create plots.
        """
        print(f"\n=== Testing {func_name} ===")
        
        # Run Gradient Descent
        print("Running Gradient Descent...")
        x_gd, f_gd, success_gd = minimize(
            func, x0, method="gd", max_iter=max_iter_gd
        )
        history_gd = get_optimization_history()
        
        # Run Newton's Method
        print("Running Newton's Method...")
        x_newton, f_newton, success_newton = minimize(
            func, x0, method="newton", max_iter=max_iter_newton
        )
        history_newton = get_optimization_history()
        
        # Print results
        print(f"\nGradient Descent:")
        print(f"  Final point: {x_gd}")
        print(f"  Final value: {f_gd:.6e}")
        print(f"  Success: {success_gd}")
        print(f"  Iterations: {len(history_gd)-1}")
        
        print(f"\nNewton's Method:")
        print(f"  Final point: {x_newton}")
        print(f"  Final value: {f_newton:.6e}")
        print(f"  Success: {success_newton}")
        print(f"  Iterations: {len(history_newton)-1}")
        
        # Create contour plot with both paths
        plt.figure(figsize=(12, 10))
        
        # Plot function contours
        x = np.linspace(xlims[0], xlims[1], 100)
        y = np.linspace(ylims[0], ylims[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                Z[i, j], _, _ = func(point, hessian=False)
        
        # Create appropriate contour levels based on function characteristics
        z_min, z_max = np.min(Z), np.max(Z)
        
        # Use linear spacing for linear functions or when values include negatives
        if z_min <= 0 or (z_max - z_min) < 1e-10:
            contour_levels = np.linspace(z_min, z_max, 20)
        else:
            # Use logarithmic spacing for positive functions with good range
            contour_levels = np.logspace(np.log10(z_min + 1e-10), np.log10(z_max + 1), 20)
        
        plt.contour(X, Y, Z, levels=contour_levels, colors='gray', alpha=0.6)
        plt.contourf(X, Y, Z, levels=contour_levels, alpha=0.3, cmap='viridis')
        
        # Plot optimization paths
        if len(history_gd) > 1:
            path_gd = np.array(history_gd)
            plt.plot(path_gd[:, 0], path_gd[:, 1], 'ro-', linewidth=2, markersize=4, 
                    label=f'GD ({len(history_gd)-1} iter)', alpha=0.7)
        
        if len(history_newton) > 1:
            path_newton = np.array(history_newton)
            plt.plot(path_newton[:, 0], path_newton[:, 1], 'bs-', linewidth=2, markersize=4, 
                    label=f'Newton ({len(history_newton)-1} iter)', alpha=0.7)
        
        # Mark start and end points
        plt.plot(x0[0], x0[1], 'go', markersize=10, label='Start')
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'{func_name} - Optimization Paths')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.colorbar(label='Function Value')
        plt.tight_layout()
        plt.show()
        
        # Plot function values over iterations
        plt.figure(figsize=(12, 5))
        
        # GD function values
        if len(history_gd) > 1:
            values_gd = [func(x, hessian=False)[0] for x in history_gd]
            plt.subplot(1, 2, 1)
            plt.plot(range(len(values_gd)), values_gd, 'r-o', linewidth=2, markersize=4)
            plt.xlabel('Iteration')
            plt.ylabel('Function Value')
            plt.title(f'{func_name} - GD Convergence')
            plt.grid(True, alpha=0.3)
            if np.all(np.array(values_gd) > 0):
                plt.yscale('log')
        
        # Newton function values  
        if len(history_newton) > 1:
            values_newton = [func(x, hessian=False)[0] for x in history_newton]
            plt.subplot(1, 2, 2)
            plt.plot(range(len(values_newton)), values_newton, 'b-s', linewidth=2, markersize=4)
            plt.xlabel('Iteration')
            plt.ylabel('Function Value')
            plt.title(f'{func_name} - Newton Convergence')
            plt.grid(True, alpha=0.3)
            if np.all(np.array(values_newton) > 0):
                plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        return (x_gd, f_gd, success_gd, history_gd), (x_newton, f_newton, success_newton, history_newton)

    def test_quadratic_circle(self):
        """
        Run both GD and Newton on the circular quadratic function.
        """
        x0 = np.array([1.0, 1.0])
        xlims, ylims = [-2, 2], [-2, 2]
        self.run_and_plot_optimization(
            examples.quadratic_circle, "Quadratic Circle", x0, xlims, ylims
        )

    def test_quadratic_ellipse(self):
        """
        Test on the elliptical quadratic function.
        """
        x0 = np.array([1.0, 1.0])
        xlims, ylims = [-2, 2], [-0.5, 0.5]
        self.run_and_plot_optimization(
            examples.quadratic_ellipse, "Quadratic Ellipse", x0, xlims, ylims
        )

    def test_quadratic_rotated(self):
        """
        Test on the rotated elliptical quadratic function.
        """
        x0 = np.array([1.0, 1.0])
        xlims, ylims = [-2, 2], [-2, 2]
        self.run_and_plot_optimization(
            examples.quadratic_rotated, "Quadratic Rotated", x0, xlims, ylims
        )

    def test_rosenbrock(self):
        """
        Test on the Rosenbrock function with high iteration count for GD.
        """
        x0 = np.array([-1.0, 2.0])
        xlims, ylims = [-2, 2], [-1, 3]
        self.run_and_plot_optimization(
            examples.rosenbrock, "Rosenbrock", x0, xlims, ylims, 
            max_iter_gd=10000, max_iter_newton=100
        )

    def test_linear_func(self):
        """
        Test on the linear function.
        """
        x0 = np.array([1.0, 1.0])
        xlims, ylims = [-2, 2], [-2, 2]
        self.run_and_plot_optimization(
            examples.linear_func, "Linear Function", x0, xlims, ylims
        )

    def test_triangle_exp(self):
        """
        Test on the exponential triangle function.
        """
        x0 = np.array([1.0, 1.0])
        xlims, ylims = [-2, 2], [-2, 2]
        self.run_and_plot_optimization(
            examples.triangle_exp, "Triangle Exponential", x0, xlims, ylims
        )

if __name__ == "__main__":
    unittest.main()
