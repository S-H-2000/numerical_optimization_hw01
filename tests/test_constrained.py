import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tests'))

from src.constrained_min import interior_pt
import examples

class TestConstrainedMin(unittest.TestCase):
    
    def test_qp(self):
        """Test QP: min x² + y² + (z+1)² s.t. x+y+z=1, x,y,z≥0"""
        print("\n" + "="*60)
        print("TESTING QUADRATIC PROGRAMMING PROBLEM")
        print("="*60)
        print("minimize x² + y² + (z+1)²")
        print("subject to: x + y + z = 1, x ≥ 0, y ≥ 0, z ≥ 0")
        print("initial point: (0.1, 0.2, 0.7)")
        
        x0 = np.array([0.1, 0.2, 0.7])
        
        # Verify initial feasibility
        print(f"Initial feasibility check:")
        print(f"  Sum constraint: {np.sum(x0)} (should be 1)")
        print(f"  Non-negativity: {x0} (all should be > 0)")
        
        # Run interior point method
        result, obj_values, central_path = interior_pt(
            examples.qp_objective, 
            examples.qp_ineq_constraints, 
            examples.qp_eq_matrix, 
            examples.qp_eq_rhs, 
            x0
        )
        
        # Print results
        print(f"\nFINAL RESULTS:")
        print(f"Solution: x={result[0]:.6f}, y={result[1]:.6f}, z={result[2]:.6f}")
        
        final_obj, _, _ = examples.qp_objective(result, hessian=False)
        print(f"Objective value: {final_obj:.6f}")
        
        print(f"Constraint verification:")
        print(f"  x+y+z = {np.sum(result):.6f} (should be 1)")
        for i, constraint in enumerate(examples.qp_ineq_constraints):
            val, _, _ = constraint(result, hessian=False)
            print(f"  Constraint {i+1}: {val:.6f} (should be ≥ 0)")
        
        # Create plots
        self.plot_qp_results(result, central_path, obj_values)
        
        # Test assertions
        self.assertAlmostEqual(np.sum(result), 1.0, places=4)
        self.assertTrue(np.all(result >= -1e-4))
        
    def test_lp(self):
        """Test LP: max x + y subject to polygon constraints"""
        print("\n" + "="*60)
        print("TESTING LINEAR PROGRAMMING PROBLEM")
        print("="*60)
        print("maximize x + y")
        print("subject to: y ≥ -x + 1, y ≤ 1, x ≤ 2, y ≥ 0")
        print("initial point: (0.5, 0.75)")
        
        x0 = np.array([0.5, 0.75])
        
        # Verify initial feasibility
        print(f"Initial feasibility check:")
        for i, constraint in enumerate(examples.lp_ineq_constraints):
            val, _, _ = constraint(x0, hessian=False)
            print(f"  Constraint {i+1}: {val:.6f} (should be ≥ 0)")
        
        # Run interior point method
        result, obj_values, central_path = interior_pt(
            examples.lp_objective, 
            examples.lp_ineq_constraints, 
            None, None, x0
        )
        
        # Print results
        print(f"\nFINAL RESULTS:")
        print(f"Solution: x={result[0]:.6f}, y={result[1]:.6f}")
        
        final_obj, _, _ = examples.lp_objective(result, hessian=False)
        print(f"Objective value (maximized): {-final_obj:.6f}")
        
        print(f"Constraint verification:")
        for i, constraint in enumerate(examples.lp_ineq_constraints):
            val, _, _ = constraint(result, hessian=False)
            print(f"  Constraint {i+1}: {val:.6f} (should be ≥ 0)")
        
        # Create plots
        self.plot_lp_results(result, central_path, obj_values)
        
        # Test assertions
        for constraint in examples.lp_ineq_constraints:
            val, _, _ = constraint(result, hessian=False)
            self.assertGreaterEqual(val, -1e-4)
    
    def plot_qp_results(self, result, central_path, obj_values):
        """Plot QP results: 3D feasible region and convergence"""
        fig = plt.figure(figsize=(15, 6))
        
        # 3D plot of simplex and central path
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Draw simplex (triangle) vertices
        vertices = np.array([[1,0,0], [0,1,0], [0,0,1]])
        
        # Draw simplex edges
        edges = [[0,1], [1,2], [2,0]]
        for edge in edges:
            points = vertices[edge]
            ax1.plot(points[:,0], points[:,1], points[:,2], 'b-', linewidth=2)
        
        # Draw simplex face
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        triangle = [vertices]
        ax1.add_collection3d(Poly3DCollection(triangle, alpha=0.3, facecolor='lightblue'))
        
        # Plot central path
        if central_path:
            path = np.array(central_path)
            ax1.plot(path[:,0], path[:,1], path[:,2], 'ro-', linewidth=2, markersize=4, label='Central Path')
        
        # Plot final solution and target
        ax1.scatter(*result, color='red', s=100, marker='*', label='Solution')
        ax1.scatter(0, 0, -1, color='green', s=100, marker='s', label='Target (0,0,-1)')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.set_title('QP: Feasible Region & Central Path')
        ax1.legend()
        
        # Convergence plot
        ax2 = fig.add_subplot(122)
        ax2.plot(obj_values, 'b-o', linewidth=2, markersize=4)
        ax2.set_xlabel('Outer Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.set_title('QP: Convergence')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_lp_results(self, result, central_path, obj_values):
        """Plot LP results: 2D feasible region and convergence"""
        fig = plt.figure(figsize=(15, 6))
        
        # 2D feasible region
        ax1 = fig.add_subplot(121)
        
        # Draw constraint lines
        x_vals = np.linspace(-0.5, 2.5, 100)
        ax1.plot(x_vals, -x_vals + 1, 'g-', linewidth=2, label='y = -x + 1')
        ax1.axhline(y=1, color='r', linewidth=2, label='y = 1')
        ax1.axhline(y=0, color='b', linewidth=2, label='y = 0')
        ax1.axvline(x=2, color='m', linewidth=2, label='x = 2')
        
        # Fill feasible region
        vertices = np.array([[0,0], [1,0], [2,0], [2,1], [0,1]])
        from matplotlib.patches import Polygon
        polygon = Polygon(vertices, alpha=0.3, facecolor='lightgray')
        ax1.add_patch(polygon)
        
        # Plot central path
        if central_path:
            path = np.array(central_path)
            ax1.plot(path[:,0], path[:,1], 'ro-', linewidth=2, markersize=4, label='Central Path')
        
        # Plot solution and optimal corner
        ax1.scatter(*result, color='red', s=100, marker='*', label='Solution')
        ax1.scatter(2, 1, color='green', s=100, marker='s', label='Optimal (2,1)')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('LP: Feasible Region & Central Path')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlim(-0.5, 2.5)
        ax1.set_ylim(-0.5, 1.5)
        
        # Convergence plot (convert to maximization)
        ax2 = fig.add_subplot(122)
        max_values = [-val for val in obj_values]
        ax2.plot(max_values, 'b-o', linewidth=2, markersize=4)
        ax2.set_xlabel('Outer Iteration')
        ax2.set_ylabel('Objective Value (x + y)')
        ax2.set_title('LP: Convergence')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    unittest.main()