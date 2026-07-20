"""
graph.py
--------
Plotting functions for the Numerical Methods Solver.
Generates: function graph (with root & initial guess), error-vs-iteration graph,
and an optional convergence graph.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.Symbol('x')


def plot_function(expr, root, x0, x_range=None):
    """
    Plot y = f(x) (or phi(x)) over a sensible range, marking the root and
    the initial guess. Returns a matplotlib Figure.
    """
    if x_range is None:
        center = root if root is not None else x0
        x_range = (center - 10, center + 10)

    xs = np.linspace(x_range[0], x_range[1], 800)
    f_lambdified = sp.lambdify(x, expr, modules=["numpy"])

    ys = []
    for val in xs:
        try:
            y = f_lambdified(val)
            # Guard against complex or invalid results
            if isinstance(y, complex):
                ys.append(np.nan)
            else:
                ys.append(y)
        except Exception:
            ys.append(np.nan)
    ys = np.array(ys, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.plot(xs, ys, label="f(x)", color="#4C6EF5", linewidth=2)

    if root is not None:
        ax.scatter([root], [0], color="#E64980", zorder=5, label=f"Root ≈ {root}")
    if x0 is not None:
        try:
            y0 = f_lambdified(x0)
            ax.scatter([x0], [y0], color="#12B886", zorder=5, label=f"Initial Guess = {x0}")
        except Exception:
            pass

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Function Graph")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_error_vs_iteration(iterations):
    """
    Plot |difference| vs iteration number, on a log scale, to visualize convergence speed.
    'iterations' is the list of dicts returned by our algorithm functions.
    """
    iter_numbers = [rec["iteration"] for rec in iterations]
    diffs = [abs(rec["difference"]) if rec["difference"] != 0 else 1e-16 for rec in iterations]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(iter_numbers, diffs, marker="o", color="#F76707", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Absolute Difference (log scale)")
    ax.set_title("Error vs Iteration")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_convergence(iterations):
    """
    Plot x_n value at each iteration, to visualize how the sequence converges toward the root.
    """
    iter_numbers = [rec["iteration"] for rec in iterations]
    x_values = [rec["new_x"] for rec in iterations]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(iter_numbers, x_values, marker="o", color="#7048E8", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("x value")
    ax.set_title("Convergence Graph")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig