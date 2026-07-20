"""
utils.py
--------
Helper functions: input validation, number formatting, sign-change checking
(for "Root Between"), and export helpers (CSV/PDF).
"""

import io
import pandas as pd
import sympy as sp

x = sp.Symbol('x')


class InputError(Exception):
    """Raised when the user's input is invalid, with a friendly message."""
    pass


def validate_interval(f_expr, left, right, trig_mode="radian"):
    """
    For 'Root Between' mode: check that f(left) and f(right) have opposite
    signs (Intermediate Value Theorem) — a basic sanity check that a root
    likely exists in [left, right].
    Returns (f_left, f_right) if valid; raises InputError if not.
    """
    from algorithms import evaluate  # local import avoids circular import issues

    if left >= right:
        raise InputError("Left Boundary must be smaller than Right Boundary.")

    f_left = evaluate(f_expr, left, trig_mode)
    f_right = evaluate(f_expr, right, trig_mode)

    if f_left == 0:
        raise InputError(f"x = {left} is already an exact root of f(x).")
    if f_right == 0:
        raise InputError(f"x = {right} is already an exact root of f(x).")

    if f_left * f_right > 0:
        raise InputError(
            f"f({left}) = {f_left:.6g} and f({right}) = {f_right:.6g} have the same sign. "
            "This means the Intermediate Value Theorem cannot guarantee a root in this interval. "
            "Try a different interval."
        )

    return f_left, f_right


def validate_max_iterations(max_iter):
    if max_iter is None or max_iter <= 0:
        raise InputError("Maximum Iterations must be a positive whole number.")
    if max_iter > 10000:
        raise InputError("Maximum Iterations is too large (limit: 10000) — this protects against infinite loops.")


def validate_tolerance(tol):
    if tol is None or tol <= 0:
        raise InputError("Tolerance must be a positive number (e.g. 0.0001).")


def format_number(value, digits):
    """Format a float for display with a fixed number of digits."""
    try:
        return f"{value:.{digits}f}"
    except Exception:
        return str(value)


def iterations_to_dataframe(iterations, method):
    """
    Convert the list of iteration dicts into a pandas DataFrame with
    friendly column names, ready for display and CSV export.
    """
    df = pd.DataFrame(iterations)

    if method == "Fixed Point":
        df = df.rename(columns={
            "iteration": "Iteration",
            "x_n": "xₙ",
            "phi_x_n": "φ(xₙ)",
            "new_x": "Update (xₙ₊₁)",
            "difference": "Difference",
        })
    else:  # Newton-Raphson
        df = df.rename(columns={
            "iteration": "Iteration",
            "x_n": "xₙ",
            "f_x_n": "f(xₙ)",
            "f_prime_x_n": "f'(xₙ)",
            "new_x": "New x",
            "difference": "Difference",
        })

    return df


def dataframe_to_csv_bytes(df):
    """Convert a DataFrame into CSV bytes, ready for Streamlit's download button."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def generate_pdf_report(problem_text, method, summary, df):
    """
    Build a simple PDF report of the solution using matplotlib's PDF backend
    (avoids needing an extra heavyweight PDF library).
    Returns PDF bytes ready for Streamlit's download button.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Page 1: summary
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis("off")
        lines = [
            "Numerical Methods Solver — Report",
            "",
            f"Problem: {problem_text}",
            f"Method: {method}",
            "",
            "Results:",
            f"  Approximate Root: {summary['root']}",
            f"  Iterations: {summary['iterations']}",
            f"  Absolute Error: {summary['absolute_error']}",
            f"  Relative Error: {summary['relative_error']}",
            f"  Execution Time: {summary['execution_time']} s",
            f"  Status: {summary['status']}",
        ]
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: iteration table
        fig2, ax2 = plt.subplots(figsize=(8.27, 11.69))
        ax2.axis("off")
        table = ax2.table(
            cellText=df.values,
            colLabels=df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        pdf.savefig(fig2)
        plt.close(fig2)

    buffer.seek(0)
    return buffer.getvalue()