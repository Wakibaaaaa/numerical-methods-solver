"""
algorithms.py
--------------
Core numerical methods engine: Fixed-Point Iteration and Newton-Raphson.
Uses SymPy for symbolic parsing and automatic differentiation.
"""

import sympy as sp
import time

x = sp.Symbol('x')


class ConvergenceError(Exception):
    """Raised when a method fails to converge or hits a mathematical problem."""
    pass


def parse_expression(expr_str):
    """
    Convert a string like 'cos(x) - x*exp(x)' or 'x^2 - 2 = 0'
    into a SymPy expression.
    Raises ConvergenceError with a friendly message if the input is invalid.
    """
    if not expr_str or not expr_str.strip():
        raise ConvergenceError("The equation field is empty. Please enter a function of x.")

    cleaned = expr_str.strip()

    # Allow "something = something" by moving everything to one side.
    # Example: "x^2 - 2 = 0"  ->  "(x^2 - 2) - (0)"
    if "=" in cleaned:
        parts = cleaned.split("=")
        if len(parts) != 2:
            raise ConvergenceError("Please use at most one '=' sign in your equation.")
        lhs, rhs = parts[0].strip(), parts[1].strip()
        cleaned = f"({lhs}) - ({rhs})"

    try:
        transformations = sp.parsing.sympy_parser.standard_transformations + (
            sp.parsing.sympy_parser.implicit_multiplication_application,
            sp.parsing.sympy_parser.convert_xor,
        )
        expr = sp.parsing.sympy_parser.parse_expr(cleaned, transformations=transformations)
    except Exception:
        raise ConvergenceError(
            "We couldn't understand that equation. Please check the syntax "
            "(example: x^2 - 2, or cos(x) - x*exp(x))."
        )

    if x not in expr.free_symbols and len(expr.free_symbols) > 0:
        raise ConvergenceError("Please use 'x' as the only variable in your equation.")

    return expr


def auto_convert_to_phi(f_expr):
    """
    Try to automatically rearrange f(x) = 0 into x = phi(x).
    Returns the phi expression if successful, or None if it isn't possible
    automatically (caller should then ask the user for phi(x) directly).
    """
    try:
        solutions = sp.solve(sp.Eq(f_expr, 0), x)
        # We only accept it if SymPy found a clean explicit expression for x
        # that still depends on x (i.e. a genuine phi(x), not a numeric root).
        for sol in solutions:
            if x in sol.free_symbols:
                return sp.simplify(sol)
    except Exception:
        pass
    return None


def evaluate(expr, value, trig_mode="radian"):
    """
    Numerically evaluate a SymPy expression at a given x value.
    Handles Degree mode by converting the input to radians for trig functions.
    """
    try:
        v = value
        if trig_mode == "degree":
            v = value * sp.pi / 180
            # substitute only inside trig-affecting evaluation;
            # simplest safe approach: substitute x with (value in radians)
        result = expr.subs(x, v).evalf()
        if result.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            raise ConvergenceError("The function is undefined at this point (division by zero or infinity).")
        return float(result)
    except ConvergenceError:
        raise
    except Exception:
        raise ConvergenceError(f"Could not evaluate the function at x = {value}.")


def round_to_digits(value, digits):
    """Round a float to a given number of significant/decimal digits."""
    return round(value, digits)


def fixed_point_iteration(phi_expr, x0, tol, max_iter, print_digits,
                           correct_digit, use_rounded, trig_mode="radian"):
    """
    Run Fixed-Point Iteration: x_{n+1} = phi(x_n)
    Returns a list of iteration records (dicts) and a summary dict.
    """
    start_time = time.time()
    iterations = []
    xn = x0
    stopping_tol = 0.5 * 10 ** (-correct_digit)

    for i in range(1, max_iter + 1):
        try:
            phi_val = evaluate(phi_expr, xn, trig_mode)
        except ConvergenceError as e:
            raise ConvergenceError(f"Iteration {i}: {str(e)}")

        new_x = phi_val
        diff = abs(new_x - xn)

        record = {
            "iteration": i,
            "x_n": round_to_digits(xn, print_digits),
            "phi_x_n": round_to_digits(phi_val, print_digits),
            "new_x": round_to_digits(new_x, print_digits),
            "difference": round_to_digits(diff, print_digits),
        }
        iterations.append(record)

        if diff < stopping_tol or diff < tol:
            elapsed = time.time() - start_time
            return iterations, {
                "root": round_to_digits(new_x, print_digits),
                "iterations": i,
                "absolute_error": round_to_digits(diff, print_digits),
                "relative_error": round_to_digits(diff / abs(new_x) if new_x != 0 else diff, print_digits),
                "execution_time": round(elapsed, 6),
                "status": "Converged"
            }

        # This is the key setting from the spec: rounded value feeds the next iteration
        xn = round_to_digits(new_x, print_digits) if use_rounded else new_x

        if abs(xn) > 1e15:
            raise ConvergenceError(f"The sequence is diverging (values growing without bound) at iteration {i}.")

    raise ConvergenceError(
        f"Did not converge within {max_iter} iterations. "
        "Try a different initial guess, or increase Maximum Iterations."
    )


def newton_raphson(f_expr, x0, tol, max_iter, print_digits,
                    correct_digit, use_rounded, trig_mode="radian"):
    """
    Run Newton-Raphson: x_{n+1} = x_n - f(x_n)/f'(x_n)
    Returns a list of iteration records (dicts) and a summary dict.
    """
    start_time = time.time()
    f_prime = sp.diff(f_expr, x)
    iterations = []
    xn = x0
    stopping_tol = 0.5 * 10 ** (-correct_digit)

    for i in range(1, max_iter + 1):
        try:
            fx = evaluate(f_expr, xn, trig_mode)
            fpx = evaluate(f_prime, xn, trig_mode)
        except ConvergenceError as e:
            raise ConvergenceError(f"Iteration {i}: {str(e)}")

        if abs(fpx) < 1e-14:
            raise ConvergenceError(
                f"Iteration {i}: The derivative is zero (f'(x) = 0) at x = {xn}. "
                "Newton-Raphson cannot continue from here — try a different initial guess."
            )

        new_x = xn - fx / fpx
        diff = abs(new_x - xn)

        record = {
            "iteration": i,
            "x_n": round_to_digits(xn, print_digits),
            "f_x_n": round_to_digits(fx, print_digits),
            "f_prime_x_n": round_to_digits(fpx, print_digits),
            "new_x": round_to_digits(new_x, print_digits),
            "difference": round_to_digits(diff, print_digits),
        }
        iterations.append(record)

        if diff < stopping_tol or diff < tol:
            elapsed = time.time() - start_time
            return iterations, {
                "root": round_to_digits(new_x, print_digits),
                "iterations": i,
                "absolute_error": round_to_digits(diff, print_digits),
                "relative_error": round_to_digits(diff / abs(new_x) if new_x != 0 else diff, print_digits),
                "execution_time": round(elapsed, 6),
                "status": "Converged"
            }

        xn = round_to_digits(new_x, print_digits) if use_rounded else new_x

        if abs(xn) > 1e15:
            raise ConvergenceError(f"The sequence is diverging (values growing without bound) at iteration {i}.")

    raise ConvergenceError(
        f"Did not converge within {max_iter} iterations. "
        "Try a different initial guess, or increase Maximum Iterations."
    )