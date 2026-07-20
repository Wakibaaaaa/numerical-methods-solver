"""
app.py
------
Numerical Methods Solver — Streamlit web app.
Solves nonlinear equations using Fixed-Point Iteration and Newton-Raphson.
"""

import streamlit as st
import sympy as sp

from algorithms import (
    parse_expression, auto_convert_to_phi, evaluate,
    fixed_point_iteration, newton_raphson, ConvergenceError
)
from utils import (
    validate_interval, validate_max_iterations, validate_tolerance,
    iterations_to_dataframe, dataframe_to_csv_bytes, generate_pdf_report,
    InputError
)
from graph import plot_function, plot_error_vs_iteration, plot_convergence

x = sp.Symbol('x')

st.set_page_config(page_title="Numerical Methods Solver", layout="wide")

st.title("🧮 Numerical Methods Solver")
st.caption("Solve nonlinear equations using Fixed-Point Iteration and Newton-Raphson — with full step-by-step working.")

# ----------------------------------------------------------------
# SIDEBAR: all inputs
# ----------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    method = st.radio("Method", ["Fixed Point", "Newton-Raphson"])

    st.divider()

    if method == "Fixed Point":
        fp_input_type = st.radio("Fixed Point Input Type", ["x = φ(x)", "f(x)"])
        if fp_input_type == "x = φ(x)":
            phi_str = st.text_input("φ(x) =", value="cos(x)/exp(x)")
            f_str = None
        else:
            f_str = st.text_input("f(x) =", value="cos(x) - x*exp(x)")
            phi_str = None
    else:
        f_str = st.text_input("f(x) =", value="x**2 - 2")
        phi_str = None

    st.divider()

    find_mode = st.radio("Find", ["Any Root", "Root Between"])
    left_bound, right_bound = None, None
    if find_mode == "Root Between":
        left_bound = st.number_input("Left Boundary", value=0.0)
        right_bound = st.number_input("Right Boundary", value=2.0)

    st.divider()

    precision_mode = st.radio(
        "For next iteration calculation",
        ["Use Full Precision", "Use Rounded Value"]
    )
    use_rounded = (precision_mode == "Use Rounded Value")

    print_digits = st.selectbox("Print Digits", list(range(2, 11)), index=4)
    trig_mode_label = st.radio("Trigonometric Mode", ["Radian", "Degree"])
    trig_mode = trig_mode_label.lower()
    correct_digit = st.selectbox("Solution Correct Up To Digit", list(range(2, 11)), index=4)

    st.divider()

    tolerance = st.number_input("Tolerance", value=0.0001, format="%.10f")
    max_iterations = st.number_input("Maximum Iterations", value=50, step=1)
    initial_guess = st.number_input("Initial Guess", value=1.0)

    st.divider()
    solve_clicked = st.button("Solve", type="primary", use_container_width=True)


# ----------------------------------------------------------------
# MAIN AREA: solving + output
# ----------------------------------------------------------------
if solve_clicked:
    try:
        # --- Parse equation(s) ---
        if method == "Fixed Point":
            if fp_input_type == "f(x)":
                f_expr = parse_expression(f_str)
                phi_expr = auto_convert_to_phi(f_expr)
                if phi_expr is None:
                    st.warning(
                        "We couldn't automatically rearrange this f(x) into the form x = φ(x). "
                        "Please switch the Fixed Point Input Type to 'x = φ(x)' and enter it manually "
                        "(hint: isolate x on one side of f(x) = 0)."
                    )
                    st.stop()
            else:
                phi_expr = parse_expression(phi_str)
                f_expr = phi_expr - x  # for graphing / display purposes
        else:
            f_expr = parse_expression(f_str)
            phi_expr = None

        # --- Validate numeric settings ---
        validate_max_iterations(int(max_iterations))
        validate_tolerance(tolerance)

        # --- Root Between check ---
        if find_mode == "Root Between":
            active_expr = f_expr if f_expr is not None else phi_expr
            validate_interval(active_expr, left_bound, right_bound, trig_mode)
            x0 = (left_bound + right_bound) / 2  # sensible starting point inside the interval
            st.info(f"Interval check passed — using midpoint x₀ = {x0} as the starting guess.")
        else:
            x0 = float(initial_guess)

        # ------------------------------------------------------------
        # PROBLEM STATEMENT
        # ------------------------------------------------------------
        st.header("Problem")
        if method == "Fixed Point":
            st.latex(f"\\text{{Find a root of }} x = \\varphi(x) = {sp.latex(phi_expr)}")
            st.write("using the **Fixed-Point Iteration Method**.")
        else:
            st.latex(f"\\text{{Find a root of }} f(x) = {sp.latex(f_expr)}")
            st.write("using the **Newton-Raphson Method**.")

        # ------------------------------------------------------------
        # SOLUTION / DERIVATION
        # ------------------------------------------------------------
        st.header("Solution")

        if method == "Fixed Point":
            if fp_input_type == "f(x)":
                st.write("**Step 1 — Rearranging f(x) = 0 into x = φ(x):**")
                st.latex(f"f(x) = {sp.latex(f_expr)} = 0 \\quad\\Rightarrow\\quad x = {sp.latex(phi_expr)}")
            st.write(f"**Initial guess:** x₀ = {x0}")
            st.write("**Iteration formula:**")
            st.latex(f"x_{{n+1}} = \\varphi(x_n) = {sp.latex(phi_expr)}")
        else:
            f_prime = sp.diff(f_expr, x)
            st.write("**Step 1 — Computing the derivative f'(x):**")
            st.latex(f"f'(x) = {sp.latex(f_prime)}")
            st.write(f"**Initial guess:** x₀ = {x0}")
            st.write("**Iteration formula:**")
            st.latex("x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}")

        # ------------------------------------------------------------
        # RUN THE ALGORITHM
        # ------------------------------------------------------------
        common_args = dict(
            x0=x0, tol=tolerance, max_iter=int(max_iterations),
            print_digits=int(print_digits), correct_digit=int(correct_digit),
            use_rounded=use_rounded, trig_mode=trig_mode
        )

        if method == "Fixed Point":
            iterations, summary = fixed_point_iteration(phi_expr, **common_args)
        else:
            iterations, summary = newton_raphson(f_expr, **common_args)

        # ------------------------------------------------------------
        # SHOW EACH ITERATION IN DETAIL
        # ------------------------------------------------------------
        st.subheader("Iteration Details")
        for rec in iterations:
            with st.expander(f"Iteration {rec['iteration']}", expanded=(rec['iteration'] <= 2)):
                if method == "Fixed Point":
                    st.write(f"xₙ = {rec['x_n']}")
                    st.write(f"φ(xₙ) = {rec['phi_x_n']}")
                    st.write(f"New x = {rec['new_x']}")
                else:
                    st.write(f"xₙ = {rec['x_n']}")
                    st.write(f"f(xₙ) = {rec['f_x_n']}")
                    st.write(f"f'(xₙ) = {rec['f_prime_x_n']}")
                    st.write(f"New x = xₙ − f(xₙ)/f'(xₙ) = {rec['new_x']}")
                st.write(f"**Difference:** {rec['difference']}")

        # ------------------------------------------------------------
        # ITERATION TABLE
        # ------------------------------------------------------------
        st.subheader("Iteration Table")
        df = iterations_to_dataframe(iterations, method)
        st.dataframe(df, use_container_width=True)

        # ------------------------------------------------------------
        # SUMMARY
        # ------------------------------------------------------------
        st.subheader("Result Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Approximate Root", summary["root"])
        c1.metric("Iterations", summary["iterations"])
        c2.metric("Absolute Error", summary["absolute_error"])
        c2.metric("Relative Error", summary["relative_error"])
        c3.metric("Execution Time (s)", summary["execution_time"])
        c3.metric("Status", summary["status"])

        # ------------------------------------------------------------
        # GRAPHS
        # ------------------------------------------------------------
        st.subheader("Graphs")
        g1, g2, g3 = st.tabs(["Function Graph", "Error vs Iteration", "Convergence Graph"])

        graph_expr = f_expr if method == "Newton-Raphson" else phi_expr
        with g1:
            fig1 = plot_function(graph_expr, summary["root"], x0)
            st.pyplot(fig1)
        with g2:
            fig2 = plot_error_vs_iteration(iterations)
            st.pyplot(fig2)
        with g3:
            fig3 = plot_convergence(iterations)
            st.pyplot(fig3)

        # ------------------------------------------------------------
        # EXPORT
        # ------------------------------------------------------------
        st.subheader("Export")
        e1, e2 = st.columns(2)
        with e1:
            csv_bytes = dataframe_to_csv_bytes(df)
            st.download_button("⬇ Download CSV", csv_bytes, file_name="iterations.csv", mime="text/csv")
        with e2:
            problem_text = f"f(x) = {sp.latex(f_expr) if f_expr is not None else sp.latex(phi_expr)}"
            pdf_bytes = generate_pdf_report(problem_text, method, summary, df)
            st.download_button("⬇ Download PDF", pdf_bytes, file_name="report.pdf", mime="application/pdf")

    except (ConvergenceError, InputError) as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else:
    st.info("👈 Fill in the settings on the left, then click **Solve** to see the full step-by-step solution.")