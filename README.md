# Numerical Methods Solver

A web application for solving nonlinear equations using the **Fixed-Point Iteration Method** and the **Newton-Raphson Method**, with full step-by-step working, iteration tables, graphs, and exportable reports.

🔗 **Live app:** https://diu-numerical-solver.streamlit.app

---

## What this project does

You type in a mathematical equation (like `x**2 - 2` or `cos(x) - x*exp(x)`), choose a method, and the app:
- Shows the mathematical derivation (like a textbook)
- Runs the algorithm and shows every iteration in detail
- Displays a professional iteration table
- Plots the function, error, and convergence graphs
- Lets you download the results as CSV or PDF

---

## Project structure

```
numerical-methods-solver/
├── app.py              # The Streamlit website (user interface)
├── algorithms.py        # Fixed-Point Iteration and Newton-Raphson math engine
├── graph.py              # Plotting functions (function graph, error graph, convergence graph)
├── utils.py               # Input validation, formatting, CSV/PDF export helpers
├── requirements.txt   # List of Python libraries this project needs
├── .gitignore              # Tells Git which files/folders to ignore
└── README.md          # This file
```

---

## Beginner's guide: running this project on your own computer

This section assumes you know **nothing** about any of these tools. Follow it top to bottom.

### 1. What is Python?

Python is a programming language. Your computer needs the Python *program* installed to understand and run `.py` files.

**Install it:**
1. Go to https://www.python.org/downloads/
2. Download and run the installer.
3. ⚠️ On the first screen, check the box **"Add python.exe to PATH"** before clicking Install.
4. Verify it worked by opening a terminal (see below) and typing `python --version`.

### 2. What is VS Code?

VS Code is a text editor built for writing code — it highlights syntax, points out errors, and has a built-in terminal.

**Install it:**
1. Go to https://code.visualstudio.com/
2. Download and run the installer (default options are fine).
3. Once open, install the **Python extension**: click the Extensions icon (four squares) in the left sidebar, search "Python," install the one by Microsoft.

### 3. What is a Terminal?

A terminal lets you control your computer by typing commands instead of clicking. In VS Code: **Terminal → New Terminal** opens one at the bottom of the window.

### 4. What is a virtual environment, and why do we need one?

It's an isolated "box" for this project's Python libraries, so they don't conflict with other projects on your computer.

**Create and activate one** (run inside the project folder, in the terminal):
```
python -m venv venv
.\venv\Scripts\Activate.ps1
```
If you get an "execution policy" error on Windows, run this once:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
You'll know it worked when your terminal prompt starts with `(venv)`.

### 5. What is pip, and what is requirements.txt?

`pip` is Python's tool for installing extra libraries. `requirements.txt` is a plain text list of the libraries this project needs (streamlit, sympy, numpy, pandas, matplotlib).

**Install everything the project needs** with one command:
```
pip install -r requirements.txt
```
This reads the list in `requirements.txt` and installs each library into your `(venv)` box.

### 6. What is Streamlit, and how do I run the app?

Streamlit is the library that turns `app.py` into an actual website, with no HTML/CSS needed.

**Run it:**
```
streamlit run app.py
```
This starts a local web server and opens the app in your browser at `http://localhost:8501`. This only works while the command is running — press `Ctrl+C` in the terminal to stop it.

---

## What is Git and GitHub?

- **Git** tracks changes to your code over time, like a save/undo system.
- **GitHub** is a website that stores your code online, backs it up, and lets you deploy it.

**The routine for saving and uploading a change:**
```
git add .
git commit -m "Describe what you changed"
git push
```

---

## Deployment

This app is deployed on **Streamlit Community Cloud** (https://share.streamlit.io/), a free hosting service for public Streamlit apps connected to a GitHub repository. Every time changes are pushed to GitHub, the live app automatically updates within about a minute — no manual redeploy needed.

---

## Technology used

- **Python** — programming language
- **Streamlit** — web interface
- **SymPy** — symbolic math (equation parsing, automatic differentiation)
- **NumPy** — numerical computation
- **Pandas** — iteration tables
- **Matplotlib** — graphs and PDF report generation

---

## Author

Built as a university project by Wakibaaaaa.