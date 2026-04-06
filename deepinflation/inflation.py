"""Inflation cosmology calculations - computes observables (ns, r, A_s)"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Physics constants
M_P = 1.0
M_P2 = M_P**2
N_OBS_VALUES = [50, 55, 60]  # Default N_obs values to compute
EPSILON_SMALL = 1e-20


def parse_potential(expression: str):
    """Parse expression to callable V(phi). Raises ValueError for invalid expressions."""
    import re

    # Check for unknown identifiers
    identifiers = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expression))
    allowed = {"phi", "exp", "log", "sqrt", "sin", "cos", "tan", "tanh", "abs", "pi", "square", "cube", "neg"}
    unknown = identifiers - allowed
    if unknown:
        raise ValueError(f"Unknown identifier(s): {list(unknown)}")

    expr_python = expression.replace("^", "**")
    namespace = {
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "tanh": np.tanh,
        "abs": np.abs,
        "pi": np.pi,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
        "neg": lambda x: -x,
    }

    # Test evaluation
    namespace["phi"] = 1.0
    try:
        result = float(eval(expr_python, {"__builtins__": {}}, namespace))
        if not np.isfinite(result):
            raise ValueError("Expression returns non-finite value at phi=1")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}") from e

    def V(phi):
        namespace["phi"] = phi
        return float(eval(expr_python, {"__builtins__": {}}, namespace))

    return V


def numerical_derivative(f, x, h=1e-6, order=1):
    """Central difference derivative."""
    if order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    elif order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)
    raise ValueError("Only order 1 and 2 supported")


def epsilon(V, V_prime, phi):
    """Slow-roll parameter: ε = (M_P²/2)(V'/V)²"""
    val, prime = V(phi), V_prime(phi)
    if not np.isfinite(val) or not np.isfinite(prime) or abs(val) < EPSILON_SMALL:
        return np.inf
    return (M_P2 / 2) * (prime / val) ** 2


def epsilon_derivative(V, V_prime, V_double_prime, phi):
    """dε/dφ"""
    val, prime, double_prime = V(phi), V_prime(phi), V_double_prime(phi)
    if not (np.isfinite(val) and np.isfinite(prime) and np.isfinite(double_prime)) or abs(val) < EPSILON_SMALL:
        return np.inf
    return M_P2 * (prime / val) * ((double_prime * val - prime**2) / val**2)


def _solve_phi_trajectory(V, V_prime, phi_end, bound, N_max):
    """Solve φ(N) with dense output from inflation end back to earlier e-folds."""

    def dphi_dN(phi):
        v, vp = V(phi), V_prime(phi)
        return M_P2 * vp / v if abs(v) >= EPSILON_SMALL else 0.0

    try:
        sol = solve_ivp(
            lambda N, phi: dphi_dN(phi[0]),
            [0.0, N_max],
            [phi_end],
            method="RK45",
            rtol=1e-4,
            atol=1e-6,
            dense_output=True,
        )
    except Exception:
        return None, None, None

    if not sol.success or sol.sol is None:
        return None, None, None

    lo, hi = min(phi_end, bound), max(phi_end, bound)
    return sol, lo, hi


def find_phi_N(V, V_prime, N_values, phi_end, bound):
    """Find φ_N via ODE: dφ/dN = M_P² V'/V

    Args:
        N_values: list of e-fold values to compute
    Returns:
        dict {N: phi_N} for valid trajectories
    """
    N_max = max(N_values)
    sol, lo, hi = _solve_phi_trajectory(V, V_prime, phi_end, bound, N_max)
    if sol is None:
        return {}

    phi_values = np.atleast_1d(sol.sol(N_values)[0])
    return {
        N: float(phi_N)
        for N, phi_N in zip(N_values, phi_values, strict=True)
        if np.isfinite(phi_N) and lo <= phi_N <= hi
    }


def compute_observables(V, V_prime, V_double_prime, phi_min, phi_max, N_values=None):
    """Compute observables (ns, r, A_s) for multiple N_obs values."""
    if N_values is None:
        N_values = N_OBS_VALUES

    # Find inflation end points (ε = 1)
    grid = np.linspace(phi_min, phi_max, 200)
    eps_grid = np.array([epsilon(V, V_prime, phi) for phi in grid])
    eps_minus_one = eps_grid - 1.0

    phi_ends = []
    sign_changes = np.where(np.diff(np.sign(eps_minus_one)) != 0)[0]

    for i in sign_changes:
        try:
            root = brentq(lambda phi: epsilon(V, V_prime, phi) - 1.0, grid[i], grid[i + 1], rtol=np.float64(1e-3))
            phi_ends.append(root)
        except Exception:
            continue

    if not phi_ends:
        return []

    candidates = []

    for phi_end in phi_ends:
        vp_end = V_prime(phi_end)
        if abs(vp_end) < EPSILON_SMALL:
            continue

        direction = 1 if vp_end > 0 else -1

        # Check epsilon is increasing at end
        if epsilon_derivative(V, V_prime, V_double_prime, phi_end) * direction >= 0:
            continue

        # Find phi_N for all N values in one solve
        bound = phi_max if direction > 0 else phi_min
        phi_N_dict = find_phi_N(V, V_prime, N_values, phi_end, bound)

        if not phi_N_dict:
            continue

        # Compute observables for each N
        observables = {}
        for N, phi_N in phi_N_dict.items():
            # Validate trajectory
            if not (
                phi_min <= phi_N <= phi_max and direction * (phi_N - phi_end) > 0 and direction * V_prime(phi_N) > 0
            ):
                continue

            # Calculate observables
            eps_N = epsilon(V, V_prime, phi_N)
            eta_N = M_P2 * V_double_prime(phi_N) / V(phi_N)

            ns = 1.0 - 6.0 * eps_N + 2.0 * eta_N
            r = 16.0 * eps_N
            A_s = V(phi_N) / (24.0 * np.pi**2 * M_P2**2 * eps_N) * 1e9

            if not (np.isfinite(ns) and np.isfinite(r) and np.isfinite(A_s)):
                continue
            if A_s <= 0 or r < 0:
                continue

            observables[N] = {
                "phi_N": float(phi_N),
                "ns": float(ns),
                "r": float(r),
                "A_s": float(A_s) / 1e9,
                "epsilon": float(eps_N),
                "eta": float(eta_N),
            }

        if observables:
            candidates.append(
                {
                    "phi_end": float(phi_end),
                    "observables": observables,
                }
            )

    return candidates


def compute_observables_all_trajectories(expression: str, phi_min: float = 0.001, phi_max: float = 25.0, N=None):
    """Compute observables for all inflation trajectories.

    Args:
        expression: Potential V(phi) expression
        phi_min, phi_max: Field range
        N: e-fold values. Can be:
           - None: use default [50, 55, 60], return grouped structure
           - float/int: single value, return flat structure
           - list: multiple values, return grouped structure

    Returns:
        If N is float/int: list of flat dicts {trajectory_id, phi_end, N, ns, r, ...}
        If N is list/None: list of dicts {trajectory_id, phi_end, observables: {N: {...}}}
    """
    V = parse_potential(expression)
    V_prime = lambda phi: numerical_derivative(V, phi, order=1)
    V_double_prime = lambda phi: numerical_derivative(V, phi, order=2)

    # Determine mode based on N type
    single_mode = isinstance(N, (int, float))
    N_list = [N] if single_mode else (N if N is not None else N_OBS_VALUES)

    candidates = compute_observables(V, V_prime, V_double_prime, phi_min, phi_max, N_values=N_list)

    trajectories = []
    for traj_id, cand in enumerate(candidates):
        if single_mode:
            # Flat structure for single N
            if N in cand["observables"]:
                obs = cand["observables"][N]
                trajectories.append(
                    {
                        "trajectory_id": traj_id + 1,
                        "phi_end": cand["phi_end"],
                        "N": N,
                        **obs,
                    }
                )
        else:
            # Grouped structure for multiple N
            cand["trajectory_id"] = traj_id + 1
            trajectories.append(cand)

    return trajectories


def generate_plot_data(expression: str, phi_min: float = 0.001, phi_max: float = 25.0, n_points: int = 400):
    """Generate plot data: (phi, V, epsilon, eta)"""
    V = parse_potential(expression)
    V_prime = lambda phi: numerical_derivative(V, phi, order=1)
    V_double_prime = lambda phi: numerical_derivative(V, phi, order=2)

    phi_array = np.linspace(phi_min, phi_max, n_points)
    V_array = np.array([V(phi) for phi in phi_array])
    eps_array = np.array([epsilon(V, V_prime, phi) for phi in phi_array])
    eta_array = np.array(
        [M_P2 * V_double_prime(phi) / V(phi) if abs(V(phi)) > EPSILON_SMALL else np.inf for phi in phi_array]
    )

    return phi_array, V_array, eps_array, eta_array
