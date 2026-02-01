"""Symbolic Regression Search Tool

Uses ProcessPoolExecutor with spawn + max_tasks_per_child=1 to ensure
each PySR run executes in a fresh process (avoids Julia threading issues).
"""

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

VERBOSE = True


def _print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


# =============================================================================
# Julia Module Template (embedded in worker process)
# =============================================================================

JULIA_MODULE = '''
using Roots, Interpolations, OrdinaryDiffEq
using Logging: disable_logging, Warn
disable_logging(Warn)

# -----------------------------
# Constants
# -----------------------------

const T = Float64
const M_P2 = T(1.0)
const SMALL = T(1e-30)

const EPS_EQ_TOL = T(1e-3)          # |epsilon(phi) - 1| tolerance for accepting roots
const ROOT_XRTOL = T(1e-6)          # root refinement tolerance (bisection)
const ROOT_DEDUP_TOL = T(1e-6)      # dedup tolerance for close roots

const NS_OBS = T({ns_target})
const NS_SIGMA = T({ns_sigma})
const R_OBS = T({r_target})
const R_SIGMA = T({r_sigma})
const N_OBS = T({N_obs})

# -----------------------------
# Slow-roll parameters
# -----------------------------

"""ε = (M_P²/2)(V'/V)²"""
function epsilon(V, V_prime, phi, phi_min, phi_max)
    (phi < phi_min || phi > phi_max || !isfinite(phi)) && return T(Inf)
    V_val, V_p = V(phi), V_prime(phi)
    (!isfinite(V_val) || !isfinite(V_p) || abs(V_val) < SMALL) && return T(Inf)
    return (M_P2 / 2) * (V_p / V_val)^2
end

"""dε/dφ"""
function epsilon_derivative(V, V_prime, V_double_prime, phi, phi_min, phi_max)
    (phi < phi_min || phi > phi_max || !isfinite(phi)) && return T(Inf)
    V_val, V_p, V_pp = V(phi), V_prime(phi), V_double_prime(phi)
    (!isfinite(V_val) || !isfinite(V_p) || !isfinite(V_pp) || abs(V_val) < SMALL) && return T(Inf)
    return M_P2 * (V_p / V_val) * ((V_pp * V_val - V_p^2) / V_val^2)
end

# -----------------------------
# E-fold integration
# -----------------------------

"""Find φ_N by integrating dφ/dN = M_P² V'/V backwards from φ_end."""
function find_phi_N(V, V_prime, phi_end, bound)
    direction = sign(bound - phi_end)
    direction == 0 && return nothing

    function dphi_dN!(dphi, phi, p, N)
        V_val, V_p = V(phi[1]), V_prime(phi[1])
        dphi[1] = (isfinite(V_val) && isfinite(V_p) && abs(V_val) > SMALL) ? M_P2 * V_p / V_val : zero(T)
    end

    boundary_cb = DiscreteCallback(
        (phi, N, integrator) -> direction > 0 ? phi[1] > bound : phi[1] < bound,
        integrator -> terminate!(integrator)
    )

    try
        prob = ODEProblem(dphi_dN!, [phi_end], (zero(T), N_OBS))
        sol = solve(prob; callback=boundary_cb, reltol=1e-4, maxiters=1000)
        (sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Terminated) || return nothing

        phi_N = sol.u[end][1]
        in_bounds = direction > 0 ? (phi_end < phi_N <= bound) : (bound <= phi_N < phi_end)
        return in_bounds ? phi_N : nothing
    catch
        return nothing
    end
end

# -----------------------------
# Root finding (epsilon = 1)
# -----------------------------

"""
Find inflation endpoints where ε(φ) = 1 within [phi_min, phi_max] using
grid-bracketing + bisection (avoids `find_zeros` numerical instability).
"""
function find_inflation_endpoints(V, V_prime, phi_min, phi_max; ngrid::Int=256)
    epsilon_eq(phi) = epsilon(V, V_prime, phi, phi_min, phi_max) - T(1.0)

    phis = range(phi_min, phi_max; length=ngrid)
    roots = T[]

    phi_prev = first(phis)
    f_prev = try
        epsilon_eq(phi_prev)
    catch
        oftype(phi_prev, NaN)
    end
    if isfinite(f_prev) && iszero(f_prev)
        push!(roots, phi_prev)
    end

    for i in 2:length(phis)
        phi = phis[i]
        f = try
            epsilon_eq(phi)
        catch
            oftype(phi, NaN)
        end

        if isfinite(f_prev) && isfinite(f)
            if iszero(f)
                push!(roots, phi)
            elseif f_prev * f < 0
                root = try
                    find_zero(epsilon_eq, (phi_prev, phi), Bisection(); xrtol=ROOT_XRTOL)
                catch
                    nothing
                end

                if !(root isa Nothing) && isfinite(root) && (phi_min < root < phi_max)
                    fr = try
                        epsilon_eq(root)
                    catch
                        oftype(root, NaN)
                    end
                    if isfinite(fr) && abs(fr) <= EPS_EQ_TOL
                        push!(roots, root)
                    end
                end
            end
        end

        phi_prev, f_prev = phi, f
    end

    isempty(roots) && return roots

    sort!(roots)
    dedup = T[]
    for r in roots
        if isempty(dedup) || abs(r - last(dedup)) > ROOT_DEDUP_TOL
            push!(dedup, r)
        end
    end
    return dedup
end

# -----------------------------
# Observables (ns, r)
# -----------------------------

"""Find all valid inflation trajectories and compute (ns, r) for each."""
function compute_observables(V, V_prime, V_double_prime, phi_min, phi_max)
    phi_ends = find_inflation_endpoints(V, V_prime, phi_min, phi_max; ngrid=256)
    isempty(phi_ends) && return Tuple{{T,T}}[]

    results = Tuple{{T,T}}[]
    for phi_end in phi_ends
        V_p_end = V_prime(phi_end)
        abs(V_p_end) < SMALL && continue
        direction = V_p_end > 0 ? 1 : -1

        # ε must be increasing at end (inflation ending)
        epsilon_derivative(V, V_prime, V_double_prime, phi_end, phi_min, phi_max) * direction >= 0 && continue

        # Integrate to find φ_N
        bound = direction > 0 ? phi_max : phi_min
        phi_N = find_phi_N(V, V_prime, phi_end, bound)
        isnothing(phi_N) && continue

        # Validate trajectory
        !(phi_min <= phi_N <= phi_max) && continue
        direction * (phi_N - phi_end) <= 0 && continue
        direction * V_prime(phi_N) <= 0 && continue
        direction * epsilon_derivative(V, V_prime, V_double_prime, phi_N, phi_min, phi_max) >= 0 && continue
        # Trajectory should not cross another ε=1 point between φ_end and φ_N.
        a, b = minmax(phi_N, phi_end)
        has_crossing = any(pe -> (a < pe < b) && (abs(pe - phi_end) > T(1e-9)), phi_ends)
        has_crossing && continue

        # Compute observables
        eps_N = epsilon(V, V_prime, phi_N, phi_min, phi_max)
        V_val, V_pp = V(phi_N), V_double_prime(phi_N)
        (!isfinite(eps_N) || eps_N <= 0 || abs(V_val) < SMALL) && continue
        eta_N = M_P2 * V_pp / V_val

        ns = 1.0 - 6.0 * eps_N + 2.0 * eta_N
        r = 16.0 * eps_N
        (isfinite(ns) && isfinite(r) && r >= 0) && push!(results, (ns, r))
    end

    return results
end

"""Compute min χ² and best (ns, r) from observations."""
function compute_min_chi2(obs_list)
    isempty(obs_list) && return T(Inf), nothing
    min_chi2, best_obs = typemax(T), nothing
    for (ns, r) in obs_list
        chi2 = ((ns - NS_OBS) / NS_SIGMA)^2 + ((r - R_OBS) / R_SIGMA)^2
        if chi2 < min_chi2
            min_chi2, best_obs = chi2, (ns, r)
        end
    end
    return min_chi2, best_obs
end

# -----------------------------
# Spline helpers
# -----------------------------

function make_spline(phis, values)
    itp = cubic_spline_interpolation(phis, values; extrapolation_bc=Line())
    V(phi) = itp(phi)
    V_prime(phi) = Interpolations.gradient(itp, phi)[1]
    V_double_prime(phi) = Interpolations.hessian(itp, phi)[1]
    return V, V_prime, V_double_prime
end

# -----------------------------
# PySR loss function
# -----------------------------

"""PySR loss: χ²/2 with numerical stability penalty."""
function compute_loss_julia(tree, dataset::Dataset{{T,L}}, options) where {{T,L}}
    prediction, success = eval_tree_array(tree, dataset.X, options)
    (!success || !all(isfinite, prediction)) && return typemax(T)

    phi_grid = view(dataset.X, 1, :)
    phi_min, phi_max = extrema(phi_grid)
    phis = range(phi_min, phi_max; length=length(phi_grid))

    try
        V, V_prime, V_double_prime = make_spline(phis, prediction)

        obs_list = compute_observables(V, V_prime, V_double_prime, phi_min, phi_max)
        chi2, best_obs = compute_min_chi2(obs_list)
        !isfinite(chi2) && return typemax(T)

        # Coarse grid stability check for promising candidates
        if chi2 < 3
            phis_c = phis[1:2:end]
            pred_c = prediction[1:2:end]
            phi_min_c, phi_max_c = first(phis_c), last(phis_c)

            V_c, V_prime_c, V_double_prime_c = make_spline(phis_c, pred_c)

            obs_c = compute_observables(V_c, V_prime_c, V_double_prime_c, phi_min_c, phi_max_c)
            chi2_c, best_obs_c = compute_min_chi2(obs_c)

            if !isfinite(chi2_c) || isnothing(best_obs_c)
                chi2 += 10.0
            else
                ns_diff = abs(best_obs[1] - best_obs_c[1])
                chi2 += (ns_diff / NS_SIGMA)^2
            end
        end

        return chi2 / 2
    catch
        return typemax(T)
    end
end
'''


# =============================================================================
# Worker Function (runs in isolated process)
# =============================================================================


def _run_pysr(config: dict) -> dict:
    """Execute PySR in worker process. Returns result dict."""

    from pysr import PySRRegressor, jl

    from .inflation import compute_observables_all_trajectories

    # Extract config
    binary_ops = config["binary_operators"]
    unary_ops = config.get("unary_operators", [])
    ns_target = config.get("ns_target", 0.9649)
    ns_sigma = config.get("ns_sigma", 0.0042)
    r_target = config.get("r_target", 0.0)
    r_sigma = config.get("r_sigma", 0.014)
    N_obs = config.get("N_obs", 60.0)
    maxsize = config.get("maxsize", 15)
    populations = config.get("populations", 31)
    population_size = config.get("population_size", 27)
    niterations = config.get("niterations", 40)

    # Process constraints
    constraints = config.get("constraints")
    if constraints:
        constraints = {k: tuple(v) if isinstance(v, list) else v for k, v in constraints.items()}

    print(f"[SR] Targets: ns={ns_target}±{ns_sigma}, r={r_target}±{r_sigma}, N={N_obs}")
    print(f"[SR] Ops: {binary_ops} + {unary_ops}, maxsize={maxsize}")
    print(f"[SR] Evolution: {populations}×{population_size}, {niterations} iters")

    # Load Julia module
    julia_code = JULIA_MODULE.format(
        ns_target=ns_target,
        ns_sigma=ns_sigma,
        r_target=r_target,
        r_sigma=r_sigma,
        N_obs=N_obs,
    )
    jl.seval(julia_code)
    print("[SR] Julia module loaded")

    # Configure PySR
    pysr_config = {
        "niterations": niterations,
        "populations": populations,
        "population_size": population_size,
        "binary_operators": binary_ops,
        "maxsize": maxsize,
        "loss_function": "compute_loss_julia",
        "precision": 64,
        "model_selection": "best",
        "timeout_in_seconds": 600,
        "early_stop_condition": "f(loss, complexity) = (loss < 1e-4) && (complexity < 10)",
        "turbo": True,
        "bumper": True,
        "verbosity": 1 if VERBOSE else 0,
        "progress": True,
    }
    if unary_ops:
        pysr_config["unary_operators"] = unary_ops
    if constraints:
        pysr_config["constraints"] = constraints
    if config.get("nested_constraints"):
        pysr_config["nested_constraints"] = config["nested_constraints"]
    if config.get("complexity_of_operators"):
        pysr_config["complexity_of_operators"] = config["complexity_of_operators"]

    # Run PySR
    print(f"[SR] Starting (~{niterations // 10} min)...")
    phi = np.linspace(0.001, 25.0, 1000)
    model = PySRRegressor(**pysr_config)
    model.fit(phi.reshape(-1, 1), np.zeros(1000), variable_names=["phi"])

    # Verify and collect results
    print("[SR] Verifying candidates...")
    equations = model.equations_.sort_values("score", ascending=False)
    results = []

    for i in range(min(20, len(equations))):
        eq = equations.iloc[i]
        expr = str(eq["equation"])
        try:
            obs_list = compute_observables_all_trajectories(expr, N=N_obs)
            if not obs_list:
                continue
            obs = obs_list[0]
            chi2 = ((obs["ns"] - ns_target) / ns_sigma) ** 2 + ((obs["r"] - r_target) / r_sigma) ** 2
            loss = chi2 / 2
            if loss <= 50.0:
                results.append(
                    {
                        "rank": len(results) + 1,
                        "expression": expr,
                        "score": float(eq["score"]),
                        "loss": round(loss, 4),
                        "ns": round(obs["ns"], 5),
                        "r": round(obs["r"], 5),
                        "N": N_obs,
                        "complexity": int(eq["complexity"]),
                    }
                )
            if len(results) >= 10:
                break
        except Exception:
            pass

    if not results:
        return {"success": False, "error": "No valid candidates found"}

    print(f"[SR] Found {len(results)} candidates")
    return {"success": True, "num_results": len(results), "results": results}


# =============================================================================
# Tool Function (called by agent)
# =============================================================================


def search_potential(config_json: str) -> str:
    """Find potentials matching target observables via symbolic regression (PySR). Takes 1-5 min.

    Returns ranked candidates with loss and complexity.

    Args:
        config_json: JSON config with: binary_operators, unary_operators,
            ns_target, ns_sigma, r_target, r_sigma, N_obs, maxsize, niterations,
            populations, constraints, nested_constraints, complexity_of_operators.
            Example: {"binary_operators": ["+", "*", "^"], "ns_target": 0.9649, ...}
    """
    # Parse config
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        return json.dumps({"success": False, "error": f"Invalid JSON: {e}"})

    # Validate required fields
    binary_ops = config.get("binary_operators")
    if not binary_ops:
        return json.dumps(
            {
                "success": False,
                "error": "Missing 'binary_operators'",
                "hint": '{"binary_operators": ["+", "*", "^"]}',
            }
        )

    # Validate constraints reference valid operators
    all_ops = set(binary_ops) | set(config.get("unary_operators", []))

    if config.get("constraints"):
        invalid = set(config["constraints"].keys()) - all_ops
        if invalid:
            return json.dumps(
                {
                    "success": False,
                    "error": f"constraints references undefined operators: {list(invalid)}",
                }
            )

    if config.get("nested_constraints"):
        # Check outer keys
        invalid = set(config["nested_constraints"].keys()) - all_ops
        if invalid:
            return json.dumps(
                {
                    "success": False,
                    "error": f"nested_constraints references undefined operators: {list(invalid)}",
                }
            )
        # Check inner keys (nested operators must also be valid)
        for outer_op, inner_dict in config["nested_constraints"].items():
            if isinstance(inner_dict, dict):
                invalid_inner = set(inner_dict.keys()) - all_ops
                if invalid_inner:
                    return json.dumps(
                        {
                            "success": False,
                            "error": f"nested_constraints['{outer_op}'] references undefined operators: {list(invalid_inner)}",
                        }
                    )

    # Run in isolated process
    _print("[SR] Launching worker process...")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1, mp_context=ctx) as executor:
        future = executor.submit(_run_pysr, config)
        try:
            result = future.result(timeout=660)
        except TimeoutError:
            return json.dumps({"success": False, "error": "Timeout (11 min)"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    return json.dumps(result, indent=2)
