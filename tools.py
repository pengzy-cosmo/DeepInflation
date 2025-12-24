"""Tools for Inflation Agent

Two main tools for analyzing inflation potentials:
- analyze_potential: Compute observables (ns, r, A_s) for all trajectories (~1s)
- plot_potential: Generate 3-panel diagnostic plot (~2s)
  - Panel 1: V(φ) with trajectory markers (φ_end, N=50, N=60)
  - Panel 2: Slow-roll parameters (ε, η) vs φ
  - Panel 3: Predicted (ns, r) vs Planck+BK18 posterior
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from inflation import compute_observables_all_trajectories, generate_plot_data

# Load BK18+Planck posterior data
_BK18_DATA_PATH = Path("data/bk18_planck_posterior.npz")
_BK18_DATA = np.load(_BK18_DATA_PATH) if _BK18_DATA_PATH.exists() else None

VERBOSE = True
IMAGE_MODEL = "dall-e-3"


def _print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def analyze_potential(expression: str) -> str:
    """Compute ns, r, A_s for all inflation trajectories. Returns JSON with observables.

    Expression must use only 'phi' and concrete numbers (NO symbolic parameters like M, V0).

    Args:
        expression: V(φ) with concrete values only.
            Valid: 'phi^2' or 'phi**2', '(1-exp(-sqrt(2/3)*phi))^2'.
            Invalid: 'M*phi^2', 'V0*exp(-phi)'.
    """
    _print(f"[Analyze] V(φ) = {expression}")
    try:
        trajectories = compute_observables_all_trajectories(expression)
        if not trajectories:
            return json.dumps({"success": False, "error": "No valid trajectories"}, indent=2)
        return json.dumps({
            "success": True,
            "expression": expression,
            "num_trajectories": len(trajectories),
            "trajectories": trajectories
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)



# Tool display config update in agent.py will be needed, but here we define the tools.

def plot_potential(expression: str, panels: list[str] = None, output_path: str = "./potential_plot.png") -> str:
    """Generate diagnostic plots for a potential.

    Args:
        expression: V(φ) with concrete values.
        panels: List of panels to generate. Options:
            - 'potential': V(φ) with trajectory markers
            - 'slow_roll': ε and η parameters
            - 'constraints': ns-r plane with Planck+BK18
            Default (None) generates all three.
        output_path: Save path (default: './potential_plot.png').
    """
    _print(f"[Plot] V(φ) = {expression}, Panels={panels or 'All'}")
    
    # Default to all panels
    if not panels:
        panels = ["potential", "slow_roll", "constraints"]
    
    # Validate panels
    valid_panels = {"potential", "slow_roll", "constraints"}
    panels = [p for p in panels if p in valid_panels]
    if not panels:
        return json.dumps({"success": False, "error": "No valid panels specified"}, indent=2)

    try:
        phi, V, eps, eta = generate_plot_data(expression)
        
        # Only compute trajectories if needed for potential markers or constraints
        trajectories_60 = []
        trajectories_50 = []
        if "potential" in panels or "constraints" in panels:
            trajectories_60 = compute_observables_all_trajectories(expression, N=60.0)
            trajectories_50 = compute_observables_all_trajectories(expression, N=50.0)

        # Helper function to get V value
        def get_V(p):
            return V[np.argmin(np.abs(phi - p))]

        # Setup figure based on number of panels
        n_panels = len(panels)
        fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
        if n_panels == 1:
            axes = [axes]
        
        fig.suptitle(f'V(φ) = {expression}', fontsize=11, y=0.98)
        
        current_ax_idx = 0

        # --- Panel: Potential ---
        if "potential" in panels:
            ax = axes[current_ax_idx]
            
            # Determine range
            phi_min, phi_max = phi[0], phi[-1]
            if trajectories_60:
                all_phi = [
                    p for t60, t50 in zip(trajectories_60, trajectories_50, strict=False)
                    for p in [t60['phi_end'], t50['phi_N'], t60['phi_N']]
                ]
                phi_min = max(phi[0], min(all_phi) - 3)
                phi_max = min(phi[-1], max(all_phi) + 3)
            
            # Restrict to V > 0
            positive_V_indices = np.where(V > 0)[0]
            if len(positive_V_indices) > 0:
                phi_V_positive_min = phi[positive_V_indices[0]]
                phi_V_positive_max = phi[positive_V_indices[-1]]
                phi_min = max(phi_min, phi_V_positive_min)
                phi_max = min(phi_max, phi_V_positive_max)

            mask = (phi >= phi_min) & (phi <= phi_max)
            ax.plot(phi[mask], V[mask], linewidth=2, color='#2E86AB', alpha=0.8)

            if trajectories_60:
                colors = plt.cm.tab10(np.arange(len(trajectories_60)))
                for i, (t60, t50) in enumerate(zip(trajectories_60, trajectories_50, strict=False)):
                    ax.scatter(t60['phi_end'], get_V(t60['phi_end']), s=60, c=[colors[i]], marker='x', linewidths=2.5, zorder=10)
                    ax.scatter([t50['phi_N'], t60['phi_N']], [get_V(t50['phi_N']), get_V(t60['phi_N'])], 
                               s=[40, 60], c=[colors[i], colors[i]], marker='o', edgecolors='black', linewidths=1, zorder=9)
            
            ax.set_xlabel('φ', fontsize=12)
            ax.set_ylabel('V(φ)', fontsize=12)
            ax.set_title('Potential', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle=':')
            current_ax_idx += 1

        # --- Panel: Slow-roll ---
        if "slow_roll" in panels:
            ax = axes[current_ax_idx]
            # Use data from phi_min/max if determined by potential panel, else full range
            # ideally we reuse the range if potential panel ran, but simple logic: plot full valid range
            
            valid_eps = np.isfinite(eps) & (eps > 0) & (eps < 1e2)
            valid_eta = np.isfinite(eta) & (np.abs(eta) > 0) & (np.abs(eta) < 1e2)
            
            if np.any(valid_eps):
                ax.plot(phi[valid_eps], eps[valid_eps], label='ε', linewidth=2, color='#A23B72', alpha=0.8)
            if np.any(valid_eta):
                ax.plot(phi[valid_eta], np.abs(eta[valid_eta]), label='|η|', linewidth=2, color='#F18F01', alpha=0.8)
            
            ax.axhline(1, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.set_xlabel('φ', fontsize=12)
            ax.set_ylabel('Slow-roll parameters', fontsize=12)
            ax.set_yscale('log')
            ax.set_ylim([1e-4, 1e2])
            ax.legend(fontsize=10)
            ax.set_title('Slow-roll Parameters', fontsize=11)
            ax.grid(True, alpha=0.3, which='both', linestyle=':')
            current_ax_idx += 1

        # --- Panel: Constraints ---
        if "constraints" in panels:
            ax = axes[current_ax_idx]
            posterior_color = plt.cm.tab10(0)
            if _BK18_DATA is not None:
                ns_data, r_data, P = _BK18_DATA['ns'], _BK18_DATA['r'], _BK18_DATA['P_bk18']
                levels = _BK18_DATA['levels_bk18']
                ax.contourf(ns_data, r_data, P, levels=[levels[0], levels[1]], colors=[posterior_color], alpha=0.4, zorder=1)
                ax.contourf(ns_data, r_data, P, levels=[levels[1], P.max()], colors=[posterior_color], alpha=0.8, zorder=2)
                ax.contour(ns_data, r_data, P, levels=levels, colors=[posterior_color], linewidths=1.2, alpha=0.9, zorder=3)

            if trajectories_60:
                for i, (t60, t50) in enumerate(zip(trajectories_60, trajectories_50, strict=False)):
                    color = plt.cm.tab10((i + 1) % 10)
                    # Compute line
                    ns_line, r_line = [], []
                    for N_val in np.linspace(50, 60, 11):
                        traj = compute_observables_all_trajectories(expression, N=N_val)
                        if traj and len(traj) > i:
                            ns_line.append(traj[i]['ns'])
                            r_line.append(traj[i]['r'])
                    
                    if len(ns_line) > 1:
                        ax.plot(ns_line, r_line, '-', color=color, alpha=0.7, linewidth=2.5, zorder=5)
                    
                    ax.scatter(t50['ns'], t50['r'], s=40, c=[color], marker='o', edgecolors='black', linewidths=1.0, zorder=10)
                    ax.scatter(t60['ns'], t60['r'], s=60, c=[color], marker='o', edgecolors='black', linewidths=1.2, zorder=11)

                r_all = [t['r'] for t in trajectories_60 + trajectories_50]
                ax.set_xlim([0.945, 0.99])
                ax.set_ylim([0.0, min(0.26, max(max(r_all) * 1.3, 0.06))])
            else:
                ax.text(0.5, 0.5, 'No valid trajectories', ha='center', va='center', transform=ax.transAxes, color='gray')
                ax.set_xlim([0.945, 1.0])
                ax.set_ylim([0.0, 0.26])

            ax.set_xlabel('$n_s$', fontsize=12)
            ax.set_ylabel('$r$', fontsize=12)
            ax.set_title('Observables vs BK18+Planck', fontsize=11)
            current_ax_idx += 1

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        _print(f"[Plot] Saved to {output_path.absolute()}")

        return json.dumps({"success": True, "plot_path": str(output_path.absolute())}, indent=2)

    except Exception as e:
        return json.dumps({"success": False, "error": f"Plot error: {e}"}, indent=2)


def generate_schematic(prompt: str, output_path: str = "./schematic.png") -> str:
    """Generate a schematic/conceptual image using DALL-E 3.
    
    Use this when the user requests a visualization that cannot be plotted mathematically,
    such as "draw the multiverse", "illustrate slow roll inflation", or artistic concepts.
    
    Args:
        prompt: Detailed description of the image to generate.
        output_path: Path to save the image (default: ./schematic.png)
    """
    _print(f"[Schematic] Generating: {prompt}")
    
    import os
    import requests
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    if not api_key:
         return json.dumps({"success": False, "error": "OPENAI_API_KEY not found"}, indent=2)

    try:
        # Use standard OpenAI client
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Call API (removed problematic parameters)
        response = client.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        
        image_obj = response.data[0]
        
        # Robustly check for image data
        if getattr(image_obj, 'url', None):
            _print(f"[Schematic] Found URL")
            img_data = requests.get(image_obj.url).content
        elif getattr(image_obj, 'b64_json', None):
            _print(f"[Schematic] Found Base64 JSON")
            import base64
            img_data = base64.b64decode(image_obj.b64_json)
        else:
            # Fallback debug: what did we actually get?
            debug_info = f"Response keys: {image_obj.__dict__ if hasattr(image_obj, '__dict__') else dir(image_obj)}"
            _print(f"[Schematic] No standard data found. Debug: {debug_info}")
            return json.dumps({"success": False, "error": f"API returned unknown format. Debug info: {debug_info}"}, indent=2)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(img_data)
            
        _print(f"[Schematic] Saved to {output_path.absolute()}")
        return json.dumps({"success": True, "plot_path": str(output_path.absolute())}, indent=2)

    except Exception as e:
        _print(f"[Schematic] Error: {e}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


if __name__ == "__main__":
    print("Testing tools...")
    print("=" * 60)
    print("\n1. Analyzing V(φ) = phi^2:")
    print(analyze_potential("phi^2"))
    print("\n2. Plotting (Potential only):")
    print(plot_potential("phi^2", panels=["potential"]))

