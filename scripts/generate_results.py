"""
scripts/generate_results.py
============================
Produces the logging/plotting artifacts for the paper's Results section:
tracking-error-over-time, manipulability-over-time, and damping-factor
plots from a full closed-loop run on iiwa7_r800 through a near-singular
trajectory -- plus the fixed-chart-vs-log-map comparison figure for the
gap identified against prior art (SEW-angle paper).

Usage
-----
    python scripts/generate_results.py

Outputs (written to results/)
------------------------------
    run_log.csv                 per-timestep numeric log (t, |xi_err|, w, lam)
    run_summary.json            scalar summary stats for the paper text
    tracking_error.png
    manipulability_and_damping.png
    chart_comparison.png
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(ROOT, "..", "src"))
RESULTS_DIR = os.path.abspath(os.path.join(ROOT, "..", "results"))
sys.path.insert(0, SRC)

from kinematics.robot_param import get_robot
from kinematics.forward_kinematics import forward_kinematics
from kinematics.lie_algebra import exp_map, log_map_se3
from simulation.controller import PIDControlLaw
from simulation.trajectory import screw_trajectory, check_log_admissible
from simulation.trajectory_tracker import run_control_loop


def run_near_singular_demo():
    iiwa7 = get_robot("iiwa7_r800")
    q_start = np.array([0.1, 0.3, 0.0, -0.05, 0.0, 0.05, 0.0])
    q_end = np.array([0.4, 0.9, 0.3, 1.0, 0.2, -0.6, 0.3])
    g_start = forward_kinematics(q_start, iiwa7["S_list"], iiwa7["M"])
    g_end = forward_kinematics(q_end, iiwa7["S_list"], iiwa7["M"])

    admissible, angle = check_log_admissible(g_start, g_end)
    print(f"segment log-admissible: {admissible} (rotation angle={angle:.4f} rad)")

    T_total = 3.0
    traj = screw_trajectory(g_start, g_end, T=T_total)
    pid = PIDControlLaw(Kp=8.0, Ki=0.0, Kd=0.5)

    log = run_control_loop(
        q_start, iiwa7["S_list"], iiwa7["M"], traj, pid,
        dt=0.005, T_total=T_total, k_null=1.0,
        w_threshold=0.05, lam_max=0.05,
        joint_limits=iiwa7["joint_limits"], verbose=True,
    )
    return log


def write_csv(log, path):
    xi_err_norm = np.linalg.norm(log["xi_err_history"], axis=1)
    with open(path, "w") as f:
        f.write("t,xi_err_norm,manipulability_w,damping_lambda\n")
        for i in range(len(xi_err_norm)):
            f.write(f"{log['t_history'][i]:.6f},{xi_err_norm[i]:.8f},"
                    f"{log['w_history'][i]:.8f},{log['lam_history'][i]:.8f}\n")


def write_summary(log, path):
    xi_err_norm = np.linalg.norm(log["xi_err_history"], axis=1)
    summary = {
        "final_tracking_error_norm": float(xi_err_norm[-1]),
        "max_tracking_error_norm": float(np.max(xi_err_norm)),
        "mean_manipulability": float(np.mean(log["w_history"])),
        "min_manipulability": float(np.min(log["w_history"])),
        "max_damping_lambda": float(np.max(log["lam_history"])),
        "num_steps": int(len(xi_err_norm)),
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def plot_tracking_error(log, path):
    xi_err_norm = np.linalg.norm(log["xi_err_history"], axis=1)
    plt.figure(figsize=(7, 4))
    plt.plot(log["t_history"][:-1], xi_err_norm)
    plt.xlabel("time (s)")
    plt.ylabel(r"$\|\xi_{err}\|$")
    plt.title("Task-space tracking error, iiwa7_r800, near-singular segment")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_manipulability_and_damping(log, path):
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(log["t_history"][:-1], log["w_history"])
    axes[0].set_ylabel("Yoshikawa manipulability w")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(log["t_history"][:-1], log["lam_history"], color="tab:orange")
    axes[1].set_ylabel(r"damping $\lambda$")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Manipulability and adaptive damping over the run")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_chart_comparison(path):
    axis = np.array([1.0, 0.0, 0.0])
    thetas = np.linspace(0.05, np.pi - 1e-3, 200)
    fixed_chart_condition = []
    log_map_error = []
    for theta in thetas:
        xi = np.concatenate([theta * axis, np.array([0.1, 0.0, 0.0])])
        g = exp_map(xi)
        R = g[:3, :3]
        sin_theta = np.sin(theta)
        fixed_chart_condition.append(1.0 / max(sin_theta, 1e-12))
        xi_hat = log_map_se3(g)
        log_map_error.append(np.max(np.abs(xi - xi_hat)))

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(thetas, fixed_chart_condition, color="tab:red", label="fixed-chart condition proxy (1/sin θ)")
    ax1.set_yscale("log")
    ax1.set_xlabel("relative rotation angle θ (rad)")
    ax1.set_ylabel("fixed-chart condition proxy", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(thetas, log_map_error, color="tab:blue", label="log-map roundtrip error")
    ax2.set_yscale("log")
    ax2.set_ylabel("log-map roundtrip error", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Fixed-chart conditioning vs. log-map roundtrip error near θ=π")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log = run_near_singular_demo()

    csv_path = os.path.join(RESULTS_DIR, "run_log.csv")
    json_path = os.path.join(RESULTS_DIR, "run_summary.json")
    write_csv(log, csv_path)
    summary = write_summary(log, json_path)

    plot_tracking_error(log, os.path.join(RESULTS_DIR, "tracking_error.png"))
    plot_manipulability_and_damping(log, os.path.join(RESULTS_DIR, "manipulability_and_damping.png"))
    plot_chart_comparison(os.path.join(RESULTS_DIR, "chart_comparison.png"))

    print("\nWrote:")
    for fname in ["run_log.csv", "run_summary.json", "tracking_error.png",
                  "manipulability_and_damping.png", "chart_comparison.png"]:
        print(f"  results/{fname}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
