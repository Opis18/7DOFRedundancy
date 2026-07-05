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
from scipy.spatial.transform import Rotation

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
    return log, traj, iiwa7


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


def write_joint_csv(log, path):
    n = log["q_history"].shape[1]
    header = "t," + ",".join(f"q{i+1}" for i in range(n))
    with open(path, "w") as f:
        f.write(header + "\n")
        for k in range(len(log["t_history"])):
            row = ",".join(f"{log['q_history'][k, i]:.8f}" for i in range(n))
            f.write(f"{log['t_history'][k]:.6f},{row}\n")


def plot_joint_angles(log, joint_limits, path):
    n = log["q_history"].shape[1]
    t = log["t_history"]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i in range(n):
        ax.plot(t, np.degrees(log["q_history"][:, i]), color=colors[i], label=f"q{i+1}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint angle (deg)")
    ax.set_title("Joint angles q1..q7 over the run, iiwa7_r800")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_ee_pose_history(log, traj, S_list, M):
    """
    Recompute the desired pose g_des(t) (from the trajectory function -- it
    is NOT stored by run_control_loop, only the pose-error twist is) and
    the actual pose T_sb(q(t)) (via forward_kinematics on the logged joint
    history) at every timestep, for direct position/orientation comparison.

    Returns
    -------
    dict with keys:
        actual_pos, desired_pos   : (num_steps+1, 3)
        actual_R, desired_R       : (num_steps+1, 3, 3)
    """
    t_history = log["t_history"]
    q_history = log["q_history"]
    n_steps = len(t_history)

    actual_pos = np.zeros((n_steps, 3))
    desired_pos = np.zeros((n_steps, 3))
    actual_R = np.zeros((n_steps, 3, 3))
    desired_R = np.zeros((n_steps, 3, 3))

    for k in range(n_steps):
        T_actual = forward_kinematics(q_history[k], S_list, M)
        actual_pos[k] = T_actual[:3, 3]
        actual_R[k] = T_actual[:3, :3]

        g_des, _ = traj(t_history[k])
        g_des = np.asarray(g_des, dtype=float)
        desired_pos[k] = g_des[:3, 3]
        desired_R[k] = g_des[:3, :3]

    return {
        "actual_pos": actual_pos, "desired_pos": desired_pos,
        "actual_R": actual_R, "desired_R": desired_R,
    }


def plot_position_tracking(t_history, pose_hist, path):
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for i, label in enumerate(labels):
        axes[i].plot(t_history, pose_hist["desired_pos"][:, i],
                     "--", color="tab:gray", label="desired (g_des)", linewidth=1.5)
        axes[i].plot(t_history, pose_hist["actual_pos"][:, i],
                     "-", color="tab:blue", label="actual (FK of q(t))", linewidth=1.5)
        axes[i].set_ylabel(f"{label} (m)")
        axes[i].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("End-effector position: desired vs. actual (iiwa7_r800)")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_orientation_tracking(t_history, pose_hist, path):
    # NOTE: Euler angles are used here ONLY for human-readable plotting.
    # They are not part of the control law anywhere in this project (which
    # uses the log map specifically to avoid fixed-chart representations)
    # and they carry their own well-known singularity (gimbal lock) at
    # pitch = +/-90 deg -- a different chart, a different singular locus,
    # but a singular locus nonetheless. Don't read anything into an
    # apparent jump here without checking it against xi_err_history first.
    actual_euler = Rotation.from_matrix(pose_hist["actual_R"]).as_euler("xyz", degrees=True)
    desired_euler = Rotation.from_matrix(pose_hist["desired_R"]).as_euler("xyz", degrees=True)

    labels = ["roll (x)", "pitch (y)", "yaw (z)"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for i, label in enumerate(labels):
        axes[i].plot(t_history, desired_euler[:, i],
                     "--", color="tab:gray", label="desired (g_des)", linewidth=1.5)
        axes[i].plot(t_history, actual_euler[:, i],
                     "-", color="tab:red", label="actual (FK of q(t))", linewidth=1.5)
        axes[i].set_ylabel(f"{label} (deg)")
        axes[i].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("End-effector orientation (Euler xyz, viz-only): desired vs. actual")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()



def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log, traj, iiwa7 = run_near_singular_demo()

    csv_path = os.path.join(RESULTS_DIR, "run_log.csv")
    json_path = os.path.join(RESULTS_DIR, "run_summary.json")
    joint_csv_path = os.path.join(RESULTS_DIR, "joint_angles.csv")
    write_csv(log, csv_path)
    summary = write_summary(log, json_path)
    write_joint_csv(log, joint_csv_path)

    plot_tracking_error(log, os.path.join(RESULTS_DIR, "tracking_error.png"))
    plot_manipulability_and_damping(log, os.path.join(RESULTS_DIR, "manipulability_and_damping.png"))
    plot_chart_comparison(os.path.join(RESULTS_DIR, "chart_comparison.png"))
    plot_joint_angles(log, iiwa7["joint_limits"], os.path.join(RESULTS_DIR, "joint_angles.png"))

    pose_hist = compute_ee_pose_history(log, traj, iiwa7["S_list"], iiwa7["M"])
    plot_position_tracking(log["t_history"], pose_hist, os.path.join(RESULTS_DIR, "position_tracking.png"))
    plot_orientation_tracking(log["t_history"], pose_hist, os.path.join(RESULTS_DIR, "orientation_tracking.png"))

    print("\nWrote:")
    for fname in ["run_log.csv", "run_summary.json", "joint_angles.csv", "tracking_error.png",
                  "manipulability_and_damping.png", "chart_comparison.png", "joint_angles.png",
                  "position_tracking.png", "orientation_tracking.png"]:
        print(f"  results/{fname}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
