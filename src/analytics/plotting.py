"""
plotting.py
Publication-quality analytical charts for presentation slides and technical reports.
Visualizes throughput, queue delay distributions, Pareto frontiers, and network resilience.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300
})

PALETTE = {
    "FIFO (Centralized)": "#E74C3C",        # Coral Red
    "IDM-TTC (Decentralized)": "#E67E22",   # Orange
    "IPPO (No Comms)": "#95A5A6",           # Gray
    "Comm-MAPPO (Proposed)": "#2ECC71"      # Emerald Green
}


def generate_presentation_plots(results_dir: str = "results"):
    benchmark_csv = os.path.join(results_dir, "benchmark_results.csv")
    packet_csv = os.path.join(results_dir, "packet_loss_sweep.csv")

    if not os.path.exists(benchmark_csv):
        print(f"Error: {benchmark_csv} not found. Run evaluate.py first.")
        return

    df = pd.read_csv(benchmark_csv)

    # -------------------------------------------------------------
    # Plot 1: System Throughput Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=df,
        x="strategy",
        y="throughput_per_1000_steps",
        palette=PALETTE,
        capsize=0.1,
        err_kws={'linewidth': 1.8}
    )
    plt.title("Fleet Bottleneck Throughput by Operational Strategy", pad=15, fontweight="bold")
    plt.xlabel("Operational Control Strategy", fontweight="semibold")
    plt.ylabel("Throughput (Vehicles / 1,000 Steps)", fontweight="semibold")
    plt.xticks(rotation=10)
    plt.tight_layout()
    p1_path = os.path.join(results_dir, "throughput_comparison.png")
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"✓ Generated: {p1_path}")

    # -------------------------------------------------------------
    # Plot 2: Average Queue Waiting Time (Bottleneck Delay)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    ax = sns.boxplot(
        data=df,
        x="strategy",
        y="avg_wait_steps",
        palette=PALETTE,
        boxprops=dict(alpha=0.85),
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    plt.title("Queue Waiting Time Distribution (Bottleneck Stalling)", pad=15, fontweight="bold")
    plt.xlabel("Operational Control Strategy", fontweight="semibold")
    plt.ylabel("Mean Stalled Steps per Vehicle (Speed < 0.5 m/s)", fontweight="semibold")
    plt.xticks(rotation=10)
    plt.tight_layout()
    p2_path = os.path.join(results_dir, "queue_delay_distribution.png")
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"✓ Generated: {p2_path}")

    # -------------------------------------------------------------
    # Plot 3: Pareto Frontier: Average Speed vs. Energy Jerk
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5.5))
    summary_df = df.groupby("strategy").agg({
        "avg_speed_mps": "mean",
        "avg_jerk": "mean"
    }).reset_index()

    for _, row in summary_df.iterrows():
        strat = row["strategy"]
        plt.scatter(
            row["avg_jerk"],
            row["avg_speed_mps"],
            color=PALETTE.get(strat, "#34495E"),
            s=220,
            edgecolors="black",
            linewidth=1.5,
            label=strat,
            zorder=5
        )
        plt.annotate(
            strat.split()[0],
            (row["avg_jerk"] + 0.3, row["avg_speed_mps"] + 0.05),
            fontsize=10,
            fontweight="bold"
        )

    plt.title("Pareto Efficiency: Fleet Flow Speed vs. Energy Jerk Dissipation", pad=15, fontweight="bold")
    plt.xlabel("Kinetic Jerk Dissipation Metric (Battery & Brake Wear)", fontweight="semibold")
    plt.ylabel("Average Fleet Speed (m/s)", fontweight="semibold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()
    p3_path = os.path.join(results_dir, "pareto_speed_vs_jerk.png")
    plt.savefig(p3_path, dpi=300)
    plt.close()
    print(f"✓ Generated: {p3_path}")

    # -------------------------------------------------------------
    # Plot 4: Cyber-Physical Resilience: Packet Loss Sweep
    # -------------------------------------------------------------
    if os.path.exists(packet_csv):
        df_loss = pd.read_csv(packet_csv)
        plt.figure(figsize=(8, 5))
        plt.plot(
            df_loss["packet_drop_rate"] * 100,
            df_loss["throughput_per_1000"],
            marker="o",
            color="#2ECC71",
            linewidth=2.5,
            markersize=8,
            label="Comm-MAPPO (Proposed)"
        )
        # Reference baseline: IPPO (no communication)
        ippo_tput = df[df["strategy"] == "IPPO (No Comms)"]["throughput_per_1000_steps"].mean()
        plt.axhline(
            y=ippo_tput,
            color="#95A5A6",
            linestyle="--",
            linewidth=2.0,
            label="Non-Cooperative Baseline (IPPO)"
        )

        plt.title("Cyber-Physical Network Resilience Under Wireless Packet Drop", pad=15, fontweight="bold")
        plt.xlabel("Simulated V2V Wireless Packet Drop Rate (%)", fontweight="semibold")
        plt.ylabel("Throughput (Vehicles / 1,000 Steps)", fontweight="semibold")
        plt.ylim(bottom=0)
        plt.legend(frameon=True, facecolor="white")
        plt.tight_layout()
        p4_path = os.path.join(results_dir, "packet_loss_resilience.png")
        plt.savefig(p4_path, dpi=300)
        plt.close()
        print(f"✓ Generated: {p4_path}")


if __name__ == "__main__":
    generate_presentation_plots()
