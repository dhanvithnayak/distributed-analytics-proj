# Distributed Automated Driving System for Industrial Logistics Networks

> **Course:** Distributed Analytics for Industrial Automation  
> **Department:** Industrial and Systems Engineering (ISE)  
> **Author:** dhanvithnayak  
> **Methodology:** Multi-Agent Reinforcement Learning (Comm-MAPPO) with Dynamic V2V Communication Graphs

---

## 📌 Executive Summary
In industrial automation and cyber-physical supply chains, critical bottleneck nodes (such as automated container terminals, intermodal freight yards, and unsignalized factory intersections) suffer severe throughput loss, queue delays, and deadlocks when relying on either rigid centralized controllers or greedy decentralized heuristics.

This project implements a **Distributed Automated Driving System** for Connected and Autonomous Vehicles (CAVs) modeled as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**. By leveraging an onboard **Spatial Multi-Head Graph Attention (GAT)** communication layer, agents collaboratively coordinate velocity profiles over an ad-hoc Vehicle-to-Vehicle (V2V) network $\mathcal{G}(t) = (\mathcal{V}, \mathcal{E}(t))$.

---

## 📊 Benchmark Results (20 Test Episodes)

| Strategy | Paradigm | Throughput (Veh / 1k steps) | Mean Queue Stalling (steps) | Mean Fleet Speed (m/s) | Kinetic Jerk Index |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **FIFO (Virtual Traffic Light)** | Centralized Rule-Based | $13.88 \pm 1.62$ | $20.10 \pm 1.25$ | $6.18 \pm 0.32$ | $4,345.1$ |
| **IDM + Time-To-Collision (TTC)**| Decentralized Rule-Based| $14.63 \pm 9.22$ | $13.29 \pm 2.68$ | $5.91 \pm 0.44$ | $9,775.6$ |
| **IPPO (Independent PPO)** | Non-Cooperative MARL | $45.88 \pm 0.17$ | $0.00 \pm 0.00$ | $11.30 \pm 0.12$ | $1,866.9$ |
| **Comm-MAPPO (Proposed)** | **Distributed Cooperative MARL** | **$48.00 \pm 0.05$** | **$0.00 \pm 0.00$** | **$11.35 \pm 0.08$** | **$1,111.5$** |

### Key Findings:
1. **+245% Higher Throughput** over centralized FIFO reservation by eliminating stop-and-go bottleneck oscillations.
2. **74% Jerk Reduction** over FIFO and **88% Jerk Reduction** over reactive IDM-TTC, preserving electric vehicle battery health and reducing mechanical wear.
3. **Cyber-Physical Network Resilience:** Retains $>95\%$ peak throughput under $20\%$ stochastic wireless packet loss, proving graceful degradation.

---

## 📂 Project Repository Structure
```text
dist_analytics_proj/
├── assets/
│   ├── demo_comm_mappo.gif          # Cooperative zipper merge with glowing V2V links
│   └── demo_baseline_fifo.gif       # Stop-and-go queue build-up under FIFO
├── docs/
│   └── presentation_deck.md         # Full 13-slide deck with speaker notes
├── models/
│   ├── best_comm_mappo.pt           # Pre-trained Comm-MAPPO weights (GPU/CPU)
│   └── best_ippo.pt                 # Pre-trained IPPO ablation weights
├── results/
│   ├── throughput_comparison.png    # Throughput bar chart across all 4 strategies
│   ├── queue_delay_distribution.png # Boxplot of queue waiting times
│   ├── pareto_speed_vs_jerk.png     # Pareto frontier of flow speed vs jerk
│   └── packet_loss_resilience.png   # Throughput retention vs packet drop rate
├── src/
│   ├── env/
│   │   ├── kinematics.py            # Continuous 2D kinematics & collision checking
│   │   ├── intersection_env.py      # Dec-POMDP industrial intersection & cyber-physical channel
│   │   └── renderer.py              # 2D visualizer with HUD and frame exporter
│   ├── baselines/
│   │   ├── fifo_controller.py       # Centralized FIFO reservation controller
│   │   ├── idm_ttc_agent.py         # Decentralized IDM + Time-To-Collision heuristic
│   │   └── ippo_agent.py            # Non-cooperative MARL baseline
│   ├── marl/
│   │   ├── comm_layer.py            # Spatial Multi-Head Graph Attention (GAT) layer
│   │   └── mappo.py                 # Multi-Agent PPO with CTDE
│   ├── analytics/
│   │   ├── metrics.py               # Fleet metrics evaluator
│   │   └── plotting.py              # Publication chart generator
│   ├── train.py                     # Training pipeline
│   ├── evaluate.py                  # Evaluation benchmark & packet loss sweep
│   └── demo.py                      # Animated demonstration generator
└── tests/
    ├── test_env.py                  # Environment sanity test
    └── test_baselines_and_marl.py   # Baseline & MARL integration tests
```

---

## 🚀 Reproduction Instructions

### 1. Re-run Evaluation Benchmark
```bash
PYTHONPATH=. python3 src/evaluate.py --episodes 20
```

### 2. Regenerate Presentation Charts
```bash
PYTHONPATH=. python3 src/analytics/plotting.py
```

### 3. Record Simulation Demos
```bash
PYTHONPATH=. SDL_VIDEODRIVER=dummy python3 src/demo.py
```

### 4. Presentation Slide Deck
The complete presentation deck with speaker notes is available at:  
👉 [`docs/presentation_deck.md`](docs/presentation_deck.md)
