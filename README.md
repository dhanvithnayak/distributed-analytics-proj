# Distributed Automated Driving System for Industrial Logistics Networks

> **Course:** Distributed Analytics for Industrial Automation  
> **Department:** Industrial and Systems Engineering (ISE)  
> **Author:** dhanvithnayak  
> **Methodology:** Multi-Agent Reinforcement Learning (Comm-MAPPO) with Dynamic V2V Communication Graphs

---

## 📌 Project Overview
In industrial automation and cyber-physical supply chains, bottleneck nodes (e.g., automated container terminals, intermodal freight yards, and unsignalized factory intersections) suffer severe throughput loss, queue delays, and deadlocks when relying on either rigid centralized controllers or greedy decentralized heuristics.

This project implements a **Distributed Automated Driving System** for Connected and Autonomous Vehicles (CAVs) modeled as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**. The vehicles coordinate their velocity profiles through an attention-based Vehicle-to-Vehicle (V2V) communication graph to achieve:
1. **Bottleneck Throughput Maximization**
2. **Zero-Collision Safety Guarantees**
3. **Energy & Jerk Minimization** (reducing EV battery wear and stop-and-go waves)
4. **Cyber-Physical Network Resilience** (robustness against packet loss and latency)

---

## 🏗️ System Architecture & Formulation
* **Centralized Training with Decentralized Execution (CTDE):** Agents train with a centralized critic that observes the global fleet state, while execution is fully decentralized on onboard edge Electronic Control Units (ECUs).
* **Graph-Attentive V2V Communication:** Dynamic message passing over local Euclidean neighbor graphs $\mathcal{G}(t) = (\mathcal{V}, \mathcal{E}(t))$.
* **Cyber-Physical Disturbance Layer:** Simulates real-world V2V channel conditions including packet drop rates ($p_{\text{loss}} \in [0.0, 0.5]$) and transmission delays.

---

## 📊 Comparative Baselines
1. **FIFO / Virtual Traffic Light:** Centralized rule-based reservation baseline.
2. **IDM + Time-To-Collision (TTC):** Decentralized car-following and cross-traffic yield heuristic.
3. **Independent PPO (IPPO):** Non-cooperative MARL without V2V communication.
4. **Comm-MAPPO (Proposed):** Cooperative MARL with spatial attention message passing.

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Sanity Check & Physics Verification
```bash
PYTHONPATH=. SDL_VIDEODRIVER=dummy python3 tests/test_env.py
```
This will run 50 simulation steps and generate an animation in `assets/`.
