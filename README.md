# Execution-Aware VRP Solver and Benchmark Studio

> A production-grade Vehicle Routing Problem solver and live comparison studio with **8 solver backends**, **18 operational scenarios**, and an **Agentic AI layer** powered by Google Gemini 2.0 Flash — all from a single Streamlit web app.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io)
[![OR-Tools](https://img.shields.io/badge/OR--Tools-9.x-green)](https://developers.google.com/optimization)
[![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-blue)](https://aistudio.google.com)

---

## Table of Contents

- [Overview](#overview)
- [What Makes This Different](#what-makes-this-different)
- [Benchmark Results](#benchmark-results)
- [Agentic AI Layer](#agentic-ai-layer)
- [8 Solver Backends](#8-solver-backends)
- [18 Test Scenarios](#18-test-scenarios)
- [Project Structure](#project-structure)
- [Input Data Schema](#input-data-schema)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [AI Agent Setup](#ai-agent-setup)
- [Streamlit UI Tabs](#streamlit-ui-tabs)
- [BackendConfig Reference](#backendconfig-reference)
- [Saving and Loading Artifacts](#saving-and-loading-artifacts)
- [Dataset](#dataset)
- [Known Limitations](#known-limitations)
- [Architecture](#architecture)
- [Citation](#citation)

---

## Overview

Field-service routing is not a one-time optimisation problem. Technicians skip jobs, shifts change, urgent orders arrive mid-day, and customers cancel — all while routes are already in execution. This studio addresses that reality with **execution-aware scheduling**: solvers that know which stops are committed, lock them in place, and re-optimise only the open work.


## Project Links

Explore the project here:

- **GitHub:** [Execution-Aware-Agentic-VRP-Solver-and-Benchmark-Studio](https://github.com/dranubhaparashar/Execution-Aware-Agentic-VRP-Solver-and-Benchmark-Studio/tree/main)
- **Hugging Face Demo:** [Execution-Aware-Agentic-VRP-Solver-and-Benchmark-Studio](https://huggingface.co/spaces/AnubhaParashar/Execution-Aware-Agentic-VRP-Solver-and-Benchmark-Studio)
- **YouTube Demo:** [Watch here](https://www.youtube.com/watch?v=FqVuVjW20yo)
  
**What you can do with this app:**

- Upload your own orders, routes, depots, and goal profiles
- Run any combination of 8 solver backends in parallel
- Stress-test across 18 real-world operational scenarios
- Compare distance, lateness, overtime, assignment rate, and runtime side by side
- Ask the AI agent plain-English questions about your results
- Let the agent autonomously detect violations and trigger re-optimisation
- Export everything to multi-sheet Excel with interactive route maps

---

## What Makes This Different

### Execution-Aware Split-Solve Architecture

The OR-Tools Execution-Aware backend uses a novel split-solve approach that **guarantees 100% assignment with zero violations**:

```
Standard approach (fails):        Split-solve (OR-Tools EA):
Submit all 150 orders              Submit only ~10 open orders
130 with pinned timestamps    vs   Locked orders copied from baseline
PATH_CHEAPEST_ARC fails            Small problem = deeper GLS search
0 orders assigned                  150/150 assigned, 0 violations
```

Steps:
1. **Separate** — locked (in-progress) orders never enter the model
2. **Anchor** — each vehicle starts at its last locked stop position and time
3. **Solve** — OR-Tools GLS on open orders only with full time budget
4. **Merge** — locked stops verbatim from baseline + OR-Tools open stops
5. **Fallback** — greedy catch for any dropped orders (guarantees 100%)

### Agentic AI Orchestration

The app includes a genuine AI agent (not just a chatbot) powered by Google Gemini 2.0 Flash with function-calling. The agent autonomously chains tool calls to answer questions, detect problems, and trigger re-optimisation without human intervention.

<img width="1911" height="906" alt="image" src="https://github.com/user-attachments/assets/4dab3aae-89ee-407b-8050-6c53913dc1b3" />

---

## Benchmark Results

**Dataset:** 40,000 orders · 400 routes · 8 depots · 4 goal profiles
**Solve scope:** 150 orders / 20 routes per run · 30 km/h haversine
**Scenarios:** 18 (BASE + EAS 1-4 + RT 1-7 + GB 1-6)

### Summary — mean across all 18 scenarios

| Backend | Assigned | Routes | BASE km | EAS km | RT km | GB km | Late min | OT min | Runtime s |
|---|---|---|---|---|---|---|---|---|---|
| **OR-Tools EA** | **150** | 19 | **232** | **233** | **232** | **240** | **0** | **0** | **770** |
| OR-Tools | 150 | 19 | 233 | 233 | 235 | 243 | 0 | 0 | 868 |
| Esri | 142 | 50 | 239 | 239 | 206 | 240 | 2 | 0 | 973 |
| Adaptive | 150 | 20 | 424 | 350 | 364 | 342 | 0 | 0 | 6,472 |
| Hybrid | 150 | 22 | 424 | 465 | 462 | 356 | 0 | 0 | 1,790 |
| Greedy | 150 | 21 | 394 | 387 | 387 | 391 | 12,837 | 2,566 | 22 |
| PyVRP | 150 | 5 | 114 | 112 | 113 | 118 | 7,471 | 3,923 | 545 |
| OSRM | 150 | 15 | 519 | 523 | 519 | 510 | 13,657 | 2,904 | 4,949 |

> PyVRP and OSRM ignore time windows. Their low/high distance numbers come from constraint violations — not suitable for production use.

### OR-Tools EA vs OR-Tools — head to head

| Metric | OR-Tools | OR-Tools EA | Improvement |
|---|---|---|---|
| BASE distance | 233.2 km | 231.8 km | -1.4 km |
| EAS avg distance | 233.4 km | 232.8 km | -0.6 km |
| RT avg distance | 234.6 km | 232.1 km | -2.5 km |
| GB avg distance | 243.2 km | 239.8 km | -3.4 km |
| Late minutes | 0 | 0 | same |
| Overtime minutes | 0 | 0 | same |
| Total runtime | 868 s | 770 s | 11% faster |

---

## Agentic AI Layer

The `🤖 AI Agent` tab in Streamlit uses `VRPOrchestrator` in `vrp_agent.py` — a Google Gemini 2.0 Flash agent with 8 callable tools that it chains autonomously.

### 4 Agentic Behaviors

**1. Natural language interface**
Ask in plain English — the agent picks the right tools, runs them, and explains the results.
```
"Which backend should I use if SLA adherence is my top priority?"
"Why is OR-Tools EA better than plain OR-Tools for real-time scenarios?"
```

**2. Autonomous re-optimization**
Detects violations, figures out the fix, re-runs, and returns a before/after comparison — all without human input.
```
"Fix any violations in the Greedy backend automatically"
"Run an improvement cycle on OR-Tools EA"
```

**3. Multi-step reasoning**
Chains multiple tool calls to answer complex comparative questions.
```
"Run OR-Tools EA and Adaptive back to back and tell me which is better"
"Compare all backends for the RT_6 scenario and recommend one"
```

**4. Real-time reactive**
One-click event buttons trigger the agent to handle operational disruptions autonomously.
- Technician sick (RT_6)
- Urgent order injected (RT_1)
- Order cancellation (RT_2)
- Shift change (RT_4)

### 8 Agent Tools

| Tool | What it does |
|---|---|
| `get_scenario_list` | Returns all 18 scenario IDs and descriptions |
| `explain_scenario` | Explains what a scenario tests and what mutation it applies |
| `analyze_results` | Reads loaded bundles — distance, late, OT, routes, runtime |
| `detect_violations` | Finds every late order and overtime route with amounts |
| `compare_backends` | Ranks all run backends by chosen priority criterion |
| `suggest_config` | Recommends specific BackendConfig changes with reasoning |
| `run_backend` | Triggers a live solve and returns summary metrics |
| `trigger_reoptimization` | Re-runs with improved config, returns before/after diff |

---

## 8 Solver Backends

| Backend | Function | Technique | Runtime | Violations |
|---|---|---|---|---|
| **OR-Tools EA** | `run_ortools_execution_aware_backend` | Split-solve GLS | 770 s | None |
| OR-Tools | `run_ortools_backend` | CVRPTW GLS | 868 s | None |
| Esri | `run_esri_backend` | ArcGIS Network Analyst | 973 s | Rare |
| Adaptive | `run_adaptive_execution_aware_metaheuristic_backend` | Multi-candidate regret | 6,472 s | None |
| Hybrid | `run_hybrid_execution_aware_backend` | Regret + rolling repair | 1,790 s | None |
| Greedy | `run_current_backend` | Greedy + 2-opt | 22 s | High |
| PyVRP | `run_pyvrp_backend` | Metaheuristic VRP | 545 s | High |
| OSRM | `run_osrm_backend` | Road-routed greedy | 4,949 s | High |

---

## 18 Test Scenarios

### Execution-Aware Scheduling (EAS)

| ID | Description | Mutation applied |
|---|---|---|
| BASE | Baseline solve | None |
| EAS_1 | Job starts earlier or later | TimeWindow +60 min on top order |
| EAS_2 | Job finishes earlier or later | ServiceTime +40 min on top order |
| EAS_3 | Technician skips job | PreferredDepot changed, Priority raised to 5 |
| EAS_4 | Auto re-optimise downstream | ServiceTime +30 on order 1, TimeWindow +45 on order 2 |

### Real-Time Scheduling (RT)

| ID | Description | Mutation applied |
|---|---|---|
| RT_1 | New urgent activity | Inject order: Priority 5, SLAFlag 1, 2-hour window |
| RT_2 | Cancellation | Drop top-priority order |
| RT_3 | Duration changes | ServiceTime x1.35 on 3 orders |
| RT_4 | Technician shift changes | EarliestStart +90 min, MaxOrderCount -6 |
| RT_5 | Location hours changes | TimeWindowEnd = Start +90 min on 3 orders |
| RT_6 | Technician calls in sick | Drop top-rated route entirely |
| RT_7 | All EAS in real-time flow | Combine EAS_2 + EAS_1 + RT_4 |

### Goal-Based Scheduling (GB)

| ID | Description | Weight override |
|---|---|---|
| GB_1 | Minimise travel time | travel=2.2, rating=0.1 |
| GB_2 | Prefer higher-rated technicians | rating=2.2, travel=0.45 |
| GB_3 | Prefer specialised technicians | skill=2.6, travel=0.5 |
| GB_4 | Maximise SLA adherence | sla=2.8, overtime=0.9 |
| GB_5 | Reduce overtime | overtime=3.0, sla=1.1 |
| GB_6 | Customer-defined weights | Reads from goal_profiles.csv |

---

## Project Structure

```
.
├── app.py                    # Streamlit UI — 6 tabs, sidebar, artifact IO
├── base_engine.py            # Core domain — agents, haversine, scenarios, Folium maps
├── compare_backends.py       # All 8 backends + shared helpers + Excel export
├── vrp_agent.py              # Agentic AI layer — Gemini 2.0 Flash + 8 tools
├── ARCHITECTURE.md           # Full system architecture with Mermaid diagrams
├── benchmark_comparison_chart.html  # Standalone benchmark results chart
├── requirements.txt
├── synthetic_data/
│   ├── orders_40000.csv
│   ├── routes_400.csv
│   ├── depots_8.csv
│   ├── accounts.csv
│   └── goal_profiles.csv
└── notebook_artifacts/       # Auto-created on first save
    └── {backend_name}/
        ├── vrp_scenario_results_full.xlsx
        ├── run_meta.json
        ├── maps_html/
        └── maps_png/
```

---

## Input Data Schema

### orders_*.csv

| Column | Type | Notes |
|---|---|---|
| Name | string | Becomes OrderId if no OrderId column |
| Latitude, Longitude | float | WGS84. Null rows dropped. |
| ServiceTime | float | Minutes. Default 45. |
| TimeWindowStart1 | datetime | Earliest service start |
| TimeWindowEnd1 | datetime | Latest service start |
| Priority | int 1-5 | 5 = highest urgency. Default 3. |
| SLAFlag | int 0/1 | 1 = SLA-critical |
| PreferredDepot | string | Match to depots.Name |
| SpecialtyNames | string | Comma-separated skill tags |
| MaxViolationTime1 | float | Allowed lateness beyond TimeWindowEnd1 (min) |
| AccountId | string | Joined to accounts.csv for GoalProfile |
| WorkType | string | Fallback for skill matching |

### routes_*.csv

| Column | Type | Notes |
|---|---|---|
| Name | string | Unique technician/vehicle ID |
| StartDepotName | string | Match to depots.Name |
| EarliestStartTime | datetime | Shift start time |
| LatestStartTime | datetime | Latest allowed dispatch |
| MaxOrderCount | int | Route capacity. Default 40. |
| OvertimeStartTime | float | Minutes from shift start before overtime |
| MaxTotalTime | float | Hard shift limit in minutes. Default 720. |
| TechnicianRating | float 1-5 | Quality score |
| SpecialtyNames | string | Comma-separated skill tags |

### depots_*.csv

| Column | Type |
|---|---|
| Name | string (must match routes.StartDepotName) |
| Latitude, Longitude | float |

### accounts.csv

| Column | Type |
|---|---|
| AccountId | string |
| GoalProfile | string (matches goal_profiles.GoalProfile) |

### goal_profiles.csv

| Column | Type | Notes |
|---|---|---|
| GoalProfile | string | Key matched to orders |
| minimize_travel_time | float | Becomes travel weight |
| prefer_higher_rated_technicians | float | Becomes rating weight |
| maximize_sla_adherence | float | Becomes sla weight |
| reduce_overtime | float | Becomes overtime weight |

---

## Installation

```bash
# Clone
git clone https://github.com/dranubhaparashar/Execution-Aware-VRP-Solver-and-Benchmark-Studio.git
cd Execution-Aware-VRP-Solver-and-Benchmark-Studio

# Virtual environment
python -m venv vrp
source vrp/bin/activate        # Linux / macOS / WSL
# vrp\Scripts\activate         # Windows CMD

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
streamlit>=1.28
pandas>=2.0
numpy>=1.24
ortools>=9.7
pyvrp>=0.6
folium>=0.15
streamlit-folium>=0.15
plotly>=5.18
openpyxl>=3.1
requests>=2.31
```

---

## Running the App

```bash
streamlit run app.py
# or on a custom port
streamlit run app.py --server.port 8504
```

Open `http://localhost:8504` in your browser.

> **WSL users:** The `gio: Operation not supported` message is cosmetic — the app is running fine. Use the Network URL shown in the terminal (e.g. `http://172.x.x.x:8504`) to open it in your Windows browser.

---

## AI Agent Setup

The AI Agent tab requires a free Google Gemini API key.

**Step 1 — Get a free key**

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **Create API key** — free tier, no credit card needed
4. Copy the key (starts with `AIza...`)

**Step 2 — Install the package**

```bash
pip install google-generativeai
```

**Step 3 — Set the key**

```bash
# Temporary (current terminal session only)
export GOOGLE_API_KEY="AIzaSy...your-key-here"

# Permanent (survives restarts)
echo 'export GOOGLE_API_KEY="AIzaSy...your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Free tier limits:** 1,500 requests/day · 15 RPM · No credit card

Each agent conversation uses roughly $0.001-0.003 worth of quota — the free tier supports weeks of normal testing.

---

## Streamlit UI Tabs

| Tab | Description |
|---|---|
| Overview | Backend run count, scenario counts, uploaded dataset metrics, full scenario_summary table |
| KPI Comparison | Scenario selector + 6 Plotly bar charts: AssignedOrders, RoutesUsed, TotalDistanceKm, TotalTravelMinutes, LateMinutes, RuntimeSeconds |
| Map Explorer | Interactive Folium map — stop markers per route, polylines in sequence, OrderId/late popup, depot icons |
| Scenario Tables | Backend + scenario selectors — Summary, Actions, Routes, Stops DataFrames |
| Export | Download multi-sheet XLSX (one sheet group per backend) + run_meta JSON |
| AI Agent | Chat interface, quick action buttons, reactive event buttons, tool calls log |

---

## BackendConfig Reference

All parameters are set in the Streamlit sidebar and passed as a `BackendConfig` dataclass.

### Core

| Parameter | Default | Description |
|---|---|---|
| speed_kmph | 35.0 | Haversine fallback speed (km/h) |
| use_two_opt | True | Enable 2-opt in Greedy and OSRM |
| strict_skill_bias | True | Hard-bias toward skill-matched routes |
| random_seed | 42 | Seed for all random sampling |
| max_orders | 350 | Orders in solve scope |
| max_routes | 24 | Routes in solve scope |
| depot_focus | None | Restrict scope to a named depot |
| sample_by_priority | True | Bias subset toward SLA/priority orders |

### Hybrid / Adaptive

| Parameter | Default | Description |
|---|---|---|
| hybrid_candidate_routes | 10 | Candidate routes per order in insertion |
| hybrid_frontier_size | 28 | Regret frontier width |
| hybrid_local_search_passes | 4 | Local search passes per route |
| hybrid_max_hard_late_minutes | 30.0 | Hard lateness cap |
| hybrid_route_activation_penalty | 42.0 | Cost for opening a new route |
| hybrid_compact_passes | 3 | Route compaction passes |
| hybrid_global_relocate_passes | 3 | Cross-route relocate passes |
| hybrid_global_swap_passes | 2 | Cross-route swap passes |
| hybrid_require_zero_late_overtime | True | Reject any-violation solutions |

### OSRM / PyVRP / OR-Tools / Esri

| Parameter | Default | Description |
|---|---|---|
| osrm_base_url | router.project-osrm.org | OSRM server URL |
| osrm_auto_reduce | True | Auto-clamp scope on public demo server |
| osrm_demo_max_orders | 30 | Max orders on public OSRM |
| pyvrp_max_runtime_seconds | 5 | PyVRP time budget per scenario |
| pyvrp_allow_drop_orders | False | Allow dropped orders in PyVRP |
| ortools_max_runtime_seconds | 5 | OR-Tools GLS budget per scenario |
| ortools_allow_drop_orders | True | Allow dropped orders (greedy fallback catches them) |
| esri_portal_url | "" | ArcGIS portal URL |
| esri_travel_mode | Driving Time | ArcGIS Network travel mode |

---

## Saving and Loading Artifacts

When **Save each live backend run** is enabled in the sidebar, each run saves to:

```
notebook_artifacts/
└── {backend_name}/
    ├── vrp_scenario_results_full.xlsx   # All 8 output sheets
    ├── run_meta.json                    # Config + paths
    ├── maps_html/
    │   ├── BASE__all_routes.html
    │   └── {SCENARIO}__{RouteName}.html
    └── maps_png/
        └── {SCENARIO}__all_routes.png
```

**Load a prior run:** Sidebar > Load saved workbook from notebook_artifacts for that backend.

**Upload a pre-computed Esri workbook:** Sidebar > Upload already-inferenced Esri workbook.

---

## Dataset

Synthetic dataset included in `synthetic_data/`:

| File | Rows | Description |
|---|---|---|
| orders_40000.csv | 40,000 | Work orders with location, time window, priority, skills, SLA |
| routes_400.csv | 400 | Technician routes with depot, shift, capacity, rating, skills |
| depots_8.csv | 8 | Depot locations centred on Richardson, TX area |
| accounts.csv | 120 | Customer accounts with GoalProfile assignments |
| goal_profiles.csv | 4 | TRAVEL_FIRST, SLA_FIRST, SKILL_FIRST, OVERTIME_FIRST |

---

## Known Limitations

| Issue | Backend | Notes |
|---|---|---|
| Ignores time windows | PyVRP, OSRM | Distance numbers are not production-valid |
| OSRM public demo rate-limited | OSRM | Use self-hosted OSRM for production |
| Requires ArcGIS licence | Esri | Commercial dependency, no workaround |
| Very slow (6,472 s total) | Adaptive | Trade-off for zero-violation quality |
| Drops ~8 orders on average | Esri | ArcGIS solver constraint |

---

## Architecture

Full system architecture and technical details:

- [Complete Technical Specification](https://github.com/dranubhaparashar/Execution-Aware-Agentic-VRP-Solver-and-Benchmark-Studio/wiki/Execution%E2%80%90Aware-VRP-Solver-%E2%80%94-Complete-Technical-Specification)
- [Architecture Overview](https://github.com/dranubhaparashar/Execution-Aware-Agentic-VRP-Solver-and-Benchmark-Studio/wiki/Architecture-%E2%80%94-Execution%E2%80%90Aware--Agentic-VRP-Solver-and-Benchmark-Studio)

Covers: data flow, all 5 agents, 8 backends, OR-Tools EA split-solve deep dive, locked-prefix mechanism, Gemini agent loop, 8 tools, 4 agentic behaviors, output bundle, Streamlit tabs, policy weights, BackendConfig, benchmark results, and deployment.

---

## Citation

```bibtex
@software{parashar2026vrp,
  title  = {Execution-Aware VRP Solver and Benchmark Studio},
  author = {Parashar, Anubha},
  year   = {2026},
  url    = {https://github.com/dranubhaparashar/Execution-Aware-VRP-Solver-and-Benchmark-Studio}
}
```

---


