# Project SWING (Smart Wide-area INtelligent Grid)
**AI-Driven Autonomous Grid Resilience and Adaptive Power Flow Optimization Platform**

Project SWING is an AI-powered operating platform for electrical grid resilience. It combines a high-fidelity power-system Digital Twin, statistical and deep-learning forecasting, a multi-agent LangGraph decision pipeline, a reinforcement-learning-assisted optimization engine, and a Retrieval-Augmented Generation (RAG) compliance layer - all with a human operator kept firmly in the loop.

The Digital Twin is infrastructure, not the project itself. It provides the physical ground truth every AI component reasons against. Nothing bypasses it, and every proposed control action is re-validated against real AC power flow before it is ever shown to an operator.

---

## 1. Architecture & Separation of Concerns

The system is deliberately split into independent services that communicate **only over HTTP**, never via direct Python imports of each other's internals. 

```
┌──────────────────────────┐        HTTP         ┌──────────────────────────────┐
│   DIGITAL TWIN SERVICE   │  <────────────────> │   AI ARCHITECTURE SERVICE    │
│   modules/physics_engine │                     │   modules/ai_agents          │
│   api.py  (port 8001)    │                     │   agent_api.py (port 8002)   │
│                          │                     │                              │
│ - pandapower IEEE-39/118 │                     │ - LangGraph multi-agent      │
│ - Newton-Raphson AC PF   │                     │   pipeline                   │
│ - GRI / security metrics │                     │ - Gemini (fallback chain)    │
│ - N-1 contingency scan   │                     │ - LightGBM + TFT forecasting │
│ - Physics validator/     │                     │ - RL-assisted optimizer      │
│   optimizer endpoints    │                     │   (SAC -> bisection search)  │
│ - WebSocket telemetry    │                     │ - FAISS RAG (IEGC compliance)│
└──────────────────────────┘                     └──────────────────────────────┘
            ▲                                                  ▲
            │ HTTP/WS                                          │ HTTP
            │                                                  │
┌─────────────────────────┐                       ┌─────────────────────────────────┐
│   Digital Twin UI       │                       │   Agent Dashboard UI            │
│   (Digital_Twin_UI/)    │                       │   (Agent_Dashboard_UI/)         │
│   Vanilla HTML/CSS/JS,  │                       │   Agent Copilot, Ask the Grid,  │
│   Cytoscape.js graph    │                       │   RL Status                     │
└─────────────────────────┘                       └─────────────────────────────────┘
Either service can be redeployed independently, pointed at a different counterpart,
or run on a separate machine without touching the other's code.
The only coupling is the DIGITAL_TWIN_API_URL environment variable.
```

## 2. The Multi-Agent Pipeline (LangGraph)

Observer → Forecast → Risk Assessment → RAG Retrieval → Decision → Physics Validator → Operator Brief
              (skipped if risk is LOW/WATCH - routes straight to a no-action or monitor brief)

- *Observer Agent*: Reads the Digital Twin's live Grid Resilience Index (GRI) and security snapshot. If the grid is collapsed or lines are actively overloaded, risk assessment escalates immediately regardless of any forecast.

- *Forecast Agent*: Attempts a multi-horizon Temporal Fusion Transformer (TFT) forecast, falling back to a single-horizon LightGBM quantile model. SHAP feature importance is always computed from the LightGBM model for explainability.

- *Risk Assessment Agent*: Classifies risk (CRITICAL / HIGH / MEDIUM / WATCH / LOW) based on live snapshots and predictive p90 bounds.

- *RAG Retrieval*: Grounds the Decision Agent's justification in real Indian Electricity Grid Code (IEGC) excerpts via FAISS.

- *Decision Agent*: Proposes a strategy (action type and target buses), never the exact MW value.

- *Optimization Engine*: Computes the exact MW amount via an RL policy (SAC) or bisection search fallback.

- *Physics Validator Agent*: Re-tests the proposal against a real AC power flow on a deep copy of the live grid. Nothing is trusted directly.

- *Operator Brief*: The final human-readable recommendation, including SHAP drivers, expected GRI changes, and an explicit Approve/Reject trigger. Nothing is executed without human approval.


## 3. Repository Structure

```

├── Agent_Dashboard_UI/        # AI Copilot, Conversational Chat, and Forecast HTML Dashboards
├── Digital_Twin_UI/           # IEEE-39/118 Cytoscape.js Physics Dashboards & Control Interfaces
├── data/
│   ├── knowledge_base/        # Indian Electricity Grid Code (IEGC) PDF and FAISS Vector Index
│   ├── raw_data/              # hourlyLoadDataIndia.csv, Northern_Region_Weather.csv
│   └── processed_data/        # tft_ready_data.csv (Engineered statistical features)
├── models/                    # Serialized Machine Learning Assets (.ckpt, .zip, .txt, .csv logs)
├── modules/
│   ├── ai_agents/             # LangGraph Multi-Agent Core, API routing, and RAG Builder
│   ├── forecasting/           # LightGBM Quantile Regressors and TFT Predictors
│   ├── optimization_engine/   # Reinforcement Learning Gym Environments and SAC Optimizer
│   └── physics_engine/        # Pandapower Twin, FastAPI Endpoints, and Scenario Stress Engine
│   └── event_store.py         # Shared SQLite Event Timeline logging both Physics and AI actions
├── notebooks/                 # Cloud Training Logic (01_TFT_Multi_Horizon, 02_RL_Optimization)
├── scripts/                   # Local RL Training (train_rl_agent.py) and Model Evaluation metrics
├── main.py                    # Multi-process Development Orchestrator for local deployment
└── requirements.txt           # Explicit Python dependencies and library versions
```


## 4. Key Features & Scoping Decisions

**Digital Twin Engine**
- *IEEE 39 & 118-Bus*: Modeled via pandapower (Newton-Raphson AC power flow).

- *N-1 Contingency Scanner*: Runs asynchronously on a background worker thread (asyncio.to_thread) to prevent event-loop freezing.

- *Idempotent Stress Testing*: Sliders scale load and thermal capacities based on an absolute percentage of the original nameplate baseline, preventing compounding errors.

- *Strict Collapse Detection*: A non-convergent power flow is explicitly caught as collapsed: true (GRI = 0.0), not silently ignored as empty matrices.

**Reinforcement Learning (SAC/PPO)**
- *Grid-Size Specific*: Policies are strictly bound to their observation dimensions. Four independent models are trained and saved: (SHED_LOAD, REDISPATCH_GEN) × (IEEE-39, IEEE-118).

- *Single-Step Episodes*: Deliberately designed as a contextual-bandit-shaped MDP. Outcome quality lives entirely in the reward magnitude (heavily penalizing remaining overloads quadratically), not in episode continuity.

## 5. Setup & Installation

1. Install Core Dependencies

```
pip install -r requirements.txt
```
2. Environment Configuration

Create a .env file at the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```
3. Initialize the Vector Database (RAG)

Ensure your source PDFs are located in data/knowledge_base/, then build the FAISS index:

```
python modules/ai_agents/rag_builder.py
```

6. Execution & Tooling

Launch the Platform (Local Development).

The orchestrator script launches all APIs and UI static servers concurrently:

```
python main.py
```

- *Digital Twin UI*: http://127.0.0.1:3000/home.html

- *Agent Dashboard UI*: http://127.0.0.1:3001/ai_home.html

**Evaluate Metrics**

Run the empirical scoring script to evaluate LightGBM RMSE, SAC convergence rates, and RAG hallucination metrics:

```
python scripts/evaluate_metrics.py
```

**Offline RL Training**

Policies must be trained manually per grid/action type. 

*Note*: For cloud-based execution, use the interactive notebook located at `notebooks/02_RL_Optimization_Cloud_Training.ipynb` .

```
python scripts/train_rl_agent.py --grid 118 --action SHED_LOAD
```

**TFT Multi-Horizon Training**

To train or fine-tune the Temporal Fusion Transformer on cloud GPUs (T4/A100), use the interactive notebook located at `notebooks/01_TFT_Multi_Horizon_Cloud_Training.ipynb`.


## 7. Maintenance & Service Continuation

**LLM Architecture**
The AI Architecture Service utilizes a multi-tiered fallback chain to ensure high availability and gracefully handle API rate limits (HTTP 429) or transient network errors. This routing is handled transparently via LangChain's .with_fallbacks() method.

Current Fallback Chain (As of July 2026):

- *Primary*: gemini-3.6-flash (Latest stable model)

- *Secondary*: gemini-3.5-flash (Previous frontier model)

- *Tertiary*: gemini-3.5-flash-lite (Fastest, cost-effective safety net)

**Maintenance Requirement: Model Deprecations**
Google regularly updates the Gemini model family and retires older endpoints. If an alias in the fallback chain points to a retired endpoint, the LangChain invocation will fail.

- *Monitor*: Check the official Gemini API Deprecations page for upcoming shutdown dates.

- *Update*: Navigate to modules/ai_agents/agent_api.py and update the model="..." strings within the ChatGoogleGenerativeAI declarations to the newest active releases.


## 8. Known Limitations
- *Single-Action Optimization Ceiling*: Both the RL optimizer and the bisection search compute the best single action (one bus, one MW value). At severe overloads, a single action may be mathematically insufficient to clear the constraint. The pipeline correctly reports VALIDATION_EXHAUSTED rather than hallucinating an invalid solution.

- *LOAD_REDISTRIBUTION Limit*: This action shifts load between two discrete buses while determining a continuous MW amount. This hybrid action space does not fit the current RL formulation and relies entirely on the bisection search fallback.

- *TFT Historical Window*: The TFT's "recent history" window reads the tail of the historical training CSV (tft_ready_data.csv). It is a pragmatic stand-in to exercise multi-horizon bounds end-to-end, not a live streaming telemetry buffer.

- *LLM Rate Limits*: Free-tier Gemini limits are real. The multi-model fallback chain and a hardcoded autopilot fallback successfully bypass quota exhaustion to ensure the Reinforcement Learning pipeline remains testable during API throttling.


## 9. Academic Scope

Project SWING focuses strictly on grid operation, resilience, power flow optimization, contingency analysis, and AI-assisted operational decision support. It is intentionally distinct from predictive-maintenance or equipment-health-monitoring systems (no RUL, DGA, or component diagnostics are integrated into the final LangGraph decision loop).

Author
Utthan Singh Roy

B.Tech in Electrical Engineering, Madan Mohan Malaviya University of Technology (MMMUT)

B.Sc. (Hons) in Data Science and Artificial Intelligence, IIT Guwahati
