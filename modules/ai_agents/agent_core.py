# =============================================
#              MODULE IMPORTS
# =============================================

import os
import sys
import json
from pathlib import Path
from typing import Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize local environment variables
load_dotenv()

# --- Path Routing ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from modules.forecasting.grid_forecaster import GridForecaster
from modules.forecasting.tft_forecaster import TFTForecaster

# =============================================
#                   ARCHITECTURE
# =============================================

'''
THE DIGITAL TWIN IS AN EXTERNAL SERVICE, NOT A LIBRARY
This module never imports TwinModel, pandapower, or anything else from modules/physics_engine. 
Every question about physics - "is the grid secure right now", "would this action be valid" -
is answered by calling the Digital Twin's own HTTP API, exactly the way the React dashboard or a human operator would. 
The AI architecture is a completely separate, swappable client of that API; 
it could be pointed at a different digital twin entirely without changing a line here.
'''

DIGITAL_TWIN_API_URL = os.getenv("DIGITAL_TWIN_API_URL", "http://127.0.0.1:8001")

MAX_VALIDATION_RETRIES = 2

# ==========================================================
#            STATE SCHEMA (Pydantic)
# ==========================================================

class ProposedAction(BaseModel):
    action_type: Literal["SHED_LOAD", "REDISPATCH_GEN", "LOAD_REDISTRIBUTION", "NO_ACTION"] = "NO_ACTION"
    target_bus: Optional[int] = None
    secondary_bus: Optional[int] = None  # LOAD_REDISTRIBUTION only: the bus load moves TO (target_bus is the source)
    mw_amount: Optional[float] = None
    justification: str = ""


class ValidationResult(BaseModel):
    valid: bool = False
    resulting_max_loading_percent: Optional[float] = None
    gri_before: Optional[float] = None
    gri_after: Optional[float] = None
    gri_improvement: Optional[float] = None
    error: Optional[str] = None


class GridState(BaseModel):
    grid_id: str = "118"                       # which Digital Twin grid this run targets
    telemetry: dict = Field(default_factory=dict)
    capacity_max: float = 0.0
    current_situation: dict = Field(default_factory=dict)  # Observer Agent's live GRI/security snapshot
    forecast_bounds: dict = Field(default_factory=dict)
    feature_importance: dict = Field(default_factory=dict)
    risk_level: str = "UNKNOWN"
    compliance_context: str = ""
    proposed_action: ProposedAction = Field(default_factory=ProposedAction)
    validation_result: ValidationResult = Field(default_factory=ValidationResult)
    retry_count: int = 0
    decision: str = "PENDING"
    operator_brief: str = ""


# ==========================================================
#   NODE 0: OBSERVER AGENT (Grid Situation Awareness)
# ==========================================================

'''
Per the multi-agent chain: Observer -> Forecast -> Risk -> Decision -> Optimization -> Physics Validator -> Operator. 
This node answers "what is the grid's situation right now" as its own explicit step, before any
forecasting happens - reading the Digital Twin's live GRI and security snapshot over HTTP, same as everything else in this file.
'''

def observer_node(state: GridState) -> dict:
    print("--- NODE 0: OBSERVING CURRENT GRID SITUATION ---")
    try:
        response = requests.get(f"{DIGITAL_TWIN_API_URL}/grid_status", params={"grid": state.grid_id}, timeout=10)
        response.raise_for_status()
        status = response.json()
        situation = {
            "capacity_max_mw": status.get("capacity_max_mw"),
            "resilience": status.get("resilience", {}),
            "security": status.get("security", {}),
        }
        print(f"Current GRI: {situation['resilience'].get('grid_resilience_index')}, "
              f"overloaded lines now: {situation['security'].get('overloaded_line_count')}")
    except requests.RequestException as e:
        print(f"[OBSERVER AGENT] Could not reach Digital Twin: {e}")
        situation = {}

    # capacity_max flows from here now, rather than requiring the caller to already know it - 
    # observer_node is the one place that reads it live off the Twin.
    update = {"current_situation": situation}
    if situation.get("capacity_max_mw"):
        update["capacity_max"] = situation["capacity_max_mw"]
    return update


# ==========================================================
#           NODE 1: FORECASTING (THE SENSOR)
# ==========================================================

#   Purely statistical - reads historical CSVs and trained LightGBM models, has nothing to do with the live Digital Twin.

def forecasting_node(state: GridState) -> dict:
    print("--- NODE 1: EXECUTING FORECAST ---")

    '''
    Try the TFT multi-horizon model first. Every failure mode here is caught and falls back to the LightGBM single-horizon model rather than crashing the whole graph 
    - the TFT integration is genuinely more fragile right now (checkpoint compatibility, historical CSV freshness, pytorch-forecasting version drift), 
    and this node's job is to always produce SOME usable forecast, not to insist on the fancier one.
    '''

    tft_bounds = None
    try:
        tft = TFTForecaster()
        tft.load_model()
        training_dataset = tft.load_training_dataset_config()
        window_df = tft.build_recent_history_window()
        raw = tft.forecast_multi_horizon(window_df, training_dataset)
        tft_bounds = raw
        print("Using TFT multi-horizon forecast.")
    except Exception as e:
        print(f"[FORECASTING NODE] TFT unavailable, falling back to LightGBM: {e}")

    # LightGBM always runs regardless - it's also the source of SHAP feature importance below, independent of which model supplied bounds.
    forecaster = GridForecaster()
    forecaster.load_models()
    lgbm_bounds = forecaster.forecast_next_hour(
        recent_features=state.telemetry,
        current_grid_baseline_mw=state.capacity_max
    )

    if tft_bounds:
        '''
        TFT's target was trained normalized (load / rolling_max), so denormalize the same way GridForecaster does, 
        using the live grid's current baseline as the scale reference. 
        TFT outputs one value per hour across the prediction horizon (24h); 
        index 0 is "1 hour out", index 23 is "24 hours out" - genuinely multi-horizon, unlike LightGBM's single "next hour" estimate.
        '''
        bounds = {
            "p10_1h": tft_bounds["p10"][0] * state.capacity_max,
            "p50_1h": tft_bounds["p50"][0] * state.capacity_max,
            "p90_1h": tft_bounds["p90"][0] * state.capacity_max,
            "p90_2h": tft_bounds["p90"][1] * state.capacity_max,
            "p90_6h": tft_bounds["p90"][5] * state.capacity_max,
            "p90_24h": tft_bounds["p90"][23] * state.capacity_max,
            "source": "TFT",
        }
    else:
        '''
        LightGBM only ever forecasts one horizon (next hour) - use the SAME key names as the TFT branch (p10_1h/p50_1h/p90_1h) 
        so nothing downstream needs to check `source` to know which keys are safe to read. 
        The longer horizons are explicitly None, not just absent, so the frontend can render "--" instead of guessing.
        '''
        bounds = {
            "p10_1h": lgbm_bounds.get("p10"),
            "p50_1h": lgbm_bounds.get("p50"),
            "p90_1h": lgbm_bounds.get("p90"),
            "p90_2h": None,
            "p90_6h": None,
            "p90_24h": None,
            "source": "LightGBM",
        }

    try:
        importance = forecaster.explain_prediction(state.telemetry)
    except Exception as e:
        print(f"[FORECASTING NODE] SHAP explanation unavailable: {e}")
        importance = {}

    return {"forecast_bounds": bounds, "feature_importance": importance}


# ==========================================================
#           NODE 2: RISK ASSESSMENT AGENT 
# ==========================================================

def risk_assessment_node(state: GridState) -> dict:
    print("--- NODE 2: ASSESSING RISK ---")

    '''
    THE OBSERVER AGENT'S LIVE SNAPSHOT TAKES PRIORITY. A forecast is a projection about the future; if the grid is ALREADY collapsed or
    ALREADY has overloaded lines right now, that is a more certain and more urgent problem than anything a forecast could say, 
    and no forecast-vs-capacity comparison should be able to mask it.
    '''
    resilience = state.current_situation.get("resilience", {})
    security = state.current_situation.get("security", {})

    if resilience.get("collapsed") or security.get("collapsed"):
        print("Digital Twin reports a NON-CONVERGENT (collapsed) power flow RIGHT NOW -- CRITICAL.")
        return {"risk_level": "CRITICAL"}

    current_overloaded = security.get("overloaded_line_count", 0)
    if current_overloaded and current_overloaded > 0:
        print(f"{current_overloaded} line(s) ALREADY overloaded right now -- HIGH, regardless of forecast.")
        return {"risk_level": "HIGH"}

    # Only reached if the CURRENT state is clean - now it's meaningful to ask "what does the forecast project."
    safety_threshold = state.capacity_max * 0.98

    # Horizon-aware: distinguish immediate risk (next hour) from emerging risk further out. 
    # Falls back to "p90" for LightGBM-only runs from before the multi-horizon rename, so older cached responses don't silently break.
    p90_1h = state.forecast_bounds.get("p90_1h", state.forecast_bounds.get("p90", 0.0))
    p90_6h = state.forecast_bounds.get("p90_6h")
    p90_24h = state.forecast_bounds.get("p90_24h")

    print(f"Projected P90 (1h): {p90_1h:.2f} MW | Safety Threshold: {safety_threshold:.2f} MW")
    if p90_6h is not None:
        print(f"Projected P90 (6h): {p90_6h:.2f} MW | Projected P90 (24h): {p90_24h:.2f} MW")

    if p90_1h >= safety_threshold:
        risk_level = "HIGH"
    elif p90_1h >= safety_threshold * 0.90:
        risk_level = "MEDIUM"
    elif p90_6h is not None and p90_6h >= safety_threshold:
        # Not urgent yet, but a real problem is coming within the next few hours -- worth the same 
        # MEDIUM treatment (triggers RAG + decision) rather than being dismissed as LOW just because it's not this hour.
        risk_level = "MEDIUM"
    elif p90_24h is not None and p90_24h >= safety_threshold:
        risk_level = "WATCH"
    else:
        risk_level = "LOW"

    return {"risk_level": risk_level}


def route_after_risk(state: GridState) -> str:
    if state.risk_level in ("CRITICAL", "HIGH", "MEDIUM"):
        return "rag_retrieval"
    elif state.risk_level == "WATCH":
        return "watch"
    return "no_action"


def watch_node(state: GridState) -> dict:
    '''
    A 24h-out risk shouldn't force the Decision Agent to propose an action NOW - too much can change in a day, 
    and prematurely shedding load or redispatching for something a day away risks unnecessary intervention. 
    This just surfaces the warning for the operator to keep an eye on and re-check as the horizon shortens.
    '''
    p90_24h = state.forecast_bounds.get("p90_24h")
    p90_24h_text = f"{p90_24h:.1f} MW" if p90_24h is not None else "unavailable"
    brief = (
        f"WATCH: No immediate action needed. The 24-hour forecast (P90: {p90_24h_text}) "
        f"projects the grid approaching its safety threshold, but nearer-term horizons are "
        f"still within safe margins. Re-run this check periodically as the horizon shortens."
    )
    return {"decision": "MONITOR", "operator_brief": brief}


# ==========================================================
#           NODE 3: RAG KNOWLEDGE SYSTEM 
# ==========================================================

def rag_retrieval_node(state: GridState) -> dict:
    print("--- NODE 3: RETRIEVING IEGC COMPLIANCE CONTEXT ---")
    db_path = project_root / "data" / "knowledge_base" / "faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(
        folder_path=str(db_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    query = (
        "What are the mandated load shedding and grid security protocols "
        "when operational demand approaches or exceeds physical capacity limits?"
    )
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"compliance_context": context}


# ==========================================================================================
#           NODE 4: DECISION AGENT (strategy) + OPTIMIZATION ENGINE call 
# ==========================================================================================

# The LLM decides WHAT KIND of action and WHERE (strategy only). 
# It never guesses the exact MW value - that's offloaded to the Digital Twin's /optimize_action endpoint, 
# which bisection-searches for the minimum MW that actually clears the violation. 
# This keeps the language model out of the one place a wrong guess is most costly: the number itself.


def decision_node(state: GridState) -> dict:
    print(f"--- NODE 4: PROPOSING STRATEGY (attempt {state.retry_count + 1}) ---")

    # --- MULTI-MODEL FALLBACK CHAIN ---

    # Attach the callback directly to LLMs in the fallback chain
    primary_llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash") 
    secondary_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash") 
    tertiary_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite") 
    
    # LangChain will automatically route to the next model if a rate limit or API error occurs
    llm = primary_llm.with_fallbacks([secondary_llm, tertiary_llm])
    

    error_feedback = ""
    if state.validation_result.error:
        error_feedback = (
            f"\nYour previous proposal FAILED physics validation with this error: "
            f"'{state.validation_result.error}'. Propose a DIFFERENT bus (or bus pair) or a "
            f"different action_type -- do not repeat the same choice."
        )

    overloaded_lines = state.current_situation.get("security", {}).get("overloaded_line_ids", [])
    overloaded_note = f"\nCurrently overloaded lines: {overloaded_lines}" if overloaded_lines else ""

    # --- GIVE THE LLM A LIST OF VALID TARGETS ---
    bus_loads = state.current_situation.get("bus_loads", {})
    # Filter out buses with 0 load and sort by heaviest MW demand
    valid_load_buses = sorted(
        [(bus, mw) for bus, mw in bus_loads.items() if mw > 0],
        key=lambda x: x[1], reverse=True
    )[:15]  # Take the top 15 heaviest loads
    
    valid_buses_str = ", ".join([f"Bus {b} ({mw:.1f} MW)" for b, mw in valid_load_buses])
    

    # Dynamic strategy rule based on line loading severity
    max_loading = state.current_situation.get("security", {}).get("max_line_loading_percent", 0.0)
    
    strategy_guidance = """
    STRATEGY SELECTION RULES:
    1. PRIMARY DIRECTIVE: Always prioritize "LOAD_REDISTRIBUTION" to ensure uninterrupted power supply. Attempt to shift load between buses first, regardless of the severity of the overload.
    2. TARGETING: You MUST select your `target_bus` and `secondary_bus` STRICTLY from the exact numbers listed in the "Top available load buses" array. Do not invent bus numbers.
    3. LAST RESORT: Only if "LOAD_REDISTRIBUTION" is physically INFEASIBLE, you MUST propose "SHED_LOAD".
    4. DO NOT propose "REDISPATCH_GEN" under any circumstances.
    """

    prompt = f"""
    You are a Grid Digital Twin Operator's decision-support agent.
    Risk level: {state.risk_level}
    Current Max Line Loading: {max_loading:.1f}%
    Top available load buses (for SHED_LOAD or REDISTRIBUTION source/dest): {valid_buses_str}
    {overloaded_note}

    {strategy_guidance}
    
    Relevant IEGC excerpts:
    {state.compliance_context}
    {error_feedback}

    Propose EXACTLY ONE corrective STRATEGY as strict JSON matching this exact format:
    {{
        "action_type": "LOAD_REDISTRIBUTION",
        "target_bus": <integer of the SOURCE bus where load is reduced>,
        "secondary_bus": <integer of the DESTINATION bus where load is increased>,
        "justification": "<short string citing IEGC>"
    }}
    (Note: If falling back to "SHED_LOAD", omit the "secondary_bus" key entirely).

    Do NOT propose an MW amount -- exact MW is computed by the grid optimization engine.
    Return ONLY the JSON object.
    """

    try:
        response = llm.invoke(prompt)
        
        # 1. Safely extract the raw text, handling both list blocks and flat strings
        content = response.content
        if isinstance(content, list):
            # Extract the text key from the first dictionary in the list
            raw = content[0].get("text", "") if isinstance(content[0], dict) else str(content)
        else:
            # Fallback for standard flat string responses
            raw = str(content)
            
        # 2. Clean and parse the resulting string as normal
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        
    except Exception as e:
        error_msg = str(e)
        print(f"[DECISION AGENT] LLM API Error: {error_msg}")
        
        # --- THE DEMO AUTOPILOT FALLBACK ---
        # If the LLM is rate-limited, bypass it and inject a hardcoded valid action so the Reinforcement Learning pipeline can still run.
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            print("[DECISION AGENT] Quota exhausted. Engaging Autopilot to demonstrate RL engine.")
            return {
                "proposed_action": ProposedAction(
                    action_type="SHED_LOAD",
                    target_bus=59,
                    justification="AI Quota Exhausted. Autopilot engaged: Shedding load at Bus 59 to trigger the Reinforcement Learning Optimization pipeline."
                )
            }
        else:
            return {
                "proposed_action": ProposedAction(
                    action_type="NO_ACTION",
                    justification=f"AI Engine Error: {error_msg[:50]}"
                )
            }

    try:
        parsed = json.loads(raw)
        strategy_action_type = parsed.get("action_type", "SHED_LOAD")
        
        target_val = parsed.get("target_bus")
        secondary_val = parsed.get("secondary_bus")
        
        if target_val is None and strategy_action_type != "NO_ACTION":
            raise ValueError("target_bus is missing from LLM output.")
            
        strategy_target_bus = int(target_val)
        strategy_secondary_bus = int(secondary_val) if secondary_val is not None else None
        justification = parsed.get("justification", "")
        
    except Exception as e:
        print(f"[DECISION AGENT] Failed to parse LLM strategy: {e}")
        # DO NOT default to NO_ACTION, as Node 5 will validate it as "safe".
        # Instead, return an explicitly invalid proposal so Node 5 catches it and correctly increments the retry_count to trigger Attempt 2 or Exhaustion.
        return {
            "proposed_action": ProposedAction(
                action_type="SHED_LOAD",
                target_bus=-1,  # Intentional invalid bus
                mw_amount=0.0,
                justification=f"AI Parsing Error: {str(e)[:50]}"
            )
        }

    # Offload the exact math to the Digital Twin's Optimization Engine --
    # 1. SANITIZE PAYLOAD: Ensure types are strictly cast and None values are stripped
    opt_payload = {
        "action_type": str(strategy_action_type),
        "target_bus": int(strategy_target_bus)
    }
    if strategy_secondary_bus is not None:
        opt_payload["secondary_bus"] = int(strategy_secondary_bus)

    try:
        opt_response = requests.post(
            f"{DIGITAL_TWIN_API_URL}/optimize_action/{state.grid_id}",
            json=opt_payload,
            timeout=15
        )
        opt_response.raise_for_status()
        opt_result = opt_response.json()
    except requests.RequestException as e:
        return {
            "proposed_action": ProposedAction(
                action_type=strategy_action_type,
                target_bus=strategy_target_bus,
                secondary_bus=strategy_secondary_bus,
                mw_amount=0.0,
                justification=f"{justification} [Optimizer unreachable: {e}]"
            )
        }

    if not opt_result.get("feasible", False):
        print(f"[DECISION AGENT] Strategy {strategy_action_type} at Bus {strategy_target_bus} is INFEASIBLE.")
        # Return an invalid action with explicit feedback so Node 5 increments retry_count and Attempt 2 forces a strategy shift to SHED_LOAD
        return {
            "proposed_action": ProposedAction(
                action_type=strategy_action_type,
                target_bus=strategy_target_bus,
                secondary_bus=strategy_secondary_bus,
                mw_amount=0.0,
                justification=f"Infeasible: {opt_result.get('note', 'Cannot clear overload at this bus.')}"
            ),
            "validation_result": ValidationResult(
                valid=False,
                error=f"Strategy '{strategy_action_type}' at Bus {strategy_target_bus} is physically infeasible. Pick SHED_LOAD at a heavy load bus instead."
            )
        }
    else:
        mw_amount = opt_result["mw_amount"]
        optimizer_method = opt_result.get("method", "bisection")
        justification = f"{justification} [Optimizer: {optimizer_method}]"

        # --- ACCEPT OVERRIDES FROM THE SMART ORDER ROUTER ---
        strategy_target_bus = opt_result.get("target_bus", strategy_target_bus)
        strategy_action_type = opt_result.get("action_type", strategy_action_type)


    action = ProposedAction(
        action_type=strategy_action_type,
        target_bus=strategy_target_bus,
        secondary_bus=strategy_secondary_bus,
        mw_amount=mw_amount,
        justification=justification
    )
    
    return {"proposed_action": action}


# ==========================================================
#           NODE 5: PHYSICS VALIDATOR AGENT 
# ==========================================================

# The LLM's proposed plan is NEVER trusted directly. Instead of touching pandapower itself, this node calls the Digital Twin's own
# POST /validate_action/{grid} endpoint - the Twin runs the test against a deep copy on its side and reports back whether it's physically valid.
# This node has no idea what pandapower even is; it just calls an HTTP endpoint, same as if a human curl'd it.

def physics_validator_node(state: GridState) -> dict:
    print("--- NODE 5: VALIDATING PROPOSAL AGAINST DIGITAL TWIN (via HTTP) ---")

    action = state.proposed_action

    if action.action_type == "NO_ACTION":
        return {"validation_result": ValidationResult(valid=True, resulting_max_loading_percent=None)}

    # Pre-flight check: If the proposal lacks a valid MW amount, fail safely and increment retry_count
    if action.mw_amount is None or action.mw_amount <= 0:
        validation = ValidationResult(
            valid=False, 
            error=f"Invalid action proposal: mw_amount is {action.mw_amount}."
        )
        return {
            "validation_result": validation,
            "retry_count": state.retry_count + 1
        }

    try:
        # Strictly format types to avoid FastAPI 422 schema errors
        payload = {
            "action_type": str(action.action_type),
            "target_bus": int(action.target_bus),
            "mw_amount": float(action.mw_amount),
        }
        if action.secondary_bus is not None:
            payload["secondary_bus"] = int(action.secondary_bus)

        response = requests.post(
            f"{DIGITAL_TWIN_API_URL}/validate_action/{state.grid_id}",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        validation = ValidationResult(
            valid=result.get("valid", False),
            resulting_max_loading_percent=result.get("resulting_max_loading_percent"),
            gri_before=result.get("gri_before"),
            gri_after=result.get("gri_after"),
            gri_improvement=result.get("gri_improvement"),
            error=result.get("error"),
        )

    except requests.RequestException as e:
        validation = ValidationResult(valid=False, error=f"Could not reach Digital Twin API: {e}")

    # Build response state update
    updates = {"validation_result": validation}
    
    # If physics validation failed, increment retry_count so LangGraph moves from attempt 1 -> attempt 2 -> EXHAUSTED
    if not validation.valid:
        updates["retry_count"] = state.retry_count + 1

    return updates


def route_after_validation(state: GridState) -> str:
    if state.validation_result.valid:
        return "operator"
    if state.retry_count < MAX_VALIDATION_RETRIES:
        return "decision"
    return "operator"


# ==========================================================
#               NODE 6: OPERATOR BRIEF 
# ==========================================================

def operator_node(state: GridState) -> dict:
    print("--- NODE 6: PREPARING OPERATOR BRIEF ---")

    action = state.proposed_action
    validation = state.validation_result

    top_features = state.feature_importance.get("ranked_features", [])[:3]
    if top_features:
        feature_lines = "\n".join(
            f"  - {name}: {value:+.2f} MW impact on forecast" for name, value in top_features
        )
        feature_section = f"\nTOP FORECAST DRIVERS (SHAP):\n{feature_lines}\n"
    else:
        feature_section = ""

    forecast_source = state.forecast_bounds.get("source", "LightGBM")
    p90_24h = state.forecast_bounds.get("p90_24h")
    if forecast_source == "TFT" and p90_24h is not None:
        forecast_section = f"FORECAST SOURCE: TFT (multi-horizon) -- 24h-out P90: {p90_24h:.1f} MW\n"
    else:
        forecast_section = f"FORECAST SOURCE: {forecast_source} (single-horizon, next hour only)\n"

    if action.action_type == "LOAD_REDISTRIBUTION":
        action_line = (
            f"PROPOSED ACTION: Redistribute {action.mw_amount} MW from Bus {action.target_bus} "
            f"(source) to Bus {action.secondary_bus} (destination) -- total demand unchanged\n"
        )
    else:
        action_line = f"PROPOSED ACTION: {action.action_type} of {action.mw_amount} MW at Bus {action.target_bus}\n"

    if action.action_type == "NO_ACTION":
        # Check if this NO_ACTION is due to an API error/rate limit
        if "Rate Limited" in action.justification or "Error" in action.justification:
            brief = f"{forecast_section}⚠️ SYSTEM OFFLINE: {action.justification}\nESCALATE TO HUMAN OPERATOR."
            decision = "AGENT_UNAVAILABLE"
        else:
            brief = f"{forecast_section}Grid operating within safe margins. No regulatory action required."
            decision = "MAINTAIN_BASELINE"
    elif validation.valid:
        if validation.gri_improvement is not None:
            gri_line = (
                f"EXPECTED RESULT: Grid Resilience Index {validation.gri_before:.1f} -> "
                f"{validation.gri_after:.1f} ({validation.gri_improvement:+.1f} points)\n"
            )
        else:
            gri_line = ""
        brief = (
            f"{forecast_section}"
            f"{action_line}"
            f"WHY: {action.justification}\n"
            f"{feature_section}"
            f"VALIDATED RESULT: post-action max line loading = {validation.resulting_max_loading_percent}%\n"
            f"{gri_line}"
            f"STATUS: Awaiting operator approval."
        )
        decision = f"{action.action_type}_PROPOSED"
    else:
        brief = (
            f"{forecast_section}"
            f"NO VALIDATED ACTION FOUND after {state.retry_count} attempt(s).\n"
            f"LAST ERROR: {validation.error}\n"
            f"{feature_section}"
            f"ESCALATE TO HUMAN OPERATOR for manual dispatch decision."
        )
        decision = "VALIDATION_EXHAUSTED"

    return {"operator_brief": brief, "decision": decision}


def no_action_node(state: GridState) -> dict:
    return {
        "decision": "MAINTAIN_BASELINE",
        "operator_brief": "Grid operating within safe margins. No regulatory action required."
    }


# ==========================================================
#                       GRAPH ASSEMBLY
# ==========================================================

# No factory, no twin binding needed anymore - the graph is stateless with respect to any particular grid; 
# grid_id travels inside GridState and the validator node reads it to know which Digital Twin endpoint to call. 
# One compiled graph now serves every grid.

workflow = StateGraph(GridState)

workflow.add_node("observer", observer_node)
workflow.add_node("sensor", forecasting_node)
workflow.add_node("risk_assessment", risk_assessment_node)
workflow.add_node("rag_retrieval", rag_retrieval_node)
workflow.add_node("decision", decision_node)
workflow.add_node("physics_validator", physics_validator_node)
workflow.add_node("operator", operator_node)
workflow.add_node("no_action", no_action_node)
workflow.add_node("watch", watch_node)

workflow.add_edge(START, "observer")
workflow.add_edge("observer", "sensor")
workflow.add_edge("sensor", "risk_assessment")

workflow.add_conditional_edges(
    "risk_assessment",
    route_after_risk,
    {"rag_retrieval": "rag_retrieval", "watch": "watch", "no_action": "no_action"}
)

workflow.add_edge("rag_retrieval", "decision")
workflow.add_edge("decision", "physics_validator")

workflow.add_conditional_edges(
    "physics_validator",
    route_after_validation,
    {"decision": "decision", "operator": "operator"}
)

workflow.add_edge("operator", END)
workflow.add_edge("no_action", END)
workflow.add_edge("watch", END)

app = workflow.compile()

# ==================================================
#              TESTING & EXECUTION
# ==================================================

if __name__ == "__main__":
    print("Initializing LangGraph Digital Twin Agent Core...")
    print(f"Targeting Digital Twin API at: {DIGITAL_TWIN_API_URL}")
    print("(Make sure api.py is already running in a separate process before this.)\n")

    mock_telemetry = {
        'hour_sin': 0.0,
        'hour_cos': 1.0,
        'dayofweek': 2,
        'month': 7,
        'load_lag_1': 0.85,
        'load_lag_2': 0.86,
        'load_lag_24': 0.82,
        'Northern_Region_Avg_T2M': 32.5
    }

    initial_state = GridState(
        grid_id="118",
        telemetry=mock_telemetry,
        capacity_max=24000.0
    )

    final_state = app.invoke(initial_state.model_dump())

    print("\n=== OPERATOR BRIEF ===")
    print(final_state["operator_brief"])