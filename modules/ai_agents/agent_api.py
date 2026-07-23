# =============================================
#              MODULE IMPORTS
# =============================================

import os
import sys
from pathlib import Path
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.exceptions import OutputParserException
import uvicorn

# --- Path Routing (explicit, not relying on agent_core's import side-effect) ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from agent_core import app as agent_graph, GridState, DIGITAL_TWIN_API_URL
from modules.event_store import log_event

# ===============================================================
#                         API ARCHITECTURE
# ===============================================================

# This is the ONLY place the AI stack (LangGraph, Gemini, RAG, LightGBM forecasting) is exposed to the outside world. 
# It imports nothing from modules/physics_engine - it doesn't even know pandapower exists. 
# Every interaction with the grid, read or write, goes through the Digital Twin's own HTTP API (api.py, normally on port 8001). 
# Run this on a different port (8002 by default) so it's trivially a separate process, separate deployment, even a separate machine if required.


app = FastAPI(title="AI Architecture Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory holding area for the latest proposal per grid, so a human operator can review it before /execute is ever called.
pending_proposals = {}


@app.get("/")
def read_root():
    return {"message": "AI Architecture Service Online", "digital_twin_target": DIGITAL_TWIN_API_URL}


@app.post("/propose/{grid}")
def propose_action(grid: str, telemetry: dict):
    '''
    Runs the full LangGraph pipeline (forecast -> risk -> RAG -> decision -> physics validation via the Digital Twin's HTTP API) and 
    returns the resulting brief. Nothing is applied to the live grid at this point --
    the Physics Validator only ever called the Twin's read-only /validate_action endpoint.
    '''
    # Pull live grid capacity from the Digital Twin itself, rather than assuming or duplicating that state locally.
    try:
        status = requests.get(f"{DIGITAL_TWIN_API_URL}/grid_status", params={"grid": grid}, timeout=10)
        status.raise_for_status()
        grid_status = status.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Digital Twin API: {e}")

    # Telemetry can still override this explicitly if the caller wants to test a hypothetical capacity, otherwise use the Twin's live figure.
    capacity_max = telemetry.pop("capacity_max", grid_status["capacity_max_mw"])

    initial_state = GridState(grid_id=grid, telemetry=telemetry, capacity_max=capacity_max)
    result = agent_graph.invoke(initial_state.model_dump())

    action = result["proposed_action"]
    validation = result["validation_result"]
    action_dict = action.model_dump() if hasattr(action, "model_dump") else action
    validation_dict = validation.model_dump() if hasattr(validation, "model_dump") else validation

    normalized_result = {
        "decision": result["decision"],
        "operator_brief": result["operator_brief"],
        "proposed_action": action_dict,
        "validation_result": validation_dict,
        "feature_importance": result.get("feature_importance", {}),
        "forecast_bounds": result.get("forecast_bounds", {}),
    }
    pending_proposals[grid] = normalized_result
    log_event("ai_agent", grid, "PROPOSAL", normalized_result)
    return normalized_result


@app.post("/execute/{grid}")
def execute_action(grid: str):
    '''
    Commits the LAST proposal returned by /propose to the live grid ONLY after a human operator has reviewed it here. This calls the
    Digital Twin's existing public actuation endpoints (/change_load or /redispatch_gen) exactly the way the React dashboard's "Apply Load
    Command" button does. This service never touches pandapower directly.
    '''
    if grid not in pending_proposals:
        raise HTTPException(status_code=400, detail="No pending proposal for this grid. Call /propose first.")

    proposal = pending_proposals[grid]
    action = proposal["proposed_action"]
    validation = proposal["validation_result"]

    if action["action_type"] == "NO_ACTION":
        raise HTTPException(status_code=400, detail="No action to execute -- grid was within safe margins.")

    if not validation["valid"]:
        raise HTTPException(status_code=409, detail=f"Cannot execute an unvalidated action: {validation['error']}")

    try:
        if action["action_type"] == "SHED_LOAD":
            # /change_load takes an absolute new_mw, not a delta, so we need the bus's current load first - 
            # fetch it from the Twin's own topology endpoint rather than assuming/duplicating state here.
            topo = requests.get(f"{DIGITAL_TWIN_API_URL}/topology", params={"grid": grid}, timeout=10).json()
            current_load = next(
                (n["data"]["load"] for n in topo["nodes"] if n["data"]["id"] == str(action["target_bus"])),
                None
            )
            if current_load is None:
                raise HTTPException(status_code=400, detail=f"Bus {action['target_bus']} not found in topology.")

            new_mw = max(0.0, current_load - action["mw_amount"])
            resp = requests.post(
                f"{DIGITAL_TWIN_API_URL}/change_load/{action['target_bus']}",
                params={"new_mw": new_mw, "grid": grid},
                timeout=10
            )

        elif action["action_type"] == "REDISPATCH_GEN":
            resp = requests.post(
                f"{DIGITAL_TWIN_API_URL}/redispatch_gen/{action['target_bus']}",
                params={"delta_mw": action["mw_amount"], "grid": grid},
                timeout=10
            )

        elif action["action_type"] == "LOAD_REDISTRIBUTION":
            resp = requests.post(
                f"{DIGITAL_TWIN_API_URL}/redistribute_load/{grid}",
                params={
                    "source_bus": action["target_bus"],
                    "destination_bus": action["secondary_bus"],
                    "mw_amount": action["mw_amount"],
                },
                timeout=10
            )

        resp.raise_for_status()
        del pending_proposals[grid]
        execution_record = {"action": action, "twin_response": resp.json()}
        log_event("ai_agent", grid, "EXECUTION", execution_record)
        return {"status": "EXECUTED", **execution_record}

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Digital Twin rejected or was unreachable for execution: {e}")


@app.get("/events/{grid}")
def get_events(grid: str, source: str = None, limit: int = 50):
    '''
    Returns the unified event timeline for a grid - both AI-side events (PROPOSAL, EXECUTION) and 
    Digital Twin-side events (LOAD_CHANGE, FAULT_INJECTED, GEN_REDISPATCH), ordered newest first. 
    Source can be 'digital_twin' or 'ai_agent' to filter to just one side.
    '''
    from modules.event_store import get_events as _get_events
    return {"grid": grid, "events": _get_events(grid_id=grid, source=source, limit=limit)}


# ===============================================================
#                   CONVERSATIONAL INTERFACE
# ===============================================================

# Grounded chat - every answer is built from LIVE context pulled from the Digital Twin (GRI, demand, contingency status) 
# plus the last proposal this service generated, not from the LLM's imagination. 
# This is explicitly NOT an actuation path: the model is told to point back to the Check for Risks / 
# Approve flow for anything that changes the grid, rather than trying to act from inside the chat.


class ChatRequest(BaseModel):
    message: str
    history: list = []  # [{"role": "user"|"assistant", "content": str}, ...]


@app.post("/chat/{grid}")
def chat(grid: str, request: ChatRequest):
    try:
        status = requests.get(f"{DIGITAL_TWIN_API_URL}/grid_status", params={"grid": grid}, timeout=10).json()
    except requests.RequestException:
        status = {}

    try:
        contingency = requests.get(f"{DIGITAL_TWIN_API_URL}/contingency/{grid}", timeout=10).json()
    except requests.RequestException:
        contingency = {}

    latest_proposal = pending_proposals.get(grid)

    context_summary = f"""
    Current grid: IEEE {grid}-Bus
    Grid Resilience Index: {status.get('resilience', {}).get('grid_resilience_index')}
    Total demand: {status.get('total_demand_mw')} MW
    Overloaded lines right now: {status.get('security', {}).get('overloaded_line_count')}
    N-1 secure: {contingency.get('n_minus_1_secure')}
    Insecure contingencies: {contingency.get('insecure_contingency_count')}
    Latest AI proposal: {latest_proposal['operator_brief'] if latest_proposal else "None yet -- 'Check for Risks' hasn't been run this session."}
    """

    history_text = "\n".join(f"{h.get('role', 'user')}: {h.get('content', '')}" for h in request.history[-10:])

    primary_llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash") 
    secondary_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash") 
    tertiary_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite") 

    # LangChain will automatically route to the next model if a rate limit or API error occurs
    llm = primary_llm.with_fallbacks([secondary_llm, tertiary_llm])
    
    prompt = f"""
    You are the conversational assistant for Project SWING's AI Architecture dashboard -- an AI-driven grid resilience decision-support platform
    (NOT an equipment health-monitoring tool). Answer the operator's question using ONLY the live context below; do not invent numbers you
    don't have, and say so plainly if something isn't in the context.

    If the operator asks you to actually DO something to the grid (shed load, redispatch, redistribute), explain what you'd recommend and why,
    but tell them to use the "Check for Risks" -> Approve flow to actually execute it -- you do not take actions directly from this chat.

    LIVE CONTEXT:
    {context_summary}

    CONVERSATION SO FAR:
    {history_text}

    OPERATOR'S QUESTION: {request.message}

    Respond conversationally and concisely, grounded in the live context above.
    """

    try:
        response = llm.invoke(prompt)
        
        content = response.content
        if isinstance(content, list):
            reply_text = content[0].get("text", "") if isinstance(content[0], dict) else str(content)
        else:
            reply_text = str(content)
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            reply_text = "I am currently receiving too many requests (API rate limit exceeded). Please wait a minute before asking another question."
        else:
            reply_text = f"An internal connection error occurred: {error_msg[:50]}..."

    log_event("ai_agent", grid, "CHAT", {"message": request.message, "response": reply_text})
    return {"response": reply_text}

# ==================================================
#              TESTING & EXECUTION
# ==================================================

if __name__ == "__main__":
    print("\nStarting AI Architecture Service")
    print(f"Digital Twin target: {DIGITAL_TWIN_API_URL}")
    uvicorn.run(app, host="127.0.0.1", port=8002)