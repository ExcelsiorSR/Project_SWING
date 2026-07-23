# =============================================
#              MODULE IMPORTS
# =============================================

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandapower as pp
import uvicorn
import os
import sys
from pathlib import Path
import warnings
import datetime
import asyncio
import copy
import time
import math
import pandas as pd

# Ignore specific internal pandapower deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandapower")

# --- Path Routing ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from twin_model import TwinModel
from grid_metrics import get_resilience_index, get_security_metrics, get_health_metrics
from modules.optimization_engine.rl_optimizer import RLOptimizer
from modules.optimization_engine.rl_env import build_observation_vector

from modules.event_store import log_event

# =============================================
#              API ARCHITECTURE
# =============================================
# NOTE: This file is the DIGITAL TWIN service only. It has zero knowledge of LangGraph, Gemini, RAG, or anything else in the AI architecture. 
# It exposes physics (topology, telemetry, contingency testing, actuation) over plain HTTP/WebSocket so that ANY external client 
# - the AI agent service, a human operator's browser, a test harness - can drive it purely through this API.

app = FastAPI(title="Digital Twin API")

# CORS Middleware to allow the React frontend (and the separate AI service) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# 1. Instantiate the grid objects globally (Unified State)
grids = {
    "39": TwinModel(grid_type="39"),
    "118": TwinModel(grid_type="118")
}

# 2. Run baselines and apply scenarios before the server opens
print("Setting up grid scenarios...")
grids["39"].run_baseline_simulation()

'''
STARTUP STRESS LEVEL for IEEE-118. derate_multiplier=0.05 means every line's thermal rating is cut to 5% of nominal 
- extremely aggressive, which is why nudging a single bus's load by any large amount tends to blow past 100% loading or fail to converge at all. 
Raise this (e.g. to 0.15-0.25) for more headroom to test bigger interventions without the grid being perpetually on the edge of collapse. 
Lower it for overload scenarios to be trivially easy to trigger for demos.
'''

STARTUP_DERATE_MULTIPLIER = float(os.getenv("SWING_STARTUP_DERATE", "0.15"))
STARTUP_LOAD_MULTIPLIER = float(os.getenv("SWING_STARTUP_LOAD_MULT", "1.4"))

grids["118"].apply_scenario(load_multiplier=STARTUP_LOAD_MULTIPLIER, derate_multiplier=STARTUP_DERATE_MULTIPLIER)
grids["118"].run_baseline_simulation()


@app.get('/')
def read_root():
    return {'message': 'Digital Twin Engine is Online'}


# =============================================
#      N-1 CONTINGENCY SCANNER (BACKGROUND)
# =============================================
# Continuously simulates the loss of each line and each generator, one at a time, on a deep copy of the live grid 
# - never the live grid itself - and caches whether the system stays secure (converges, no line >100% loading) under each contingency.

# IMPORTANT PERFORMANCE NOTE: pp.runpp() is synchronous, CPU-bound code.
# Scanning ~170+ contingencies (118 lines + ~54 gens on the 118-bus case) takes several real seconds. 
# Running that directly inside an `async def` body would monopolize the single-threaded event loop for that whole stretch, 
# freezing the WebSocket telemetry stream and any HTTP request (like /change_load) that arrives during the scan. 
# To avoid that, the actual scanning work is a plain synchronous function, and the async loop only ever calls it via asyncio.to_thread(), 
# which runs it on a worker thread and lets the event loop keep serving telemetry and requests concurrently.

contingency_cache = {grid_key: {} for grid_key in grids.keys()}


def _scan_contingencies_sync(grid_key: str) -> dict:
    base_net = grids[grid_key].net
    line_results = {}

    for line_idx in base_net.line.index:
        test_net = copy.deepcopy(base_net)
        test_net.line.at[line_idx, 'in_service'] = False
        try:
            pp.runpp(test_net, solver='nr')
            max_loading = float(test_net.res_line.loading_percent.max())
            line_results[int(line_idx)] = {
                "converged": True,
                "max_loading_percent": round(max_loading, 2),
                "secure": bool(max_loading <= 100.0)
            }
        except pp.LoadflowNotConverged:
            line_results[int(line_idx)] = {"converged": False, "max_loading_percent": None, "secure": False}

    gen_results = {}
    for gen_idx in base_net.gen.index:
        test_net = copy.deepcopy(base_net)
        test_net.gen.at[gen_idx, 'in_service'] = False
        try:
            pp.runpp(test_net, solver='nr')
            max_loading = float(test_net.res_line.loading_percent.max())
            gen_results[int(gen_idx)] = {
                "converged": True,
                "max_loading_percent": round(max_loading, 2),
                "secure": bool(max_loading <= 100.0)
            }
        except pp.LoadflowNotConverged:
            gen_results[int(gen_idx)] = {"converged": False, "max_loading_percent": None, "secure": False}

    insecure_count = sum(
        1 for r in list(line_results.values()) + list(gen_results.values()) if not r["secure"]
    )

    return {
        "lines": line_results,
        "generators": gen_results,
        "n_minus_1_secure": insecure_count == 0,
        "insecure_contingency_count": insecure_count,
        "last_scan_timestamp": time.time()
    }


async def contingency_scanner_loop(grid_key: str, interval_seconds: int = 30):
    while True:
        try:
            result = await asyncio.to_thread(_scan_contingencies_sync, grid_key)
            contingency_cache[grid_key] = result
        except Exception as e:
            print(f"[N-1 SCANNER] Error scanning grid {grid_key}: {e}")
        await asyncio.sleep(interval_seconds)


@app.on_event("startup")
async def launch_background_scanners():
    for grid_key in grids.keys():
        asyncio.create_task(contingency_scanner_loop(grid_key, interval_seconds=30))


@app.get("/rl_status")
def get_rl_status():
    '''
    Reports which RL policies actually exist on disk, for both grids and both RL-eligible action types. 
    '''
    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    grids_to_check = ["39", "118"]
    actions_to_check = ["shed_load", "redispatch_gen"]

    status = {}
    for g in grids_to_check:
        status[g] = {}
        for a in actions_to_check:
            path = models_dir / f"rl_optimizer_{a}_{g}.zip"
            if path.exists():
                stat = path.stat()
                status[g][a] = {
                    "trained": True,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "last_trained": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            else:
                status[g][a] = {"trained": False}

    return status


@app.get("/contingency/{grid}")
def get_contingency_status(grid: str):
    '''Returns the latest N-1 security scan for the given grid. May be empty
    for a few seconds after startup until the first background scan completes.'''
    if grid not in contingency_cache:
        raise HTTPException(status_code=404, detail='Grid not Found')
    return contingency_cache[grid]


# ==============================================================
#      PHYSICS ORACLE (validate-only, no actuation)
# ==============================================================
# A pure physics test: "if this action were applied, would the grid stay secure?" 
# This endpoint NEVER mutates the live grid - it always tests against a deep copy, exactly like the contingency scanner above. 
# This is the one endpoint the AI architecture's Physics Validator agent calls; the Digital Twin has no idea an LLM exists on the other end of that call, 
# it's just answering a physics question over HTTP.

class ActionRequest(BaseModel):
    action_type: str   # "SHED_LOAD", "REDISPATCH_GEN", or "LOAD_REDISTRIBUTION"
    target_bus: int
    mw_amount: float
    secondary_bus: int = None  # only used by LOAD_REDISTRIBUTION


def _apply_action(test_net, action_type: str, target_bus: int, mw_amount: float, secondary_bus: int = None) -> bool:
    """
    Mutates test_net (a deep copy, never the live grid) to reflect the given action. 
    Returns False if the action_type/bus combination is invalid (e.g. no matching facility), True if applied. 
    Shared by _test_action and the GRI-recompute step in /validate_action so both are guaranteed to apply the exact same physics change.
    """
    if action_type == "SHED_LOAD":
        matching = test_net.load[test_net.load.bus == target_bus].index
        if matching.empty:
            return False
        idx = matching[0]
        current_mw = test_net.load.at[idx, "p_mw"]
        test_net.load.at[idx, "p_mw"] = max(0.0, current_mw - mw_amount)
        return True

    elif action_type == "REDISPATCH_GEN":
        matching = test_net.gen[test_net.gen.bus == target_bus].index
        if matching.empty:
            return False
        idx = matching[0]
        test_net.gen.at[idx, "p_mw"] += mw_amount
        return True

    elif action_type == "LOAD_REDISTRIBUTION":
        '''
        Moves mw_amount of load FROM target_bus TO secondary_bus - net zero change in total system demand. 
        This is the action type that actually does what "relieve an overloaded corridor by shifting load elsewhere" means, 
        as opposed to SHED_LOAD which just reduces total demand outright.
        '''
        if secondary_bus is None:
            return False
        source_matching = test_net.load[test_net.load.bus == target_bus].index
        dest_matching = test_net.load[test_net.load.bus == secondary_bus].index
        if source_matching.empty or dest_matching.empty:
            return False
        source_idx = source_matching[0]
        dest_idx = dest_matching[0]
        current_source_mw = test_net.load.at[source_idx, "p_mw"]
        actual_shift = min(mw_amount, current_source_mw)  # can't shift more than the source bus is carrying
        test_net.load.at[source_idx, "p_mw"] = current_source_mw - actual_shift
        test_net.load.at[dest_idx, "p_mw"] += actual_shift
        return True

    return False


def _test_action(base_net, action_type: str, target_bus: int, mw_amount: float, secondary_bus: int = None):
    """
    Shared physics-testing core used by BOTH /validate_action and  /optimize_action. 
    Always operates on a deep copy - base_net itself is never mutated. 
    Returns (converged: bool|None, max_loading_percent: float|None). 
    converged=None means the action_type/target_bus itself was invalid (e.g. no matching facility), not a power-flow failure.
    """
    test_net = copy.deepcopy(base_net)

    if not _apply_action(test_net, action_type, target_bus, mw_amount, secondary_bus):
        return None, None

    try:
        pp.runpp(test_net, solver="nr")
        return True, float(test_net.res_line.loading_percent.max())
    except pp.LoadflowNotConverged:
        return False, None


@app.post("/validate_action/{grid}")
def validate_action(grid: str, action: ActionRequest):
    if grid not in grids:
        raise HTTPException(status_code=404, detail='Grid not Found')

    base_net = grids[grid].net
    gri_before = get_resilience_index(base_net)["grid_resilience_index"]

    converged, max_loading = _test_action(base_net, action.action_type, action.target_bus, action.mw_amount, action.secondary_bus)

    if converged is None:
        return {"valid": False, "resulting_max_loading_percent": None,
                "gri_before": gri_before, "gri_after": None, "gri_improvement": None,
                "error": f"Bus {action.target_bus} (or secondary bus {action.secondary_bus}) has no matching facility for '{action.action_type}'."}

    if not converged:
        return {"valid": False, "resulting_max_loading_percent": None,
                "gri_before": gri_before, "gri_after": None, "gri_improvement": None,
                "error": "Power flow did not converge for this action."}

    # Recompute GRI on the same post-action state used for the loading check
    test_net = copy.deepcopy(base_net)
    _apply_action(test_net, action.action_type, action.target_bus, action.mw_amount, action.secondary_bus)
    pp.runpp(test_net, solver="nr")
    gri_after = get_resilience_index(test_net)["grid_resilience_index"]

    # Allow a micro-tolerance of 100.2% to account for pandapower float precision formatting
    secure = max_loading <= 100.2
    return {
        "valid": secure,
        "resulting_max_loading_percent": round(max_loading, 2),
        "gri_before": gri_before,
        "gri_after": gri_after,
        "gri_improvement": round(gri_after - gri_before, 2),
        "error": None if secure else f"Max line loading {max_loading:.1f}% -- still overloaded."
    }


# =============================================
#             OPTIMIZATION ENGINE 
# =============================================
# The Decision Agent (LLM) decides the STRATEGY -- which action_type, which bus (or bus PAIR, for LOAD_REDISTRIBUTION). 
# It should NOT be guessing the exact MW value; that's a physics optimization problem, not a language one. 
# This endpoint runs a bisection search over mw_amount to find the MINIMUM value that clears the overload (max line loading <= 100%), 
# so the agent always proposes the smallest corrective action that actually  works - no more, no less. 
# Every trial point goes through _test_action on a deep copy; the live grid is never touched by this search.

class OptimizeRequest(BaseModel):
    action_type: str
    target_bus: int
    secondary_bus: int = None  # required for LOAD_REDISTRIBUTION
    search_ceiling_mw: float = 300.0
    tolerance_mw: float = 0.5

def _emergency_grid_rescue(base_net):
    """
    Iteratively sheds 10% of total grid load globally until the 
    Newton-Raphson power flow matrix regains mathematical stability.
    """
    rescue_net = copy.deepcopy(base_net)
    total_original_load = float(rescue_net.load['p_mw'].sum())
    
    for iteration in range(1, 10):
        # Scale the active power load vector down by 10%
        rescue_net.load['p_mw'] *= 0.90
        
        try:
            pp.runpp(rescue_net, solver="nr")
            # If we reach here, the Jacobian matrix is no longer singular!
            current_load = float(rescue_net.load['p_mw'].sum())
            total_mw_shed = total_original_load - current_load
            max_loading = float(rescue_net.res_line.loading_percent.max())
            
            return True, total_mw_shed, max_loading
        except pp.LoadflowNotConverged:
            continue
            
    return False, 0.0, None

@app.post("/optimize_action/{grid}")
def optimize_action(grid: str, request: OptimizeRequest):
    if grid not in grids:
        raise HTTPException(status_code=404, detail='Grid not Found')

    base_net = grids[grid].net

    # Generate the Order Books
    heavy_buses = base_net.load.sort_values(by='p_mw', ascending=False)['bus'].astype(int).tolist()
    light_buses = base_net.load.sort_values(by='p_mw', ascending=True)['bus'].astype(int).tolist()
    candidate_buses = heavy_buses[:10]
    
    # --- AGGRESSIVE LLM SANITIZER ---
    request.search_ceiling_mw = float(base_net.load[base_net.load['bus'] == request.target_bus]['p_mw'].sum())
    
    # 1. Force the target_bus to be a valid load node
    if request.target_bus not in heavy_buses:
        print(f"[OPTIMIZER] LLM target_bus {request.target_bus} is invalid. Auto-correcting to Heaviest Load Bus: {heavy_buses[0]}.")
        request.target_bus = heavy_buses[0]
        
    # 2. Force the secondary_bus to be a valid, distinct destination for REDISTRIBUTION
    if request.action_type == "LOAD_REDISTRIBUTION":
        if request.secondary_bus not in heavy_buses or request.secondary_bus == request.target_bus:
            # Route the load to the lightest-loaded bus in the grid
            for lb in light_buses:
                if lb != request.target_bus:
                    request.secondary_bus = lb
                    break
            print(f"[OPTIMIZER] LLM secondary_bus was invalid. Auto-correcting to Lightest Load Bus: {request.secondary_bus}.")

    # Try the RL-trained optimizer first...
    if request.action_type in ("SHED_LOAD", "REDISPATCH_GEN"):
        try:
            rl = RLOptimizer(grid_type=grid)
            rl.load_model(request.action_type)
            obs = build_observation_vector(base_net, request.target_bus)
            
            start_time = time.time()
            rl_mw = rl.predict_mw(obs)
            inference_time_ms = (time.time() - start_time) * 1000

            converged, max_loading = _test_action(base_net, request.action_type, request.target_bus, rl_mw)
            if converged and max_loading is not None and max_loading <= 100.0:
                print(f"[OPTIMIZER] RL policy succeeded: {rl_mw:.2f} MW, resulting loading {max_loading:.1f}%")
                print(f"Total RL Inference time = {inference_time_ms}")
                return {
                    "feasible": True,
                    "mw_amount": round(rl_mw, 2),
                    "resulting_max_loading_percent": round(max_loading, 2),
                    "method": "RL",
                }
            else:
                print(f"[OPTIMIZER] RL policy's prediction failed the physics re-check "
                      f"(mw={rl_mw:.2f}, loading={max_loading}) -- falling back to bisection search.")

        except FileNotFoundError:
            print(f"[OPTIMIZER] No trained RL policy for {request.action_type}/IEEE-{grid} yet -- using bisection search.")
        except ImportError:
            print("[OPTIMIZER] stable-baselines3/gymnasium not installed -- using bisection search.")
        except Exception as e:
            print(f"[OPTIMIZER] RL optimizer error ({e}) -- falling back to bisection search.")

    # First confirm the ceiling is even enough to clear the overload at all on the LLM's original choice
    converged, max_loading = _test_action(base_net, request.action_type, request.target_bus, request.search_ceiling_mw, request.secondary_bus)

    # --- DO NOT THROW A 400 ERROR! LET THE SOR FIX IT ---
    if converged is None:
        print("[OPTIMIZER] LLM hallucinated an invalid bus topology. Forwarding to Smart Order Router...")
        max_loading = 999.0  # Force it into the SOR loop
        converged = True     # Prevent it from triggering the emergency collapse rescue

    # --- EMERGENCY GRID RESCUE (Jacobian Singularity Intercept) ---
    if not converged:
        print("[OPTIMIZER] Grid Collapse Detected (Jacobian Singular). Initiating Emergency Rescue...")
        rescue_converged, total_mw_shed, rescue_loading = _emergency_grid_rescue(base_net)
        
        if rescue_converged:
            return {
                "feasible": False,
                "mw_amount": None,
                "resulting_max_loading_percent": rescue_loading,
                "note": (
                    f"GRID COLLAPSE AVERTED via GLOBAL SHED. The requested action failed to converge. "
                    f"A broad-brush emergency shed of {total_mw_shed:.2f} MW was required just to restore "
                    f"mathematical stability. Propose SHED_LOAD at the heaviest bus immediately."
                )
            }
        else:
            return {
                "feasible": False,
                "mw_amount": None,
                "resulting_max_loading_percent": None,
                "note": "TOTAL GRID COLLAPSE. Even shedding 90% of global demand failed to restore matrix stability."
            }

    routed_justification = ""

    # --- INITIATE SMART ORDER ROUTING (SOR) ---
    if max_loading is None or max_loading > 100.0:
        print(f"[OPTIMIZER] Target Bus {request.target_bus} insufficient. Initiating SOR fallback loop...")
        found_viable_bus = False
        
        # DOUBLE-PASS SOR: If REDISTRIBUTION fails across the board, auto-degrade to SHED_LOAD
        actions_to_try = [request.action_type]
        if request.action_type == "LOAD_REDISTRIBUTION":
            actions_to_try.append("SHED_LOAD")
            
        for current_action in actions_to_try:
            for candidate in candidate_buses:
                if candidate == request.target_bus and current_action == request.action_type:
                    continue # Skip the venue already tested

                # 1. Dynamically calculate the maximum physical load sitting at this specific candidate bus
                candidate_max_mw = float(base_net.load[base_net.load['bus'] == candidate]['p_mw'].sum())
                    
                # 2. Sweep the order book using the actual physical limit
                c_converged, c_loading = _test_action(base_net, current_action, candidate, candidate_max_mw, request.secondary_bus)
    
                # 3. Check against safety margin
                if c_converged and c_loading is not None and c_loading <= 100.0:
                    print(f"[OPTIMIZER] SOR found viable venue: {current_action} at Bus {candidate} (Loading: {c_loading:.2f}%)")
                    request.target_bus = candidate  # OVERWRITE the target
                    request.action_type = current_action # OVERWRITE the action
                    max_loading = c_loading
                    found_viable_bus = True
                    routed_justification = f" [SOR dynamically routed to {current_action} at Bus {candidate}]"
                    break
                    
            if found_viable_bus:
                break # Break out of the action loop if found a solution
                
        # If the entire order book was swept and nothing cleared the constraint
        if not found_viable_bus:
            return {
                "feasible": False,
                "mw_amount": None,
                "resulting_max_loading_percent": max_loading,
                "note": (
                    f"SOR EXHAUSTED: Tested top 10 heaviest buses. "
                    f"None could physically clear the overload."
                )
            }

    # Proceed to Bisection search for the minimum feasible mw_amount on the successful bus
    low, high = 0.0, request.search_ceiling_mw
    best_mw = request.search_ceiling_mw
    best_loading = max_loading

    for _ in range(25):
        if (high - low) < request.tolerance_mw:
            break
        mid = (low + high) / 2.0
        converged, loading = _test_action(base_net, request.action_type, request.target_bus, mid, request.secondary_bus)
        if converged and loading is not None and loading <= 100.2:
            best_mw = mid
            best_loading = loading
            high = mid
        else:
            low = mid

    return {
        "feasible": True,
        "mw_amount": round(best_mw, 2),
        "resulting_max_loading_percent": round(best_loading, 2),
        "method": f"bisection{routed_justification}",
        "target_bus": request.target_bus,
        "action_type": request.action_type
    }


@app.get('/grid_status')
def get_grid_status(grid: str = '39'):
    '''
    Returns topological stats, current MW capacity, AND a live resilience snapshot (GRI + security metrics) for the given grid. 
    '''
    if grid not in grids:
        raise HTTPException(status_code=404, detail='Grid not Found')
        
    target_grid = grids[grid].net
    total_demand_mw = float(target_grid.load['p_mw'].sum())
    total_gen_capacity_mw = float(target_grid.gen['p_mw'].sum()) if len(target_grid.gen) else 0.0
    resilience = get_resilience_index(target_grid)
    security = get_security_metrics(target_grid)

    # --- Compute per-bus loads to pass to the AI Agent ---
    bus_loads = {str(idx): 0.0 for idx in target_grid.bus.index}
    for _, row in target_grid.load.iterrows():
        bus_id = str(int(row['bus']))
        if bus_id in bus_loads:
            bus_loads[bus_id] += row['p_mw']

    return {
        'active_grid': f'IEEE {grid}-Bus',
        'buses': len(target_grid.bus),
        'lines': len(target_grid.line),
        'total_demand_mw': round(total_demand_mw, 2),
        'total_generation_capacity_mw': round(total_gen_capacity_mw, 2),
        # capacity_max_mw is the reference the Risk Assessment Agent compares forecast p90 against - current total system load, 
        # not nameplate generation capacity, since that's what "the grid is at X% of what it's currently carrying" should be measured against.
        'capacity_max_mw': round(total_demand_mw, 2),
        'resilience': resilience,
        'security': security,
        'bus_loads': bus_loads,
    }


@app.post('/redistribute_load/{grid}')
def redistribute_load(grid: str, source_bus: int, destination_bus: int, mw_amount: float):
    '''
    Moves mw_amount of load FROM source_bus TO destination_bus in ONE atomic power-flow solve - net zero change in total system demand.
    This is the actuation counterpart to the LOAD_REDISTRIBUTION action type: relieves a specific overloaded corridor by shifting load away
    from it rather than curtailing demand outright.
    '''
    if grid not in grids:
        raise HTTPException(status_code=404, detail='Grid not Found')

    twin = grids[grid]
    try:
        source_matching = twin.net.load[twin.net.load.bus == source_bus].index
        dest_matching = twin.net.load[twin.net.load.bus == destination_bus].index
        if source_matching.empty:
            raise Exception(f"Source bus {source_bus} has no load facility.")
        if dest_matching.empty:
            raise Exception(f"Destination bus {destination_bus} has no load facility.")

        source_idx = source_matching[0]
        dest_idx = dest_matching[0]
        current_source_mw = twin.net.load.at[source_idx, "p_mw"]
        actual_shift = min(mw_amount, current_source_mw)

        twin.net.load.at[source_idx, "p_mw"] = current_source_mw - actual_shift
        twin.net.load.at[dest_idx, "p_mw"] += actual_shift

        pp.runpp(twin.net, solver='nr')
        summary = twin.get_current_state()
        twin.update_state(summary, "COMPLETED", f"Redistributed {actual_shift:.1f} MW: Bus {source_bus} -> Bus {destination_bus}")
        log_event("digital_twin", grid, "LOAD_REDISTRIBUTION", {
            "source_bus": source_bus, "destination_bus": destination_bus, "mw_amount": actual_shift
        })

        return {
            'status': 'Command Received',
            'action': f'Redistributed {actual_shift:.1f} MW from Bus {source_bus} to Bus {destination_bus}'
        }

    except pp.LoadflowNotConverged:
        raise HTTPException(status_code=400, detail="Redistribution caused a non-convergent power flow.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/topology")
def get_topology(grid: str = '118'):
    net = grids[grid].net

    # Which buses are actually generation, transformer-connected, or pure transmission junctions - computed once, not per-node, 
    # since it's the same set of buses regardless of which bus we're currently labeling.
    generator_buses = set(net.gen['bus'].tolist())
    transformer_buses = set(net.trafo['hv_bus'].tolist()) | set(net.trafo['lv_bus'].tolist())

    # Include current load in the node data
    nodes = []
    for idx, row in net.bus.iterrows():
        # Find if this bus has a load
        bus_loads = net.load[net.load.bus == idx]
        current_load = bus_loads['p_mw'].sum() if not bus_loads.empty else 0

        # Classify the node - generator takes priority over transformer. 
        # If a bus happens to be both (rare, but generators are the more visually/operationally significant fact about that bus).
        if idx in generator_buses:
            node_type = "GENERATOR"
        elif idx in transformer_buses:
            node_type = "TRANSFORMER"
        elif current_load > 0:
            node_type = "LOAD"
        else:
            node_type = "JUNCTION"

        nodes.append({'data': {'id': str(idx), 'label': f'Bus {idx}', 'load': current_load, 'node_type': node_type}})
        
    edges = [{'data': {'id': f'line{idx}', 'source': str(int(row['from_bus'])), 'target': str(int(row['to_bus']))}} 
             for idx, row in net.line.iterrows()]
             
    return {'nodes': nodes, 'edges': edges}

@app.post('/inject_fault/{line_id}')
def trigger_fault(line_id: int, grid: str = '39'):
    '''Trips a line on the specifically requested grid.'''
    if grid not in grids:
        raise HTTPException(status_code=404, detail='Grid not Found')
        
    grids[grid].inject_line_fault(line_id)
    log_event("digital_twin", grid, "FAULT_INJECTED", {"line_id": line_id})
    return {'status': 'Command Received', 'action': f'Tripped Line {line_id} on IEEE {grid}'}

@app.post('/change_load/{bus_id}')
def update_grid_load(bus_id: int, new_mw: float, grid: str = '118'):
    '''
    Updates the MW load on a specific bus ID. This is the sole actuation surface for load-side actions - called by the React dashboard directly,
    or by the separate AI agent service once a human has approved its proposal. The Digital Twin doesn't care which one calls it.
    '''
    if grid not in grids:
        raise HTTPException(status_code=404, detail='Grid not Found')
        
    try:
        # change_load() already does its own correct lookup by BUS NUMBER internally - calling it directly with the original bus_id 
        # (not a load-table row index) is both simpler and correct.
        grids[grid].change_load(bus_id, new_mw)
        log_event("digital_twin", grid, "LOAD_CHANGE", {"bus_id": bus_id, "new_mw": new_mw})
        
        return {'status': 'Command Received', 'action': f'Updated Facility {bus_id} to {new_mw}MW'}
    except Exception as e:
        # Pass the exact error string to the frontend
        raise HTTPException(status_code=400, detail=str(e))


class StressTestRequest(BaseModel):
    load_multiplier: float = 1.0     # 1.0 = 100% of nameplate load
    derate_multiplier: float = 1.0   # 1.0 = 100% of nominal line thermal capacity


@app.post('/stress_test/{grid}')
def apply_stress_test(grid: str, request: StressTestRequest):
    '''
    Sets the grid's load and line capacity to an absolute percentage of its ORIGINAL baseline - the endpoint behind the frontend's stress-test slider. 
    Unlike /change_load (single bus) or the startup-only /apply_scenario logic, this is idempotent: 
    calling it repeatedly with different values always means "X% of nameplate," never compounds.
    '''
    if grid not in grids:
        raise HTTPException(status_code=404, detail='Grid not Found')

    if request.load_multiplier <= 0 or request.derate_multiplier <= 0:
        raise HTTPException(status_code=400, detail='Multipliers must be positive.')

    twin = grids[grid]
    twin.apply_stress_test(request.load_multiplier, request.derate_multiplier)
    log_event("digital_twin", grid, "STRESS_TEST_APPLIED", {
        "load_multiplier": request.load_multiplier,
        "derate_multiplier": request.derate_multiplier
    })

    if twin.simulation_status == "FAILED":
        raise HTTPException(status_code=422, detail="Grid collapsed (power flow did not converge) at this stress level.")

    return {
        "status": "Stress Test Applied",
        "load_multiplier": request.load_multiplier,
        "derate_multiplier": request.derate_multiplier,
        "max_line_loading_percent": round(float(twin.net.res_line.loading_percent.max()), 2)
    }


@app.post('/redispatch_gen/{bus_id}')
def redispatch_generator(bus_id: int, delta_mw: float, grid: str = '118'):
    '''
    Adjusts generator output at a specific bus by delta_mw (+/-). 
    This is the generator-side counterpart to /change_load - the other half of the actuation surface an approved REDISPATCH_GEN action commits through.
    '''
    if grid not in grids:
        raise HTTPException(status_code=404, detail='Grid not Found')

    try:
        twin = grids[grid]
        matching_gens = twin.net.gen[twin.net.gen.bus == bus_id].index
        if matching_gens.empty:
            raise Exception(f"Bus {bus_id} has no generator to redispatch.")

        gen_idx = matching_gens[0]
        twin.net.gen.at[gen_idx, "p_mw"] += delta_mw

        pp.runpp(twin.net, solver='nr')
        summary = twin.get_current_state()
        twin.update_state(summary, "COMPLETED", f"Generator redispatched at Bus {bus_id} by {delta_mw} MW")
        log_event("digital_twin", grid, "GEN_REDISPATCH", {"bus_id": bus_id, "delta_mw": delta_mw})

        return {'status': 'Command Received', 'action': f'Redispatched Bus {bus_id} by {delta_mw}MW'}

    except pp.LoadflowNotConverged:
        raise HTTPException(status_code=400, detail="Redispatch caused a non-convergent power flow.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws/telemetry/{grid}")
async def grid_telemetry(websocket: WebSocket, grid: str):
    # 1. Accept the incoming connection from the frontend
    await websocket.accept()
    
    # Verify grid exists
    if grid not in grids:
        await websocket.close(code=1008)
        return
        
    try:
        # 2. Enter the infinite telemetry loop
        while True:
            # 3. Grab the latest state dictionary from the unified dictionary
            current_state = grids[grid].get_current_state()
            
            # 4. Map the actual Bus IDs to their current loads
            net = grids[grid].net
            # Initialize all buses to 0.0 MW
            bus_loads = {str(idx): 0.0 for idx in net.bus.index}
            
            # Add the actual loads to their respective physical buses
            for _, row in net.load.iterrows():
                bus_id = str(int(row['bus']))
                if bus_id in bus_loads:
                    bus_loads[bus_id] += row['p_mw']
                    
            current_state['bus_loads'] = bus_loads

            # --- CALCULATE AND APPEND LINE METRICS ---
            line_metrics = {}
            if "res_line" in net and not net.res_line.empty:
                for line_idx in net.line.index:
                    try:
                        # Extract active (P) and reactive (Q) power
                        p = float(net.res_line.p_from_mw.at[line_idx])
                        q = float(net.res_line.q_from_mvar.at[line_idx])
                        
                        # Calculate apparent power (S) and Power Factor (PF)
                        s = math.sqrt(p**2 + q**2)
                        pf = abs(p) / s if s > 0 else 1.0
                        
                        # Extract loading and current
                        loading = float(net.res_line.loading_percent.at[line_idx])
                        current = float(net.res_line.i_from_ka.at[line_idx])
                        
                        line_metrics[str(line_idx)] = {
                            "loading_percent": loading if not pd.isna(loading) else 0.0,
                            "i_ka": current if not pd.isna(current) else 0.0,
                            "pf": pf
                        }
                    except Exception:
                        # Fallback for disconnected or non-converged lines
                        line_metrics[str(line_idx)] = {"loading_percent": 0.0, "i_ka": 0.0, "pf": 1.0}
                        
            current_state["line_metrics"] = line_metrics

            # --- CALCULATE AND APPEND BUS METRICS ---
            bus_metrics = {}
            if "res_bus" in net and not net.res_bus.empty:
                for bus_idx in net.bus.index:
                    try:
                        v_pu = float(net.res_bus.vm_pu.at[bus_idx])
                        v_ang = float(net.res_bus.va_degree.at[bus_idx])
                        
                        bus_metrics[str(bus_idx)] = {
                            "v_pu": v_pu if not pd.isna(v_pu) else 0.0,
                            "v_ang": v_ang if not pd.isna(v_ang) else 0.0
                        }
                    except Exception:
                        bus_metrics[str(bus_idx)] = {"v_pu": 0.0, "v_ang": 0.0}
                        
            current_state["bus_metrics"] = bus_metrics
            
            # 5. Push it through the socket as JSON
            await websocket.send_json(current_state)
            
            # 6. Rest for 1 second before sending the next frame
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"Client disconnected from IEEE {grid}: {e}")

@app.post("/restore_line/{grid}/{line_id}")
def restore_line(grid: str, line_id: int):
    if grid not in grids:
        raise HTTPException(status_code=404, detail="Grid not found")
        
    net = grids[grid].net
    if line_id not in net.line.index:
        raise HTTPException(status_code=404, detail="Line not found")
        
    # Reconnect the line
    net.line.loc[line_id, 'in_service'] = True
    
    # Re-run power flow to update the grid state
    import pandapower as pp
    pp.runpp(net)
    
    return {"status": "restored", "line_id": line_id}

# ==================================================
#              TESTING & EXECUTION
# ==================================================

if __name__ == '__main__':
    # Runs the server locally
    print('\nStarting FastAPI Server')
    uvicorn.run(app, host='127.0.0.1', port=8001)