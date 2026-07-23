# =============================================
#              MODULE IMPORTS
# =============================================

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

load_dotenv()
warnings.filterwarnings("ignore")

from modules.forecasting.grid_forecaster import GridForecaster
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.event_store import get_events

# =============================================
#              FUNCTIONAL SCRIPT
# =============================================


def extract_text(content):
    """Safely extracts text from LangChain's Gemini response content."""
    if isinstance(content, list):
        return content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    return str(content)

def evaluate_forecasting():
    print("\n" + "="*55)
    print(" 1. FORECASTING METRICS (LightGBM P50 Model)")
    print("="*55)
    
    forecaster = GridForecaster()
    try:
        df = forecaster.load_and_harmonize()
        df_features = forecaster.engineer_features(df, 'Northen Region Hourly Demand')
        
        split_idx = int(len(df_features) * 0.8)
        test_df = df_features.iloc[split_idx:]
        
        forecaster.load_models()
        p50_model = forecaster.models[0.50]
        
        feature_cols = ['hour_sin', 'hour_cos', 'dayofweek', 'month', 
                        'load_lag_1', 'load_lag_2', 'load_lag_24', 'Northern_Region_Avg_T2M']
        
        X_test = test_df[feature_cols]
        y_test = test_df['normalized_target']
        
        predictions = p50_model.predict(X_test)
        
        rmse = np.sqrt(np.mean((predictions - y_test)**2))
        mae = np.mean(np.abs(predictions - y_test))
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        print(f"Root Mean Square Error (RMSE): {rmse:.4f} (Normalized)")
        print(f"Mean Absolute Error (MAE):     {mae:.4f} (Normalized)")
        print(f"Mean Absolute Pct Error (MAPE): {mape:.2f}%")
              
    except Exception as e:
        print(f"Forecasting Eval Error: {e}")

def evaluate_rl_optimization():
    print("\n" + "="*55)
    print(" 2. REINFORCEMENT LEARNING METRICS (SAC Policies)")
    print("="*55)
    
    models_dir = PROJECT_ROOT / "models"
    reward_files = list(models_dir.glob("*_rewards.csv"))
    
    if not reward_files:
        print("No RL reward CSVs found. Train an agent first using train_rl_agent.py.")
        return
        
    for rf in reward_files:
        df = pd.read_csv(rf)
        policy_name = rf.stem.replace("_rewards", "")
        
        total_episodes = len(df)
        converged_episodes = df['converged'].sum()
        convergence_rate = (converged_episodes / total_episodes) * 100
        
        chunk = max(1, total_episodes // 10)
        initial_reward = df['reward'].head(chunk).mean()
        final_reward = df['reward'].tail(chunk).mean()
        
        print(f"\nPolicy: {policy_name}")
        print(f"Total Training Episodes:  {total_episodes}")
        print(f"Physics Convergence Rate: {convergence_rate:.1f}%")
        print(f"Initial Avg Reward:       {initial_reward:.2f}")
        print(f"Final Avg Reward:         {final_reward:.2f}")

def evaluate_proposal_rag_compliance():
    print("\n" + "="*55)
    print(" 3. RAG & LLM DECISION EVALUATION (Top 7 Proposals)")
    print("="*55)
    
    db_path = PROJECT_ROOT / "data" / "knowledge_base" / "faiss_index"
    if not db_path.exists():
        print("FAISS index not found. Run rag_builder.py first.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(str(db_path), embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Multi-model fallback chain
    primary_llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.0)
    secondary_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
    tertiary_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)
    judge_llm = primary_llm.with_fallbacks([secondary_llm, tertiary_llm])

    events = get_events(source="ai_agent", limit=200)
    proposals = [ev for ev in events if ev["event_type"] == "PROPOSAL"]
    
    action_proposals = [
        p for p in proposals 
        if p["payload"].get("proposed_action", {}).get("action_type") != "NO_ACTION"
    ]
    
    if not action_proposals:
        print("No action proposals found in event_log.db.")
        return

    sample_proposals = action_proposals[:7]
    n_samples = len(sample_proposals)
    print(f"Found {len(action_proposals)} total logs. Evaluating Top {n_samples} recent proposals...\n", flush=True)
    
    total_relevance, total_faithfulness, total_actionable = 0, 0, 0
    successful_evals = 0
    
    for i, prop in enumerate(sample_proposals):
        payload = prop["payload"]
        action = payload.get("proposed_action", {})
        validation = payload.get("validation_result", {})
        
        action_type = action.get("action_type")
        target_bus = action.get("target_bus")
        secondary_bus = action.get("secondary_bus")
        justification = action.get("justification", "")
        brief = payload.get("operator_brief", "")
        
        # Check if the proposal passed physics check so the Approve button was enabled for the human operator
        is_actionable = validation.get("valid", False)
        if is_actionable:
            total_actionable += 1

        # Fix display string depending on whether secondary_bus is present
        bus_str = f"Bus {target_bus}" + (f" -> Bus {secondary_bus}" if secondary_bus and action_type == "LOAD_REDISTRIBUTION" else "")
        
        print(f"[Evaluating {i+1}/{n_samples}] {action_type} at {bus_str}...", end="", flush=True)
        
        query = f"Mandated IEGC protocols for {action_type} at Bus {target_bus} during grid thermal overload"
        docs = retriever.invoke(query)
        context = "\n".join(doc.page_content for doc in docs)[:1200]
        
        combined_prompt = f"""
Evaluate the following grid operation proposal against the retrieved IEGC rules.

RETRIEVED CONTEXT:
{context}

PROPOSAL JUSTIFICATION:
{justification}

BRIEF:
{brief}

Questions:
1. Is the retrieved context relevant to grid security, load shedding, or power flow control? (1 for Yes, 0 for No)
2. Is the proposed justification faithful to power grid rules without hallucinating fake policies? (1 for Yes, 0 for No)

Reply strictly in this format:
RELEVANCE: <1 or 0>
FAITHFULNESS: <1 or 0>
"""
        try:
            resp_text = extract_text(judge_llm.invoke(combined_prompt).content)
            
            rel_score = 1.0 if "RELEVANCE: 1" in resp_text or "RELEVANCE:1" in resp_text else 0.0
            faith_score = 1.0 if "FAITHFULNESS: 1" in resp_text or "FAITHFULNESS:1" in resp_text else 0.0
            
            total_relevance += rel_score
            total_faithfulness += faith_score
            successful_evals += 1
            
            print(f" DONE")
            print(f"   ├─ IEGC Context Relevant:          {'Yes' if rel_score == 1.0 else 'No'}")
            print(f"   ├─ Faithful to Grid Code:           {'Yes' if faith_score == 1.0 else 'No'}")
            print(f"   └─ Actionable (Approve Enabled):    {'Yes' if is_actionable else 'No (Blocked by Physics)'}\n")
            
        except Exception as e:
            print(f" SKIPPED ({str(e)[:40]}...)")
            break

    if successful_evals > 0:
        print("--- Aggregate RAG & Operator Safety Scores (Top 7) ---")
        print(f"IEGC Retrieval Relevance:    {(total_relevance / successful_evals) * 100:.1f}%")
        print(f"Regulatory Faithfulness:     {(total_faithfulness / successful_evals) * 100:.1f}%")
        print(f"Operator Actionability Rate:  {(total_actionable / n_samples) * 100:.1f}% (Valid proposals presented for Approval)")

def evaluate_proposal_physics_metrics():
    print("\n" + "="*55)
    print(" 4. PROPOSAL PERFORMANCE & PHYSICS METRICS")
    print("="*55)
    
    events = get_events(source="ai_agent", limit=500)
    proposals = [ev for ev in events if ev["event_type"] == "PROPOSAL"]
    
    action_proposals = [
        p for p in proposals 
        if p["payload"].get("proposed_action", {}).get("action_type") != "NO_ACTION"
    ]
    
    if not action_proposals:
        print("No action proposals found in event_log.db.")
        return

    # Strategy Breakdown
    redistribution_count = sum(
        1 for p in action_proposals 
        if p["payload"].get("proposed_action", {}).get("action_type") == "LOAD_REDISTRIBUTION"
    )
    shed_count = sum(
        1 for p in action_proposals 
        if p["payload"].get("proposed_action", {}).get("action_type") == "SHED_LOAD"
    )
    
    # Physics Pass Rate (Digital Twin AC Power Flow Validation)
    valid_proposals = [
        p for p in action_proposals 
        if p["payload"].get("validation_result", {}).get("valid") == True
    ]
    validity_rate = (len(valid_proposals) / len(action_proposals)) * 100

    # SOR / Optimization Method Breakdown
    sor_routed_count = sum(
        1 for p in action_proposals 
        if "sor" in p["payload"].get("proposed_action", {}).get("justification", "").lower()
    )
    
    # Average GRI Improvement for validated proposals
    gri_improvements = [
        p["payload"].get("validation_result", {}).get("gri_improvement")
        for p in valid_proposals 
        if p["payload"].get("validation_result", {}).get("gri_improvement") is not None
    ]
    avg_gri_delta = np.mean(gri_improvements) if gri_improvements else 0.0

    print(f"Total Action Proposals Logged:   {len(action_proposals)}")
    print(f"  - Load Redistribution Ratio:  {(redistribution_count / len(action_proposals))*100:.1f}% ({redistribution_count} proposals)")
    print(f"  - Load Shedding Ratio:        {(shed_count / len(action_proposals))*100:.1f}% ({shed_count} proposals)")
    print(f"Digital Twin Physics Pass Rate: {validity_rate:.1f}%")
    print(f"Smart Order Router Interventions: {sor_routed_count}")
    print(f"Avg Grid Resilience Index Boost: +{avg_gri_delta:.2f} points")

# ==================================================
#              TESTING & EXECUTION
# ==================================================

if __name__ == "__main__":
    evaluate_forecasting()
    evaluate_rl_optimization()
    evaluate_proposal_rag_compliance()
    evaluate_proposal_physics_metrics()
    print("\n")