# ============================================
#              MODULE IMPORTS
# ============================================

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ==================================================
#              FUNCTIONAL SCRIPT
# ==================================================

# What this script IS: a developer convenience that launches all of the four services together, in a sensible order, 
# and shuts them all down cleanly on Ctrl+C - so one doesn't have to juggle four separate terminal windows every
# time one wants to test the full stack.

DIGITAL_TWIN_UI_DIR = "Digital_Twin_UI"
AGENT_DASHBOARD_UI_DIR = "Agent_Dashboard_UI"

SERVICES = [
    {
        "name": "Digital Twin API (:8001)",
        "cmd": [sys.executable, "api.py"],
        "cwd": PROJECT_ROOT / "modules" / "physics_engine",
    },
    {
        "name": "AI Architecture API (:8002)",
        "cmd": [sys.executable, "agent_api.py"],
        "cwd": PROJECT_ROOT / "modules" / "ai_agents",
    },
    {
        "name": "Digital Twin UI (:3000)",
        "cmd": [sys.executable, "-m", "http.server", "3000"],
        "cwd": PROJECT_ROOT / DIGITAL_TWIN_UI_DIR,
    },
    {
        "name": "Agent Dashboard UI (:3001)",
        "cmd": [sys.executable, "-m", "http.server", "3001"],
        "cwd": PROJECT_ROOT / AGENT_DASHBOARD_UI_DIR,
    },
]


def main():
    processes = []
    try:
        for service in SERVICES:
            if not service["cwd"].exists():
                print(f"[SKIP] {service['name']} -- directory not found: {service['cwd']}")
                continue
            print(f"[STARTING] {service['name']}  (cwd={service['cwd']})")
            proc = subprocess.Popen(service["cmd"], cwd=str(service["cwd"]))
            processes.append((service["name"], proc))
            # Stagger startup - the Digital Twin should be up and answering HTTP requests before the AI service's own startup code (which may call it) has a chance to run.
            time.sleep(1.5)

        if not processes:
            print("\nNo services started -- check the folder names/paths above.")
            return

        print("\nAll available services started. Press Ctrl+C to stop everything.\n")
        print("Digital Twin UI:    http://127.0.0.1:3000/home.html")            # Copy this URL and paste in preferred browser(doesn't auto launch)
        print("Agent Dashboard UI: http://127.0.0.1:3001/ai_home.html\n")       # Copy this URL and paste in preferred browser(doesn't auto launch)

        while True:
            time.sleep(1)
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"[WARNING] {name} exited unexpectedly (code {proc.returncode}). "
                          f"Check its own terminal output if you started it separately, or re-run main.py.")

    except KeyboardInterrupt:
        print("\nShutting down all services...")
        for name, proc in processes:
            print(f"  stopping {name}...")
            proc.terminate()
        for name, proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("All services stopped.")

# ==================================================
#              TESTING & EXECUTION
# ==================================================

# main.py -- Development orchestrator for Project SWING.
# Usage:
#    python main.py
#    (then Ctrl+C to stop everything)

if __name__ == "__main__":
    main()