/* ============================================================
   Digital Twin Interface -- Shared Dashboard Logic
   Expects window.GRID_ID to be set ('39' or '118') by the page
   that includes this script, before this file runs.

   NOTE: Agent Copilot lives in ai_dashboard.html / ai_app.js now,
   not here. This file is the Digital Twin only -- it has no idea
   the AI Architecture service exists, matching the same clean
   separation the backend already enforces.
   ============================================================ */

const GRID_ID = window.GRID_ID;
const API_URL = "http://127.0.0.1:8001";     // Digital Twin service only

let cy = null;
let selectedBus = null;
let latestGridState = null;

// ------------------------------------------------------------
// ICONS -- generator/transformer buses ALWAYS get their correct
// icon (from /topology's node_type field); only genuine load
// buses get a randomized consumer icon for visual variety.
// ------------------------------------------------------------
const LOAD_ICONS = [
  "icons/factory.png", "icons/home.png", "icons/sky-scrappers.png",
  "icons/school.png", "icons/college.png", "icons/gas-station.png",
  "icons/hospital.png", "icons/museum.png", "icons/office.png",
  "icons/banks.png", "icons/post-office.png", "icons/shop.png",
  "icons/shopping-mall.png",
];

function iconForNode(nodeType) {
  switch (nodeType) {
    case "GENERATOR": return "icons/power-plant.png";
    case "TRANSFORMER": return "icons/transformer.png";
    case "LOAD": return LOAD_ICONS[Math.floor(Math.random() * LOAD_ICONS.length)];
    default: return "icons/transformer.png"; // JUNCTION
  }
}

// ------------------------------------------------------------
// CYTOSCAPE GRAPH
// ------------------------------------------------------------
async function loadTopology() {
  const res = await fetch(`${API_URL}/topology?grid=${GRID_ID}`);
  const data = await res.json();
  if (!data || !data.nodes || !data.edges) return;

  const enrichedNodes = data.nodes.map(node => ({
    ...node,
    data: { ...node.data, image: iconForNode(node.data.node_type) }
  }));

  cy = cytoscape({
    container: document.getElementById("cy"),
    elements: [...enrichedNodes, ...data.edges],
    layout: { name: "cose", animate: false },
    style: [
      {
        selector: "node",
        style: {
          "background-image": "data(image)",
          "background-fit": "cover",
          "background-color": "transparent",
          "label": "data(id)",
          "color": "#ffffff",
          "text-valign": "bottom",
          "text-margin-y": 5,
          "font-size": "12px",
          "width": 35,
          "height": 35
        }
      },
      {
        selector: "edge",
        style: {
          "width": 2,
          "line-color": "#444444",
          "curve-style": "bezier",
          "line-style": "dashed",
          "line-dash-pattern": [5, 5]
        }
      },
      {
        // CURRENTLY overloaded right now (>100% loading in live telemetry)
        selector: ".overloaded",
        style: {
          "line-color": "#e74c3c",
          "width": 5,
          "shadow-blur": 15,
          "shadow-color": "#e74c3c",
          "shadow-opacity": 0.9,
          "transition-property": "line-color, width",
          "transition-duration": "0.3s"
        }
      },
      {
        // WOULD become insecure if lost (N-1 contingency risk) -- distinct
        // amber color so it's visually different from an actual live
        // overload. Lower z-index priority than .overloaded if a line is
        // somehow both (cytoscape applies the later-defined style last).
        selector: ".contingency-risk",
        style: {
          "line-color": "#f39c12",
          "width": 4,
          "line-style": "dashed",
          "shadow-blur": 10,
          "shadow-color": "#f39c12",
          "shadow-opacity": 0.7
        }
      },
      {
        // Highest-demand facilities right now -- a gold glow, recomputed
        // every telemetry tick in renderTelemetry(). Any bus within 90% of
        // the current peak load lights up, not just a single "#1" bus, so
        // a genuine cluster of hotspots is visible together.
        selector: ".high-demand",
        style: {
          "border-width": 3,
          "border-color": "#ffd700",
          "border-opacity": 1,
          "shadow-blur": 25,
          "shadow-color": "#ffd700",
          "shadow-opacity": 0.9,
          "width": 42,
          "height": 42,
          "transition-property": "shadow-blur, shadow-opacity, width, height",
          "transition-duration": "0.4s"
        }
      },
      {
        // Visual state for a disconnected line
        selector: ".line-tripped",
        style: {
          "line-color": "#475569",
          "line-style": "dotted",
          "opacity": 0.3,
          "shadow-blur": 0
        }
      }
    ]
  });

  // --- LINE HOVER TOOLTIP ---
  const tooltip = document.getElementById("line-tooltip");

  cy.on("mouseover", "edge", (evt) => {
    const edge = evt.target;
    const lineId = edge.id().replace("line", "");
    const isTripped = edge.data("tripped") === true;
    
    // Extract telemetry (Defaults to 0 if the backend doesn't send line_metrics yet)
    const load = latestGridState?.line_metrics?.[lineId]?.loading_percent || 0;
    const current = latestGridState?.line_metrics?.[lineId]?.i_ka || 0;
    const pf = latestGridState?.line_metrics?.[lineId]?.pf || 1.0;

    const actionText = isTripped ? 
      '<span style="color:#4ade80;">Click to RESTORE this line</span>' : 
      '<span style="color:#f87171;">Click to TRIP this line</span>';

    tooltip.innerHTML = `
      <strong style="color: var(--accent-cyan);">Line ${lineId}</strong><br>
      Status: ${isTripped ? 'TRIPPED' : 'ACTIVE'}<br>
      Loading: ${Number(load).toFixed(1)}%<br>
      Current: ${Number(current).toFixed(2)} kA<br>
      Power Factor: ${Number(pf).toFixed(3)}<br>
      <hr>
      ${actionText}
    `;
    tooltip.classList.remove("hidden");
  });

  cy.on("mousemove", "edge", (evt) => {
    // Make the HTML tooltip follow the Cytoscape rendered mouse position
    tooltip.style.left = `${evt.originalEvent.pageX}px`;
    tooltip.style.top = `${evt.originalEvent.pageY}px`;
  });

  cy.on("mouseout", "edge", () => {
    tooltip.classList.add("hidden");
  });

  // --- BUS/NODE HOVER TOOLTIP ---
  cy.on("mouseover", "node", (evt) => {
    const node = evt.target;
    const busId = node.id();
    const nodeType = node.data("node_type"); // GENERATOR, LOAD, or TRANSFORMER
    
    // Extract telemetry
    const v_pu = latestGridState?.bus_metrics?.[busId]?.v_pu || 0;
    const v_ang = latestGridState?.bus_metrics?.[busId]?.v_ang || 0;
    const load = latestGridState?.bus_loads?.[busId] || 0;

    tooltip.innerHTML = `
      <strong style="color: #fde047;">Bus ${busId} (${nodeType})</strong><br>
      Voltage: ${Number(v_pu).toFixed(3)} p.u.<br>
      Angle: ${Number(v_ang).toFixed(1)}&deg;<br>
      Live Load: ${Number(load).toFixed(2)} MW<br>
      <hr>
      <span style="color:#94a3b8;">Click to open Facility Control</span>
    `;
    tooltip.classList.remove("hidden");
  });

  cy.on("mousemove", "node", (evt) => {
    tooltip.style.left = `${evt.originalEvent.pageX}px`;
    tooltip.style.top = `${evt.originalEvent.pageY}px`;
  });

  cy.on("mouseout", "node", () => {
    tooltip.classList.add("hidden");
  });

  // --- BI-DIRECTIONAL TRIP/RESTORE ---
  cy.on("tap", "edge", (evt) => {
    const edge = evt.target;
    const lineId = parseInt(edge.id().replace("line", ""), 10);
    const isTripped = edge.data("tripped") === true;

    if (isTripped) {
      // RESTORE LINE
      edge.data("tripped", false);
      edge.removeClass("line-tripped");
      fetch(`${API_URL}/restore_line/${GRID_ID}/${lineId}`, { method: "POST" })
        .catch(err => console.error(err));
    } else {
      // TRIP LINE
      edge.data("tripped", true);
      edge.addClass("line-tripped");
      fetch(`${API_URL}/inject_fault/${lineId}?grid=${GRID_ID}`, { method: "POST" })
        .catch(err => console.error(err));
    }
  });

  cy.on("tap", "node", (evt) => {
    const busId = evt.target.id();
    const liveLoad = latestGridState?.bus_loads?.[busId] ?? evt.target.data("load") ?? 0;
    selectedBus = busId;
    document.getElementById("targetFacility").textContent = `Facility ${busId}`;
    document.getElementById("currentLoad").textContent = `${Number(liveLoad).toFixed(2)} MW`;
    document.getElementById("loadInput").placeholder = `Current: ${Number(liveLoad).toFixed(2)} MW`;
    document.getElementById("loadInput").value = "";
    document.getElementById("loadCommandResult").textContent = "";
  });

  // animated power-flow dash offset
  let offset = 0;
  setInterval(() => {
    if (!cy) return;
    offset = (offset + 1) % 20;
    // Only animate edges that are NOT tripped
    cy.edges('[!tripped]').style("line-dash-offset", offset);
  }, 50);
}

// ------------------------------------------------------------
// TELEMETRY WEBSOCKET
// A "generation" counter guards against a subtle but real bug: if the
// Digital Twin backend is restarted while a tab is open, the old socket
// can remain half-open and its onmessage handler can still fire late,
// overwriting the display with stale data from BEFORE the restart, right
// after a newer connection already rendered current data. Each call to
// connectTelemetry() gets its own generation number; a message is only
// applied to the DOM if its socket's generation is still the current one.
// ------------------------------------------------------------
let wsGeneration = 0;

function connectTelemetry() {
  wsGeneration += 1;
  const myGeneration = wsGeneration;
  const ws = new WebSocket(`ws://127.0.0.1:8001/ws/telemetry/${GRID_ID}`);

  ws.onopen = () => console.log(`WebSocket connected to Digital Twin (gen ${myGeneration})`);
  ws.onerror = (err) => console.error("WebSocket error:", err);
  ws.onclose = () => {
    console.log(`WebSocket disconnected (gen ${myGeneration}) -- retrying in 2s`);
    setTimeout(connectTelemetry, 2000);
  };

  ws.onmessage = (event) => {
    if (myGeneration !== wsGeneration) return; // stale socket, ignore
    const state = JSON.parse(event.data);
    latestGridState = state;
    renderTelemetry(state);
  };
}

function renderTelemetry(state) {
  // System status
  setText("statStatus", state.simulation_status || "UNKNOWN");
  document.getElementById("statStatus").className =
    state.simulation_status === "COMPLETED" ? "badge-green" : "badge-red";
  setText("statTimeStep", `t=${state.time_step || 0}`);
  setText("statLastEvent", state.last_event || "None");
  setText("statScenario", state.current_scenario || "Baseline");

  // Electrical metrics
  setText("statDemand", `${(state.current_demand_mw ?? 0).toFixed(2)} MW`);
  setText("statGeneration", `${(state.total_generation_mw ?? 0).toFixed(2)} MW`);
  const maxLoad = state.max_line_loading_percent ?? 0;
  const maxLoadEl = document.getElementById("statMaxLoad");
  maxLoadEl.textContent = `${maxLoad.toFixed(1)}%`;
  maxLoadEl.className = maxLoad > 100 ? "badge-red" : "badge-green";

  // Security metrics
  setText("statOverloadCount", state.overloaded_line_count || 0);
  const linesEl = document.getElementById("statOverloadLines");
  if (state.overloaded_line_count > 0 && state.overloaded_line_ids) {
    linesEl.textContent = `Lines: ${state.overloaded_line_ids.join(", ")}`;
    linesEl.classList.remove("hidden");
  } else {
    linesEl.classList.add("hidden");
  }

  // Grid Resilience Index
  if (state.resilience) {
    const gri = state.resilience.grid_resilience_index;
    const griEl = document.getElementById("griScore");
    griEl.textContent = gri.toFixed(1);
    griEl.className = "gri-score " + (gri >= 80 ? "badge-green" : gri >= 50 ? "badge-amber" : "badge-red");
    setText("griVoltage", state.resilience.voltage_score.toFixed(1));
    setText("griLoading", state.resilience.loading_score.toFixed(1));
    setText("griSecurity", state.resilience.security_score.toFixed(1));
  }

  // Currently-overloaded edges on the graph
  if (cy) {
    cy.edges().removeClass("overloaded");
    if (state.overloaded_line_count > 0 && state.overloaded_line_ids) {
      state.overloaded_line_ids.forEach(id => {
        const el = cy.getElementById(`line${id}`);
        if (el) el.addClass("overloaded");
      });
    }
  }

  // Highest-demand facilities -- glow any bus within 90% of the current
  // peak load, so a genuine cluster of hotspots lights up together, not
  // just a single arbitrary "#1" bus.
  if (cy && state.bus_loads) {
    cy.nodes().removeClass("high-demand");
    const loads = Object.values(state.bus_loads).filter(v => v > 0);
    if (loads.length > 0) {
      const peak = Math.max(...loads);
      const threshold = peak * 0.9;
      Object.entries(state.bus_loads).forEach(([busId, load]) => {
        if (load >= threshold && load > 0) {
          const el = cy.getElementById(busId);
          if (el) el.addClass("high-demand");
        }
      });
    }
  }

  // Selected facility's live load, if any -- this is the ongoing
  // background refresh; the OPTIMISTIC update after a successful
  // /change_load call (see applyLoadCommand) is what makes the number
  // change instantly instead of waiting up to 1s for this tick.
  if (selectedBus) {
    const liveLoad = state.bus_loads?.[selectedBus] ?? 0;
    document.getElementById("currentLoad").textContent = `${Number(liveLoad).toFixed(2)} MW`;
  }
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

// ------------------------------------------------------------
// FACILITY CONTROL (single-bus load command)
// ------------------------------------------------------------
function applyLoadCommand() {
  const val = document.getElementById("loadInput").value;
  const resultEl = document.getElementById("loadCommandResult");
  const btn = document.getElementById("applyLoadBtn");

  if (!selectedBus) { resultEl.textContent = "Click a facility on the graph first."; resultEl.className = "gri-sub badge-amber"; return; }
  if (!val) { resultEl.textContent = "Enter a value first."; resultEl.className = "gri-sub badge-amber"; return; }

  // Freeze the target bus at click time -- if the user clicks a DIFFERENT
  // node while this request is still in flight, this variable (not the
  // possibly-changed `selectedBus`) is what the response applies to, and
  // the button being disabled below prevents overlapping requests
  // entirely, which is what caused the confusing "wrong bus in the error"
  // report: a slow request for an earlier bus resolving after a newer bus
  // was already selected on screen.
  const targetBus = selectedBus;
  const newMw = val;

  btn.disabled = true;
  btn.textContent = "Applying...";
  resultEl.textContent = "";

  fetch(`${API_URL}/change_load/${targetBus}?new_mw=${newMw}&grid=${GRID_ID}`, { method: "POST" })
    .then(async (res) => {
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Command failed");
      return data;
    })
    .then(data => {
      resultEl.textContent = data.action;
      resultEl.className = "gri-sub badge-green";
      document.getElementById("loadInput").value = "";

      // Optimistic immediate update -- don't wait for the next WebSocket
      // tick (up to 1s away) to reflect what we already know just
      // succeeded. Only applies if the user is still looking at the same
      // facility they just changed.
      if (selectedBus === targetBus) {
        document.getElementById("currentLoad").textContent = `${Number(newMw).toFixed(2)} MW`;
      }
    })
    .catch(err => {
      resultEl.textContent = `Bus ${targetBus}: ${err.message}`;
      resultEl.className = "gri-sub badge-red";
    })
    .finally(() => {
      btn.disabled = false;
      btn.textContent = "Apply Load Command";
    });
}

// ------------------------------------------------------------
// STRESS TEST (idempotent, calls /stress_test/{grid})
// ------------------------------------------------------------
function initStressTestControls() {
  const loadSlider = document.getElementById("loadSlider");
  const derateSlider = document.getElementById("derateSlider");
  const loadValueEl = document.getElementById("loadSliderValue");
  const derateValueEl = document.getElementById("derateSliderValue");

  loadSlider.addEventListener("input", () => {
    loadValueEl.textContent = `${loadSlider.value}%`;
  });
  derateSlider.addEventListener("input", () => {
    derateValueEl.textContent = `${derateSlider.value}%`;
  });

  document.getElementById("applyStressBtn").addEventListener("click", () => {
    const loadMultiplier = Number(loadSlider.value) / 100;
    const derateMultiplier = Number(derateSlider.value) / 100;
    const resultEl = document.getElementById("stressResult");
    const btn = document.getElementById("applyStressBtn");

    btn.disabled = true;
    resultEl.textContent = "Applying stress test...";
    resultEl.className = "gri-sub";

    fetch(`${API_URL}/stress_test/${GRID_ID}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ load_multiplier: loadMultiplier, derate_multiplier: derateMultiplier })
    })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Stress test failed");
        return data;
      })
      .then(data => {
        resultEl.textContent = `Applied. Max line loading now ${data.max_line_loading_percent}%.`;
        resultEl.className = data.max_line_loading_percent > 100 ? "gri-sub badge-red" : "gri-sub badge-green";
      })
      .catch(err => {
        // If you see "Failed to fetch" here specifically (as opposed to a
        // clean error message), it almost always means the Digital Twin
        // server crashed on an unhandled exception for this request --
        // check that terminal's traceback, not just this message.
        resultEl.textContent = `FAILED: ${err.message}`;
        resultEl.className = "gri-sub badge-red";
      })
      .finally(() => {
        btn.disabled = false;
      });
  });
}

// ------------------------------------------------------------
// N-1 CONTINGENCY STATUS (polled, not streamed)
// Shows WHICH specific lines/generators are insecure, not just a count,
// and highlights those lines on the graph in amber so you can see exactly
// where the grid's hidden risk is sitting, not just that risk exists.
// ------------------------------------------------------------
function pollContingencyStatus() {
  fetch(`${API_URL}/contingency/${GRID_ID}`)
    .then(res => res.json())
    .then(data => {
      const summaryEl = document.getElementById("contingencyStatus");
      const detailEl = document.getElementById("contingencyDetail");

      if (!data || data.n_minus_1_secure === undefined) {
        summaryEl.innerHTML = `<span class="gri-sub">Scanning... (first scan can take up to 30s after startup)</span>`;
        detailEl.innerHTML = "";
        return;
      }

      if (data.n_minus_1_secure) {
        summaryEl.innerHTML = `<span class="badge-green">&#9679; N-1 SECURE</span>`;
        detailEl.innerHTML = "";
        if (cy) cy.edges().removeClass("contingency-risk");
        return;
      }

      summaryEl.innerHTML = `<span class="pulse-dot"></span><span class="badge-red">${data.insecure_contingency_count} insecure contingenc${data.insecure_contingency_count === 1 ? "y" : "ies"}</span>`;

      // List exactly which lines and generators are insecure
      const insecureLines = Object.entries(data.lines || {}).filter(([, v]) => !v.secure).map(([id]) => id);
      const insecureGens = Object.entries(data.generators || {}).filter(([, v]) => !v.secure).map(([id]) => id);

      let detailHtml = "";
      if (insecureLines.length) {
        detailHtml += `<p class="gri-sub" style="margin-top:8px;">Lines at risk if lost: <strong>${insecureLines.join(", ")}</strong></p>`;
      }
      if (insecureGens.length) {
        detailHtml += `<p class="gri-sub">Generators at risk if lost: <strong>${insecureGens.join(", ")}</strong></p>`;
      }
      detailEl.innerHTML = detailHtml;

      // Highlight those specific lines on the graph
      if (cy) {
        cy.edges().removeClass("contingency-risk");
        insecureLines.forEach(id => {
          const el = cy.getElementById(`line${id}`);
          if (el) el.addClass("contingency-risk");
        });
      }
    })
    .catch(() => {});
}

// ------------------------------------------------------------
// PANEL TOGGLE
// ------------------------------------------------------------
function initPanelToggle() {
  const panel = document.getElementById("sidePanel");
  const btn = document.getElementById("togglePanelBtn");
  btn.addEventListener("click", () => {
    const isHidden = panel.classList.toggle("hidden");
    btn.textContent = isHidden ? "Show Command Center" : "Hide Command Center";
  });
}

// ------------------------------------------------------------
// BOOTSTRAP
// ------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  loadTopology();
  connectTelemetry();
  initPanelToggle();
  initStressTestControls();
  document.getElementById("applyLoadBtn").addEventListener("click", applyLoadCommand);

  pollContingencyStatus();
  setInterval(pollContingencyStatus, 15000);
});