/* ============================================================
   AI Architecture Dashboard - talks ONLY to agent_api.py (:8002).
   This is intentionally its own file/page, separate from the Digital
   Twin's app.js: the Digital Twin dashboards should be usable as a pure
   physics test harness with no AI concepts anywhere in them, and this
   page should be usable (in principle) against ANY digital twin that
   implements the same /validate_action, /optimize_action, /change_load,
   /redispatch_gen contract -- not just this one.
   ============================================================ */

const AI_API_URL = "http://127.0.0.1:8002";

function getSelectedGrid() {
  return document.getElementById("gridSelect").value;
}

// ------------------------------------------------------------
// RUN THE AGENT PIPELINE
// ------------------------------------------------------------
function initCheckRisks() {
  document.getElementById("checkRisksBtn").addEventListener("click", () => {
    const btn = document.getElementById("checkRisksBtn");
    const loader = document.getElementById("agentLoader");
    const briefContainer = document.getElementById("agentBriefContainer");


    btn.disabled = true;
    loader.classList.remove("hidden"); // Show the thinking animation
    briefContainer.classList.add("hidden"); // Hide previous results

    btn.textContent = "Running Agent Pipeline...";

    const now = new Date();
    const mockTelemetry = {
      hour_sin: Math.sin(2 * Math.PI * now.getHours() / 24),
      hour_cos: Math.cos(2 * Math.PI * now.getHours() / 24),
      dayofweek: now.getDay(),
      month: now.getMonth() + 1,
      load_lag_1: 0.85,
      load_lag_2: 0.86,
      load_lag_24: 0.82,
      Northern_Region_Avg_T2M: 32.5
    };

    fetch(`${AI_API_URL}/propose/${getSelectedGrid()}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(mockTelemetry)
    })
      .then(res => res.json())
      .then(data => {
        renderAgentProposal(data);
        loadEventTimeline(); // the propose call just logged an event -- reflect it
      })
      .catch(err => {
        console.error(err);
        alert(`Could not reach AI Architecture service: ${err.message}`);
      })
      .finally(() => {
        btn.disabled = false;
        loader.classList.add("hidden"); // Hide the thinking animation
        btn.textContent = "Check for Risks";
      });
  });
}

function renderAgentProposal(proposal) {
  const container = document.getElementById("agentBriefContainer");
  container.classList.remove("hidden");
  document.getElementById("agentDecision").textContent = proposal.decision;
  document.getElementById("agentBriefText").textContent = proposal.operator_brief;

  renderForecastSource(proposal.forecast_bounds);
  renderShapBars(proposal.feature_importance);
  renderGriBar(proposal.validation_result);

  const pillRow = document.getElementById("agentPillRow");
  if (proposal.proposed_action && proposal.proposed_action.action_type !== "NO_ACTION") {
    pillRow.classList.remove("hidden");
    const approveBtn = document.getElementById("approveBtn");
    approveBtn.disabled = !proposal.validation_result?.valid;
    approveBtn.className = "action-btn " + (proposal.validation_result?.valid ? "btn-green" : "btn-dark");
  } else {
    pillRow.classList.add("hidden");
  }
}

function renderForecastSource(forecastBounds) {
  const el = document.getElementById("forecastSourceBadge");
  if (!forecastBounds || !forecastBounds.source) { el.innerHTML = ""; return; }

  const isTft = forecastBounds.source === "TFT";

  if (isTft) {
    // Multi-Horizon Breakdown
    el.innerHTML = `
      <div style="margin-bottom:10px;">
        <span class="badge-green" style="font-size:11px;">&#9679; TFT (Multi-Horizon P90 Peak)</span>
      </div>
      <div style="display:flex; gap:10px; font-size:12px; color:var(--text-dim); background:#2a2a2a; padding:10px; border-radius:6px;">
        <div><strong>1 Hour:</strong><br><span style="color:#fff;">${forecastBounds.p90_1h?.toFixed(1) || '--'} MW</span></div>
        <div><strong>2 Hours:</strong><br><span style="color:#fff;">${forecastBounds.p90_2h?.toFixed(1) || '--'} MW</span></div>
        <div><strong>6 Hours:</strong><br><span style="color:#fff;">${forecastBounds.p90_6h?.toFixed(1) || '--'} MW</span></div>
        <div><strong>24 Hours:</strong><br><span style="color:#fff;">${forecastBounds.p90_24h?.toFixed(1) || '--'} MW</span></div>
      </div>
    `;
  } else {
    // Fallback LightGBM View
    el.innerHTML = `<span class="badge-amber" style="font-size:11px;">&#9679; LightGBM (Next Hour Only): ${forecastBounds.p90_1h?.toFixed(1) || '--'} MW</span>`;
  }
}

function renderShapBars(featureImportance) {
  const el = document.getElementById("shapBars");
  const ranked = featureImportance?.ranked_features || [];
  if (!ranked.length) { el.innerHTML = `<p class="gri-sub">No feature importance available for this run.</p>`; return; }

  const maxAbs = Math.max(...ranked.map(([, v]) => Math.abs(v)));
  el.innerHTML = ranked.slice(0, 5).map(([name, value]) => {
    const pct = maxAbs > 0 ? (Math.abs(value) / maxAbs) * 100 : 0;
    const color = value >= 0 ? "var(--accent-red)" : "var(--accent-cyan)";
    return `
      <div style="margin-bottom:8px;">
        <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--text-dim); margin-bottom:3px;">
          <span>${name}</span><span>${value >= 0 ? "+" : ""}${value.toFixed(2)} MW</span>
        </div>
        <div style="background:#2a2a2a; border-radius:3px; height:6px; overflow:hidden;">
          <div style="width:${pct}%; background:${color}; height:100%;"></div>
        </div>
      </div>
    `;
  }).join("");
}

function renderGriBar(validationResult) {
  const el = document.getElementById("griBarContainer");
  
  // Keep your existing safety check for the base case
  if (!validationResult || validationResult.gri_before === null || validationResult.gri_before === undefined) {
    el.innerHTML = "";
    return;
  }
  
  const before = validationResult.gri_before;
  const after = validationResult.gri_after;
  const improvement = validationResult.gri_improvement;
  
  // Safely handle the non-convergent (null) cases
  const afterStr = after !== null ? after.toFixed(1) : "N/A";
  const improvementStr = improvement !== null 
    ? `(${improvement >= 0 ? "+" : ""}${improvement.toFixed(1)})` 
    : "(N/A)";
    
  // Only turn it green if it actually improved AND didn't collapse
  const improved = after !== null && after >= before;
  
  el.innerHTML = `
    <div style="display:flex; align-items:center; gap:14px; margin-top:6px;">
      <div style="text-align:center;">
        <div style="font-size:11px; color:var(--text-dim);">BEFORE</div>
        <div style="font-size:20px; font-weight:bold;">${before.toFixed(1)}</div>
      </div>
      <div style="font-size:18px; color:var(--text-dim);">&rarr;</div>
      <div style="text-align:center;">
        <div style="font-size:11px; color:var(--text-dim);">AFTER</div>
        <div style="font-size:20px; font-weight:bold; color:${improved ? 'var(--accent-green)' : 'var(--accent-red)'};">${afterStr}</div>
      </div>
      <div style="font-size:13px; color:${improved ? 'var(--accent-green)' : 'var(--accent-red)'};">
        ${improvementStr}
      </div>
    </div>
  `;
}

function initAgentActions() {
  document.getElementById("approveBtn").addEventListener("click", () => {
    fetch(`${AI_API_URL}/execute/${getSelectedGrid()}`, { method: "POST" })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Execution failed");
        return data;
      })
      .then(data => {
        alert(`Executed: ${JSON.stringify(data.action)}`);
        document.getElementById("agentBriefContainer").classList.add("hidden");
        loadEventTimeline();
      })
      .catch(err => alert(`EXECUTION FAILED: ${err.message}`));
  });

  document.getElementById("rejectBtn").addEventListener("click", () => {
    document.getElementById("agentBriefContainer").classList.add("hidden");
  });
}

// ------------------------------------------------------------
// EVENT TIMELINE
// ------------------------------------------------------------
function loadEventTimeline() {
  const el = document.getElementById("eventTimeline");
  fetch(`${AI_API_URL}/events/${getSelectedGrid()}?limit=30`)
    .then(res => res.json())
    .then(data => {
      const events = data.events || [];
      if (!events.length) {
        el.innerHTML = `<p class="gri-sub">No events yet for this grid.</p>`;
        return;
      }
      el.innerHTML = events.map(ev => {
        const date = new Date(ev.timestamp * 1000).toLocaleTimeString();
        const payloadSummary = JSON.stringify(ev.payload).slice(0, 120);
        return `
          <div class="timeline-entry source-${ev.source}">
            <span class="ts">${date}</span> --
            <span class="type">${ev.event_type}</span>
            <span class="gri-sub">(${ev.source})</span>
            <div class="gri-sub">${payloadSummary}</div>
          </div>
        `;
      }).join("");
    })
    .catch(() => {
      el.innerHTML = `<p class="gri-sub badge-red">Could not load event timeline.</p>`;
    });
}

// ------------------------------------------------------------
// BOOTSTRAP
// ------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  initCheckRisks();
  initAgentActions();
  document.getElementById("refreshEventsBtn").addEventListener("click", loadEventTimeline);
  document.getElementById("gridSelect").addEventListener("change", () => {
    document.getElementById("agentBriefContainer").classList.add("hidden");
    loadEventTimeline();
  });
  loadEventTimeline();
});
