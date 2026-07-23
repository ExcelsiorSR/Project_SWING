const DIGITAL_TWIN_API_URL = "http://127.0.0.1:8001";

const GRID_LABELS = { "39": "IEEE 39", "118": "IEEE 118" };
const ACTION_LABELS = { "shed_load": "SHED_LOAD", "redispatch_gen": "REDISPATCH_GEN" };

function renderPolicyGrid(status) {
  const el = document.getElementById("policyGrid");
  let html = "";

  for (const grid of Object.keys(GRID_LABELS)) {
    for (const action of Object.keys(ACTION_LABELS)) {
      const info = status[grid]?.[action] || { trained: false };
      html += `
        <div class="policy-card">
          <h3>${GRID_LABELS[grid]} -- ${ACTION_LABELS[action]}</h3>
          ${info.trained
            ? `
              <div class="status-line badge-green">&#9679; Trained and active</div>
              <div class="status-line">Size: ${info.size_kb} KB</div>
              <div class="status-line">Last trained: ${new Date(info.last_trained).toLocaleString()}</div>
            `
            : `
              <div class="status-line badge-amber">&#9679; Not trained yet</div>
              <div class="status-line">Falling back to bisection search for this combination.</div>
            `
          }
        </div>
      `;
    }
  }

  el.innerHTML = html;
}

window.addEventListener("DOMContentLoaded", () => {
  fetch(`${DIGITAL_TWIN_API_URL}/rl_status`)
    .then(res => res.json())
    .then(status => renderPolicyGrid(status))
    .catch(() => {
      document.getElementById("policyGrid").innerHTML =
        `<p class="status-line badge-red">Could not reach the Digital Twin API -- is api.py running?</p>`;
    });
});
