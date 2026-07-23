const AI_API_URL = "http://127.0.0.1:8002";
let forecastChart = null;

function getSelectedGrid() {
  return document.getElementById("gridSelect").value;
}

function initChart() {
  const ctx = document.getElementById('forecastChart').getContext('2d');
  
  // Initialize an empty chart
  forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['Now', '+1 Hour', '+2 Hours', '+6 Hours', '+24 Hours'],
      datasets: []
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#aaa' } },
        tooltip: { backgroundColor: 'rgba(0,0,0,0.8)' }
      },
      scales: {
        x: { grid: { color: '#333' }, ticks: { color: '#888' } },
        y: { 
            grid: { color: '#333' }, 
            ticks: { color: '#888' },
            title: { display: true, text: 'Demand (MW)', color: '#888' }
        }
      }
    }
  });
}

function fetchAndRenderForecast() {
  const btn = document.getElementById("refreshForecastBtn");
  btn.disabled = true;
  btn.textContent = "Running Inference...";

  // Provide basic telemetry to trigger the forecast node
  const mockTelemetry = {
    hour_sin: 0.5, hour_cos: 0.5, dayofweek: 3, month: 7,
    load_lag_1: 0.85, load_lag_2: 0.86, load_lag_24: 0.82, Northern_Region_Avg_T2M: 32.5
  };

  fetch(`${AI_API_URL}/propose/${getSelectedGrid()}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(mockTelemetry)
  })
    .then(res => res.json())
    .then(data => {
      const bounds = data.forecast_bounds;
      const currentMw = data.proposed_action ? data.forecast_bounds.capacity_max : (bounds.p50_1h || 0); // Approximate current
      
      document.getElementById("modelSource").textContent = bounds.source === "TFT" ? "TFT (Multi-Horizon)" : "LightGBM (Next-Hour)";

      // Build the datasets
      let p90Data = [null, bounds.p90_1h, bounds.p90_2h, bounds.p90_6h, bounds.p90_24h];
      let p50Data = [null, bounds.p50_1h, null, null, null]; // LightGBM typically gives p50 for 1h
      let p10Data = [null, bounds.p10_1h, null, null, null]; 

      forecastChart.data.datasets = [
        {
          label: 'P90 (High Risk)',
          data: p90Data,
          borderColor: '#ff4757', // accent-red
          backgroundColor: 'rgba(255, 71, 87, 0.1)',
          fill: 1, // Fill to the next dataset (P50)
          tension: 0.4
        },
        {
          label: 'P50 (Expected)',
          data: p50Data,
          borderColor: '#00d4ff', // accent-cyan
          backgroundColor: 'transparent',
          borderDash: [5, 5],
          tension: 0.4
        },
        {
          label: 'P10 (Low Risk)',
          data: p10Data,
          borderColor: '#2ed573', // accent-green
          backgroundColor: 'rgba(0, 212, 255, 0.1)',
          fill: 1,
          tension: 0.4
        }
      ];
      
      forecastChart.update();
    })
    .catch(err => console.error("Forecast fetch failed:", err))
    .finally(() => {
      btn.disabled = false;
      btn.textContent = "Pull Latest Forecast";
    });
}

window.addEventListener("DOMContentLoaded", () => {
  initChart();
  document.getElementById("refreshForecastBtn").addEventListener("click", fetchAndRenderForecast);
  document.getElementById("gridSelect").addEventListener("change", fetchAndRenderForecast);
  
  // Auto-fetch on load
  fetchAndRenderForecast();
});