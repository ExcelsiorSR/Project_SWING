const AI_API_URL = "http://127.0.0.1:8002";

let chatHistory = [];

function getSelectedGrid() {
  return document.getElementById("gridSelect").value;
}

function appendMessage(role, content) {
  const chatWindowEl = document.getElementById("chatWindow");
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.innerHTML = `
    <div class="label">${role === "user" ? "YOU" : "AGENT"}</div>
    <div class="bubble">${content}</div>
  `;
  chatWindowEl.appendChild(div);
  chatWindowEl.scrollTop = chatWindowEl.scrollHeight;
}

function sendMessage(messageOverride) {
  const input = document.getElementById("chatInput");
  const btn = document.getElementById("sendBtn");
  const message = messageOverride || input.value.trim();
  if (!message) return;

  appendMessage("user", message);
  chatHistory.push({ role: "user", content: message });
  input.value = "";
  btn.disabled = true;
  btn.textContent = "Thinking...";

  fetch(`${AI_API_URL}/chat/${getSelectedGrid()}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history: chatHistory })
  })
    .then(async (res) => {
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Chat request failed");
      return data;
    })
    .then(data => {
      appendMessage("assistant", data.response);
      chatHistory.push({ role: "assistant", content: data.response });
    })
    .catch(err => {
      appendMessage("assistant", `Could not reach the AI Architecture service: ${err.message}`);
    })
    .finally(() => {
      btn.disabled = false;
      btn.textContent = "Send";
    });
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("sendBtn").addEventListener("click", () => sendMessage());
  document.getElementById("chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });
  document.querySelectorAll(".suggestion-chip").forEach(chip => {
    chip.addEventListener("click", () => sendMessage(chip.dataset.q));
  });
  document.getElementById("gridSelect").addEventListener("change", () => {
    chatHistory = [];
    document.getElementById("chatWindow").innerHTML = `
      <div class="chat-msg assistant">
        <div class="label">AGENT</div>
        <div class="bubble">Switched grid context. Ask me anything about the current situation.</div>
      </div>
    `;
  });
});
