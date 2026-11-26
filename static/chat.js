const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const messagesEl = document.getElementById("chat-messages");

function addMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `message ${sender}-message`;
  div.style.whiteSpace = "pre-line";
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  addMessage(text, "user");
  input.value = "";
  input.focus();

  const thinkingMsg = addMessage("Searching knowledge base...", "bot");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();

    if (!res.ok) {
      thinkingMsg.textContent = data.error || "Something went wrong.";
      return;
    }

    thinkingMsg.textContent = data.reply;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } catch (err) {
    thinkingMsg.textContent = "Could not reach the server. Is app.py running?";
  }
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.dataset.q;
    form.requestSubmit();
  });
});