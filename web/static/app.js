(() => {
  const messages = document.getElementById("messages");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const badgeMode = document.getElementById("badge-mode");
  const badgeHealth = document.getElementById("badge-health");

  function addMsg(role, text, meta) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.textContent = text;
    if (meta) {
      const m = document.createElement("span");
      m.className = "meta";
      m.textContent = meta;
      el.appendChild(m);
    }
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
  }

  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) throw new Error(`${path} → ${res.status}`);
    return res.json();
  }

  async function refreshHealth() {
    try {
      const h = await api("/health");
      badgeMode.textContent = `LLM: ${h.llm_mode}`;
      badgeHealth.textContent = `ok · engramas ${h.engrams}`;
    } catch (e) {
      badgeHealth.textContent = "sin conexión";
    }
  }

  function renderTelemetry(t) {
    document.getElementById("tr-status").textContent = t.train.training ? "entrenando…" : "idle";
    document.getElementById("tr-epochs").textContent = t.train.epochs_done;
    document.getElementById("tr-acc").textContent =
      t.train.accuracy == null ? "—" : (100 * t.train.accuracy).toFixed(1) + "%";
    document.getElementById("tr-log").textContent = (t.train.events_tail || [])
      .map((e) => `[${e.kind}] ${e.detail} (eng=${e.engrams})`)
      .join("\n") || "(sin eventos)";

    document.getElementById("li-q").textContent = t.liquid.queries;
    document.getElementById("li-score").textContent = Number(t.liquid.score_last).toFixed(4);
    document.getElementById("li-avg").textContent = Number(t.liquid.score_avg).toFixed(4);
    document.getElementById("li-lat").textContent = Number(t.liquid.latency_us_last).toFixed(2);
    document.getElementById("li-pct").textContent = Number(t.liquid.route_pct).toFixed(1) + "%";
    document.getElementById("li-route").textContent = t.liquid.last_route || "—";

    document.getElementById("cd-eng").textContent = t.cdt.engram_count;
    document.getElementById("cd-wake").textContent = t.cdt.wake_buffer;
    document.getElementById("cd-sleeps").textContent = t.cdt.sleeps;
    document.getElementById("cd-last").textContent = t.cdt.last_sleep
      ? JSON.stringify(t.cdt.last_sleep, null, 2)
      : "(aún no hay sueño)";

    document.getElementById("rq-inf").textContent = t.rqm.infer_calls;
    document.getElementById("rq-tr").textContent = t.rqm.train_calls;
    document.getElementById("rq-pct").textContent = Number(t.rqm.route_pct).toFixed(1) + "%";
    document.getElementById("rq-cues").textContent = t.rqm.relational_cues;
  }

  async function refreshTelemetry() {
    try {
      const t = await api("/api/telemetry");
      renderTelemetry(t);
      badgeMode.textContent = `LLM: ${t.llm_mode}`;
    } catch (_) {}
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    addMsg("user", message);
    try {
      const r = await api("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const meta = `ruta=${r.route} · in=${r.concept_in} → out=${r.concept_out} · líquido=${Number(r.liquid_score).toFixed(3)} · eng=${r.engrams}`;
      addMsg("agent", r.reply, meta);
      refreshTelemetry();
      refreshHealth();
    } catch (e) {
      addMsg("agent", "Error: " + e.message);
    }
  });

  document.getElementById("btn-train").addEventListener("click", async () => {
    addMsg("user", "[UI] Iniciar entrenamiento");
    try {
      const r = await api("/api/train/start", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ epochs: 3 }),
      });
      addMsg(
        "agent",
        r.ok
          ? `Entrenamiento OK · épocas=${r.epochs} · engramas=${r.engrams} · acc=${r.accuracy}`
          : "Entrenamiento ocupado o fallido",
      );
      refreshTelemetry();
      refreshHealth();
    } catch (e) {
      addMsg("agent", "Error train: " + e.message);
    }
  });

  document.getElementById("btn-sleep").addEventListener("click", async () => {
    addMsg("user", "[UI] Sueño / consolidar");
    try {
      const r = await api("/api/sleep", { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
      addMsg("agent", `Sueño: ${r.episodes_consolidated} episodios · engramas ${r.engrams_before}→${r.engrams_after}`);
      refreshTelemetry();
      refreshHealth();
    } catch (e) {
      addMsg("agent", "Error sueño: " + e.message);
    }
  });

  document.getElementById("btn-refresh").addEventListener("click", () => {
    refreshTelemetry();
    refreshHealth();
  });

  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("panel-" + btn.dataset.tab).classList.add("active");
    });
  });

  addMsg("agent", "Listo. Di «hola», «entrena», «sueño» o «estado». GGUF opcional (GEMMA2_GGUF).");
  refreshHealth();
  refreshTelemetry();
  setInterval(() => { refreshTelemetry(); refreshHealth(); }, 4000);
})();
