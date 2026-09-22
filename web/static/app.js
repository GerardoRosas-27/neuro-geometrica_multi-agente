(() => {
  const messages = document.getElementById("messages");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const badgeMode = document.getElementById("badge-mode");
  const badgeHealth = document.getElementById("badge-health");

  let trainPollTimer = null;
  let eventAfter = 0;
  let liveEvents = [];
  let useSse = true;
  let eventSource = null;

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
      badgeHealth.textContent = `ok · engramas ${h.engrams}` + (h.training ? " · entrenando" : "");
    } catch (e) {
      badgeHealth.textContent = "sin conexión";
    }
  }

  function renderLiveJob(job) {
    if (!job) return;
    const cur = job.current_batch || 0;
    const infinite = !!(job.infinite || job.total_batches == null);
    if (infinite) {
      document.getElementById("tr-progress-bar").style.width = job.running ? "100%" : "0%";
      document.getElementById("tr-progress-bar").classList.toggle("infinite", !!job.running);
      document.getElementById("tr-progress-label").textContent =
        `Lote ${cur} (∞ infinito)` + (job.running ? " (en curso)" : "");
    } else {
      const total = Math.max(1, job.total_batches || 1);
      const pct = Math.min(100, (100 * cur) / total);
      document.getElementById("tr-progress-bar").style.width = pct + "%";
      document.getElementById("tr-progress-bar").classList.remove("infinite");
      document.getElementById("tr-progress-label").textContent =
        `Lote ${cur} / ${job.total_batches}` + (job.running ? " (en curso)" : "");
    }
    document.getElementById("tr-status").textContent = job.running
      ? (job.cancelled ? "deteniendo…" : (infinite ? "entrenando ∞…" : "entrenando…"))
      : (job.cancelled ? "detenido" : (job.job_id ? "idle / listo" : "idle"));
    document.getElementById("tr-job").textContent = job.job_id || "—";
    document.getElementById("tr-epochs").textContent = job.epochs != null ? job.epochs : "—";
    document.getElementById("tr-acc").textContent =
      job.accuracy == null ? "—" : (100 * job.accuracy).toFixed(1) + "%";
    document.getElementById("tr-eng").textContent = job.engrams != null ? job.engrams : "—";
    document.getElementById("tr-dataset").textContent =
      (job.dataset_size != null ? job.dataset_size : "—") +
      (job.dataset_source ? ` (${job.dataset_source})` : "");
    const dsSaved = document.getElementById("tr-ds-saved");
    if (dsSaved) dsSaved.textContent = job.datasets_saved != null ? job.datasets_saved : "—";
    document.getElementById("tr-ckpt").textContent =
      job.last_dataset_path ||
      (job.last_checkpoint && job.last_checkpoint.path) ||
      "—";
    document.getElementById("tr-decoded").textContent = job.last_decoded || "—";

    if (Array.isArray(job.events) && job.events.length) {
      // Preferir log acumulado de polling de /events
      const merged = liveEvents.length ? liveEvents : job.events;
      const log = document.getElementById("tr-log");
      log.textContent = merged
        .map((e) => {
          const b = e.batch != null ? ` b${e.batch}` : "";
          const eng = e.engrams != null ? ` eng=${e.engrams}` : "";
          return `[${e.kind}]${b}${eng} ${e.message}`;
        })
        .join("\n");
      log.scrollTop = log.scrollHeight;
    }
  }

  function renderTelemetry(t) {
    const job = t.live_job || (t.train && t.train.live) || null;
    if (job) renderLiveJob(job);
    else {
      document.getElementById("tr-status").textContent = t.train.training ? "entrenando…" : "idle";
      document.getElementById("tr-epochs").textContent = t.train.epochs_done;
      document.getElementById("tr-acc").textContent =
        t.train.accuracy == null ? "—" : (100 * t.train.accuracy).toFixed(1) + "%";
    }

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

  function appendEvents(events) {
    if (!events || !events.length) return;
    for (const e of events) {
      if (liveEvents.length && liveEvents[liveEvents.length - 1].seq === e.seq) continue;
      liveEvents.push(e);
      if (e.seq != null) eventAfter = Math.max(eventAfter, e.seq);
    }
    if (liveEvents.length > 300) liveEvents = liveEvents.slice(-300);
  }

  async function pollTrainOnce() {
    try {
      const st = await api("/api/train/status");
      renderLiveJob(st);
      const ev = await api("/api/train/events?after=" + eventAfter);
      appendEvents(ev.events || []);
      renderLiveJob(st);
      if (!st.running) {
        stopTrainPolling();
        refreshTelemetry();
        refreshHealth();
      }
    } catch (_) {}
  }

  function startTrainPolling() {
    if (trainPollTimer) return;
    trainPollTimer = setInterval(pollTrainOnce, 500);
    pollTrainOnce();
    if (useSse) startSse();
  }

  function stopTrainPolling() {
    if (trainPollTimer) {
      clearInterval(trainPollTimer);
      trainPollTimer = null;
    }
    stopSse();
  }

  function startSse() {
    stopSse();
    try {
      eventSource = new EventSource("/api/train/stream");
      eventSource.addEventListener("train", (msg) => {
        try {
          const arr = JSON.parse(msg.data);
          if (Array.isArray(arr)) appendEvents(arr);
          pollTrainOnce();
        } catch (_) {}
      });
      eventSource.onerror = () => {
        // Fallback a solo polling
        useSse = false;
        stopSse();
      };
    } catch (_) {
      useSse = false;
    }
  }

  function stopSse() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
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
      const meta = `ruta=${r.route} · in=${r.concept_in} → out=${r.concept_out} · líquido=${Number(r.liquid_score).toFixed(3)} · eng=${r.engrams} · decoded=${r.decoded}`;
      addMsg("agent", r.reply, meta);
      if (r.route === "train") {
        eventAfter = 0;
        liveEvents = [];
        startTrainPolling();
      }
      refreshTelemetry();
      refreshHealth();
    } catch (e) {
      addMsg("agent", "Error: " + e.message);
    }
  });

  document.getElementById("btn-train").addEventListener("click", async () => {
    const infinite = document.getElementById("chk-infinite")?.checked !== false;
    addMsg("user", `[UI] Iniciar entrenamiento en vivo${infinite ? " (∞)" : ""}`);
    eventAfter = 0;
    liveEvents = [];
    try {
      const body = infinite
        ? { batches: null, batch_size: 8, epochs: 1 }
        : { batches: 4, batch_size: 8, epochs: 1 };
      const r = await api("/api/train/start", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      addMsg(
        "agent",
        r.ok
          ? `Entrenamiento ${r.infinite ? "∞" : "finito"} OK · job=${r.job_id}`
          : `No iniciado: ${r.message || "ocupado"}`,
      );
      if (r.ok) startTrainPolling();
      refreshTelemetry();
      refreshHealth();
    } catch (e) {
      addMsg("agent", "Error train: " + e.message);
    }
  });

  document.getElementById("btn-stop").addEventListener("click", async () => {
    addMsg("user", "[UI] Detener entrenamiento");
    try {
      const r = await api("/api/train/stop", { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
      addMsg("agent", r.ok ? "Cancelación pedida" : "No había job activo");
      pollTrainOnce();
    } catch (e) {
      addMsg("agent", "Error stop: " + e.message);
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
    pollTrainOnce();
  });

  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("panel-" + btn.dataset.tab).classList.add("active");
    });
  });

  addMsg("agent", "Listo. Entrenamiento ∞ por defecto: dataset → líquido → CDT + checkpoint por dataset. LLM = decoder. Di «entrena», «sueño» o «estado».");
  refreshHealth();
  refreshTelemetry();
  setInterval(() => { refreshTelemetry(); refreshHealth(); }, 4000);
})();
