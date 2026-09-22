(() => {
  const $ = (id) => document.getElementById(id);
  const messages = $("messages");
  const form = $("chat-form");
  const input = $("chat-input");
  const badgeMode = $("badge-mode");
  const badgeHealth = $("badge-health");

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
      badgeHealth.textContent =
        `ok · engramas ${h.engrams}` + (h.training ? " · entrenando" : "");
    } catch (_) {
      badgeHealth.textContent = "sin conexión";
    }
  }

  function renderLiveJob(job) {
    if (!job) return;
    const cur = job.current_batch || 0;
    const infinite = !!(job.infinite || job.total_batches == null);
    const bar = $("tr-progress-bar");
    if (infinite) {
      bar.style.width = job.running ? "100%" : "0%";
      bar.classList.toggle("infinite", !!job.running);
      $("tr-progress-label").textContent =
        `Lote ${cur} (∞ infinito)` + (job.running ? " (en curso)" : "");
    } else {
      const total = Math.max(1, job.total_batches || 1);
      bar.style.width = Math.min(100, (100 * cur) / total) + "%";
      bar.classList.remove("infinite");
      $("tr-progress-label").textContent =
        `Lote ${cur} / ${job.total_batches}` + (job.running ? " (en curso)" : "");
    }
    $("tr-status").textContent = job.running
      ? job.cancelled
        ? "deteniendo…"
        : infinite
          ? "entrenando ∞…"
          : "entrenando…"
      : job.cancelled
        ? "detenido"
        : job.job_id
          ? "idle / listo"
          : "idle";
    $("tr-job").textContent = job.job_id || "—";
    $("tr-epochs").textContent = job.epochs != null ? job.epochs : "—";
    $("tr-acc").textContent =
      job.accuracy == null ? "—" : (100 * job.accuracy).toFixed(1) + "%";
    $("tr-eng").textContent = job.engrams != null ? job.engrams : "—";
    $("tr-dataset").textContent =
      (job.dataset_size != null ? job.dataset_size : "—") +
      (job.dataset_source ? ` (${job.dataset_source})` : "");
    $("tr-ds-saved").textContent =
      job.datasets_saved != null ? job.datasets_saved : "—";
    $("tr-ckpt").textContent =
      job.last_dataset_path ||
      (job.last_checkpoint && job.last_checkpoint.path) ||
      "—";
    $("tr-decoded").textContent = job.last_decoded || "—";

    if (Array.isArray(job.events) && job.events.length) {
      const merged = liveEvents.length ? liveEvents : job.events;
      const log = $("tr-log");
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

  function renderSleepReport(r) {
    if (!r) return;
    $("sl-f-before").textContent = Number(r.free_energy_before).toFixed(4);
    $("sl-f-after").textContent = Number(r.free_energy_after).toFixed(4);
    $("sl-sym-before").textContent = Number(r.symmetry_before).toFixed(4);
    $("sl-sym-after").textContent = Number(r.symmetry_after).toFixed(4);
    $("sl-hs").textContent = Number(r.handshake).toFixed(4);
    $("sl-pruned").textContent = r.routes_pruned;
    $("sl-phasors").textContent = r.phasors_compacted;
    $("sl-edges").textContent = `${r.edges_compacted} / ${r.nodes_compacted}`;
    $("sl-eng").textContent = r.engrams;
    $("sl-report").textContent = JSON.stringify(r, null, 2);
  }

  function renderTests(r) {
    if (!r) {
      $("te-report").textContent = "(aún no hay informe)";
      return;
    }
    $("te-eng").textContent = r.engrams;
    $("te-had").textContent = r.had_engrams ? "sí" : "no";
    $("te-id-acc").textContent =
      (100 * (r.identity_accuracy || 0)).toFixed(1) +
      `% (${r.identity_correct}/${r.identity_total})`;
    $("te-sh-acc").textContent =
      r.shifted_accuracy == null
        ? "— (sin cues RQM)"
        : (100 * r.shifted_accuracy).toFixed(1) +
          `% (${r.shifted_correct}/${r.shifted_total})`;
    $("te-lat").textContent =
      Number(r.liquid_latency_us_mean).toFixed(2) +
      ` (p50 ${Number(r.liquid_latency_us_p50).toFixed(2)})`;
    $("te-recall").textContent =
      r.engram_recall_mean == null
        ? "—"
        : Number(r.engram_recall_mean).toFixed(3);
    $("te-f").textContent =
      r.free_energy == null ? "—" : Number(r.free_energy).toFixed(4);
    $("te-sym").textContent =
      r.symmetry == null ? "—" : Number(r.symmetry).toFixed(4);
    $("te-hs").textContent =
      r.handshake == null ? "—" : Number(r.handshake).toFixed(4);
    $("te-routes").textContent = JSON.stringify(r.route_histogram || {});
    $("te-concepts").textContent = JSON.stringify(r.per_concept || [], null, 2);
    $("te-report").textContent = JSON.stringify(r, null, 2);
  }

  function renderTelemetry(t) {
    const job = t.live_job || (t.train && t.train.live) || null;
    if (job) renderLiveJob(job);

    $("li-q").textContent = t.liquid.queries;
    $("li-score").textContent = Number(t.liquid.score_last).toFixed(4);
    $("li-avg").textContent = Number(t.liquid.score_avg).toFixed(4);
    $("li-lat").textContent = Number(t.liquid.latency_us_last).toFixed(2);
    $("li-pct").textContent = Number(t.liquid.route_pct).toFixed(1) + "%";
    $("li-route").textContent = t.liquid.last_route || "—";

    $("cd-eng").textContent = t.cdt.engram_count;
    $("cd-wake").textContent = t.cdt.wake_buffer;
    $("cd-sleeps").textContent = t.cdt.sleeps;

    $("rq-inf").textContent = t.rqm.infer_calls;
    $("rq-tr").textContent = t.rqm.train_calls;
    $("rq-pct").textContent = Number(t.rqm.route_pct).toFixed(1) + "%";
    $("rq-cues").textContent = t.rqm.relational_cues;

    if (t.last_sleep_optimize) renderSleepReport(t.last_sleep_optimize);
    if (t.last_field_eval) renderTests(t.last_field_eval);
    badgeMode.textContent = `LLM: ${t.llm_mode}`;
  }

  async function refreshTelemetry() {
    try {
      const t = await api("/api/telemetry");
      renderTelemetry(t);
    } catch (_) {}
  }

  function appendEvents(events) {
    if (!events || !events.length) return;
    for (const e of events) {
      if (liveEvents.length && liveEvents[liveEvents.length - 1].seq === e.seq)
        continue;
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

  // Tabs
  document.querySelectorAll(".main-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document
        .querySelectorAll(".main-tab")
        .forEach((b) => b.classList.remove("active"));
      document
        .querySelectorAll(".main-panel")
        .forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $("main-" + btn.dataset.main).classList.add("active");
    });
  });

  // Chat — solo campo / decoder (sin controles de train)
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
      const meta = `ruta=${r.route} · in=${r.concept_in}→out=${r.concept_out} · líquido=${Number(r.liquid_score).toFixed(3)} · eng=${r.engrams} · decoded=${r.decoded}`;
      addMsg("agent", r.reply, meta);
      refreshTelemetry();
      refreshHealth();
    } catch (e) {
      addMsg("agent", "Error: " + e.message);
    }
  });

  // Train
  $("btn-train").addEventListener("click", async () => {
    const infinite = $("chk-infinite")?.checked !== false;
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
      $("tr-status").textContent = r.ok
        ? r.infinite
          ? "entrenando ∞…"
          : "entrenando…"
        : r.message || "ocupado";
      if (r.ok) startTrainPolling();
      refreshTelemetry();
      refreshHealth();
    } catch (e) {
      $("tr-status").textContent = "Error: " + e.message;
    }
  });

  $("btn-stop").addEventListener("click", async () => {
    try {
      await api("/api/train/stop", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{}",
      });
      pollTrainOnce();
    } catch (_) {}
  });

  $("btn-refresh-train").addEventListener("click", () => {
    refreshTelemetry();
    refreshHealth();
    pollTrainOnce();
  });

  // Sleep sliders
  function bindSlider(id, valId) {
    const sl = $(id);
    const val = $(valId);
    const sync = () => {
      val.textContent = (sl.value / 100).toFixed(2);
    };
    sl.addEventListener("input", sync);
    sync();
  }
  bindSlider("sl-prune", "sl-prune-val");
  bindSlider("sl-compact", "sl-compact-val");

  $("btn-sleep").addEventListener("click", async () => {
    const prune = $("sl-prune").value / 100;
    const compact = $("sl-compact").value / 100;
    try {
      const r = await api("/api/sleep", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          prune_intensity: prune,
          compact_intensity: compact,
          consolidate_first: true,
        }),
      });
      renderSleepReport(r);
      refreshTelemetry();
      refreshHealth();
    } catch (e) {
      $("sl-report").textContent = "Error: " + e.message;
    }
  });

  // Tests
  $("btn-tests").addEventListener("click", async () => {
    $("te-report").textContent = "Ejecutando batería…";
    try {
      const r = await api("/api/tests/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{}",
      });
      renderTests(r);
      refreshTelemetry();
      refreshHealth();
    } catch (e) {
      $("te-report").textContent = "Error: " + e.message;
    }
  });

  $("btn-tests-refresh").addEventListener("click", async () => {
    try {
      const st = await api("/api/tests/last");
      renderTests(st.last || null);
    } catch (e) {
      $("te-report").textContent = "Error: " + e.message;
    }
  });

  addMsg(
    "agent",
    "Listo. Chat = interpretación del modelo de campo (decoder). Entrenamiento y Sueño están en pestañas aparte. Pruebas evalúa el modelo ya consolidado.",
  );
  refreshHealth();
  refreshTelemetry();
  setInterval(() => {
    refreshTelemetry();
    refreshHealth();
  }, 4000);
})();
