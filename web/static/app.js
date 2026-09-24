(() => {
  const $ = (id) => document.getElementById(id);
  const messages = $("messages");
  const form = $("chat-form");
  const input = $("chat-input");
  const badgeMode = $("badge-mode");
  const badgeHealth = $("badge-health");
  const badgeProcs = $("badge-procs");

  let trainPollTimer = null;
  let sleepPollTimer = null;
  let testsPollTimer = null;
  let eventAfter = 0;
  let liveEvents = [];
  let sleepEventAfter = 0;
  let sleepEvents = [];
  let testsEventAfter = 0;
  let testsEvents = [];
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

  function setTabStatus(id, running) {
    const el = $(id);
    if (!el) return;
    el.dataset.state = running ? "running" : "idle";
  }

  function isTabRunning(id) {
    const el = $(id);
    return el?.dataset?.state === "running";
  }

  function updateProcessBadges(flags) {
    const parts = [];
    if (flags.train) parts.push("entrenando");
    if (flags.sleep) parts.push("durmiendo");
    if (flags.tests) parts.push("pruebas");
    badgeProcs.textContent = parts.length ? parts.join(" · ") : "procesos idle";
    badgeProcs.classList.toggle("active", parts.length > 0);
    setTabStatus("status-train", !!flags.train);
    setTabStatus("status-sleep", !!flags.sleep);
    setTabStatus("status-tests", !!flags.tests);
  }

  async function refreshHealth() {
    try {
      const h = await api("/health");
      badgeMode.textContent = `LLM: ${h.llm_mode}`;
      const bits = [`ok · engramas ${h.engrams}`];
      if (h.training) bits.push("entrenando");
      if (h.sleeping) bits.push("durmiendo");
      if (h.testing) bits.push("pruebas");
      badgeHealth.textContent = bits.join(" · ");
      updateProcessBadges({
        train: !!h.training,
        sleep: !!h.sleeping,
        tests: !!h.testing,
      });
    } catch (_) {
      badgeHealth.textContent = "sin conexión";
    }
  }

  function formatLog(events) {
    return (events || [])
      .map((e) => {
        const b = e.batch != null ? ` b${e.batch}` : "";
        const eng = e.engrams != null ? ` eng=${e.engrams}` : "";
        return `[${e.kind}]${b}${eng} ${e.message}`;
      })
      .join("\n");
  }

  function maxSeq(events) {
    let m = 0;
    for (const e of events || []) {
      if (e.seq != null) m = Math.max(m, e.seq);
    }
    return m;
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
    // "Épocas / lote": mostrar lote actual (progreso) + épocas por lote (config).
    // Antes solo se mostraba job.epochs (siempre 1) y un typo `j.` rompía el resto.
    const epochsCfg = job.epochs != null ? job.epochs : "—";
    $("tr-epochs").textContent =
      `lote ${cur} · ${epochsCfg} ép/lote` +
      (job.batch_size != null ? ` · bs=${job.batch_size}` : "");
    $("tr-acc").textContent =
      job.accuracy == null ? "—" : (100 * job.accuracy).toFixed(1) + "%";
    $("tr-eng").textContent = job.engrams != null ? job.engrams : "—";
    if ($("tr-family")) {
      const fam = job.last_dataset_family || "—";
      const exps = Array.isArray(job.last_experiment_ids) && job.last_experiment_ids.length
        ? ` (${job.last_experiment_ids.join(",")})`
        : "";
      $("tr-family").textContent = fam === "—" ? "—" : fam + exps;
    }
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

    const merged =
      liveEvents.length > 0
        ? liveEvents
        : Array.isArray(job.events)
          ? job.events
          : [];
    if (merged.length) {
      const log = $("tr-log");
      log.textContent = formatLog(merged);
      log.scrollTop = log.scrollHeight;
    }
  }

  function renderSleepJob(job) {
    if (!job) return;
    const cur = job.current_cycle || job.current_batch || 0;
    const infinite = !!(job.infinite || job.total_cycles == null);
    const bar = $("sl-progress-bar");
    if (bar) {
      if (infinite) {
        bar.style.width = job.running ? "100%" : "0%";
        bar.classList.toggle("infinite", !!job.running);
        $("sl-progress-label").textContent =
          `Ciclo ${cur} (∞ infinito)` + (job.running ? " (en curso)" : "");
      } else {
        const total = Math.max(1, job.total_cycles || job.total_batches || 1);
        bar.style.width = Math.min(100, (100 * cur) / total) + "%";
        bar.classList.remove("infinite");
        $("sl-progress-label").textContent =
          `Ciclo ${cur} / ${job.total_cycles || job.total_batches}` +
          (job.running ? " (en curso)" : "");
      }
    }
    $("sl-status").textContent = job.running
      ? job.cancelled
        ? "deteniendo…"
        : infinite
          ? "durmiendo ∞…"
          : "durmiendo…"
      : job.cancelled
        ? "cancelado"
        : job.job_id
          ? "idle / listo"
          : "idle";
    $("sl-job").textContent = job.job_id || "—";
    $("sl-phase").textContent = job.phase || "—";
    const merged =
      sleepEvents.length > 0
        ? sleepEvents
        : Array.isArray(job.events)
          ? job.events
          : [];
    if (merged.length) {
      const log = $("sl-log");
      log.textContent = formatLog(merged);
      log.scrollTop = log.scrollHeight;
    }
    if (job.last_report) renderSleepReport(job.last_report);
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

  function renderTestsJob(job) {
    if (!job) return;
    const cur = job.current_cycle || job.current_batch || 0;
    const infinite = !!(job.infinite || job.total_cycles == null);
    const bar = $("te-progress-bar");
    if (bar) {
      if (infinite) {
        bar.style.width = job.running ? "100%" : "0%";
        bar.classList.toggle("infinite", !!job.running);
        $("te-progress-label").textContent =
          `Batería ${cur} (∞)` + (job.running ? " (en curso)" : "");
      } else {
        const total = Math.max(1, job.total_cycles || job.total_batches || 1);
        // Dentro de una batería, combina ciclo + pasos si hay total_steps.
        let pct;
        if (job.total_steps > 0 && total === 1) {
          pct = Math.min(100, (100 * (job.step || 0)) / job.total_steps);
        } else {
          pct = Math.min(100, (100 * cur) / total);
        }
        bar.style.width = pct + "%";
        bar.classList.remove("infinite");
        $("te-progress-label").textContent =
          `Batería ${cur} / ${job.total_cycles || job.total_batches}` +
          (job.running ? " (en curso)" : "");
      }
    }
    $("te-status").textContent = job.running
      ? job.cancelled
        ? "deteniendo…"
        : infinite
          ? "ejecutando ∞…"
          : "ejecutando…"
      : job.cancelled
        ? "cancelado"
        : job.job_id
          ? "idle / listo"
          : "idle";
    $("te-job").textContent = job.job_id || "—";
    $("te-progress").textContent =
      job.total_steps > 0
        ? `${job.step || 0} / ${job.total_steps} (${job.phase || "—"})` +
          (infinite ? ` · bat ${cur} ∞` : cur ? ` · bat ${cur}` : "")
        : job.phase || "—";
    const merged =
      testsEvents.length > 0
        ? testsEvents
        : Array.isArray(job.events)
          ? job.events
          : [];
    if (merged.length) {
      const log = $("te-log");
      log.textContent = formatLog(merged);
      log.scrollTop = log.scrollHeight;
    }
    if (job.last_report) renderTests(job.last_report);
  }

  function renderTests(r) {
    if (!r) {
      $("te-report").textContent = "(aún no hay informe)";
      return;
    }
    $("te-eng").textContent = r.engrams;
    $("te-had").textContent = r.had_engrams ? "sí" : "no";
    if ($("te-suite")) {
      const suite = r.experiment_suite;
      if (suite && suite.verdict_counts) {
        const parts = Object.entries(suite.verdict_counts)
          .map(([k, v]) => `${k}=${v}`)
          .sort();
        $("te-suite").textContent = `${suite.rows?.length || 0} filas · ${parts.join(" ")} · ${Math.round(suite.elapsed_ms || 0)} ms`;
      } else {
        $("te-suite").textContent = "—";
      }
    }
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
    // Aislar errores de render de job para no bloquear líquido/CDT/RQM.
    try {
      if (job) renderLiveJob(job);
      if (t.sleep_job) renderSleepJob(t.sleep_job);
      if (t.tests_job) renderTestsJob(t.tests_job);
    } catch (err) {
      console.warn("render job metrics failed", err);
    }

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

    if (t.last_sleep_optimize && !t.sleep_job?.running)
      renderSleepReport(t.last_sleep_optimize);
    if (t.last_field_eval && !t.tests_job?.running) renderTests(t.last_field_eval);
    badgeMode.textContent = `LLM: ${t.llm_mode}`;
    if (t.processes) {
      updateProcessBadges({
        train: (t.processes.active || []).includes("train"),
        sleep: (t.processes.active || []).includes("sleep"),
        tests: (t.processes.active || []).includes("tests"),
      });
    }
  }

  async function refreshTelemetry() {
    try {
      const t = await api("/api/telemetry");
      renderTelemetry(t);
    } catch (_) {}
  }

  function appendTo(bufName, afterName, events) {
    if (!events || !events.length) return;
    let buf = bufName === "train" ? liveEvents : bufName === "sleep" ? sleepEvents : testsEvents;
    let after =
      afterName === "train"
        ? eventAfter
        : afterName === "sleep"
          ? sleepEventAfter
          : testsEventAfter;
    for (const e of events) {
      if (buf.length && buf[buf.length - 1].seq === e.seq) continue;
      buf.push(e);
      if (e.seq != null) after = Math.max(after, e.seq);
    }
    if (buf.length > 300) buf = buf.slice(-300);
    if (bufName === "train") {
      liveEvents = buf;
      eventAfter = after;
    } else if (bufName === "sleep") {
      sleepEvents = buf;
      sleepEventAfter = after;
    } else {
      testsEvents = buf;
      testsEventAfter = after;
    }
  }

  function appendEvents(events) {
    appendTo("train", "train", events);
  }

  async function pollTrainOnce() {
    try {
      const st = await api("/api/train/status");
      if (Array.isArray(st.events) && st.events.length && liveEvents.length === 0) {
        liveEvents = st.events.slice();
        eventAfter = maxSeq(liveEvents);
      }
      renderLiveJob(st);
      const ev = await api("/api/train/events?after=" + eventAfter);
      appendEvents(ev.events || []);
      renderLiveJob(st);
      updateProcessBadges({
        train: !!st.running,
        sleep: isTabRunning("status-sleep"),
        tests: isTabRunning("status-tests"),
      });
      setTabStatus("status-train", !!st.running);
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

  async function pollSleepOnce() {
    try {
      const st = await api("/api/sleep/status");
      if (Array.isArray(st.events) && st.events.length && sleepEvents.length === 0) {
        sleepEvents = st.events.slice();
        sleepEventAfter = maxSeq(sleepEvents);
      }
      renderSleepJob(st);
      const ev = await api("/api/sleep/events?after=" + sleepEventAfter);
      appendTo("sleep", "sleep", ev.events || []);
      renderSleepJob(st);
      setTabStatus("status-sleep", !!st.running);
      if (!st.running) {
        stopSleepPolling();
        refreshTelemetry();
        refreshHealth();
      }
    } catch (_) {}
  }

  function startSleepPolling() {
    if (sleepPollTimer) return;
    sleepPollTimer = setInterval(pollSleepOnce, 400);
    pollSleepOnce();
  }

  function stopSleepPolling() {
    if (sleepPollTimer) {
      clearInterval(sleepPollTimer);
      sleepPollTimer = null;
    }
  }

  async function pollTestsOnce() {
    try {
      const st = await api("/api/tests/status");
      if (Array.isArray(st.events) && st.events.length && testsEvents.length === 0) {
        testsEvents = st.events.slice();
        testsEventAfter = maxSeq(testsEvents);
      }
      renderTestsJob(st);
      const ev = await api("/api/tests/events?after=" + testsEventAfter);
      appendTo("tests", "tests", ev.events || []);
      renderTestsJob(st);
      setTabStatus("status-tests", !!st.running);
      if (!st.running) {
        stopTestsPolling();
        refreshTelemetry();
        refreshHealth();
      }
    } catch (_) {}
  }

  function startTestsPolling() {
    if (testsPollTimer) return;
    testsPollTimer = setInterval(pollTestsOnce, 400);
    pollTestsOnce();
  }

  function stopTestsPolling() {
    if (testsPollTimer) {
      clearInterval(testsPollTimer);
      testsPollTimer = null;
    }
  }

  /** Reconecta a jobs en memoria del servidor sin cancelarlos. */
  async function reconnectProcesses() {
    try {
      const p = await api("/api/processes");
      updateProcessBadges({
        train: (p.active || []).includes("train"),
        sleep: (p.active || []).includes("sleep"),
        tests: (p.active || []).includes("tests"),
      });

      if (p.train) {
        if (Array.isArray(p.train.events) && p.train.events.length) {
          liveEvents = p.train.events.slice();
          eventAfter = maxSeq(liveEvents);
        }
        renderLiveJob(p.train);
        if (p.train.running) startTrainPolling();
      }
      if (p.sleep) {
        if (Array.isArray(p.sleep.events) && p.sleep.events.length) {
          sleepEvents = p.sleep.events.slice();
          sleepEventAfter = maxSeq(sleepEvents);
        }
        renderSleepJob(p.sleep);
        if (p.sleep.running) startSleepPolling();
      }
      if (p.tests) {
        if (Array.isArray(p.tests.events) && p.tests.events.length) {
          testsEvents = p.tests.events.slice();
          testsEventAfter = maxSeq(testsEvents);
        }
        renderTestsJob(p.tests);
        if (p.tests.running) startTestsPolling();
      }
    } catch (_) {}
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
      if (r.route === "train") {
        await reconnectProcesses();
      }
      refreshTelemetry();
      refreshHealth();
    } catch (e) {
      addMsg("agent", "Error: " + e.message);
    }
  });

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
    const infinite = $("chk-infinite-sleep")?.checked !== false;
    sleepEventAfter = 0;
    sleepEvents = [];
    $("sl-log").textContent = "";
    $("sl-status").textContent = "iniciando…";
    try {
      const body = {
        prune_intensity: prune,
        compact_intensity: compact,
        consolidate_first: true,
        infinite,
        cycles: infinite ? null : 1,
      };
      const r = await api("/api/sleep/start", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      if (r.ok) {
        $("sl-job").textContent = r.job_id || "—";
        startSleepPolling();
      } else {
        $("sl-status").textContent = r.message || "ocupado";
      }
      refreshHealth();
    } catch (e) {
      $("sl-report").textContent = "Error: " + e.message;
      $("sl-status").textContent = "error";
    }
  });

  $("btn-sleep-stop").addEventListener("click", async () => {
    try {
      await api("/api/sleep/stop", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{}",
      });
      pollSleepOnce();
    } catch (_) {}
  });

  $("btn-sleep-refresh")?.addEventListener("click", () => {
    pollSleepOnce();
    refreshHealth();
  });

  $("btn-tests").addEventListener("click", async () => {
    const infinite = $("chk-infinite-tests")?.checked !== false;
    testsEventAfter = 0;
    testsEvents = [];
    $("te-log").textContent = "";
    $("te-status").textContent = "iniciando…";
    $("te-report").textContent = "Ejecutando batería…";
    try {
      const body = infinite
        ? { infinite: true, cycles: null }
        : { infinite: false, cycles: 1 };
      const r = await api("/api/tests/start", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      if (r.ok) {
        $("te-job").textContent = r.job_id || "—";
        startTestsPolling();
      } else {
        $("te-status").textContent = r.message || "ocupado";
        $("te-report").textContent = r.message || "ocupado";
      }
      refreshHealth();
    } catch (e) {
      $("te-report").textContent = "Error: " + e.message;
      $("te-status").textContent = "error";
    }
  });

  $("btn-tests-stop").addEventListener("click", async () => {
    try {
      await api("/api/tests/stop", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{}",
      });
      pollTestsOnce();
    } catch (_) {}
  });

  $("btn-tests-refresh").addEventListener("click", async () => {
    try {
      const st = await api("/api/tests/status");
      renderTestsJob(st);
      if (!st.last_report) {
        const last = await api("/api/tests/last");
        renderTests(last.last || null);
      }
    } catch (e) {
      $("te-report").textContent = "Error: " + e.message;
    }
  });

  addMsg(
    "agent",
    "Listo. Chat = interpretación del modelo de campo (decoder). Entrenamiento, Sueño y Pruebas son jobs en servidor: al refrescar la UI se reconecta sin cancelar.",
  );
  refreshHealth();
  refreshTelemetry();
  reconnectProcesses();
  setInterval(() => {
    refreshTelemetry();
    refreshHealth();
  }, 4000);
})();
