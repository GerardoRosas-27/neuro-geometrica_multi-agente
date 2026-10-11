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

  const MODE_LABEL = {
    field_decoder: "Decoder del campo",
    gemma_raw: "Gemma 2 original",
  };
  const CHAT_STORE_KEY = "chat.history.v1";
  const CHAT_STORE_CAP = 200; // últimos N mensajes (user+agent)

  function loadChatHistory() {
    try {
      const raw = localStorage.getItem(CHAT_STORE_KEY);
      if (!raw) return [];
      const arr = JSON.parse(raw);
      return Array.isArray(arr) ? arr : [];
    } catch (_) {
      return [];
    }
  }
  function saveChatHistory(list) {
    try {
      const capped = list.length > CHAT_STORE_CAP
        ? list.slice(list.length - CHAT_STORE_CAP)
        : list;
      localStorage.setItem(CHAT_STORE_KEY, JSON.stringify(capped));
      return capped;
    } catch (_) {
      return list;
    }
  }
  function pushChatTurn(entry) {
    const list = loadChatHistory();
    list.push(entry);
    saveChatHistory(list);
  }
  function clearChatHistory() {
    try {
      localStorage.removeItem(CHAT_STORE_KEY);
    } catch (_) {}
  }

  function addMsg(role, text, meta, mode, opts) {
    const o = opts || {};
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    if (o.error) el.classList.add("error");
    if (mode) {
      const b = document.createElement("span");
      const ext = mode === "gemma_raw" && (o.llm ? o.llm !== "Gemma local" : false);
      b.className = `mode-badge ${mode}${ext ? " external" : ""}`;
      b.textContent = ext ? `LLM crudo · ${o.llm}` : MODE_LABEL[mode] || mode;
      el.appendChild(b);
      el.appendChild(document.createElement("br"));
    }
    el.appendChild(document.createTextNode(text));
    if (meta) {
      const m = document.createElement("span");
      m.className = "meta";
      m.textContent = meta;
      el.appendChild(m);
    }
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
    // Persistir solo turnos definitivos (no "pending").
    if (!o.skipStore && !/\bpending\b/.test(role)) {
      pushChatTurn({
        role: role.split(/\s+/)[0],
        text,
        meta: meta || null,
        mode: mode || null,
        llm: o.llm || null,
        error: !!o.error,
        t: o.t || Date.now(),
      });
    }
    return el;
  }

  function renderStoredHistory() {
    const list = loadChatHistory();
    for (const e of list) {
      addMsg(e.role || "agent", e.text || "", e.meta || null, e.mode || null, {
        skipStore: true,
        error: !!e.error,
        llm: e.llm || null,
        t: e.t,
      });
    }
    return list.length;
  }

  // Bandera «Decoder del campo» (persistida). ON = campo + Gemma decoder; OFF = Gemma 2 original.
  const chkField = $("chk-field-decoder");
  const modeHint = $("chat-mode-hint");
  const FIELD_KEY = "chat.fieldDecoder";
  let rawAvailable = null;
  let modelStatus = null;
  // LLM activo de toda la app (Gemma local o API externa OpenAI-compatible).
  let activeLlm = { id: "gemma_local", label: "Gemma local", kind: "local", model: "" };
  function isExternal() {
    return activeLlm && activeLlm.kind === "openai_compatible";
  }
  const CHAT_TIMEOUT_MS = 120000;
  try {
    const saved = localStorage.getItem(FIELD_KEY);
    if (saved !== null) chkField.checked = saved === "1";
  } catch (_) {}
  function chatMode() {
    return chkField.checked ? "field_decoder" : "gemma_raw";
  }
  function syncChatMode() {
    const on = chkField.checked;
    const ext = isExternal();
    const name = ext ? activeLlm.label : "Gemma";
    modeHint.classList.remove("warn");
    if (on) {
      modeHint.textContent = `Activo · campo v4 (Dφ) → ${name} verbaliza (decoder-only)`;
      input.placeholder = "Pide una transformación al campo: «rota 0.3», «refleja», «escala»…";
    } else {
      modeHint.textContent = ext
        ? `Inactivo · ${name} directo (sin campo)`
        : "Inactivo · Gemma 2 congelado original (sin campo)";
      input.placeholder = ext ? `Habla con ${name}…` : "Habla con Gemma 2 original…";
      if (!ext && rawAvailable === false) {
        modeHint.textContent += " · " + modelStatusText();
        modeHint.classList.add("warn");
      }
    }
    if (on && !ext && rawAvailable === false) {
      modeHint.textContent += " · sin LLM: estado del campo (" + modelStatusText() + ")";
    }
  }
  function modelStatusText() {
    const m = modelStatus;
    if (!m) return "modelo no disponible";
    if (m.state === "downloading") {
      const mb = (m.downloaded_bytes || 0) / 1e6;
      const tot = m.total_bytes ? m.total_bytes / 1e6 : null;
      return tot
        ? `descargando modelo ${mb.toFixed(0)}/${tot.toFixed(0)} MB (${((100 * mb) / tot).toFixed(0)}%)`
        : `descargando modelo ${mb.toFixed(0)} MB`;
    }
    if (m.state === "loading") return "cargando modelo…";
    if (m.state === "error") return "error de modelo: " + m.detail;
    if (m.state === "disabled") return "GGUF ausente (descarga desactivada)";
    if (m.state === "ready") return "modelo listo";
    return "GGUF no disponible en el servidor";
  }
  chkField.addEventListener("change", () => {
    try {
      localStorage.setItem(FIELD_KEY, chkField.checked ? "1" : "0");
    } catch (_) {}
    syncChatMode();
  });
  syncChatMode();

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
      const h = await api("/api/status");
      badgeMode.textContent = `local: ${h.llm_mode}`;
      if (h.llm_active) setActiveLlm(h.llm_active);
      if (h.model) modelStatus = h.model;
      if (typeof h.raw_gemma_available === "boolean") {
        rawAvailable = h.raw_gemma_available;
        syncChatMode();
      }
      const f = h.field || {};
      const bits = [`ok · Dφ ${f.trained ? "entrenado" : "sin entrenar"} · CDT ${h.wake_buffer ?? 0}`];
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
        return `[${e.kind}]${b} ${e.message}`;
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

  function setText(id, value) {
    const el = $(id);
    if (el) el.textContent = value;
  }

  function renderLiveJob(job) {
    if (!job) return;
    try {
      const cur =
        job.current_batch != null && job.current_batch !== ""
          ? Number(job.current_batch)
          : 0;
      const infinite = !!(job.infinite || job.total_batches == null);
      const bar = $("tr-progress-bar");
      if (bar) {
        if (infinite) {
          bar.style.width = job.running ? "100%" : "0%";
          bar.classList.toggle("infinite", !!job.running);
        } else {
          const total = Math.max(1, Number(job.total_batches) || 1);
          bar.style.width = Math.min(100, (100 * cur) / total) + "%";
          bar.classList.remove("infinite");
        }
      }
      setText(
        "tr-progress-label",
        infinite
          ? `Lote ${cur} (∞ infinito)` + (job.running ? " (en curso)" : "")
          : `Lote ${cur} / ${job.total_batches}` +
            (job.running ? " (en curso)" : "")
      );
      setText(
        "tr-status",
        job.running
          ? job.cancelled
            ? "deteniendo…"
            : infinite
              ? "entrenando ∞…"
              : "entrenando…"
          : job.cancelled
            ? "detenido"
            : job.job_id
              ? "idle / listo"
              : "idle"
      );
      setText("tr-job", job.job_id || "—");
      setText("tr-steps", job.steps != null ? `${job.steps} pasos/lote` : "—");
      const r = job.last_report || null;
      const m = (r && r.metrics) || null;
      const pct = (x) => (x == null ? "—" : (100 * Number(x)).toFixed(1) + "%");
      setText("tr-acc", m ? pct(m.acc) : "—");
      setText("tr-relerr", m ? Number(m.rel_err_1step).toFixed(4) : "—");
      setText("tr-rollout", m ? `${pct(m.rollout_acc)} (h${m.rollout_h})` : "—");
      setText("tr-audit", m ? `queries=${m.test_queries} · leaked=${m.leaked}` : "—");
      setText("tr-cdt", r ? r.cdt_experiences : "—");
      setText("tr-trusted", r ? (r.trusted_families.join(", ") || "ninguna") : "—");
      setText("tr-dataset", r ? `0x${Number(r.seed).toString(16).toUpperCase()} · ${r.dataset_hash.slice(0, 12)} · ${r.audit_status}` : "—");
      setText("tr-fam", m ? m.fam_acc.map(([f, a]) => `${f} ${(100 * a).toFixed(0)}%`).join(" · ") : "—");
      setText(
        "tr-ckpt",
        job.last_checkpoint
          ? `${job.last_checkpoint.path} (${(job.last_checkpoint.weights_hash || "").slice(0, 12)})`
          : "—"
      );

      const merged =
        liveEvents.length > 0
          ? liveEvents
          : Array.isArray(job.events)
            ? job.events
            : [];
      const log = $("tr-log");
      if (log) {
        if (merged.length) {
          log.textContent = formatLog(merged);
          log.scrollTop = log.scrollHeight;
        } else {
          log.textContent = job.running
            ? "(sin eventos aún — esperando lote…)"
            : job.job_id
              ? "(sin eventos en memoria — refresca o espera el próximo lote)"
              : "(consola vacía)";
        }
      }
    } catch (err) {
      console.warn("renderLiveJob failed", err);
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
    const pct = (x) => (100 * Number(x)).toFixed(1) + "%";
    setText("sl-exp", r.experiences_consolidated);
    setText("sl-teacher", r.teacher_numbers);
    setText("sl-trusted", (r.trusted_families || []).join(", ") || "ninguna");
    setText("sl-acc", `${pct(r.before.acc)} → ${pct(r.after.acc)}`);
    setText("sl-relerr", `${Number(r.before.rel_err_1step).toFixed(4)} → ${Number(r.after.rel_err_1step).toFixed(4)}`);
    setText("sl-rollout", `${pct(r.before.rollout_acc)} → ${pct(r.after.rollout_acc)} (h${r.after.rollout_h})`);
    setText("sl-audit", `CDT borrada=${r.cdt_dropped ? "sí" : "no"} · queries=${r.after.test_queries} · leaked=${r.after.leaked}`);
    setText("sl-ms", (Number(r.sleep_ms) / 1000).toFixed(1) + " s · " + r.steps + " pasos");
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
    setText("te-verdict", `${r.pass} PASS · ${r.fail} FAIL`);
    setText("te-e35", `A=${Number(r.e35_mean_a).toFixed(3)} · B=${Number(r.e35_mean_b).toFixed(3)} · Δ=${(r.e35_mean_b - r.e35_mean_a >= 0 ? "+" : "") + (r.e35_mean_b - r.e35_mean_a).toFixed(3)}`);
    setText("te-elapsed", (Number(r.elapsed_ms) / 1000).toFixed(1) + " s");
    const rows = (r.rows || [])
      .map((x) => `${x.verdict.padEnd(4)} ${x.test} [${x.seed}] ${x.value} (umbral: ${x.threshold})`)
      .join("\n");
    $("te-rows").textContent = rows + ((r.notes || []).length ? "\n\nnotas: " + r.notes.join(" · ") : "");
    $("te-report").textContent = JSON.stringify(r, null, 2);
  }

  function renderTelemetry(t) {
    if (!t) return;
    const job = t.live_job || (t.train && t.train.live) || null;
    // Aislar errores de render de job para no bloquear el resto.
    try {
      if (job) renderLiveJob(job);
      if (t.sleep_job) renderSleepJob(t.sleep_job);
      if (t.tests_job) renderTestsJob(t.tests_job);
    } catch (err) {
      console.warn("render job metrics failed", err);
    }

    try {
      const f = t.field || {};
      const m = t.metrics || {};
      setText("fv-state", f.trained ? `entrenado (${f.origin})` : "sin entrenar");
      setText("fv-params", f.params != null ? f.params : "—");
      setText("fv-hash", f.weights_hash ? f.weights_hash.slice(0, 16) : "—");
      setText("fv-wake", f.wake_experiences != null ? f.wake_experiences : "—");
      setText("fv-batches", f.batches_total != null ? f.batches_total : "—");
      setText("fv-sleeps", f.sleeps_total != null ? f.sleeps_total : "—");
      setText("fv-chat", m.field_queries != null ? `${m.field_queries} (${m.field_correct} <5%)` : "—");
      setText("fv-relerr", t.field_rel_err_avg != null ? Number(t.field_rel_err_avg).toFixed(4) : "—");
      setText("fv-lat", m.field_latency_us_last != null ? Number(m.field_latency_us_last).toFixed(0) : "—");
    } catch (err) {
      console.warn("render field failed", err);
    }

    try {
      if (t.field && t.field.last_sleep && !t.sleep_job?.running)
        renderSleepReport(t.field.last_sleep);
      if (t.last_smoke && !t.tests_job?.running) renderTests(t.last_smoke);
      if (badgeMode) badgeMode.textContent = `LLM: ${t.llm_mode || "—"}`;
      if (t.processes) {
        updateProcessBadges({
          train: (t.processes.active || []).includes("train"),
          sleep: (t.processes.active || []).includes("sleep"),
          tests: (t.processes.active || []).includes("tests"),
        });
      }
    } catch (err) {
      console.warn("render telemetry extras failed", err);
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
      // Sembrar consola desde status (incluye historial) si el buffer local está vacío
      // o si el servidor trae un event_seq más alto que el nuestro.
      if (Array.isArray(st.events) && st.events.length) {
        const serverMax = maxSeq(st.events);
        if (liveEvents.length === 0 || serverMax > eventAfter) {
          liveEvents = st.events.slice();
          eventAfter = serverMax;
        }
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
  function showTab(name) {
    document
      .querySelectorAll(".main-tab")
      .forEach((b) => b.classList.toggle("active", b.dataset.main === name));
    document
      .querySelectorAll(".main-panel")
      .forEach((p) => p.classList.remove("active"));
    $("main-" + name)?.classList.add("active");
    if (name === "llm") refreshLlmProviders();
  }
  document.querySelectorAll(".main-tab").forEach((btn) => {
    btn.addEventListener("click", () => showTab(btn.dataset.main));
  });
  $("btn-open-llm")?.addEventListener("click", () => showTab("llm"));

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    const mode = chatMode();
    const llmAtSend = isExternal() ? activeLlm.label : "Gemma local";
    addMsg("user", message, null, mode, { llm: llmAtSend });
    const pending = addMsg(
      "agent pending",
      mode === "gemma_raw"
        ? (isExternal() ? `${activeLlm.label} generando…` : "Gemma 2 original generando…")
        : `Campo procesando · decoder (${isExternal() ? activeLlm.label : "Gemma"}) interpretando…`,
      null,
      mode
    );
    // Nunca colgar en silencio: aborta tras CHAT_TIMEOUT_MS y muestra error claro.
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), CHAT_TIMEOUT_MS);
    const started = performance.now();
    const submitBtn = form.querySelector("button[type=submit], button:not([type])");
    if (submitBtn) submitBtn.disabled = true;
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ message, mode }),
        signal: ctrl.signal,
      });
      let r = null;
      try {
        r = await res.json();
      } catch (_) {}
      pending.remove();
      if (!r || (!res.ok && !r.reply)) {
        throw new Error((r && r.error) || `/api/chat → HTTP ${res.status}`);
      }
      const secs = ((performance.now() - started) / 1000).toFixed(1);
      const rMode = r.mode || mode;
      const llmTxt = r.llm ? `llm=${r.llm}${r.fallback ? " (respaldo)" : ""} · ` : "";
      const meta =
        rMode === "gemma_raw"
          ? `modelo=${r.llm && r.llm_id !== "gemma_local" ? r.llm : "Gemma 2 original"} (sin campo) · ${r.decoded || ""} · ${secs}s`
          : r.field
            ? `${llmTxt}campo v4 · ${r.field.family} · err₁=${(100 * r.field.rel_err).toFixed(1)}% · ${r.field.trained ? "Dφ entrenado" : "Dφ sin entrenar"} · ${r.decoded} · ${secs}s`
            : `ruta=${r.route} · ${secs}s`;
      addMsg("agent", r.reply, meta, rMode, {
        error: !res.ok,
        llm: rMode === "gemma_raw" ? (r.llm || llmAtSend) : null,
      });
      if (r.route === "train" || r.route === "sleep") {
        await reconnectProcesses();
      }
      refreshTelemetry();
      refreshHealth();
    } catch (e) {
      pending.remove();
      const msg =
        e && e.name === "AbortError"
          ? `el servidor no respondió en ${Math.round(CHAT_TIMEOUT_MS / 1000)} s (modelo lento o sin memoria). Reintenta o usa respuestas más cortas.`
          : e && e.message
            ? e.message
            : String(e);
      addMsg("agent", "Error: " + msg, null, mode, { error: true, llm: llmAtSend });
      refreshHealth();
    } finally {
      clearTimeout(timer);
      if (submitBtn) submitBtn.disabled = false;
    }
  });

  $("btn-train").addEventListener("click", async () => {
    const infinite = $("chk-infinite")?.checked !== false;
    eventAfter = 0;
    liveEvents = [];
    try {
      const steps = Number($("tr-steps-in")?.value) || 4000;
      const body = infinite ? { batches: null, steps } : { batches: 4, steps };
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

  $("btn-sleep").addEventListener("click", async () => {
    const steps = Number($("sl-steps-in")?.value) || 8000;
    const infinite = $("chk-infinite-sleep")?.checked !== false;
    sleepEventAfter = 0;
    sleepEvents = [];
    $("sl-log").textContent = "";
    $("sl-status").textContent = "iniciando…";
    try {
      const body = {
        steps,
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


  // ------------------------------------------------------------ Modelos / API
  let llmState = { active: "gemma_local", providers: [], local: null };
  let llmRefreshing = false;

  function setActiveLlm(a) {
    if (!a) return;
    const changed = a.id !== activeLlm.id || a.label !== activeLlm.label;
    activeLlm = a;
    const chip = $("chat-llm");
    if (chip) {
      chip.textContent = `LLM: ${a.label}`;
      chip.classList.toggle("external", a.kind === "openai_compatible");
    }
    setText(
      "tr-llm",
      a.kind === "openai_compatible"
        ? `${a.label} (genera los datasets)`
        : "Gemma local (curriculum + sonda GGUF)"
    );
    const sel = $("llm-select");
    if (sel && sel.value !== a.id) {
      if ([...sel.options].some((o) => o.value === a.id)) {
        sel.value = a.id;
      } else if (!llmRefreshing) {
        // Otro navegador añadió/activó una API: recargar la lista.
        llmRefreshing = true;
        refreshLlmProviders().finally(() => {
          llmRefreshing = false;
        });
      }
    }
    if (changed) syncChatMode();
  }

  function renderLlmSelect() {
    const sel = $("llm-select");
    if (!sel) return;
    const local = llmState.local;
    const opts = [
      {
        id: "gemma_local",
        label:
          "Gemma local" +
          (local && !local.available ? " (no cargado)" : ""),
      },
    ].concat(
      llmState.providers.map((p) => ({
        id: p.id,
        label: `API · ${p.name}`,
      }))
    );
    sel.innerHTML = "";
    for (const o of opts) {
      const el = document.createElement("option");
      el.value = o.id;
      el.textContent = o.label;
      sel.appendChild(el);
    }
    sel.value = llmState.active;
  }

  function llmItem(p, isActive) {
    const row = document.createElement("div");
    row.className = "llm-item" + (isActive ? " active" : "");
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "llm-active-radio";
    radio.checked = isActive;
    radio.title = "Usar en toda la app";
    radio.addEventListener("change", () => selectLlm(p.id));
    const info = document.createElement("div");
    const t = document.createElement("div");
    t.className = "llm-title";
    t.textContent = p.title + (isActive ? " · activo" : "");
    const sub = document.createElement("div");
    sub.className = "llm-sub";
    sub.textContent = p.sub;
    info.append(t, sub);
    const btns = document.createElement("div");
    btns.className = "llm-btns";
    for (const b of p.buttons || []) {
      const el = document.createElement("button");
      el.type = "button";
      el.textContent = b.label;
      if (b.danger) el.className = "danger";
      if (b.disabled) el.disabled = true;
      if (b.title) el.title = b.title;
      el.addEventListener("click", b.onClick);
      btns.appendChild(el);
    }
    row.append(radio, info, btns);
    return row;
  }

  function renderLlmList() {
    const list = $("llm-list");
    if (!list) return;
    list.innerHTML = "";
    const local = llmState.local || {};
    list.appendChild(
      llmItem(
        {
          id: "gemma_local",
          title: "Gemma local (GGUF)",
          sub: `${local.probe || "gemma-2-2b-it"} · ${local.available ? "cargado" : "no cargado: " + (local.model_state || "—") + " (el decoder usa léxico)"}`,
          buttons: [],
        },
        llmState.active === "gemma_local"
      )
    );
    if (!llmState.providers.length) {
      const p = document.createElement("p");
      p.className = "hint";
      p.textContent = "Aún no hay APIs guardadas. Pega un curl a la izquierda y pulsa «Auto-configurar».";
      list.appendChild(p);
    }
    for (const pr of llmState.providers) {
      const envSrc = pr.source === "env";
      list.appendChild(
        llmItem(
          {
            id: pr.id,
            title: `API · ${pr.name}`,
            sub: `${pr.base_url} · modelo ${pr.model || "(primero de /v1/models)"} · key ${pr.api_key_masked || "—"}${envSrc ? " · variables de entorno" : ""}`,
            buttons: [
              { label: "Probar", onClick: () => testLlm({ id: pr.id }) },
              { label: "Editar", disabled: envSrc, onClick: () => editLlm(pr) },
              {
                label: "Borrar",
                danger: true,
                disabled: envSrc,
                title: envSrc ? "Definido por LLM_API_*: quítalo en las variables del servidor" : "",
                onClick: () => deleteLlm(pr),
              },
            ],
          },
          llmState.active === pr.id
        )
      );
    }
    const persist = $("llm-persist");
    if (persist) {
      persist.textContent = llmState.persisted_to
        ? `Se guarda en el servidor (${llmState.persisted_to}). En Railway sin volumen se pierde al redeploy; usa LLM_API_BASE / LLM_API_KEY / LLM_API_MODEL para dejarlo fijo.`
        : "";
    }
  }

  function applyLlmList(v) {
    if (!v) return;
    llmState = {
      active: v.active,
      providers: v.providers || [],
      local: v.local || null,
      persisted_to: v.persisted_to || null,
    };
    renderLlmSelect();
    renderLlmList();
    if (v.active_info) setActiveLlm(v.active_info);
  }

  async function llmApi(path, method, body) {
    const res = await fetch(path, {
      method: method || "GET",
      headers: { "content-type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    let r = null;
    try {
      r = await res.json();
    } catch (_) {}
    if (!res.ok || (r && r.ok === false && r.error && !("models_ok" in r))) {
      throw new Error((r && r.error) || `${path} → HTTP ${res.status}`);
    }
    return r;
  }

  async function refreshLlmProviders() {
    try {
      applyLlmList(await llmApi("/api/llm/providers"));
    } catch (e) {
      console.warn("llm providers", e);
    }
  }

  function llmResult(text, kind) {
    const el = $("llm-test-result");
    if (!el) return;
    el.textContent = text;
    el.classList.toggle("ok", kind === "ok");
    el.classList.toggle("err", kind === "err");
  }

  function formFields() {
    return {
      id: $("llm-edit-id").value || null,
      name: $("llm-name").value.trim(),
      base_url: $("llm-base").value.trim(),
      api_key: $("llm-key").value.trim(),
      model: $("llm-model").value.trim(),
    };
  }

  function clearLlmForm() {
    for (const id of ["llm-edit-id", "llm-name", "llm-base", "llm-key", "llm-model", "llm-curl"]) {
      $(id).value = "";
    }
    $("llm-key").placeholder = "obk1.…";
    $("llm-edit-note").textContent = "";
    $("llm-warnings").innerHTML = "";
    llmResult("—");
  }

  function editLlm(pr) {
    $("llm-edit-id").value = pr.id;
    $("llm-name").value = pr.name;
    $("llm-base").value = pr.base_url;
    $("llm-model").value = pr.model;
    $("llm-key").value = "";
    $("llm-key").placeholder = `${pr.api_key_masked} (vacío = conservar)`;
    $("llm-edit-note").textContent = `Editando «${pr.name}». Deja la API key vacía para conservar la guardada.`;
    llmResult("—");
    $("llm-name").focus();
  }

  function setModelOptions(models) {
    const dl = $("llm-models-list");
    if (!dl) return;
    dl.innerHTML = "";
    for (const m of models || []) {
      const o = document.createElement("option");
      o.value = m;
      dl.appendChild(o);
    }
  }

  $("llm-key-show")?.addEventListener("change", (ev) => {
    $("llm-key").type = ev.target.checked ? "text" : "password";
  });

  $("btn-llm-clear")?.addEventListener("click", clearLlmForm);

  $("btn-llm-parse")?.addEventListener("click", async () => {
    const curl = $("llm-curl").value;
    const warn = $("llm-warnings");
    warn.innerHTML = "";
    if (!curl.trim()) {
      llmResult("Pega primero el curl que te da el panel de docker-llm.", "err");
      return;
    }
    try {
      const r = await llmApi("/api/llm/parse-curl", "POST", { curl });
      const p = r.parsed;
      $("llm-edit-id").value = "";
      $("llm-edit-note").textContent = "";
      $("llm-key").placeholder = "obk1.…";
      if (p.name) $("llm-name").value = p.name;
      if (p.base_url) $("llm-base").value = p.base_url;
      if (p.api_key && !p.api_key.includes("$")) $("llm-key").value = p.api_key;
      if (p.model) $("llm-model").value = p.model;
      for (const w of p.warnings || []) {
        const li = document.createElement("li");
        li.textContent = w;
        warn.appendChild(li);
      }
      llmResult(
        `Auto-configurado: ${p.base_url || "(URL por completar)"} · modelo ${p.model || "(por completar)"} · key ${r.api_key_masked || "(por completar)"}` +
          (p.stream ? " · el curl usaba streaming; la app usa respuestas completas" : "") +
          "\nPulsa «Probar conexión» y luego «Guardar».",
        "ok"
      );
    } catch (e) {
      llmResult("No se pudo interpretar el curl: " + e.message, "err");
    }
  });

  async function testLlm(body) {
    llmResult("Probando conexión… (GET /v1/models + chat mínimo; la primera llamada puede cargar el modelo)");
    try {
      const r = await llmApi("/api/llm/providers/test", "POST", body);
      setModelOptions(r.models);
      const lines = [];
      lines.push(r.ok ? "✔ Conexión correcta" : "✖ Falló la conexión");
      lines.push(`URL base: ${r.base_url}`);
      lines.push(
        `GET /v1/models: ${r.models_ok ? "ok" : "error"} (${r.models_ms} ms)` +
          (r.models && r.models.length ? ` · ${r.models.join(", ")}` : "")
      );
      if (r.model_listed === false) lines.push(`⚠ el modelo «${r.model}» no aparece en /v1/models`);
      if (r.models_ok) {
        lines.push(`Chat (${r.model || "?"}): ${r.chat_ok ? "ok" : "error"} (${r.chat_ms} ms)` + (r.reply_preview ? ` · «${r.reply_preview}»` : ""));
      }
      if (r.error) lines.push("Error: " + r.error);
      llmResult(lines.join("\n"), r.ok ? "ok" : "err");
      if (!$("llm-model").value && r.ok && r.model && !body.id) $("llm-model").value = r.model;
      return r;
    } catch (e) {
      llmResult("Error: " + e.message, "err");
      return null;
    }
  }

  $("btn-llm-test")?.addEventListener("click", () => {
    const f = formFields();
    if (!f.base_url && $("llm-curl").value.trim()) {
      testLlm({ curl: $("llm-curl").value });
      return;
    }
    testLlm(f);
  });

  async function saveLlm(activate) {
    const f = formFields();
    if (!f.base_url) {
      llmResult("Falta la URL base (pulsa «Auto-configurar» o escríbela).", "err");
      return;
    }
    try {
      const r = await llmApi("/api/llm/providers", "POST", { ...f, activate: !!activate });
      applyLlmList(r);
      const p = r.provider;
      $("llm-edit-id").value = p.id;
      $("llm-key").value = "";
      // No dejar la clave a la vista en el curl pegado.
      $("llm-curl").value = "";
      $("llm-warnings").innerHTML = "";
      $("llm-key").placeholder = `${p.api_key_masked} (vacío = conservar)`;
      $("llm-edit-note").textContent = `Guardado «${p.name}».`;
      llmResult(
        `Guardado «${p.name}» (key ${p.api_key_masked}).` +
          (activate ? " Ahora es el LLM activo de toda la app." : " Selecciónalo arriba o en la lista para usarlo."),
        "ok"
      );
    } catch (e) {
      llmResult("No se pudo guardar: " + e.message, "err");
    }
  }
  $("btn-llm-save")?.addEventListener("click", () => saveLlm(false));
  $("btn-llm-save-activate")?.addEventListener("click", () => saveLlm(true));

  async function deleteLlm(pr) {
    if (!confirm(`¿Borrar la API «${pr.name}»?\n\nSe elimina del servidor. Si estaba activa, la app vuelve a Gemma local.`)) return;
    try {
      applyLlmList(await llmApi(`/api/llm/providers/${encodeURIComponent(pr.id)}`, "DELETE"));
      if ($("llm-edit-id").value === pr.id) clearLlmForm();
    } catch (e) {
      llmResult("No se pudo borrar: " + e.message, "err");
    }
  }

  async function selectLlm(id) {
    try {
      applyLlmList(await llmApi("/api/llm/active", "POST", { id }));
    } catch (e) {
      alert("No se pudo cambiar el LLM: " + e.message);
      refreshLlmProviders();
    }
  }
  $("llm-select")?.addEventListener("change", (ev) => selectLlm(ev.target.value));

  // Restaurar historial del navegador. Los paneles se ocultan con CSS
  // (display:none), así que cambiar de pestaña no borra el DOM del chat.
  const restored = renderStoredHistory();
  if (!restored) {
    addMsg(
      "agent",
      "Listo. Chat = interpretación del modelo de campo (decoder). Entrenamiento, Sueño y Pruebas son jobs en servidor: al refrescar la UI se reconecta sin cancelar. El historial se guarda en este navegador.",
      null,
      null,
      { skipStore: true },
    );
  }

  $("btn-new-chat").addEventListener("click", async () => {
    if (
      !confirm(
        "¿Borrar la conversación y empezar de cero?\n\nSe limpia el historial de esta pantalla y el contexto del servidor (Gemma OFF).",
      )
    ) {
      return;
    }
    const btn = $("btn-new-chat");
    btn.disabled = true;
    try {
      await api("/api/chat/reset", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{}",
      });
    } catch (e) {
      // Aun si el servidor falla, limpiamos la UI local.
      console.warn("reset chat:", e);
    }
    clearChatHistory();
    messages.innerHTML = "";
    addMsg(
      "agent",
      "Chat nuevo. Historial borrado en esta pantalla y en el servidor.",
      null,
      null,
      { skipStore: true },
    );
    btn.disabled = false;
  });

  refreshHealth();
  refreshTelemetry();
  refreshLlmProviders();
  reconnectProcesses();
  setInterval(() => {
    refreshTelemetry();
    refreshHealth();
  }, 4000);
})();
