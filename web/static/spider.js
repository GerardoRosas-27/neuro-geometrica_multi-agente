// Prompt Spider — «cada palabra, un fork».
// El crawler (servidor) camina el prompt; cada palabra recibe una pregunta
// tipada y una p. p ≥ umbral → código; p < umbral → LLM activo; aprobación →
// tú. Esta vista solo dibuja eventos de /api/spider/* (SSE con respaldo de
// sondeo); no decide nada en el navegador.
(() => {
  const $ = (id) => document.getElementById(id);
  const root = $("main-spider");
  if (!root) return;

  const PROMPT_KEY = "spider.prompt.v1";
  const Q_LABEL = {
    relevant: "¿relevante para la tarea?",
    grounded: "¿afirmación con fuente?",
    ambiguous: "¿es específico?",
    approval: "¿requiere aprobación?",
  };
  const COLORS = { code: "#39e58c", llm: "#ff9a3c", you: "#ffd23f", pink: "#ff4f8b" };

  const ta = $("sp-prompt");
  const view = $("sp-view");
  const canvas = $("sp-canvas");
  const thr = $("sp-threshold");

  let run = null; // {id, tokens, threshold, llm, state, llm_calls}
  let dec = new Map(); // index → decision
  let order = []; // índices de forks en orden de llegada
  let cursor = 0;
  let seq = 0;
  let running = false;
  let es = null;
  let pollTimer = null;
  let llmLabel = "LLM";
  let llmCalls = [];
  let spans = [];
  let dots = [];
  let dirty = true;
  let lastFork = null; // {index, t}
  let callStart = 0; // performance.now() de la llamada LLM en curso
  const MODE_KEY = "spider.mode.v1";
  const FIELD_KEY = "spider.fieldDecoder.v1";
  const fieldChk = $("sp-field-decoder");
  const fieldOn = () => !!(fieldChk && fieldChk.checked);
  let lines = 1;

  // ---------------------------------------------------------------- util
  async function api(path, opts) {
    const res = await fetch(path, opts);
    let body = null;
    try {
      body = await res.json();
    } catch (_) {}
    if (!res.ok) throw new Error((body && body.error) || `${path} → ${res.status}`);
    return body;
  }
  const post = (path, body) =>
    api(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body || {}) });
  const fmt = (p) => (typeof p === "number" ? p.toFixed(3) : "—");
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  function setStatus(t) {
    $("sp-status").textContent = t;
  }
  function setLlmName(label) {
    llmLabel = label || "LLM";
    document.querySelectorAll(".sp-llm-name").forEach((el) => (el.textContent = llmLabel));
    const short = document.querySelector(".sp-llm-short");
    if (short) short.textContent = llmLabel.length > 10 ? "llm" : llmLabel.toLowerCase();
  }
  function setRunning(on) {
    running = on;
    $("sp-run").disabled = on;
    $("sp-stop").disabled = !on;
    $("sp-edit").disabled = on;
    $("sp-sample").disabled = on;
    thr.disabled = on;
    $("sp-mode").disabled = on;
    const st = $("status-spider");
    if (st) st.dataset.state = on ? "running" : "idle";
  }
  function syncThr() {
    const v = Number(thr.value).toFixed(2);
    $("sp-thr-val").textContent = v;
    document.querySelectorAll(".sp-thr-txt").forEach((el) => (el.textContent = v));
  }
  thr.addEventListener("input", syncThr);

  // ---------------------------------------------------------- editor/view
  function showEditor() {
    view.hidden = true;
    ta.hidden = false;
  }
  function buildView() {
    if (!run) return;
    const text = run.prompt;
    const frag = document.createDocumentFragment();
    spans = [];
    let pos = 0;
    // offsets en caracteres (code points) → convertir a índices UTF-16.
    const cps = Array.from(text);
    const u16 = new Array(cps.length + 1);
    u16[0] = 0;
    for (let i = 0; i < cps.length; i++) u16[i + 1] = u16[i] + cps[i].length;
    for (const t of run.tokens) {
      const s = u16[t.start];
      const e = u16[t.end];
      if (s > pos) frag.appendChild(document.createTextNode(text.slice(pos, s)));
      const sp = document.createElement("span");
      sp.className = "tk" + (t.kind === "tag" || t.kind === "list_marker" ? " tk-tag" : "");
      sp.textContent = text.slice(s, e);
      frag.appendChild(sp);
      spans[t.index] = sp;
      pos = e;
    }
    if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
    view.textContent = "";
    view.appendChild(frag);
    view.hidden = false;
    ta.hidden = true;
    lines = Math.max(1, text.split("\n").length);
    // rejilla de palabras leídas
    const grid = $("sp-dots");
    grid.textContent = "";
    dots = run.tokens.map(() => {
      const i = document.createElement("i");
      grid.appendChild(i);
      return i;
    });
  }
  function paintToken(i) {
    const sp = spans[i];
    const dt = dots[i];
    const d = dec.get(i);
    const read = i < cursor;
    let cls = "tk";
    const t = run && run.tokens[i];
    if (t && (t.kind === "tag" || t.kind === "list_marker")) cls += " tk-tag";
    let dcls = "";
    if (read) {
      cls += " read";
      dcls = "read";
    }
    if (d) {
      cls += " r-" + d.route;
      dcls = "r-" + d.route;
      if (d.status === "escalating") cls += " st-esc";
      if (d.status === "pending") cls += " st-pending";
    }
    if (i === cursor - 1 && running) {
      cls += " cur";
      dcls += " cur";
      // Mantener la palabra actual visible dentro del editor (sin mover la página).
      if (sp && !view.hidden) {
        const top = sp.offsetTop;
        if (top < view.scrollTop + 8 || top > view.scrollTop + view.clientHeight - 24) {
          view.scrollTop = Math.max(0, top - view.clientHeight / 2);
        }
      }
    }
    if (sp) sp.className = cls;
    if (dt) dt.className = dcls;
  }

  // ------------------------------------------------------------- render
  function counts() {
    let code = 0, llmEsc = 0, llmRes = 0, you = 0, pending = 0, approved = 0, rejected = 0, sumP = 0;
    for (const i of order) {
      const d = dec.get(i);
      if (d.route === "code") code++;
      if (d.first_route === "llm") llmEsc++;
      if (d.route === "llm" && d.status === "resolved") llmRes++;
      if (d.route === "you") you++;
      if (d.status === "pending") pending++;
      if (d.status === "approved") approved++;
      if (d.status === "rejected") rejected++;
      sumP += d.final_p;
    }
    return { code, llmEsc, llmRes, you, pending, approved, rejected, avg: order.length ? sumP / order.length : null };
  }
  function bump(id, n, prev) {
    const el = $(id);
    if (String(n) !== el.textContent) {
      el.textContent = n;
      if (prev !== undefined) {
        const box = el.closest(".sp-route");
        if (box) {
          box.classList.remove("bump");
          void box.offsetWidth;
          box.classList.add("bump");
        }
      }
    }
  }
  function render() {
    if (!run) return;
    const c = counts();
    const total = run.tokens.length;
    const forksTotal = run.tokens.filter((t) => t.kind === "word" || t.kind === "number").length;
    $("sp-c-words").textContent = cursor;
    $("sp-c-forks").textContent = order.length;
    $("sp-c-code").textContent = c.code;
    $("sp-c-llm").textContent = c.llmEsc;
    $("sp-c-you").textContent = c.you;
    bump("sp-n-code", c.code, 1);
    bump("sp-n-llm", c.llmEsc, 1);
    bump("sp-n-you", c.you, 1);
    const capped = order.filter((i) => dec.get(i).capped).length;
    const isField = run.decider === "campo";
    const fieldUs = order.map((i) => dec.get(i).field).filter(Boolean).map((f) => f.micros).sort((a, b) => a - b);
    const fieldLat = fieldUs.length ? ` · ${fieldUs[fieldUs.length >> 1].toFixed(0)} µs` : "";
    $("sp-f-llm").textContent = isField
      ? `campo · ${c.llmRes} resueltas · 0 llamadas${fieldLat}`
      : `${llmCalls.length} llamadas · ${c.llmRes} resueltas` + (capped ? ` · ${capped} por límite` : "");
    $("sp-f-you").textContent = `${c.pending} pendientes · ${c.approved}✓ ${c.rejected}✗`;
    $("sp-routed-n").textContent = order.length;
    $("sp-routed-sub").textContent = `de ${forksTotal} palabras del prompt (${total} tokens)`;
    const finalLlm = order.filter((i) => dec.get(i).route === "llm").length;
    for (const [k, n] of [["code", c.code], ["llm", finalLlm], ["you", c.you]]) {
      $(`sp-bar-${k}`).style.width = (order.length ? (100 * n) / order.length : 0) + "%";
      $(`sp-bar-${k}-n`).textContent = n;
    }
    $("sp-avg").textContent = c.avg == null ? "—" : c.avg.toFixed(3);
    $("sp-conf-n").textContent = `— ${order.length}`;
    $("sp-read-n").textContent = `${cursor} / ${total}`;
    const cov = total ? Math.round((100 * cursor) / total) : 0;
    $("sp-cov").textContent = cov + "%";
    $("sp-t-code").textContent = `${c.code} if-statements`;
    const failed = llmCalls.filter((x) => !x.ok).length;
    $("sp-t-llm").textContent = isField
      ? `decoder del campo · ${c.llmRes} resueltas · ${c.llmEsc - c.llmRes} a ti${fieldLat}`
      : `${llmCalls.length} llamadas · ${c.llmRes} resueltas` + (failed ? ` · ${failed} fallidas` : "");
    $("sp-t-you").textContent = `${c.you} aprobaciones (${c.pending} pendientes)`;
    $("sp-ln").textContent = "ln " + (run.tokens[Math.max(0, cursor - 1)]?.line + 1 || 0);
    const prof = run.profile;
    const profTxt = isField
      ? " · escaladas → decoder del campo"
      : prof
      ? ` · corrida ${run.mode || prof.mode} (lotes de ${prof.batch_size}, plazo ${prof.timeout_secs}s, ${prof.max_escalations == null ? "sin límite" : "máx " + prof.max_escalations})`
      : "";
    $("sp-foot-l").textContent = `spider·${llmLabel} · ${run.state || "—"} · umbral ${Number(run.threshold).toFixed(2)}${profTxt}`;
    $("sp-foot-r").textContent = `forks ${order.length} · code ${c.code} · ${llmLabel} ${c.llmRes}/${c.llmEsc} · tú ${c.you} · p̄ ${c.avg == null ? "—" : c.avg.toFixed(3)}`;
    renderStream();
    renderPending();
    drawSpark();
  }
  function renderStream() {
    const body = $("sp-stream");
    const total = run.tokens.length || 1;
    const rows = order.slice(-40).reverse().map((i) => {
      const d = dec.get(i);
      const pct = ((100 * i) / total).toFixed(1).padStart(4, "0");
      const arrow =
        d.route === "code" ? "→ code" : d.route === "llm" ? `→ ${llmLabel}` : d.status === "pending" ? "→ tú (pend.)" : `→ tú (${d.status === "approved" ? "✓" : d.status === "rejected" ? "✗" : "…"})`;
      const esc2 = d.status === "escalating" ? " …" : "";
      return `<tr class="r-${d.route}"><td class="sp-muted">${pct}%</td><td class="w"><span>${esc(d.word)}</span></td><td class="q">${Q_LABEL[d.question] || d.question}</td><td class="p">${fmt(d.final_p)}</td><td class="r">${esc(arrow)}${esc2}</td></tr>`;
    });
    body.innerHTML = rows.join("");
  }
  function contextOf(i) {
    const toks = run.tokens;
    const lo = Math.max(0, i - 6);
    const hi = Math.min(toks.length, i + 7);
    return toks
      .slice(lo, hi)
      .filter((t) => t.kind !== "tag")
      .map((t) => (t.index === i ? `[${t.text}]` : t.text))
      .join(" ");
  }
  function renderPending() {
    const list = $("sp-pend-list");
    const items = order.map((i) => dec.get(i)).filter((d) => d.route === "you");
    $("sp-pend-n").textContent = `${items.filter((d) => d.status === "pending").length} pendientes · ${items.length} en tu cola`;
    if (!items.length) {
      list.innerHTML = '<p class="sp-muted">Nada pendiente.</p>';
      return;
    }
    items.sort((a, b) => (b.status === "pending") - (a.status === "pending") || a.index - b.index);
    list.innerHTML = items
      .map((d) => {
        const done = d.status !== "pending";
        const verdict = d.verdict ? ` · ${esc(llmLabel)}: p=${fmt(d.verdict.p)}${d.verdict.needs_approval ? ", pide aprobación" : ""}${d.verdict.ambiguous ? ", ambigua" : ""}` : "";
        return `<div class="sp-pend${done ? " done" : ""}" data-i="${d.index}">
          <div><span class="w">${esc(d.word)}</span> <span class="sp-muted">p=${fmt(d.p)} · ${Q_LABEL[d.question] || d.question}</span>
          <div class="ctx">${esc(contextOf(d.index))}</div>
          <div class="note">${esc(d.note || "")}${verdict}</div></div>
          <div class="acts">${
            done
              ? `<span class="sp-muted">${d.status === "approved" ? "aprobada ✓" : "rechazada ✗"}</span>`
              : `<button type="button" class="sp-btn ok" data-act="1">Aprobar</button><button type="button" class="sp-btn no" data-act="0">Rechazar</button>`
          }</div></div>`;
      })
      .join("");
  }
  $("sp-pend-list").addEventListener("click", async (ev) => {
    const b = ev.target.closest("button[data-act]");
    if (!b || !run) return;
    const idx = Number(b.closest(".sp-pend").dataset.i);
    b.disabled = true;
    try {
      const r = await post("/api/spider/decide", { run_id: run.id, index: idx, approve: b.dataset.act === "1" });
      applyDecision(r.decision);
      dirty = true;
    } catch (e) {
      setStatus("decidir: " + e.message);
      b.disabled = false;
    }
  });

  // ------------------------------------------------------------- canvas
  function fit(c) {
    const dpr = window.devicePixelRatio || 1;
    const r = c.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width * dpr));
    const h = Math.max(1, Math.round(r.height * dpr));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: r.width, h: r.height };
  }
  function geometry(w, h) {
    const cx = w * 0.06;
    const cy = h * 0.5;
    const R = Math.max(80, Math.min(w * 0.8, h * 1.25));
    const A = Math.min(1.15, Math.asin(Math.min(1, (h * 0.44) / R)));
    const slots = Math.max(18, Math.min(56, Math.round(h / 9)));
    return { cx, cy, R, A, slots };
  }
  function nodePos(g, k) {
    const s = k % g.slots;
    const a = -g.A + (2 * g.A * (s + 0.5)) / g.slots;
    return { x: g.cx + g.R * Math.cos(a), y: g.cy + g.R * Math.sin(a), a };
  }
  function originY(h, i) {
    const t = run && run.tokens[i];
    const ln = t ? t.line : 0;
    return h * 0.04 + (h * 0.92 * (ln + 0.5)) / lines;
  }
  function drawSpider(now) {
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    const g = geometry(w, h);
    // ticks radiales del arco
    ctx.lineWidth = 1;
    for (let k = 0; k < g.slots * 3; k++) {
      const a = -g.A + (2 * g.A * k) / (g.slots * 3);
      const r1 = g.R + 6;
      const r2 = g.R + 6 + (k % 3 === 0 ? 14 : 7);
      ctx.strokeStyle = k % 3 === 0 ? "rgba(57,229,140,0.35)" : "rgba(57,229,140,0.15)";
      ctx.beginPath();
      ctx.moveTo(g.cx + r1 * Math.cos(a), g.cy + r1 * Math.sin(a));
      ctx.lineTo(g.cx + r2 * Math.cos(a), g.cy + r2 * Math.sin(a));
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(255,79,139,0.18)";
    ctx.beginPath();
    ctx.arc(g.cx, g.cy, g.R, -g.A, g.A);
    ctx.stroke();
    if (!run || !order.length) {
      ctx.fillStyle = "#3a4757";
      ctx.font = "12px ui-monospace, monospace";
      ctx.fillText("Ejecuta para soltar la araña sobre el prompt…", 16, 24);
      return;
    }
    const n = order.length;
    const start = Math.max(0, n - 160);
    // historial: palabra → nodo, con desvanecimiento
    for (let k = start; k < n; k++) {
      const d = dec.get(order[k]);
      const age = n - 1 - k;
      const alpha = Math.max(0.05, 0.55 - age * 0.004);
      const p = nodePos(g, k);
      const oy = originY(h, d.index);
      ctx.strokeStyle = hexA(COLORS[d.route], alpha * (d.route === "code" ? 0.55 : 0.9));
      ctx.lineWidth = d.route === "code" ? 0.7 : 1.1;
      ctx.beginPath();
      ctx.moveTo(0, oy);
      ctx.bezierCurveTo(w * 0.25, oy, p.x - w * 0.2, p.y, p.x, p.y);
      ctx.stroke();
    }
    // abanico desde la palabra actual
    const cur = order[n - 1];
    const cy0 = originY(h, cur);
    const fan = Math.min(n, g.slots);
    for (let k = n - fan; k < n; k++) {
      const p = nodePos(g, k);
      ctx.strokeStyle = "rgba(255,79,139,0.10)";
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      ctx.moveTo(0, cy0);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }
    // línea animada de la última decisión
    if (lastFork) {
      const t = Math.min(1, (now - lastFork.t) / 280);
      const k = n - 1;
      const p = nodePos(g, k);
      const d = dec.get(order[k]);
      const x = p.x * t;
      const y = cy0 + (p.y - cy0) * t;
      ctx.strokeStyle = COLORS[d.route];
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, cy0);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.fillStyle = COLORS.pink;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    // nodos + etiquetas
    ctx.font = "10px ui-monospace, monospace";
    const vis = Math.min(n, g.slots);
    for (let k = n - vis; k < n; k++) {
      const d = dec.get(order[k]);
      const p = nodePos(g, k);
      ctx.fillStyle = COLORS[d.route];
      ctx.beginPath();
      ctx.arc(p.x, p.y, d.route === "code" ? 2.2 : 3.4, 0, Math.PI * 2);
      ctx.fill();
      const age = n - 1 - k;
      const label = d.route !== "code" || age < 10 || k % 4 === 0;
      if (!label) continue;
      const txt = d.word.length > 14 ? d.word.slice(0, 13) + "…" : d.word;
      const tw = ctx.measureText(txt).width + 8;
      const off = k % 2 === 0 ? -tw - 10 : 8;
      let lx = p.x + off;
      if (lx + tw > w - 2) lx = p.x - tw - 10;
      const ly = p.y - 7;
      ctx.fillStyle = age === 0 ? "#ff7aa8" : hexA(COLORS.pink, Math.max(0.35, 0.95 - age * 0.02));
      ctx.fillRect(lx, ly, tw, 14);
      ctx.strokeStyle = COLORS[d.route];
      ctx.lineWidth = 1;
      ctx.strokeRect(lx + 0.5, ly + 0.5, tw - 1, 13);
      ctx.fillStyle = "#05070a";
      ctx.fillText(txt, lx + 4, ly + 10.5);
    }
  }
  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a.toFixed(3)})`;
  }
  function drawSpark() {
    const c = $("sp-spark");
    const { ctx, w, h } = fit(c);
    ctx.clearRect(0, 0, w, h);
    if (!run) return;
    const th = Number(run.threshold);
    const lo = 0.4;
    const y = (p) => h - 6 - ((Math.max(lo, p) - lo) / (1 - lo)) * (h - 12);
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "rgba(57,229,140,0.5)";
    ctx.beginPath();
    ctx.moveTo(0, y(th));
    ctx.lineTo(w, y(th));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(57,229,140,0.7)";
    ctx.font = "9px ui-monospace, monospace";
    ctx.fillText(`p ≥ ${th.toFixed(2)}`, w - 52, y(th) - 3);
    const ps = order.map((i) => dec.get(i).final_p);
    if (ps.length < 2) return;
    ctx.strokeStyle = COLORS.pink;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ps.forEach((p, k) => {
      const x = (k / (ps.length - 1)) * w;
      k ? ctx.lineTo(x, y(p)) : ctx.moveTo(x, y(p));
    });
    ctx.stroke();
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = "rgba(255,79,139,0.08)";
    ctx.fill();
  }
  function drawSphere(now) {
    const c = $("sp-sphere");
    const { ctx, w, h } = fit(c);
    ctx.clearRect(0, 0, w, h);
    const r = Math.min(w, h) * 0.42;
    const cx = w / 2;
    const cy = h / 2;
    const cov = run && run.tokens.length ? cursor / run.tokens.length : 0;
    const rot = (now / 4000) % (Math.PI * 2);
    ctx.lineWidth = 0.8;
    for (let m = 0; m < 10; m++) {
      const a = rot + (m * Math.PI) / 10;
      const rx = Math.abs(Math.cos(a)) * r;
      const lit = m / 10 < cov;
      ctx.strokeStyle = lit ? "rgba(255,79,139,0.75)" : "rgba(120,140,160,0.2)";
      ctx.beginPath();
      ctx.ellipse(cx, cy, Math.max(0.5, rx), r, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    for (let p = 1; p < 6; p++) {
      const yy = cy - r + (2 * r * p) / 6;
      const rr = Math.sqrt(Math.max(0, r * r - (yy - cy) ** 2));
      ctx.strokeStyle = (p / 6) < cov ? "rgba(57,229,140,0.5)" : "rgba(120,140,160,0.15)";
      ctx.beginPath();
      ctx.ellipse(cx, yy, rr, rr * 0.18, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = "#e9eef4";
    ctx.font = "bold 13px ui-monospace, monospace";
    const t = Math.round(cov * 100) + "%";
    ctx.fillText(t, cx - ctx.measureText(t).width / 2, cy + 4);
  }

  // --------------------------------------------------------------- loop
  function visible() {
    return root.classList.contains("active") && !document.hidden;
  }
  function frame(now) {
    if (visible()) {
      if (dirty) {
        render();
        dirty = false;
      }
      drawSpider(now);
      drawSphere(now);
      tickProg(now);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  window.addEventListener("resize", () => (dirty = true));
  document.querySelectorAll('.main-tab[data-main="spider"]').forEach((b) =>
    b.addEventListener("click", () => {
      dirty = true;
      refreshLlm();
    })
  );

  // ------------------------------------------------------------- events
  function applyDecision(d) {
    if (!d) return;
    if (!dec.has(d.index)) order.push(d.index);
    dec.set(d.index, d);
    paintToken(d.index);
  }
  function applyEvents(evs) {
    for (const e of evs) {
      if (e.seq <= seq) continue;
      seq = e.seq;
      if (run && e.run_id && e.run_id !== run.id) continue;
      const prevCursor = cursor;
      cursor = Math.max(cursor, e.cursor || 0);
      if (e.kind === "decision") {
        applyDecision(e.decision);
        lastFork = { index: e.index, t: performance.now() };
      } else if (e.kind === "update") {
        applyDecision(e.decision);
      } else if (e.kind === "llm_progress") {
        callStart = performance.now();
        showProg(e.message);
      } else if (e.kind === "fallback") {
        showFallback(e.message);
        refreshCalls();
      } else if (e.kind === "llm_call") {
        callStart = 0;
        showProg(e.message);
        setStatus(e.message);
        refreshCalls();
      } else if (e.kind === "done") {
        setStatus(e.message);
        finish();
      }
      for (let i = Math.max(0, prevCursor - 1); i < cursor; i++) paintToken(i);
    }
    dirty = true;
  }
  function showProg(msg) {
    const el = $("sp-llm-prog");
    el.hidden = !msg;
    el.innerHTML = msg ? `${esc(msg)}<span class="t"></span>` : "";
  }
  function showFallback(msg) {
    const el = $("sp-fallback");
    el.hidden = !msg;
    el.textContent = msg ? "⚠ " + msg + " (respaldo; corrida corta si el modo es auto)" : "";
  }
  function tickProg(now) {
    if (!callStart) return;
    const t = document.querySelector("#sp-llm-prog .t");
    if (t) t.textContent = `esperando… ${((now - callStart) / 1000).toFixed(0)}s`;
  }
  async function refreshCalls() {
    try {
      const st = await api("/api/spider/status");
      if (st.run && run && st.run.id === run.id) {
        llmCalls = st.run.llm_calls || [];
        run.state = st.run.state;
        run.mode = st.run.mode;
        run.profile = st.run.profile;
        run.fallback = st.run.fallback;
        if (run.fallback) {
          setLlmName(run.fallback.to);
          showFallback(run.fallback.message);
        }
        dirty = true;
      }
    } catch (_) {}
  }
  function finish() {
    callStart = 0;
    setRunning(false);
    closeStream();
    refreshCalls();
    $("sp-export").disabled = !run;
    for (let i = 0; i < (run ? run.tokens.length : 0); i++) paintToken(i);
  }
  function closeStream() {
    if (es) {
      es.close();
      es = null;
    }
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }
  function subscribe() {
    closeStream();
    if (window.EventSource) {
      es = new EventSource("/api/spider/stream?after=" + seq);
      es.addEventListener("spider", (m) => {
        try {
          applyEvents(JSON.parse(m.data));
        } catch (_) {}
      });
      es.addEventListener("idle", () => {
        closeStream();
        if (running) poll();
      });
      es.onerror = () => {
        closeStream();
        poll();
      };
    } else poll();
  }
  function poll() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      try {
        const r = await api("/api/spider/events?after=" + seq);
        applyEvents(r.events || []);
        if (!r.running && r.seq <= seq) {
          finish();
        }
      } catch (_) {}
    }, 350);
  }

  // ------------------------------------------------------------ actions
  function loadRun(st) {
    run = st.run;
    dec = new Map();
    order = [];
    cursor = run.cursor || 0;
    llmCalls = run.llm_calls || [];
    seq = st.seq || 0;
    setLlmName(run.fallback ? run.fallback.to : run.llm && run.llm.label);
    showFallback(run.fallback ? run.fallback.message : "");
    buildView();
    for (const d of run.decisions || []) applyDecision(d);
    for (let i = 0; i < run.tokens.length; i++) paintToken(i);
    thr.value = run.threshold;
    syncThr();
    $("sp-export").disabled = false;
    dirty = true;
  }
  async function refreshLlm() {
    if (running) return;
    try {
      const st = await api("/api/spider/status");
      if (!running) setLlmName(fieldOn() ? "campo" : st.llm_active && st.llm_active.label);
      const chip = $("sp-llm-chip");
      if (st.llm_active && st.llm_active.unavailable) chip.title = "No disponible: " + st.llm_active.unavailable;
      else if (st.llm_fallback) chip.title = "Respaldo si la API no responde: " + st.llm_fallback.label;
      applyModeDefaults(st);
    } catch (_) {}
  }
  function applyModeDefaults(st) {
    const sel = $("sp-mode");
    const d = st && st.defaults;
    if (!sel || !d) return;
    const desc = (p) =>
      p ? `lotes de ${p.batch_size}, plazo ${p.timeout_secs}s, ${p.max_escalations == null ? "sin límite" : "máx " + p.max_escalations + " escaladas"}` : "";
    for (const o of sel.options) {
      if (o.value === "auto") o.textContent = `auto (${d.mode})`;
      if (o.value === "corta") o.title = desc(d.corta);
      if (o.value === "completa") o.title = desc(d.completa);
    }
  }
  $("sp-mode").addEventListener("change", () => {
    try {
      localStorage.setItem(MODE_KEY, $("sp-mode").value);
    } catch (_) {}
  });
  $("sp-run").addEventListener("click", async () => {
    const prompt = ta.hidden && run ? run.prompt : ta.value;
    if (!prompt.trim()) {
      setStatus("escribe un prompt");
      return;
    }
    try {
      localStorage.setItem(PROMPT_KEY, prompt);
    } catch (_) {}
    setRunning(true);
    setStatus("arrancando…");
    try {
      const r = await post("/api/spider/start", {
        prompt,
        threshold: Number(thr.value),
        pace_ms: Number($("sp-pace").value),
        mode: $("sp-mode").value,
        field_decoder: fieldOn(),
      });
      showProg("");
      showFallback("");
      const st = await api("/api/spider/status");
      loadRun(st);
      const p = r.profile || {};
      if (r.decider === "campo") {
        setStatus(
          `corriendo · ${r.forks_total} forks · escalado → decoder del campo (${r.llm.model})` +
            (r.llm.unavailable ? ` · ${r.llm.unavailable}` : " · sin llamadas al LLM")
        );
        subscribe();
        return;
      }
      setStatus(
        `corriendo · corrida ${r.mode} · ${r.forks_total} forks · escalado → ${r.llm.label}` +
          ` (lotes de ${p.batch_size}, plazo ${p.timeout_secs}s, ${p.max_escalations == null ? "sin límite" : "máx " + p.max_escalations + " escaladas"})` +
          (r.fallback_available ? " · respaldo Gemma local" : "") +
          (r.llm.unavailable ? " (no disponible: quedará pendiente)" : "")
      );
      subscribe();
    } catch (e) {
      setRunning(false);
      setStatus("error: " + e.message);
    }
  });
  $("sp-stop").addEventListener("click", async () => {
    try {
      await post("/api/spider/stop", {});
      setStatus("deteniendo…");
    } catch (e) {
      setStatus("error: " + e.message);
    }
  });
  $("sp-edit").addEventListener("click", () => {
    if (run && !ta.value) ta.value = run.prompt;
    showEditor();
  });
  $("sp-sample").addEventListener("click", async () => {
    try {
      const s = await api("/api/spider/sample");
      ta.value = s.prompt;
      showEditor();
    } catch (e) {
      setStatus("error: " + e.message);
    }
  });
  $("sp-export").addEventListener("click", async () => {
    if (!run) return;
    try {
      const res = await fetch("/api/spider/export?id=" + encodeURIComponent(run.id));
      if (!res.ok) throw new Error("HTTP " + res.status);
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `prompt_spider_${run.id}.json`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        URL.revokeObjectURL(a.href);
        a.remove();
      }, 500);
    } catch (e) {
      setStatus("exportar: " + e.message);
    }
  });

  // ------------------------------------------- decoder del campo (Spider)
  if (fieldChk)
    fieldChk.addEventListener("change", () => {
      try {
        localStorage.setItem(FIELD_KEY, fieldChk.checked ? "1" : "0");
      } catch (_) {}
      if (!running) {
        if (fieldChk.checked) setLlmName("campo");
        else refreshLlm();
      }
    });
  let ftSeq = 0;
  let ftTimer = null;
  const pct = (x) => (typeof x === "number" ? x.toFixed(1) + "%" : "—");
  function metricRow(name, m) {
    if (!m || !m.escalated) return `<tr><td class="k">${esc(name)}</td><td class="n" colspan="6">—</td></tr>`;
    return (
      `<tr><td class="k">${esc(name)}</td><td class="n">${m.escalated}</td><td class="n">${fmt(m.ok.accuracy)}</td>` +
      `<td class="n">${fmt(m.ok.ece)}</td><td class="n">${fmt(m.ok.brier)}</td><td class="n">${pct(m.resolved_pct)}</td>` +
      `<td class="n">${fmt(m.resolved_precision)}</td></tr>`
    );
  }
  function renderField(f) {
    if (!f) return;
    $("sp-ft-model").textContent =
      `${f.name} · ${f.trained ? "entrenado" : "sin entrenar"} · ${f.examples_seen} ejemplos · ${f.sleeps} sueños` +
      (f.wake ? ` · vigilia ${f.wake}` : "");
    const ho = f.holdout && f.holdout.synthetic;
    const j = f.job || {};
    const tb = $("sp-ft-metrics");
    if (ho) {
      const heads = ["relevant", "grounded", "ambiguous", "needs_approval", "ok"];
      const real = (f.holdout && f.holdout.real) || {};
      tb.innerHTML =
        `<tr class="h"><td>escaladas (hold-out)</td><td class="n">n</td><td class="n">exact. ok</td><td class="n">ECE</td><td class="n">Brier</td><td class="n">resueltas</td><td class="n">precisión</td></tr>` +
        metricRow("campo · H1 (vocab. visto)", ho.router_h1) +
        metricRow("campo · H2 (vocab. nuevo)", ho.router_h2) +
        metricRow("heurística · H1", ho.heuristic_h1) +
        metricRow("heurística · H2", ho.heuristic_h2) +
        `<tr class="h"><td>por pregunta (H1, todas)</td><td class="n">n</td><td class="n">exact.</td><td class="n">ECE</td><td class="n">Brier</td><td colspan="2"></td></tr>` +
        (ho.heads_h1 || [])
          .map(
            (m, i) =>
              `<tr><td class="k">${heads[i]}</td><td class="n">${m.n}</td><td class="n">${fmt(m.accuracy)}</td><td class="n">${fmt(m.ece)}</td><td class="n">${fmt(m.brier)}</td><td colspan="2"></td></tr>`
          )
          .join("") +
        `<tr><td class="k">latencia mediana</td><td class="n" colspan="6">${ho.latency_us ? ho.latency_us.toFixed(1) + " µs / decisión" : "—"}</td></tr>` +
        `<tr><td class="k">real: tus decisiones</td><td class="n" colspan="6">${real.user_n ? fmt(real.user_ok_accuracy) + " (n=" + real.user_n + ")" : "sin hold-out"}</td></tr>` +
        `<tr><td class="k">real: acuerdo con maestro LLM</td><td class="n" colspan="6">${real.teacher_n ? fmt(real.teacher_agreement) + " (n=" + real.teacher_n + ")" : "sin hold-out"}</td></tr>`;
    }
    const total = j.opts ? (j.opts.datasets === 0 ? "∞" : j.opts.datasets) : "—";
    $("sp-ft-status").textContent = j.running
      ? `entrenando · dataset ${j.current + 1}/${total} · ${j.phase}`
      : j.phase
      ? `${j.phase} · ${j.current} dataset(s)` + (j.last_checkpoint ? " · checkpoint guardado" : "")
      : "listo";
    $("sp-ft-start").disabled = !!j.running;
    $("sp-ft-stop").disabled = !j.running;
    if (j.running) ftPoll();
  }
  function ftLog(evs) {
    const box = $("sp-ft-log");
    for (const e of evs) {
      const d = document.createElement("div");
      d.textContent = `${new Date(e.ts_ms).toLocaleTimeString()} · ${e.message}`;
      box.appendChild(d);
      ftSeq = Math.max(ftSeq, e.seq);
    }
    while (box.children.length > 200) box.removeChild(box.firstChild);
    box.scrollTop = box.scrollHeight;
  }
  async function ftRefresh() {
    try {
      const r = await api("/api/spider/field/events?after=" + ftSeq);
      ftLog(r.events || []);
      const st = await api("/api/spider/field/status");
      renderField(st);
      if (!st.job.running && ftTimer) {
        clearInterval(ftTimer);
        ftTimer = null;
      }
    } catch (_) {}
  }
  function ftPoll() {
    if (!ftTimer) ftTimer = setInterval(ftRefresh, 1000);
  }
  $("sp-ft-start").addEventListener("click", async () => {
    try {
      await post("/api/spider/field/train", {
        datasets: Number($("sp-ft-datasets").value),
        prompts_per_dataset: Number($("sp-ft-ppd").value),
        teacher_live: Number($("sp-ft-teacher").value),
        include_runs: $("sp-ft-runs").checked,
        reset: $("sp-ft-reset").checked,
      });
      $("sp-ft-reset").checked = false;
      ftPoll();
      ftRefresh();
    } catch (e) {
      $("sp-ft-status").textContent = "error: " + e.message;
    }
  });
  $("sp-ft-stop").addEventListener("click", async () => {
    try {
      await post("/api/spider/field/stop", {});
      $("sp-ft-status").textContent = "deteniendo…";
    } catch (e) {
      $("sp-ft-status").textContent = "error: " + e.message;
    }
  });
  ftRefresh();

  // --------------------------------------------------------------- init
  (async () => {
    syncThr();
    try {
      if (fieldChk) fieldChk.checked = localStorage.getItem(FIELD_KEY) === "1";
    } catch (_) {}
    try {
      const m = localStorage.getItem(MODE_KEY);
      if (m && ["auto", "corta", "completa"].includes(m)) {
        $("sp-mode").value = m;
      }
    } catch (_) {}
    let saved = null;
    try {
      saved = localStorage.getItem(PROMPT_KEY);
    } catch (_) {}
    try {
      const st = await api("/api/spider/status");
      setLlmName(fieldOn() ? "campo" : st.llm_active && st.llm_active.label);
      applyModeDefaults(st);
      renderField(st.field);
      if (st.run) {
        loadRun(st);
        ta.value = st.run.prompt;
        if (st.running) {
          setRunning(true);
          setStatus("corrida en curso…");
          subscribe();
        } else {
          setStatus(`última corrida: ${st.run.state}`);
        }
      }
    } catch (_) {}
    if (!ta.value) {
      if (saved) ta.value = saved;
      else {
        try {
          ta.value = (await api("/api/spider/sample")).prompt;
        } catch (_) {}
      }
    }
    dirty = true;
  })();
})();
