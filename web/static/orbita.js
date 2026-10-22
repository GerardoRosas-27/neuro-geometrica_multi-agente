// Pestaña ÓRBITA: formulario estructurado (familia + 2 parámetros + estado de
// 12 números) → POST /api/orbita/run. Si el servidor no tiene Python, pesos,
// GGUF o el binario `orbita`, /api/orbita/status lo dice y el formulario se
// desactiva con la lista de faltantes (no rompe el resto de la GUI).
(() => {
  const $ = (id) => document.getElementById(id);
  const form = $("orb-form");
  if (!form) return;
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const fmt = (x) => (typeof x === "number" ? x.toFixed(4) : String(x ?? "—"));
  const tabDot = $("status-orbita");
  let families = [];
  let textEdited = false;

  function setDot(state) {
    if (tabDot) tabDot.dataset.state = state;
  }

  function fillSelects(fams) {
    families = fams;
    $("orb-family").innerHTML = fams.map((f, i) => `<option value="${i}">${i} · ${esc(f)}</option>`).join("");
    $("orb-family").value = "1";
    const seeds = [];
    for (const p of ["A", "B"]) for (let i = 0; i < 16; i++) seeds.push(`0x${p}4${i.toString(16).toUpperCase().padStart(2, "0")}`);
    $("orb-seed").innerHTML = seeds.map((s) => `<option>${s}</option>`).join("");
    syncText();
  }

  function numText(v) {
    return String(v).replace(".", ",");
  }
  function syncText() {
    if (textEdited) return;
    const f = families[+$("orb-family").value];
    const p1 = $("orb-p1").value, p2 = $("orb-p2").value;
    if (f && p1 !== "" && p2 !== "") $("orb-text").value = `Aplica ${+$("orb-family").value === 3 ? "un" : "una"} ${f} con parámetros ${numText(p1)} y ${numText(p2)}.`;
  }
  ["orb-family", "orb-p1", "orb-p2"].forEach((id) => $(id).addEventListener("input", syncText));
  $("orb-text").addEventListener("input", () => (textEdited = $("orb-text").value.trim() !== ""));
  $("orb-obs-mode").addEventListener("change", () => ($("orb-observed-wrap").hidden = $("orb-obs-mode").value !== "real"));

  function parseState(s, what) {
    const v = s.split(/[,;\s]+/).filter(Boolean).map(Number);
    if (v.length !== 12) throw new Error(`${what}: se esperan 12 números, hay ${v.length}`);
    if (v.some((x) => !Number.isFinite(x))) throw new Error(`${what}: hay valores no numéricos`);
    return v;
  }

  async function refreshStatus() {
    const box = $("orb-avail");
    try {
      const r = await fetch("/api/orbita/status");
      const d = await r.json();
      if (!families.length) fillSelects(d.families || []);
      if (d.available) {
        box.dataset.state = "ok";
        box.textContent = "ÓRBITA disponible en este servidor.";
        $("orb-submit").disabled = false;
        setDot("idle");
      } else {
        box.dataset.state = "off";
        box.innerHTML = "<strong>ÓRBITA no está disponible en este servidor.</strong> Faltan:<ul>" +
          (d.missing || []).map((m) => `<li>${esc(m)}</li>`).join("") + "</ul>";
        $("orb-submit").disabled = true;
        setDot("error");
      }
    } catch (e) {
      box.dataset.state = "off";
      box.textContent = "No se pudo consultar /api/orbita/status: " + e.message;
      $("orb-submit").disabled = true;
    }
  }

  function plot(x0, base, reid) {
    const pts = (v) => Array.from({ length: 6 }, (_, i) => [v[2 * i], v[2 * i + 1]]);
    const all = [x0, base[base.length - 1]].concat(reid ? [reid[reid.length - 1]] : []).flatMap(pts);
    const m = Math.max(1, ...all.map(([x, y]) => Math.max(Math.abs(x), Math.abs(y)))) * 1.1;
    const S = 220, sc = (v) => ((v / m) * 0.5 + 0.5) * S;
    const poly = (v, color, dash) =>
      `<polygon points="${pts(v).map(([x, y]) => `${sc(x)},${S - sc(y)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="1.5" ${dash ? 'stroke-dasharray="4 3"' : ""}/>`;
    return `<svg width="${S}" height="${S}" viewBox="0 0 ${S} ${S}">${poly(x0, "#888", true)}${poly(base[0], "#4a8", false)}${poly(base[base.length - 1], "#28c", false)}${reid ? poly(reid[reid.length - 1], "#c62", false) : ""}</svg>
      <div class="hint">gris: x0 · verde: h1 · azul: h16 base${reid ? " · naranja: h16 tras re-observación" : ""}</div>`;
  }

  function render(d, x0) {
    const r = d.result, yo = r.yo || {}, ro = r.re_observacion;
    const traj = r.trayectoria_h1_h16 || [];
    const rows = traj.map((s, i) => `<tr><th>h${i + 1}</th>${s.map((v) => `<td>${fmt(v)}</td>`).join("")}</tr>`).join("");
    $("orb-result").innerHTML = `
      <h3>Lectura del encoder</h3>
      <p>Familia leída: <strong>${esc(r.familia_leida)}</strong> · p = [${(r.p_leido || []).map(fmt).join(", ")}] (${esc(r.p_fuente)}) · prob. encoder ${fmt(r.prob_encoder)} · seed ${esc(r.seed)} · ${d.elapsed_ms} ms</p>
      <h3>«yo» (informa, no decide)</h3>
      <p>Confianza de acierto h16: <strong>${fmt(yo.confianza_acierto_h16)}</strong> · sorpresa vs obs: ${fmt(yo.sorpresa_vs_obs)} · σ Jacobiano: ${fmt(yo.sigma_jacobiano)} · ratio norma h16: ${fmt(yo.ratio_norma_h16)}</p>
      ${ro ? `<h3>Re-observación (k=${ro.k})</h3><p>Familia re-identificada: <strong>${esc(ro.familia_reidentificada)}</strong> · p = [${(ro.p_reidentificado || []).map(fmt).join(", ")}] · error de ajuste ${fmt(ro.error_ajuste)}</p>` : ""}
      ${plot(x0, traj, ro && ro.trayectoria_h1_h16)}
      <details><summary>Trayectoria base h1–h16 (12 números por paso)</summary><table>${rows}</table></details>
      <details><summary>JSON completo</summary><pre>${esc(JSON.stringify(r, null, 1))}</pre></details>`;
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const errBox = $("orb-error");
    errBox.textContent = "";
    let body;
    let x0;
    try {
      const p1 = Number($("orb-p1").value), p2 = Number($("orb-p2").value);
      if (!Number.isFinite(p1) || !Number.isFinite(p2) || Math.abs(p1) > 1 || Math.abs(p2) > 1)
        throw new Error("p1 y p2 deben ser números con |p| ≤ 1 (rango de entrenamiento)");
      body = { family: +$("orb-family").value, p1, p2, seed: $("orb-seed").value };
      const text = $("orb-text").value.trim();
      if (text) {
        if (!/\d/.test(text)) throw new Error("el texto debe incluir los 2 parámetros numéricos");
        body.text = text;
      }
      const xs = $("orb-x0").value.trim();
      x0 = xs ? parseState(xs, "x0") : [0.8, 0.0, 0.4, 0.69282, -0.4, 0.69282, -0.8, 0.0, -0.4, -0.69282, 0.4, -0.69282];
      if (xs) body.x0 = x0;
      const mode = $("orb-obs-mode").value, k = +$("orb-k").value;
      if (mode === "sim") body.simulate_obs = k;
      if (mode === "real") {
        const lines = $("orb-observed").value.split("\n").map((l) => l.trim()).filter(Boolean);
        if (lines.length < 1 || lines.length > 2) throw new Error("observaciones reales: 1 o 2 líneas");
        body.observed = lines.map((l, i) => parseState(l, `observación ${i + 1}`));
      }
    } catch (e) {
      errBox.textContent = e.message;
      return;
    }
    $("orb-submit").disabled = true;
    setDot("running");
    $("orb-result").innerHTML = '<p class="hint">ÓRBITA calculando (Gemma capa 13 → D2 → ÓRBITA-v1 → yo)…</p>';
    try {
      const r = await fetch("/api/orbita/run", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      const d = await r.json().catch(() => ({}));
      if (!r.ok || !d.ok) {
        $("orb-result").innerHTML = `<p class="auth-error">${esc(d.error || "error " + r.status)}</p>` +
          (d.missing ? "<ul>" + d.missing.map((m) => `<li>${esc(m)}</li>`).join("") + "</ul>" : "") +
          (d.stderr ? `<pre>${esc(d.stderr)}</pre>` : "");
        setDot("error");
        if (r.status === 503) refreshStatus();
      } else {
        render(d, x0);
        setDot("done");
      }
    } catch (e) {
      $("orb-result").innerHTML = `<p class="auth-error">${esc(e.message)}</p>`;
      setDot("error");
    } finally {
      $("orb-submit").disabled = $("orb-avail").dataset.state !== "ok";
    }
  });

  refreshStatus();
})();
