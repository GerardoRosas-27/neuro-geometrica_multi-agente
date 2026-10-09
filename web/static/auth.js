// Acceso con secreto maestro (MASTER_SECRET).
// - El secreto solo viaja en POST /api/auth/login y nunca se guarda en el
//   navegador. La sesión vive en una cookie HttpOnly (JS no la ve) que también
//   usan los EventSource de las consolas en vivo.
// - Cualquier 401 de /api/* muestra el login; 503 master_secret_not_configured
//   muestra el aviso de configuración.
// - app.js solo se carga cuando hay sesión.
(() => {
  const APP_SRC = "/app.js?v=35";
  const SPIDER_SRC = "/spider.js?v=35";
  const $ = (id) => document.getElementById(id);
  const nativeFetch = window.fetch.bind(window);
  const overlay = $("auth-overlay");
  const form = $("auth-form");
  const input = $("auth-secret");
  const errBox = $("auth-error");
  const warnBox = $("auth-unconfigured");
  const noteBox = $("auth-note");
  const badge = $("badge-session");
  const btnLogout = $("btn-logout");
  let appLoaded = false;
  let expiryTimer = null;

  function isProtectedApi(input) {
    try {
      const raw = typeof input === "string" ? input : input && input.url;
      const u = new URL(raw, location.href);
      return (
        u.origin === location.origin &&
        u.pathname.startsWith("/api/") &&
        !u.pathname.startsWith("/api/auth/")
      );
    } catch (_) {
      return false;
    }
  }

  // Envoltorio global: la app existente sigue usando fetch() tal cual.
  window.fetch = async (resource, init) => {
    const res = await nativeFetch(resource, init);
    if (isProtectedApi(resource)) {
      if (res.status === 401) {
        showLogin("La sesión caducó o se cerró. Vuelve a entrar con el secreto maestro.");
      } else if (res.status === 503) {
        res
          .clone()
          .json()
          .then((b) => {
            if (b && b.code === "master_secret_not_configured") showUnconfigured();
          })
          .catch(() => {});
      }
    }
    return res;
  };

  function fmtExpiry(sec) {
    const d = new Date(sec * 1000);
    const sameDay = d.toDateString() === new Date().toDateString();
    const time = d.toLocaleTimeString("es-MX", { hour: "2-digit", minute: "2-digit" });
    return sameDay
      ? time
      : `${d.toLocaleDateString("es-MX", { day: "2-digit", month: "short" })} ${time}`;
  }

  function setSession(expiresAt) {
    badge.textContent = `Sesión activa · hasta ${fmtExpiry(expiresAt)}`;
    badge.title = `La sesión caduca el ${new Date(expiresAt * 1000).toLocaleString("es-MX")}`;
    badge.hidden = false;
    btnLogout.hidden = false;
    if (expiryTimer) clearTimeout(expiryTimer);
    const ms = expiresAt * 1000 - Date.now();
    expiryTimer = setTimeout(
      () => {
        if (Date.now() >= expiresAt * 1000) showLogin("La sesión caducó. Vuelve a entrar.");
        else setSession(expiresAt);
      },
      Math.max(1000, Math.min(ms, 6 * 3600 * 1000)),
    );
  }

  function clearSession() {
    badge.hidden = true;
    btnLogout.hidden = true;
    if (expiryTimer) clearTimeout(expiryTimer);
    expiryTimer = null;
  }

  function showLogin(note) {
    clearSession();
    warnBox.hidden = true;
    form.hidden = false;
    noteBox.textContent = note || "";
    noteBox.hidden = !note;
    if (overlay.hidden) {
      errBox.textContent = "";
      overlay.hidden = false;
      document.body.classList.add("auth-locked");
      setTimeout(() => input.focus(), 0);
    }
  }

  function showUnconfigured() {
    clearSession();
    form.hidden = true;
    noteBox.hidden = true;
    warnBox.hidden = false;
    overlay.hidden = false;
    document.body.classList.add("auth-locked");
  }

  function hideOverlay() {
    overlay.hidden = true;
    document.body.classList.remove("auth-locked");
    errBox.textContent = "";
  }

  function loadApp() {
    if (appLoaded) return;
    appLoaded = true;
    const s = document.createElement("script");
    s.src = APP_SRC;
    document.body.appendChild(s);
    const sp = document.createElement("script");
    sp.src = SPIDER_SRC;
    document.body.appendChild(sp);
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const secret = input.value;
    if (!secret.trim()) {
      errBox.textContent = "Escribe el secreto maestro.";
      return;
    }
    const btn = $("auth-submit");
    btn.disabled = true;
    errBox.textContent = "";
    try {
      const res = await nativeFetch("/api/auth/login", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ secret }),
        credentials: "same-origin",
      });
      let r = null;
      try {
        r = await res.json();
      } catch (_) {}
      if (res.ok && r && r.ok) {
        input.value = ""; // el secreto no se conserva
        hideOverlay();
        setSession(r.expires_at);
        loadApp();
        return;
      }
      if (r && r.code === "master_secret_not_configured") {
        showUnconfigured();
        return;
      }
      input.value = "";
      errBox.textContent = (r && r.error) || `Error HTTP ${res.status}`;
    } catch (e) {
      errBox.textContent = "Sin conexión con el servidor.";
    } finally {
      btn.disabled = false;
    }
  });

  btnLogout.addEventListener("click", async () => {
    btnLogout.disabled = true;
    try {
      await nativeFetch("/api/auth/logout", { method: "POST", credentials: "same-origin" });
    } catch (_) {}
    // Recarga: detiene sondeos/streams y vuelve a la pantalla de acceso.
    location.reload();
  });

  async function init() {
    try {
      const res = await nativeFetch("/api/auth/status", { credentials: "same-origin" });
      const st = await res.json();
      if (!st.master_configured) {
        showUnconfigured();
        return;
      }
      if (st.authenticated && st.session) {
        setSession(st.session.expires_at);
        loadApp();
        return;
      }
      showLogin();
    } catch (_) {
      showLogin("No se pudo consultar el estado del servidor. Reintenta.");
    }
  }

  init();
})();
