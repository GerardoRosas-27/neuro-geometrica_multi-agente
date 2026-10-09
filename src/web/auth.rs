//! Acceso a la app con un único secreto maestro (`MASTER_SECRET`).
//!
//! Mismo esquema que `docker-llm`:
//! - `POST /api/auth/login {secret}` compara en tiempo constante y devuelve una
//!   **sesión firmada** (HMAC-SHA256 con una clave derivada del secreto por
//!   HKDF). Cambiar `MASTER_SECRET` invalida todas las sesiones de golpe.
//! - La sesión viaja en una cookie `HttpOnly; SameSite=Strict; Path=/api`
//!   (la usa también `EventSource`, que no puede mandar cabeceras) o en
//!   `Authorization: Bearer ngs1.…` para scripts. Nunca en la query string.
//! - `POST /api/auth/logout` revoca la sesión en el servidor (lista de `jti`
//!   revocados, persistida en `data/revoked_sessions.json`).
//! - Bloqueo por IP tras 5 fallos (30 s, duplicando hasta 15 min) + tope global.
//! - Sin `MASTER_SECRET` todo `/api/*` responde 503 `master_secret_not_configured`
//!   (fail-closed); `/health` lo avisa.
//!
//! El secreto y los tokens nunca se registran ni se devuelven en errores.

use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const SESSION_PREFIX: &str = "ngs1";
pub const COOKIE_NAME: &str = "ngs_session";
pub const NOT_CONFIGURED_CODE: &str = "master_secret_not_configured";
pub const NOT_CONFIGURED_MESSAGE: &str = "MASTER_SECRET no configurado: la API está cerrada hasta que se defina MASTER_SECRET en las variables de entorno del servidor.";
pub const MIN_RECOMMENDED_LEN: usize = 32;

const FREE_ATTEMPTS: u32 = 5;
const FAIL_WINDOW_SECS: u64 = 15 * 60;
const MAX_LOCK_SECS: u64 = 15 * 60;
const BASE_LOCK_SECS: u64 = 30;
const GLOBAL_FAILS_PER_MINUTE: usize = 30;
const MAX_TRACKED_IPS: usize = 10_000;

pub type SharedAuth = Arc<AuthConfig>;

// ---------------------------------------------------------------------------
// Primitivas (HMAC-SHA256, HKDF, base64url, comparación en tiempo constante)

fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        k[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let inner = Sha256::new()
        .chain_update(ipad)
        .chain_update(msg)
        .finalize();
    Sha256::new()
        .chain_update(opad)
        .chain_update(inner)
        .finalize()
        .into()
}

/// HKDF-SHA256 (RFC 5869) de 32 bytes con sal fija de la aplicación.
fn hkdf32(secret: &[u8], info: &[u8]) -> [u8; 32] {
    let prk = hmac_sha256(b"neuro-geometrica-hkdf-salt-v1", secret);
    let mut m = Vec::with_capacity(info.len() + 1);
    m.extend_from_slice(info);
    m.push(1);
    hmac_sha256(&prk, &m)
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    std::hint::black_box(diff) == 0
}

/// Comparación en tiempo constante aunque las longitudes no coincidan.
fn same_secret(a: &str, b: &str) -> bool {
    ct_eq(&Sha256::digest(a.as_bytes()), &Sha256::digest(b.as_bytes()))
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn b64e(raw: &[u8]) -> String {
    let mut out = String::with_capacity(raw.len().div_ceil(3) * 4);
    for chunk in raw.chunks(3) {
        let n = match chunk.len() {
            3 => (chunk[0] as u32) << 16 | (chunk[1] as u32) << 8 | chunk[2] as u32,
            2 => (chunk[0] as u32) << 16 | (chunk[1] as u32) << 8,
            _ => (chunk[0] as u32) << 16,
        };
        let chars = chunk.len() + 1;
        for i in 0..chars {
            out.push(B64[((n >> (18 - 6 * i)) & 63) as usize] as char);
        }
    }
    out
}

fn b64d(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            _ => return None,
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn random_id() -> String {
    use rand::RngCore;
    let mut b = [0u8; 12];
    rand::rngs::OsRng.fill_bytes(&mut b);
    b64e(&b)
}

// ---------------------------------------------------------------------------
// Límite de intentos

#[derive(Default)]
struct Limiter {
    /// ip -> (fallos, primer_fallo, bloqueado_hasta)
    fails: HashMap<String, (u32, u64, u64)>,
    global: Vec<u64>,
}

impl Limiter {
    fn check(&mut self, client: &str, now: u64) -> Result<(), AuthError> {
        self.global.retain(|t| now.saturating_sub(*t) < 60);
        if self.global.len() >= GLOBAL_FAILS_PER_MINUTE {
            return Err(AuthError::rate_limited(60, true));
        }
        if let Some((_, _, until)) = self.fails.get(client) {
            if *until > now {
                return Err(AuthError::rate_limited(until - now, false));
            }
        }
        Ok(())
    }

    fn fail(&mut self, client: &str, now: u64) {
        self.global.push(now);
        if self.fails.len() >= MAX_TRACKED_IPS {
            self.fails.retain(|_, (_, first, until)| {
                *until > now || now.saturating_sub(*first) <= FAIL_WINDOW_SECS
            });
        }
        let e = self.fails.entry(client.to_string()).or_insert((0, now, 0));
        if now.saturating_sub(e.1) > FAIL_WINDOW_SECS {
            *e = (0, now, 0);
        }
        e.0 += 1;
        if e.0 >= FREE_ATTEMPTS {
            let exp = (e.0 - FREE_ATTEMPTS).min(10);
            e.2 = now + (BASE_LOCK_SECS << exp).min(MAX_LOCK_SECS);
        }
    }

    fn success(&mut self, client: &str) {
        self.fails.remove(client);
    }
}

// ---------------------------------------------------------------------------
// Errores

#[derive(Debug)]
pub struct AuthError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
    pub retry_after: Option<u64>,
}

impl AuthError {
    fn not_configured() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: NOT_CONFIGURED_CODE,
            message: NOT_CONFIGURED_MESSAGE.into(),
            retry_after: None,
        }
    }
    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "unauthorized",
            message: "Sesión requerida: entra con el secreto maestro (POST /api/auth/login)."
                .into(),
            retry_after: None,
        }
    }
    fn invalid_secret() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "invalid_secret",
            message: "Secreto maestro incorrecto.".into(),
            retry_after: None,
        }
    }
    fn rate_limited(wait: u64, global: bool) -> Self {
        let wait = wait.max(1);
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            code: "rate_limited",
            message: if global {
                "Demasiados intentos fallidos. Espera un minuto.".into()
            } else {
                format!("Demasiados intentos fallidos. Espera {wait} s.")
            },
            retry_after: Some(wait),
        }
    }
    fn csrf() -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code: "cross_origin_rejected",
            message: "Petición de otro origen rechazada.".into(),
            retry_after: None,
        }
    }
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let mut resp = (
            self.status,
            Json(json!({ "ok": false, "error": self.message, "code": self.code })),
        )
            .into_response();
        let h = resp.headers_mut();
        h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
        if self.status == StatusCode::UNAUTHORIZED {
            h.insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        if let Some(s) = self.retry_after {
            if let Ok(v) = HeaderValue::from_str(&s.to_string()) {
                h.insert(header::RETRY_AFTER, v);
            }
        }
        resp
    }
}

// ---------------------------------------------------------------------------
// Configuración y sesiones

#[derive(Clone, Debug)]
pub struct Session {
    pub iat: u64,
    pub exp: u64,
    pub jti: String,
}

pub struct AuthConfig {
    secret: Option<String>,
    session_key: Option<[u8; 32]>,
    pub ttl_secs: u64,
    /// Retardo tras un fallo de login (frena intentos en paralelo).
    pub fail_delay: Duration,
    /// `Secure` en la cookie: auto (X-Forwarded-Proto=https) salvo que se fuerce.
    pub cookie_secure: Option<bool>,
    revoked: Mutex<HashMap<String, u64>>,
    revoked_path: Option<PathBuf>,
    limiter: Mutex<Limiter>,
}

impl std::fmt::Debug for AuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Nunca imprimir el secreto ni la clave derivada.
        f.debug_struct("AuthConfig")
            .field("configured", &self.configured())
            .field("ttl_secs", &self.ttl_secs)
            .finish()
    }
}

impl AuthConfig {
    /// `secret` vacío o `None` = no configurado (fail-closed).
    pub fn new(secret: Option<String>, ttl_secs: u64) -> Self {
        let secret = secret
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let session_key = secret
            .as_ref()
            .map(|s| hkdf32(s.as_bytes(), b"neuro-geometrica/session/v1"));
        Self {
            secret,
            session_key,
            ttl_secs: ttl_secs.max(60),
            fail_delay: Duration::from_millis(250),
            cookie_secure: None,
            revoked: Mutex::new(HashMap::new()),
            revoked_path: None,
            limiter: Mutex::new(Limiter::default()),
        }
    }

    /// `MASTER_SECRET`, `SESSION_TTL_HOURS` (12), `AUTH_REVOKED_FILE`
    /// (`data/revoked_sessions.json`), `COOKIE_SECURE` (auto).
    pub fn from_env() -> Self {
        let hours = std::env::var("SESSION_TTL_HOURS")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|h| h.is_finite() && *h > 0.0)
            .unwrap_or(12.0);
        let mut cfg = Self::new(std::env::var("MASTER_SECRET").ok(), (hours * 3600.0) as u64);
        cfg.cookie_secure = match std::env::var("COOKIE_SECURE").ok().as_deref() {
            Some("1") | Some("true") => Some(true),
            Some("0") | Some("false") => Some(false),
            _ => None,
        };
        let path = std::env::var("AUTH_REVOKED_FILE")
            .ok()
            .filter(|p| !p.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("data/revoked_sessions.json"));
        cfg.load_revoked(path);
        cfg
    }

    fn load_revoked(&mut self, path: PathBuf) {
        if let Ok(txt) = std::fs::read_to_string(&path) {
            if let Ok(map) = serde_json::from_str::<HashMap<String, u64>>(&txt) {
                let now = now_secs();
                let mut g = self.revoked.lock().unwrap_or_else(|e| e.into_inner());
                g.extend(map.into_iter().filter(|(_, exp)| *exp > now));
            }
        }
        self.revoked_path = Some(path);
    }

    fn persist_revoked(&self, map: &HashMap<String, u64>) {
        let Some(path) = &self.revoked_path else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match serde_json::to_string(map) {
            Ok(txt) => {
                if let Err(e) = std::fs::write(path, txt) {
                    tracing::warn!(error = %e, "no se pudo guardar la lista de sesiones revocadas");
                }
            }
            Err(e) => tracing::warn!(error = %e, "no se pudo serializar sesiones revocadas"),
        }
    }

    pub fn configured(&self) -> bool {
        self.secret.is_some()
    }

    pub fn mode(&self) -> &'static str {
        if self.configured() {
            "master"
        } else {
            "unconfigured"
        }
    }

    pub fn warnings(&self) -> Vec<String> {
        match &self.secret {
            None => vec![NOT_CONFIGURED_MESSAGE.into()],
            Some(s) if s.chars().count() < MIN_RECOMMENDED_LEN => vec![format!(
                "MASTER_SECRET es corto; usa al menos {MIN_RECOMMENDED_LEN} caracteres aleatorios."
            )],
            _ => vec![],
        }
    }

    fn sign(&self, message: &str) -> Option<String> {
        self.session_key
            .as_ref()
            .map(|k| b64e(&hmac_sha256(k, message.as_bytes())))
    }

    /// Emite una sesión (`now`/`ttl` explícitos para tests).
    pub fn issue_session_at(&self, now: u64, ttl: u64) -> Option<(String, Session)> {
        let s = Session {
            iat: now,
            exp: now + ttl,
            jti: random_id(),
        };
        let payload = json!({ "role": "admin", "iat": s.iat, "exp": s.exp, "jti": s.jti });
        let body = b64e(payload.to_string().as_bytes());
        let message = format!("{SESSION_PREFIX}.{body}");
        let sig = self.sign(&message)?;
        Some((format!("{message}.{sig}"), s))
    }

    pub fn issue_session(&self) -> Option<(String, Session)> {
        self.issue_session_at(now_secs(), self.ttl_secs)
    }

    pub fn verify(&self, token: &str) -> Option<Session> {
        let token = token.trim();
        let mut parts = token.split('.');
        let (prefix, body, sig) = (parts.next()?, parts.next()?, parts.next()?);
        if parts.next().is_some() || prefix != SESSION_PREFIX {
            return None;
        }
        let expected = self.sign(&format!("{prefix}.{body}"))?;
        if !ct_eq(expected.as_bytes(), sig.as_bytes()) {
            return None;
        }
        let payload: Value = serde_json::from_slice(&b64d(body)?).ok()?;
        if payload.get("role")?.as_str()? != "admin" {
            return None;
        }
        let exp = payload.get("exp")?.as_u64()?;
        let iat = payload.get("iat")?.as_u64()?;
        let jti = payload.get("jti")?.as_str()?.to_string();
        if exp <= now_secs() {
            return None;
        }
        if self
            .revoked
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&jti)
        {
            return None;
        }
        Some(Session { iat, exp, jti })
    }

    /// Revoca la sesión en el servidor. `true` si era válida.
    pub fn revoke(&self, token: &str) -> bool {
        let Some(s) = self.verify(token) else {
            return false;
        };
        let now = now_secs();
        let mut g = self.revoked.lock().unwrap_or_else(|e| e.into_inner());
        g.retain(|_, exp| *exp > now);
        g.insert(s.jti, s.exp);
        let snapshot = g.clone();
        drop(g);
        self.persist_revoked(&snapshot);
        true
    }

    /// Comprueba el secreto con límite de intentos. No duerme (lo hace el handler).
    pub fn check_login(&self, secret: &str, client: &str) -> Result<(String, Session), AuthError> {
        let master = self
            .secret
            .as_deref()
            .ok_or_else(AuthError::not_configured)?;
        let now = now_secs();
        let mut lim = self.limiter.lock().unwrap_or_else(|e| e.into_inner());
        lim.check(client, now)?;
        if !same_secret(secret.trim(), master) {
            lim.fail(client, now);
            return Err(AuthError::invalid_secret());
        }
        lim.success(client);
        drop(lim);
        self.issue_session().ok_or_else(AuthError::not_configured)
    }

    fn cookie(&self, value: &str, max_age: u64, headers: &HeaderMap) -> String {
        let secure = self.cookie_secure.unwrap_or_else(|| {
            headers
                .get("x-forwarded-proto")
                .and_then(|v| v.to_str().ok())
                .map(|v| {
                    v.split(',')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .eq_ignore_ascii_case("https")
                })
                .unwrap_or(false)
        });
        format!(
            "{COOKIE_NAME}={value}; Path=/api; Max-Age={max_age}; HttpOnly; SameSite=Strict{}",
            if secure { "; Secure" } else { "" }
        )
    }
}

// ---------------------------------------------------------------------------
// Extracción de credenciales

/// Token de la petición y si vino por cookie (`true`) o Bearer (`false`).
pub fn request_token(headers: &HeaderMap) -> Option<(String, bool)> {
    if let Some(v) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        let v = v.trim();
        if v.len() > 7 && v[..7].eq_ignore_ascii_case("bearer ") {
            return Some((v[7..].trim().to_string(), false));
        }
    }
    for raw in headers.get_all(header::COOKIE) {
        let Ok(s) = raw.to_str() else { continue };
        for part in s.split(';') {
            if let Some((k, v)) = part.trim().split_once('=') {
                if k == COOKIE_NAME && !v.is_empty() {
                    return Some((v.to_string(), true));
                }
            }
        }
    }
    None
}

/// IP del cliente: Railway añade la IP real al final de X-Forwarded-For.
fn client_ip(req_headers: &HeaderMap, peer: Option<SocketAddr>) -> String {
    if let Some(f) = req_headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
    {
        if let Some(last) = f
            .rsplit(',')
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return last.chars().take(64).collect();
        }
    }
    peer.map(|p| p.ip().to_string())
        .unwrap_or_else(|| "?".into())
}

fn host_of_origin(origin: &str) -> Option<&str> {
    let rest = origin.split_once("://")?.1;
    Some(rest.split('/').next().unwrap_or(rest))
}

/// Defensa CSRF para peticiones con cookie que cambian estado: el Origin (si
/// viene) debe coincidir con el Host; si no hay Origin, Sec-Fetch-Site (si
/// viene) debe ser same-origin/none.
fn same_origin_ok(headers: &HeaderMap) -> bool {
    let host = headers
        .get("x-forwarded-host")
        .or_else(|| headers.get(header::HOST))
        .and_then(|v| v.to_str().ok())
        .map(|h| {
            h.split(',')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase()
        });
    if let Some(origin) = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
        return match (host_of_origin(origin), host) {
            (Some(o), Some(h)) => o.eq_ignore_ascii_case(&h),
            _ => false,
        };
    }
    match headers.get("sec-fetch-site").and_then(|v| v.to_str().ok()) {
        Some(s) => matches!(s, "same-origin" | "none"),
        None => true,
    }
}

/// Rutas `/api/*` públicas (el resto exige sesión).
pub fn is_public_api(path: &str) -> bool {
    matches!(
        path,
        "/api/auth/login" | "/api/auth/logout" | "/api/auth/status"
    )
}

/// Middleware: protege **todo** `/api/*` salvo `/api/auth/*`.
pub async fn require_auth(
    State(auth): State<SharedAuth>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let path = req.uri().path();
    let protected = (path == "/api" || path.starts_with("/api/")) && !is_public_api(path);
    if !protected || req.method() == Method::OPTIONS {
        return next.run(req).await;
    }
    if !auth.configured() {
        return AuthError::not_configured().into_response();
    }
    let Some((token, via_cookie)) = request_token(req.headers()) else {
        return AuthError::unauthorized().into_response();
    };
    if auth.verify(&token).is_none() {
        return AuthError::unauthorized().into_response();
    }
    let safe = matches!(*req.method(), Method::GET | Method::HEAD);
    if via_cookie && !safe && !same_origin_ok(req.headers()) {
        return AuthError::csrf().into_response();
    }
    next.run(req).await
}

// ---------------------------------------------------------------------------
// Handlers

#[derive(Deserialize)]
pub struct LoginBody {
    #[serde(default)]
    pub secret: String,
}

fn no_store(mut resp: Response) -> Response {
    resp.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    resp
}

pub async fn login(
    State(auth): State<SharedAuth>,
    headers: HeaderMap,
    req: Request<Body>,
) -> Response {
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0);
    let client = client_ip(&headers, peer);
    // Cuerpo JSON pequeño; nunca se registra.
    let bytes = match axum::body::to_bytes(req.into_body(), 16 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            return no_store(
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"ok": false, "error": "cuerpo inválido", "code": "bad_request"})),
                )
                    .into_response(),
            )
        }
    };
    let secret = serde_json::from_slice::<LoginBody>(&bytes)
        .map(|b| b.secret)
        .unwrap_or_default();
    match auth.check_login(&secret, &client) {
        Ok((token, s)) => {
            tracing::info!(client = %client, "login correcto");
            let cookie = auth.cookie(&token, s.exp.saturating_sub(s.iat), &headers);
            let mut resp = Json(json!({
                "ok": true,
                "token": token,
                "role": "admin",
                "expires_at": s.exp,
                "expires_in": s.exp.saturating_sub(s.iat),
            }))
            .into_response();
            if let Ok(v) = HeaderValue::from_str(&cookie) {
                resp.headers_mut().insert(header::SET_COOKIE, v);
            }
            no_store(resp)
        }
        Err(e) => {
            if e.code == "invalid_secret" {
                tracing::warn!(client = %client, "login fallido");
                tokio::time::sleep(auth.fail_delay).await;
            } else if e.code == "rate_limited" {
                tracing::warn!(client = %client, "login bloqueado por intentos");
            }
            e.into_response()
        }
    }
}

pub async fn logout(State(auth): State<SharedAuth>, headers: HeaderMap) -> Response {
    let revoked = request_token(&headers)
        .map(|(t, _)| auth.revoke(&t))
        .unwrap_or(false);
    let mut resp = Json(json!({ "ok": true, "revoked": revoked })).into_response();
    if let Ok(v) = HeaderValue::from_str(&auth.cookie("", 0, &headers)) {
        resp.headers_mut().insert(header::SET_COOKIE, v);
    }
    no_store(resp)
}

pub fn status_json(auth: &AuthConfig, headers: &HeaderMap) -> Value {
    let session = request_token(headers).and_then(|(t, cookie)| {
        auth.verify(&t).map(|s| {
            json!({
                "expires_at": s.exp,
                "issued_at": s.iat,
                "via": if cookie { "cookie" } else { "bearer" },
            })
        })
    });
    json!({
        "ok": true,
        "mode": auth.mode(),
        "master_configured": auth.configured(),
        "authenticated": session.is_some(),
        "session": session,
        "session_ttl_hours": auth.ttl_secs as f64 / 3600.0,
        "warnings": auth.warnings(),
    })
}

pub async fn status(State(auth): State<SharedAuth>, headers: HeaderMap) -> Response {
    no_store(Json(status_json(&auth, &headers)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc4231_case2() {
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        let hex: String = mac.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn b64_roundtrip() {
        for n in 0..40u8 {
            let v: Vec<u8> = (0..n).map(|i| i.wrapping_mul(37)).collect();
            assert_eq!(b64d(&b64e(&v)).unwrap(), v);
        }
        assert_eq!(b64e(b"hi?"), "aGk_");
    }

    #[test]
    fn sessions_sign_verify_tamper_expire_rotate() {
        let a = AuthConfig::new(Some("a".repeat(40)), 3600);
        let (tok, _) = a.issue_session().unwrap();
        assert!(a.verify(&tok).is_some());
        // Firma alterada
        let mut bad = tok.clone();
        let last = bad.pop().unwrap();
        bad.push(if last == 'A' { 'B' } else { 'A' });
        assert!(a.verify(&bad).is_none());
        // Payload alterado (exp extendido) con la firma antigua
        let parts: Vec<&str> = tok.split('.').collect();
        let forged = b64e(br#"{"role":"admin","iat":1,"exp":99999999999,"jti":"x"}"#);
        assert!(a
            .verify(&format!("{}.{}.{}", parts[0], forged, parts[2]))
            .is_none());
        // Caducada
        let (old, _) = a.issue_session_at(now_secs() - 7200, 3600).unwrap();
        assert!(a.verify(&old).is_none());
        // Otro secreto
        let b = AuthConfig::new(Some("b".repeat(40)), 3600);
        assert!(b.verify(&tok).is_none());
        // Sin secreto
        let none = AuthConfig::new(None, 3600);
        assert!(none.issue_session().is_none());
        assert!(none.verify(&tok).is_none());
        // Revocación
        assert!(a.revoke(&tok));
        assert!(a.verify(&tok).is_none());
        assert!(!a.revoke(&tok));
    }

    #[test]
    fn limiter_locks_after_five_and_doubles() {
        let mut l = Limiter::default();
        let t0 = 1_000_000;
        for _ in 0..4 {
            l.fail("ip", t0);
            assert!(l.check("ip", t0).is_ok());
        }
        l.fail("ip", t0);
        let e = l.check("ip", t0).unwrap_err();
        assert_eq!(e.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(e.retry_after, Some(30));
        assert!(l.check("other", t0).is_ok());
        assert!(l.check("ip", t0 + 31).is_ok());
        l.fail("ip", t0 + 31);
        assert_eq!(l.check("ip", t0 + 31).unwrap_err().retry_after, Some(60));
        for _ in 0..20 {
            l.fail("ip", t0 + 40);
        }
        let wait = l.check("ip", t0 + 40).unwrap_err().retry_after.unwrap();
        assert!(wait <= MAX_LOCK_SECS);
    }

    #[test]
    fn global_cap_blocks_everyone() {
        let mut l = Limiter::default();
        for i in 0..GLOBAL_FAILS_PER_MINUTE {
            l.fail(&format!("ip{i}"), 500);
        }
        assert!(l.check("fresh", 500).is_err());
        assert!(l.check("fresh", 561).is_ok());
    }

    #[test]
    fn warnings_short_and_missing() {
        assert!(!AuthConfig::new(None, 3600).warnings().is_empty());
        assert!(!AuthConfig::new(Some("corto".into()), 3600)
            .warnings()
            .is_empty());
        assert!(AuthConfig::new(Some("x".repeat(48)), 3600)
            .warnings()
            .is_empty());
        let dbg = format!("{:?}", AuthConfig::new(Some("supersecreto".into()), 3600));
        assert!(!dbg.contains("supersecreto"));
    }

    #[test]
    fn csrf_origin_checks() {
        let mut h = HeaderMap::new();
        h.insert(header::HOST, HeaderValue::from_static("app.example"));
        assert!(same_origin_ok(&h));
        h.insert(
            header::ORIGIN,
            HeaderValue::from_static("https://app.example"),
        );
        assert!(same_origin_ok(&h));
        h.insert(
            header::ORIGIN,
            HeaderValue::from_static("https://evil.example"),
        );
        assert!(!same_origin_ok(&h));
        h.remove(header::ORIGIN);
        h.insert("sec-fetch-site", HeaderValue::from_static("cross-site"));
        assert!(!same_origin_ok(&h));
    }
}
