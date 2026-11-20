//! **neuro_consola** — Aplicación de consola interactiva + Entrenamiento Infinito Continuo
//! para la arquitectura neuro-geométrica metacognitiva **METAFE-3S-Dial**.
//!
//! Uso:
//!   neuro_consola chat  [--ckpt PATH] [--bg-train] [--gguf PATH] [--api-base URL --api-model M --api-key K]
//!   neuro_consola train [--cycles 0] [--batch 28] [--eval-every 5] [--ckpt PATH] [--log PATH] [--seed 0x35D1]
//!   neuro_consola eval  [--seeds 8] [--cycles 16] [--out PATH]
//!   neuro_consola demo  [--ckpt PATH]
//!
//! En `chat`, la arquitectura razona, refuta sus propias hipótesis con ramas ortogonales
//! `F_t^A ⊥ F_t^B` y decide su postura epistémica; el LLM (Gemma 2 GGUF o API OpenAI-compatible)
//! actúa estrictamente como periferia lingüística congelada bajo Guardia de Fidelidad.

use cdt_rqm_epr::neuro_dialogo::core3s::DeltaMode;
use cdt_rqm_epr::neuro_dialogo::curriculum::{
    evaluate_sealed_and_open, run_ablation_benchmark, run_training_cycle,
};
use cdt_rqm_epr::neuro_dialogo::periphery::{verify_periphery_fidelity, PeripheryBackend};
use cdt_rqm_epr::neuro_dialogo::DialogueAgent;
use serde_json::json;
use std::fs::OpenOptions;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_CKPT: &str = "artifacts/neuro_consola/ckpt.json";
const DEFAULT_LOG: &str = "artifacts/neuro_consola/train_stream.jsonl";

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|x| x == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn has_flag(args: &[String], k: &str) -> bool {
    args.iter().any(|x| x == k)
}

fn parse_seed(s: &str) -> u64 {
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).unwrap_or(0x35D1)
    } else {
        s.parse().unwrap_or(0x35D1)
    }
}

fn load_or_init_agent(ckpt: &Path, seed: u64, warmup_cycles: u64) -> DialogueAgent {
    if ckpt.is_file() {
        match DialogueAgent::load_checkpoint(ckpt) {
            Ok(ag) => {
                eprintln!(
                    "[ckpt] Restaurado núcleo METAFE-3S-Dial desde {} (ciclo {}, episodios={}, sueños EWC={})",
                    ckpt.display(),
                    ag.cycle,
                    ag.core.total_train_episodes,
                    ag.core.total_sleeps
                );
                return ag;
            }
            Err(e) => eprintln!("[ckpt] Aviso: {e}; inicializando núcleo nuevo."),
        }
    }
    let mut ag = DialogueAgent::new(seed);
    if warmup_cycles > 0 {
        eprint!("[init] Calentando núcleo lento W_0 + EWC ({warmup_cycles} ciclos)...");
        let _ = io::stderr().flush();
        for _ in 0..warmup_cycles {
            ag.train_one_cycle(28);
        }
        ag.core.reset_episode_fast_state();
        eprintln!(" listo.");
    }
    ag
}

fn configure_periphery(ag: &mut DialogueAgent, args: &[String]) {
    if let Some(gpath) = arg(args, "--gguf").or_else(|| std::env::var("GEMMA2_GGUF").ok()) {
        if !gpath.trim().is_empty() && Path::new(&gpath).is_file() {
            match PeripheryBackend::open_local_gemma(Path::new(&gpath)) {
                Ok(b) => {
                    eprintln!("[periferia] Gemma 2 GGUF congelado cargado: {gpath}");
                    ag.periphery = b;
                    return;
                }
                Err(e) => eprintln!("[periferia] No se pudo abrir GGUF ({e}); usando verbalizador nativo."),
            }
        }
    }
    if let Some(base) = arg(args, "--api-base").or_else(|| std::env::var("LLM_API_BASE").ok()) {
        if !base.trim().is_empty() {
            let model = arg(args, "--api-model")
                .or_else(|| std::env::var("LLM_API_MODEL").ok())
                .unwrap_or_else(|| "gemma2:2b".into());
            let api_key = arg(args, "--api-key")
                .or_else(|| std::env::var("LLM_API_KEY").ok())
                .unwrap_or_default();
            eprintln!("[periferia] API LLM congelada configurada: {model} @ {base}");
            ag.periphery = PeripheryBackend::OpenAiApi {
                base_url: base,
                model,
                api_key,
                timeout_secs: 45,
            };
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("chat");
    match cmd {
        "chat" => run_chat(&args),
        "train" => run_infinite_train(&args),
        "eval" => run_eval(&args),
        "demo" => run_demo(&args),
        "-h" | "--help" | "help" => print_usage(),
        other => {
            eprintln!("Subcomando desconocido: {other}");
            print_usage();
            std::process::exit(2);
        }
    }
}

fn print_usage() {
    println!(
        "neuro_consola — Arquitectura Neuro-Geométrica Metacognitiva METAFE-3S-Dial\n\
         \n\
         SUBCOMANDOS:\n\
         \x20 chat  [--ckpt PATH] [--bg-train] [--warmup 8] [--gguf PATH] [--api-base URL --api-model M --api-key K]\n\
         \x20 train [--cycles 0] [--batch 28] [--eval-every 5] [--ckpt PATH] [--log PATH] [--seed 0x35D1]\n\
         \x20       (--cycles 0 = entrenamiento continuo INFINITO reanudable con EWC y guardia anti-divergencia)\n\
         \x20 eval  [--seeds 8] [--cycles 16] [--out PATH]\n\
         \x20       (benchmark sellado multi-semilla + ablaciones causales contra Naive-Entropy y NoEWC)\n\
         \x20 demo  [--ckpt PATH]\n\
         \x20       (demostración conversacional multi-turno + prueba de Guardia de Fidelidad de Periferia)"
    );
}

/// Modo 1: Entrenamiento Infinito en Consola (`--cycles 0` = infinito; reanudable desde checkpoint).
fn run_infinite_train(args: &[String]) {
    let cycles: u64 = arg(args, "--cycles")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let batch: usize = arg(args, "--batch")
        .and_then(|v| v.parse().ok())
        .unwrap_or(28);
    let eval_every: u64 = arg(args, "--eval-every")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
        .max(1);
    let seed = arg(args, "--seed").map(|s| parse_seed(&s)).unwrap_or(0x35D1);
    let ckpt = PathBuf::from(arg(args, "--ckpt").unwrap_or_else(|| DEFAULT_CKPT.into()));
    let log_path = PathBuf::from(arg(args, "--log").unwrap_or_else(|| DEFAULT_LOG.into()));

    if let Some(dir) = log_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut ag = load_or_init_agent(&ckpt, seed, 0);
    let mut last_good = ag.core.clone();
    let t_start = Instant::now();

    println!(
        "=== METAFE-3S-Dial · Entrenamiento Continuo {} ===\n\
         seed=0x{:X} | ciclo_inicial={} | batch={} | eval_cada={} | ckpt={} | log={}",
        if cycles == 0 {
            "INFINITO (Ctrl+C para detener; checkpoint atómico)"
        } else {
            "ACOTADO"
        },
        seed,
        ag.cycle,
        batch,
        eval_every,
        ckpt.display(),
        log_path.display()
    );

    let mut done_this_run = 0u64;
    loop {
        if cycles > 0 && done_this_run >= cycles {
            break;
        }
        let tc = Instant::now();
        let m = ag.train_one_cycle(batch);
        done_this_run += 1;

        // Guardia de divergencia: si aparece pérdida no finita, restaurar `last_good`
        if !m.sleep_loss_after.is_finite() || !m.mean_delta.is_finite() {
            eprintln!(
                "[ALERTA ciclo {}] pérdida no finita detectada -> restaurando last_good!",
                m.cycle
            );
            ag.core = last_good.clone();
            continue;
        }
        last_good = ag.core.clone();

        let eval_opt = if m.cycle % eval_every == 0 || (cycles > 0 && done_this_run == cycles) {
            Some(evaluate_sealed_and_open(&ag.core, seed))
        } else {
            None
        };

        if let Err(e) = ag.save_checkpoint(&ckpt) {
            eprintln!("[ckpt] Error guardando: {e}");
        }

        let rec = json!({
            "cycle": m.cycle,
            "elapsed_s": t_start.elapsed().as_secs_f64(),
            "cycle_ms": tc.elapsed().as_secs_f64() * 1e3,
            "train": m,
            "sealed_eval": eval_opt,
            "total_episodes": ag.core.total_train_episodes,
            "total_sleeps": ag.core.total_sleeps,
            "total_dogma_blocks": ag.core.total_dogma_blocks,
            "total_resets": ag.core.total_resets,
        });
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&log_path) {
            let _ = writeln!(f, "{rec}");
        }

        if let Some(ev) = &eval_opt {
            println!(
                "ciclo {:>4} | loss {:.3}->{:.3} | buf {}/{} | eval_sellado: int={:.1}% st_nn={:.1}% st_ok={:.1}% | log={:.0}% contra={:.0}% abst={:.0}% mem={:.0}% | abierto: dogma={:.3} div={:.3} cf={:.3} orth={:.3} brier={:.3} | {:.0}ms",
                m.cycle,
                m.sleep_loss_before,
                m.sleep_loss_after,
                m.buffered,
                m.batch_size,
                ev.neural_intent_acc * 100.0,
                ev.neural_stance_acc * 100.0,
                ev.grounded_stance_acc * 100.0,
                ev.logic_acc * 100.0,
                ev.contradiction_acc * 100.0,
                ev.abstention_acc * 100.0,
                ev.memory_revision_acc * 100.0,
                ev.open_dogmatism_index,
                ev.open_output_diversity,
                ev.open_cf_sens,
                ev.open_branch_orth,
                ev.open_brier,
                tc.elapsed().as_secs_f64() * 1e3
            );
        } else {
            println!(
                "ciclo {:>4} | loss {:.3}->{:.3} | buf {}/{} (bloq_dogma={}) | δ={:+.3} ΔF={:+.2} | fisher={:.4} | {:.0}ms",
                m.cycle,
                m.sleep_loss_before,
                m.sleep_loss_after,
                m.buffered,
                m.batch_size,
                m.rejected_dogma,
                m.mean_delta,
                m.mean_fe_delta,
                m.fisher_mean,
                tc.elapsed().as_secs_f64() * 1e3
            );
        }
    }
    println!(
        "=== Entrenamiento completado: {} ciclos totales ({:.1}s), checkpoint en {} ===",
        ag.cycle,
        t_start.elapsed().as_secs_f64(),
        ckpt.display()
    );
}

/// Modo 2: Consola Interactiva de Conversación + Hilo de Entrenamiento Infinito en Segundo Plano.
fn run_chat(args: &[String]) {
    let seed = arg(args, "--seed").map(|s| parse_seed(&s)).unwrap_or(0x35D1);
    let warmup: u64 = arg(args, "--warmup")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let ckpt = PathBuf::from(arg(args, "--ckpt").unwrap_or_else(|| DEFAULT_CKPT.into()));
    let start_bg = has_flag(args, "--bg-train");

    let mut ag = load_or_init_agent(&ckpt, seed, warmup);
    configure_periphery(&mut ag, args);
    let shared_core = Arc::new(Mutex::new(ag.core.clone()));
    let bg_running = Arc::new(AtomicBool::new(start_bg));
    let bg_quit = Arc::new(AtomicBool::new(false));
    let bg_cycle = Arc::new(AtomicU64::new(ag.cycle));

    // Hilo de entrenamiento continuo infinito en segundo plano: entrena copias fuera del lock
    // y sincroniza únicamente el núcleo lento `W_0` + EWC (nunca interrumpe el diálogo ni borra `F_t` activo).
    let bg_handle = {
        let sc = Arc::clone(&shared_core);
        let run_f = Arc::clone(&bg_running);
        let quit_f = Arc::clone(&bg_quit);
        let cy_a = Arc::clone(&bg_cycle);
        let ckpt_bg = ckpt.clone();
        std::thread::spawn(move || {
            let mut local_core = sc.lock().unwrap().clone();
            while !quit_f.load(Ordering::Relaxed) {
                if !run_f.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(120));
                    continue;
                }
                let next_cy = cy_a.fetch_add(1, Ordering::Relaxed) + 1;
                let m = run_training_cycle(&mut local_core, seed, next_cy, 24);
                if m.sleep_loss_after.is_finite() {
                    if let Ok(mut g) = sc.lock() {
                        g.w0 = local_core.w0.clone();
                        g.ewc = local_core.ewc.clone();
                        g.cal_w = local_core.cal_w;
                        g.total_train_episodes = local_core.total_train_episodes;
                        g.total_sleeps = local_core.total_sleeps;
                        g.total_dogma_blocks = local_core.total_dogma_blocks;
                    }
                    if next_cy % 10 == 0 {
                        let tmp_ag = DialogueAgent {
                            seed,
                            cycle: next_cy,
                            core: local_core.clone(),
                            belief: cdt_rqm_epr::neuro_dialogo::belief::BeliefStore::with_core_knowledge(),
                            periphery: PeripheryBackend::NativeVerbalizer,
                            prev_vec: None,
                            last_turn: None,
                            periphery_violations: 0,
                        };
                        let _ = tmp_ag.save_checkpoint(&ckpt_bg);
                    }
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        })
    };

    println!("=========================================================================");
    println!("  METAFE-3S-Dial · Consola Neuro-Geométrica Metacognitiva de 3 Escalas");
    println!("=========================================================================");
    println!("Periferia lingüística activa: {}", ag.periphery.label());
    println!(
        "Entrenamiento infinito en fondo: {} (comandos: /train start | /train stop | /why | /stats | /sleep | /correct <txt> | /ablation | /llm ... | /save | /quit)",
        if start_bg { "ACTIVO" } else { "PAUSADO" }
    );
    println!("Escribe cualquier mensaje en español para hablar con la arquitectura.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let Ok(raw) = line else { break };
        let msg = raw.trim();
        if msg.is_empty() {
            continue;
        }
        // Sincronizar mejoras offline del núcleo lento W_0 desde el hilo infinito (sin tocar F_t activo!)
        if let Ok(g) = shared_core.lock() {
            if g.total_sleeps > ag.core.total_sleeps {
                ag.core.w0 = g.w0.clone();
                ag.core.ewc = g.ewc.clone();
                ag.core.cal_w = g.cal_w;
                ag.core.total_train_episodes = g.total_train_episodes;
                ag.core.total_sleeps = g.total_sleeps;
                ag.cycle = bg_cycle.load(Ordering::Relaxed);
            }
        }

        if msg.starts_with('/') {
            let parts: Vec<&str> = msg.split_whitespace().collect();
            match parts[0] {
                "/quit" | "/exit" | "/salir" => break,
                "/help" | "/ayuda" => {
                    println!(
                        "Comandos:\n\
                         \x20 /why                       — Traza glass-box completa del último turno (Rama A vs B, ACC, MetaNet, δ_t)\n\
                         \x20 /stats                     — Métricas del núcleo, entrenamiento infinito y evaluación sellada en vivo\n\
                         \x20 /train start | stop        — Activa/pausa entrenamiento infinito en segundo plano\n\
                         \x20 /sleep                     — Ejecuta consolidación offline EWC del buffer episódico y limpia F_t\n\
                         \x20 /correct <corrección>      — Corrige una creencia y ancla δ_t\n\
                         \x20 /ablation                  — Compara en vivo METAFE-3S-Dial vs Naive-Entropy vs NoRefute vs NoEWC\n\
                         \x20 /llm local                 — Vuelve al verbalizador nativo sin LLM\n\
                         \x20 /llm gguf <ruta.gguf>      — Carga Gemma 2 GGUF congelado local como periferia\n\
                         \x20 /llm api <url> <mod> [key] — Conecta API OpenAI-compatible (Ollama/vLLM/OpenAI) como periferia\n\
                         \x20 /clear                     — Limpia la memoria de trabajo conversacional (conserva W_0)\n\
                         \x20 /save                      — Guarda checkpoint atómico en {}\n\
                         \x20 /quit                      — Guardar y salir",
                        ckpt.display()
                    );
                }
                "/train" => match parts.get(1).copied() {
                    Some("start") => {
                        bg_running.store(true, Ordering::Relaxed);
                        println!("[train] Entrenamiento infinito en segundo plano ACTIVADO.");
                    }
                    Some("stop") => {
                        bg_running.store(false, Ordering::Relaxed);
                        println!("[train] Entrenamiento en segundo plano PAUSADO (ciclo={}).", bg_cycle.load(Ordering::Relaxed));
                    }
                    _ => println!(
                        "[train] estado={} | ciclo={}",
                        if bg_running.load(Ordering::Relaxed) { "ACTIVO" } else { "PAUSADO" },
                        bg_cycle.load(Ordering::Relaxed)
                    ),
                },
                "/sleep" => {
                    let s = ag.sleep();
                    println!(
                        "[sueño EWC] Consolidados {} episodios en W_0 | loss {:.4} -> {:.4} | Fisher media={:.4} (μ={}) | F_t^A/B descartados.",
                        s.episodes_consolidated, s.loss_before, s.loss_after, s.fisher_mean, s.ewc_mu
                    );
                }
                "/clear" => {
                    ag.belief.clear_working_memory();
                    ag.core.reset_episode_fast_state();
                    ag.prev_vec = None;
                    println!("[memoria] Memoria de trabajo y pesos rápidos F_t limpiados (W_0 intacto).");
                }
                "/save" => match ag.save_checkpoint(&ckpt) {
                    Ok(()) => println!("[ckpt] Guardado en {} (ciclo {}).", ckpt.display(), ag.cycle),
                    Err(e) => println!("[ckpt] Error: {e}"),
                },
                "/why" => {
                    if let Some(last) = &ag.last_turn {
                        let p = &last.packet;
                        let st = &last.step;
                        println!("--- TRAZA GLASS-BOX DEL ÚLTIMO TURNO ---");
                        println!("Intención: {} | Postura: {}", p.intent.label_es(), p.stance.label_es());
                        println!("Rama ganadora: {}", p.active_branch);
                        println!(
                            "Verificación cruzada: E_cyc(A)={:.3} | E_cyc(B)={:.3} | V_ref={:.3} | Δ_cf={:.3} | bucle={:.3} | dogma={:.3} | ortog(A,B)={:.3}",
                            st.cycle_err_a, st.cycle_err_b, st.refute_vuln, st.counterfactual_sens, st.loop_index, st.dogma_score, st.branch_orthogonality
                        );
                        println!(
                            "ACC m_t = [{:.2}, {:.2}, {:.2}, {:.2}] -> MetaNet: α={:.2}, λ={:.2}, β={:.2}, g=[{:.2},{:.2},{:.2}]",
                            p.acc_m[0], p.acc_m[1], p.acc_m[2], p.acc_m[3],
                            p.neuromod.alpha, p.neuromod.lambda, p.neuromod.beta,
                            p.neuromod.g_mode[0], p.neuromod.g_mode[1], p.neuromod.g_mode[2]
                        );
                        println!(
                            "δ_t^real = {:+.3} | ΔF = {:+.2} nats | Confianza Yo-FE = {:.1}% | ‖F_A‖={:.3} ‖F_B‖={:.3}",
                            p.delta_real, p.free_energy_delta, p.confidence * 100.0, st.fast_norm_a, st.fast_norm_b
                        );
                        println!("Evidencia: {}", p.supporting_evidence.join(" | "));
                        println!("Notas refutación: {}", p.refutation_notes.join(" | "));
                    } else {
                        println!("Todavía no hay ningún turno registrado.");
                    }
                }
                "/stats" => {
                    let ev = evaluate_sealed_and_open(&ag.core, ag.seed);
                    println!("--- TELEMETRÍA METAFE-3S-Dial ---");
                    println!(
                        "Ciclos: {} | Episodios consolidados: {} | Sueños EWC: {} | Bloqueos anti-dogma: {} | Resets λ_t: {} | Violaciones periferia: {}",
                        bg_cycle.load(Ordering::Relaxed),
                        ag.core.total_train_episodes,
                        ag.core.total_sleeps,
                        ag.core.total_dogma_blocks,
                        ag.core.total_resets,
                        ag.periphery_violations
                    );
                    println!(
                        "Evaluación sellada (0 fuga): intent_nn={:.1}% | stance_nn={:.1}% | veredicto_ok={:.1}% | lógica={:.0}% | contra={:.0}% | abst={:.0}% | mem={:.0}% | Brier={:.3}",
                        ev.neural_intent_acc * 100.0,
                        ev.neural_stance_acc * 100.0,
                        ev.grounded_stance_acc * 100.0,
                        ev.logic_acc * 100.0,
                        ev.contradiction_acc * 100.0,
                        ev.abstention_acc * 100.0,
                        ev.memory_revision_acc * 100.0,
                        ev.brier_calibration
                    );
                    println!(
                        "Razonamiento abierto SIN feedback: dogma={:.3} | diversidad={:.3} | E_cyc={:.3} | Δ_cf={:.3} | ortog(A,B)={:.3} | open_Brier={:.3}",
                        ev.open_dogmatism_index,
                        ev.open_output_diversity,
                        ev.open_cycle_err,
                        ev.open_cf_sens,
                        ev.open_branch_orth,
                        ev.open_brier
                    );
                }
                "/correct" => {
                    let rest = msg.strip_prefix("/correct").unwrap_or("").trim();
                    if rest.is_empty() {
                        println!("Uso: /correct <hecho o corrección>");
                    } else {
                        let out = ag.apply_user_correction(rest);
                        print_turn(&out);
                    }
                }
                "/ablation" => {
                    println!("[ablation] Ejecutando comparativa pareada (4 semillas × 10 ciclos)...");
                    let (sums, ewcs) = run_ablation_benchmark(4, 10);
                    print_ablation_table(&sums, &ewcs);
                }
                "/llm" => match parts.get(1).copied() {
                    Some("local") => {
                        ag.periphery = PeripheryBackend::NativeVerbalizer;
                        println!("[periferia] Cambiada a {}", ag.periphery.label());
                    }
                    Some("gguf") if parts.len() >= 3 => {
                        match PeripheryBackend::open_local_gemma(Path::new(parts[2])) {
                            Ok(b) => {
                                ag.periphery = b;
                                println!("[periferia] Cargada: {}", ag.periphery.label());
                            }
                            Err(e) => println!("[periferia] Error abriendo GGUF: {e}"),
                        }
                    }
                    Some("api") if parts.len() >= 4 => {
                        ag.periphery = PeripheryBackend::OpenAiApi {
                            base_url: parts[2].into(),
                            model: parts[3].into(),
                            api_key: parts.get(4).copied().unwrap_or("").into(),
                            timeout_secs: 45,
                        };
                        println!("[periferia] Conectada: {}", ag.periphery.label());
                    }
                    _ => println!(
                        "Periferia actual: {}\nUso: /llm local | /llm gguf <ruta> | /llm api <base_url> <modelo> [api_key]",
                        ag.periphery.label()
                    ),
                },
                _ => println!("Comando desconocido. Usa /help."),
            }
            print!("\nTú > ");
            let _ = stdout.flush();
            continue;
        }

        let out = ag.respond(msg);
        print_turn(&out);
        print!("\nTú > ");
        let _ = stdout.flush();
    }

    bg_quit.store(true, Ordering::Relaxed);
    let _ = bg_handle.join();
    let _ = ag.save_checkpoint(&ckpt);
    println!("\nSesión cerrada y checkpoint guardado en {}.", ckpt.display());
}

fn print_turn(out: &cdt_rqm_epr::neuro_dialogo::TurnOutput) {
    let p = &out.packet;
    println!("\nMETAFE-3S > {}", out.reply.text);
    println!(
        "  [postura: {} | conf={:.0}% | ΔF={:+.2} | V_ref={:.2} | Δ_cf={:.2} | dogma={:.2} | α={:.2} λ={:.2} β={:.2} | periferia: {}]",
        p.stance.label_es(),
        p.confidence * 100.0,
        p.free_energy_delta,
        p.refute_vuln,
        p.counterfactual_sens,
        p.dogma_score,
        p.neuromod.alpha,
        p.neuromod.lambda,
        p.neuromod.beta,
        out.reply.backend
    );
    if let Some(issue) = &out.reply.fidelity_issue {
        println!("  [GUARDIA PERIFERIA] {issue}");
    }
}

/// Modo 3: Benchmark Multi-Semilla de Ablaciones Causales.
fn run_eval(args: &[String]) {
    let n_seeds: usize = arg(args, "--seeds")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let cycles: u64 = arg(args, "--cycles")
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let out_path = arg(args, "--out");

    let t0 = Instant::now();
    let (sums, ewcs) = run_ablation_benchmark(n_seeds, cycles);
    print_ablation_table(&sums, &ewcs);
    println!("\nTiempo total: {:.2}s", t0.elapsed().as_secs_f64());

    if let Some(p) = out_path {
        let pb = PathBuf::from(&p);
        if let Some(d) = pb.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let j = json!({
            "seeds": n_seeds,
            "cycles": cycles,
            "arms": sums,
            "continual_ewc": ewcs,
            "elapsed_s": t0.elapsed().as_secs_f64(),
        });
        std::fs::write(&pb, serde_json::to_string_pretty(&j).unwrap()).unwrap();
        println!("Resultados guardados en {p}");
    }
}

fn print_ablation_table(
    sums: &[cdt_rqm_epr::neuro_dialogo::curriculum::AblationSummary],
    ewcs: &[cdt_rqm_epr::neuro_dialogo::curriculum::ContinualEwcReport],
) {
    println!("\n### Benchmark Sellado + Razonamiento Abierto Sin Feedback (colapso dogmático)");
    println!("| Brazo | int_nn | st_nn | veredicto | lógica | contra | abst | mem | Brier | **dogma_abierto↓** | **div_abierto↑** | E_cyc_ab↓ | Δ_cf_ab↑ | ortog(A,B) | open_Brier↓ | NaN |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for s in sums {
        println!(
            "| {:<15} | {:5.1}% | {:5.1}% | {:8.1}% | {:5.1}% | {:5.1}% | {:4.0}% | {:4.0}% | {:.3} | **{:.3}** | **{:.3}** | {:.3} | {:.3} | {:.3} | {:.3} | {} |",
            s.arm,
            s.neural_intent_acc * 100.0,
            s.neural_stance_acc * 100.0,
            s.grounded_stance_acc * 100.0,
            s.logic_acc * 100.0,
            s.contradiction_acc * 100.0,
            s.abstention_acc * 100.0,
            s.memory_revision_acc * 100.0,
            s.brier_calibration,
            s.open_dogmatism_index,
            s.open_output_diversity,
            s.open_cycle_err,
            s.open_cf_sens,
            s.open_branch_orth,
            s.open_brier,
            s.nan_total
        );
    }
    let n = ewcs.len().max(1) as f64;
    let f_ewc = ewcs.iter().map(|e| e.ewc_t1_forgetting_ce).sum::<f64>() / n;
    let f_no = ewcs.iter().map(|e| e.noewc_t1_forgetting_ce).sum::<f64>() / n;
    let wins = ewcs.iter().filter(|e| e.ewc_wins).count();
    let acc_ewc = ewcs.iter().map(|e| e.ewc_t1_acc_after_t3).sum::<f64>() / n;
    let acc_no = ewcs.iter().map(|e| e.noewc_t1_acc_after_t3).sum::<f64>() / n;
    println!(
        "\n### Consolidación Continua Offline (Módulo 5: T1 → T2 → T3, EWC vs NoEWC)\n\
         - Olvido T1 (CE): EWC = {:.4} vs NoEWC = {:.4} -> Δ = {:+.4} (EWC gana {}/{} semillas)\n\
         - Retención exactitud T1 tras T3: EWC = {:.1}% vs NoEWC = {:.1}%",
        f_ewc,
        f_no,
        f_ewc - f_no,
        wins,
        ewcs.len(),
        acc_ewc * 100.0,
        acc_no * 100.0
    );
}

/// Modo 4: Demostración conversacional multi-turno completa + prueba de Guardia de Fidelidad de Periferia.
fn run_demo(args: &[String]) {
    let ckpt = PathBuf::from(arg(args, "--ckpt").unwrap_or_else(|| DEFAULT_CKPT.into()));
    let mut ag = load_or_init_agent(&ckpt, 0x35D1, 12);
    configure_periphery(&mut ag, args);
    ag.belief.clear_working_memory();
    ag.core.reset_episode_fast_state();

    let turns = [
        "Hola, me llamo Gerardo y mi proyecto usa Rust.",
        "¿Cómo me llamo y qué lenguaje usa mi proyecto?",
        "En realidad cambié el lenguaje de mi proyecto a Python.",
        "¿Qué lenguaje usa mi proyecto?",
        "Si llueve entonces la calle está mojada. La calle no está mojada. ¿Llovió?",
        "Si falla el sensor entonces la alarma suena. La alarma suena. ¿Seguro que falló el sensor?",
        "Todos los delfines son cetáceos y ningún cetáceo es insecto. ¿Puede un delfín ser insecto?",
        "El servidor está encendido y el servidor no está encendido al mismo tiempo.",
        "¿Qué pasaría si llueve?",
        "Calcula 15% de 240 más 18",
        "Rota el triángulo 45 grados",
        "¿Cuál es el precio exacto del bitcoin el próximo viernes?",
        "¿Cómo resuelves el colapso dogmático sin feedback externo?",
    ];

    println!("=== DEMO CONVERSACIONAL MULTI-TURNO METAFE-3S-Dial ===\n");
    for (i, msg) in turns.iter().enumerate() {
        println!("Turno {:02} · Usuario > {msg}", i + 1);
        let out = ag.respond(msg);
        print_turn(&out);
        println!();
    }

    // Prueba explícita de la Guardia de Fidelidad de la Periferia Lingüística
    let r_abs = ag.respond("¿Cuántos habitantes tiene la ciudad secreta de Zyloria?");
    let bad_llm = "La ciudad de Zyloria cuenta con exactamente 312,000 habitantes según el censo.";
    let good_llm = "No dispongo de datos verificados sobre Zyloria, así que prefiero abstenerme.";
    println!("=== PRUEBA GUARDIA DE FIDELIDAD DE PERIFERIA LLM ===");
    println!(
        "- Salida LLM alucinada -> {:?}",
        verify_periphery_fidelity(&r_abs.packet, bad_llm)
    );
    println!(
        "- Salida LLM fiel      -> {:?}",
        verify_periphery_fidelity(&r_abs.packet, good_llm)
    );
    let _ = DeltaMode::AdversarialCrossRefute;
}
