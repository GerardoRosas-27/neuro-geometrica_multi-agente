#!/usr/bin/env python3
"""Plantilla Kaggle: BQ10 (ATLAS-FW CORE + M7) sobre CONSTELACIÓN v4, CPU/Rust, reanudable.
Ver docs/kaggle/PLAN_KAGGLE.md. NO contiene credenciales: el token GitHub (opcional) se lee de Kaggle Secrets.
Variables (env o constantes abajo): PHASE=paso0|pilot|bq_tune|bq_dev|bq_confirm, ARMS, SCALE (perfil), JOBS, BUDGET_H.
"""
import os, subprocess, time, json, shutil, glob, sys
W = "/kaggle/working"; SRC = "/kaggle/tmp/ngm"; CK = f"{W}/ckpt"
PHASE = os.environ.get("PHASE", "paso0")
ARMS = os.environ.get("ARMS", "v4,ALL,CORE,BQ10,BQR10")
SCALE = os.environ.get("SCALE", "base")          # base = v4 congelada; s1/s2/s3 = perfiles del plan (requieren rama con flags)
JOBS = int(os.environ.get("JOBS", str(os.cpu_count() or 4)))
BUDGET_H = float(os.environ.get("BUDGET_H", "11.0"))  # margen bajo el límite de 12 h por sesión
REF = os.environ.get("REF", "main")
T0 = time.time()

def sh(cmd, **kw):
    print("+", cmd, flush=True); return subprocess.run(cmd, shell=True, check=True, **kw)

# 1) restaurar checkpoint desde dataset adjunto (salida previa) si existe
os.makedirs(CK, exist_ok=True)
for d in glob.glob("/kaggle/input/*/ckpt"):
    sh(f"cp -rn {d}/. {CK}/")
# 2) setup (rust + build), código de main
sh(f"REF={REF} bash {os.path.dirname(os.path.abspath(__file__))}/setup.sh" if os.path.exists("setup.sh") or os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup.sh"))
   else f"curl -sSfL https://raw.githubusercontent.com/GerardoRosas-27/neuro-geometrica_multi-agente/{REF}/kaggle/setup.sh | REF={REF} bash")
os.chdir(SRC)
R = f"{CK}/artifacts/bio_quantum_v10"
os.makedirs(R, exist_ok=True)
# los scripts escriben en artifacts/bio_quantum_v10 → enlazarlo a /kaggle/working para que persista
shutil.rmtree("artifacts/bio_quantum_v10", ignore_errors=True)
os.symlink(R, "artifacts/bio_quantum_v10")

# 3) Paso 0: hash de referencia (obligatorio antes de cualquier fase)
ref = "1985bdbad5eee41f"
if not os.path.exists(f"{CK}/paso0.ok"):
    out = subprocess.run("orbita/target/release/orbita cst4-run --phase smoke --seed 0xC420 --arms cst --out /kaggle/tmp/p0 "
                         "--weights artifacts/orbita/dev/weights_0xA400.json", shell=True, capture_output=True, text=True)
    print(out.stdout[-2000:], out.stderr[-2000:])
    if ref not in out.stdout + out.stderr + "".join(open(f).read() for f in glob.glob("/kaggle/tmp/p0/**/*.json", recursive=True)):
        sys.exit("Paso 0 FALLA: hash != 1985bdbad5eee41f (plataforma CPU distinta → no comparable; NO seguir)")
    open(f"{CK}/paso0.ok", "w").write(ref)
if PHASE == "paso0": sys.exit(0)

# 4) fase: el script de main ya es reanudable (salta semillas con .done)
env = dict(os.environ, JOBS=str(JOBS), BQ_SCALE=SCALE)
p = subprocess.Popen(f"bash orbita/scripts/bq10_phase.sh {PHASE} {ARMS}", shell=True, env=env)
while p.poll() is None:
    time.sleep(60)
    if time.time() - T0 > BUDGET_H * 3600:
        p.terminate(); print("presupuesto agotado: se reanudará en la siguiente sesión"); break
done = len(glob.glob(f"{R}/{PHASE}/*.done")); print(f"{PHASE}: {done} semillas hechas")
json.dump({"phase": PHASE, "done": done, "scale": SCALE, "elapsed_h": (time.time()-T0)/3600,
           "sha": open(f"{CK}/git_sha.txt").read().strip()}, open(f"{CK}/status_{PHASE}.json", "w"))

# 5) opcional: push de resultados a rama git con token de Kaggle Secrets (nunca en código)
if os.environ.get("PUSH_BRANCH"):
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
    br = os.environ["PUSH_BRANCH"]
    sh(f"git checkout -B {br} && rm artifacts/bio_quantum_v10 && cp -r {R} artifacts/bio_quantum_v10 && git add artifacts/bio_quantum_v10 && "
       f"git -c user.name=kaggle -c user.email=kaggle@local commit -qm 'kaggle {PHASE} {SCALE}: {done} semillas' || true")
    subprocess.run(f"git push https://x-access-token:{tok}@github.com/GerardoRosas-27/neuro-geometrica_multi-agente.git {br}",
                   shell=True, check=True, capture_output=True)  # capture_output: no imprimir el token
