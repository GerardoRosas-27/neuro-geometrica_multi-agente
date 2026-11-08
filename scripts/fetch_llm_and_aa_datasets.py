#!/usr/bin/env python3
"""
Descargador y Generador de Datos Reales Generados por LLM + Benchmark Artificial Analysis.
1. Descarga el dataset oficial de evaluación de Artificial Analysis:
   `ArtificialAnalysis/AA-Omniscience-Public-v1.1` (300 preguntas expertas de ciencia, razonamiento, biología y física).
2. Descarga conversaciones reales de LLM (Llama 3.1 70B/8B de `HuggingFaceTB/everyday-conversations-llama3.1-2k`).
3. Genera el dataset multimodal emparejando prompts lingüísticos multilingües (ES, EN, FR, PT, IT)
   con las 11 familias dinámicas y algebraicas (0..7 conocidas, 8, 10, 11 OOD como Duffing, Lorenz y Van der Pol).
"""

import json
import os
import sys
import urllib.request
import math
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
AA_DIR = os.path.join(DATA_DIR, "artificial_analysis")
LLM_FAM_DIR = os.path.join(DATA_DIR, "llm_generated_families")

os.makedirs(AA_DIR, exist_ok=True)
os.makedirs(LLM_FAM_DIR, exist_ok=True)

def fetch_aa_omniscience():
    out_file = os.path.join(AA_DIR, "aa_omniscience_300.json")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        print(f"[*] Benchmark Artificial Analysis ya existe en {out_file}")
        with open(out_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("[*] Descargando benchmark oficial de https://artificialanalysis.ai/ (ArtificialAnalysis/AA-Omniscience-Public-v1.1)...")
    all_rows = []
    for offset in range(0, 300, 100):
        url = f"https://datasets-server.huggingface.co/rows?dataset=ArtificialAnalysis%2FAA-Omniscience-Public-v1.1&config=default&split=test&offset={offset}&limit=100"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            res = urllib.request.urlopen(req, timeout=30)
            data = json.loads(res.read().decode("utf-8"))
            rows = [r["row"] for r in data.get("rows", [])]
            all_rows.extend(rows)
            print(f"    - Descargadas {len(rows)} preguntas (offset {offset})")
        except Exception as e:
            print(f"    [!] Error al descargar offset {offset}: {e}")
            break

    if len(all_rows) > 0:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        print(f"[+] Total de preguntas de Artificial Analysis guardadas: {len(all_rows)} en {out_file}")
    return all_rows

def fetch_llm_conversations():
    out_file = os.path.join(LLM_FAM_DIR, "hf_llama3_conversations.json")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        print(f"[*] Conversaciones LLM ya existen en {out_file}")
        with open(out_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("[*] Descargando conversaciones reales de LLM (HuggingFaceTB/everyday-conversations-llama3.1-2k)...")
    all_convs = []
    for offset in range(0, 200, 100):
        url = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceTB%2Feveryday-conversations-llama3.1-2k&config=default&split=train_sft&offset={offset}&limit=100"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            res = urllib.request.urlopen(req, timeout=30)
            data = json.loads(res.read().decode("utf-8"))
            rows = [r["row"]["messages"] for r in data.get("rows", [])]
            all_convs.extend(rows)
            print(f"    - Descargadas {len(rows)} conversaciones LLM (offset {offset})")
        except Exception as e:
            print(f"    [!] Error al descargar offset {offset}: {e}")
            break

    if len(all_convs) > 0:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_convs, f, indent=2, ensure_ascii=False)
        print(f"[+] Total conversaciones LLM guardadas: {len(all_convs)} en {out_file}")
    return all_convs

def generate_llm_dynamical_family_dataset():
    """Genera un corpus exhaustivo de familias dinámicas reales generadas con contexto de LLM multilingüe."""
    out_file = os.path.join(LLM_FAM_DIR, "llm_multilingual_dynamical_dataset.json")
    
    # 11 Familias Geométricas y Dinámicas Formales
    families_meta = {
        0: {"name": "traslación uniforme", "is_ood": False, "category": "afín"},
        1: {"name": "rotación SO(2)", "is_ood": False, "category": "euclidiana"},
        2: {"name": "reflexión de paridad", "is_ood": False, "category": "simetría discreta"},
        3: {"name": "escalado homotético", "is_ood": False, "category": "conforme"},
        4: {"name": "cizalla dinámica (shear)", "is_ood": False, "category": "afín no ortogonal"},
        5: {"name": "boost de Lorentz hiperbólico", "is_ood": False, "category": "pseudo-euclidiana"},
        6: {"name": "espiral logarítmica Wheeler-Feynman", "is_ood": False, "category": "holomorfa"},
        7: {"name": "atractor en cuenca CDT", "is_ood": False, "category": "topológica"},
        8: {"name": "oscilador no lineal de Duffing", "is_ood": True, "category": "no lineal OOD"},
        10: {"name": "atractor caótico de Lorenz", "is_ood": True, "category": "caótica OOD"},
        11: {"name": "ciclo límite de Van der Pol", "is_ood": True, "category": "relajación OOD"}
    }

    prompt_templates = {
        "es": [
            "Aplica una {fam_name} con parámetro inicial {p1:.3f} y variación temporal {p2:.3f}.",
            "Simula la evolución de un sistema bajo {fam_name} considerando {p1:.3f} como constante.",
            "Calcula la trayectoria del colector gobernada por {fam_name} con parámetros [{p1:.3f}, {p2:.3f}]."
        ],
        "en": [
            "Apply a {fam_name} dynamic with parameter {p1:.3f} and rate {p2:.3f}.",
            "Simulate state trajectory governed by {fam_name} using constant {p1:.3f}.",
            "Compute the symplectic flow under {fam_name} with configuration [{p1:.3f}, {p2:.3f}]."
        ],
        "fr": [
            "Appliquez une dynamique de {fam_name} avec paramètre {p1:.3f} et taux {p2:.3f}.",
            "Simulez la trajectoire spatio-temporelle sous {fam_name} avec constante {p1:.3f}."
        ],
        "pt": [
            "Aplique uma dinâmica de {fam_name} com parâmetro {p1:.3f} e taxa {p2:.3f}.",
            "Simule a evolução do sistema geométrico sob {fam_name} com [{p1:.3f}, {p2:.3f}]."
        ],
        "it": [
            "Applica una dinamica di {fam_name} con parametro {p1:.3f} e tasso {p2:.3f}.",
            "Calcola la traiettoria geometrica secondo {fam_name} con parametri [{p1:.3f}, {p2:.3f}]."
        ]
    }

    def apply_dynamics(fam: int, x: float, y: float, param: float, dt: float = 0.1):
        if fam == 0:
            return (x + 0.01 * dt, y - 0.01 * dt)
        elif fam == 1:
            omega = param * dt
            return (x * math.cos(omega) - y * math.sin(omega), x * math.sin(omega) + y * math.cos(omega))
        elif fam == 2:
            return (-x, y)
        elif fam == 3:
            s = 1.0 + (param - 0.4) * 0.2 * dt
            return (x * s, y * s)
        elif fam == 4:
            return (x + param * dt * y, y)
        elif fam == 5:
            phi = max(-0.4, min(0.4, param * 0.3 * dt))
            return (x * math.cosh(phi) + y * math.sinh(phi), x * math.sinh(phi) + y * math.cosh(phi))
        elif fam == 6:
            omega = param * dt
            s = 1.0 - 0.05 * dt
            return (s * (x * math.cos(omega) - y * math.sin(omega)), s * (x * math.sin(omega) + y * math.cos(omega)))
        elif fam == 7:
            r2 = x * x + y * y
            dr = 0.5 * (1.0 - r2) * dt
            omega = (param + 0.5) * dt
            return ((x + dr * x) * math.cos(omega) - (y + dr * y) * math.sin(omega),
                    (x + dr * x) * math.sin(omega) + (y + dr * y) * math.cos(omega))
        elif fam == 8: # Duffing
            dx = y
            dy = x - x * x * x - 0.2 * y
            return (x + dx * dt * param, y + dy * dt * param)
        elif fam == 10: # Lorenz
            dx = 10.0 * (y - x)
            dy = x * 27.0 - y
            return (x + dx * 0.005 * param, y + dy * 0.005 * param)
        else: # 11: Van der Pol
            mu = param * 1.5
            dx = y
            dy = mu * (1.0 - x * x) * y - x
            return (x + dx * 0.04, y + dy * 0.04)

    rng = random.Random(42)
    dataset = []
    sample_id = 0

    # Generar 500 muestras estratificadas por familia e idioma
    for fam_id, meta in families_meta.items():
        for lang, templates in prompt_templates.items():
            for rep in range(10): # 11 familias * 5 idiomas * 10 = 550 episodios
                sample_id += 1
                p1 = rng.uniform(-1.0, 1.0)
                p2 = rng.uniform(-0.5, 0.5)
                template = rng.choice(templates)
                prompt = template.format(fam_name=meta["name"], p1=p1, p2=p2)

                # Generar constelación de 6 puntos
                k_points = 6
                init_points = []
                for p in range(k_points):
                    r = rng.uniform(0.4, 0.9)
                    theta = (p * 2.0 * math.pi) / k_points + rng.uniform(-0.08, 0.08)
                    init_points.append(r * math.cos(theta))
                    init_points.append(r * math.sin(theta))

                # Generar rollout de 20 pasos (4 observación + 16 horizonte)
                frames = [list(init_points)]
                cur = list(init_points)
                for _ in range(19):
                    nxt = [0.0] * (k_points * 2)
                    for p in range(k_points):
                        nx, ny = apply_dynamics(fam_id, cur[p * 2], cur[p * 2 + 1], p1)
                        nxt[p * 2] = nx
                        nxt[p * 2 + 1] = ny
                    frames.append(nxt)
                    cur = nxt

                dataset.append({
                    "id": sample_id,
                    "family_id": fam_id,
                    "family_name": meta["name"],
                    "is_ood": meta["is_ood"],
                    "category": meta["category"],
                    "language": lang,
                    "llm_prompt": prompt,
                    "params": [p1, p2],
                    "trajectory_20_frames": frames
                })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"[+] Dataset LLM Multimodal generado: {len(dataset)} muestras en {out_file}")
    return dataset

if __name__ == "__main__":
    print("=== Preparando Datos Reales de LLM y Benchmark Artificial Analysis ===")
    fetch_aa_omniscience()
    fetch_llm_conversations()
    generate_llm_dynamical_family_dataset()
    print("=== Preparación Completa ===")
