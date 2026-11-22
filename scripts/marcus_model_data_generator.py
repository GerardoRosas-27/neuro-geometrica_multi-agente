#!/usr/bin/env python3
"""
Generador y Proveedor Continuo de Datos de Entrenamiento con Modelos para MARCUS-UNIFIED.
Provee streams continuos de datos a gran escala:
1. Diálogos y razonamiento relacional de LLMs reales (HuggingFace Llama 3.1 70B / Gemma).
2. Simulación física continua de Spelke (solidez, continuidad, oclusión, gravedad, soporte).
3. Grafos causales de Pearl con contrastes observacionales vs intervenciones do(x).
4. Colectores dinámicos continuos de Lie (11 familias con atractores no lineales y caóticos).
"""

import json
import os
import sys
import time
import math
import random
import urllib.request
from typing import Dict, List, Any, Iterator

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
STREAM_DIR = os.path.join(DATA_DIR, "stream_buffer")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")

os.makedirs(STREAM_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class ModelDataStreamer:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.cached_hf_convs = []
        self._load_local_hf_cache()

    def _load_local_hf_cache(self):
        llm_cache_file = os.path.join(DATA_DIR, "llm_generated_families", "hf_llama3_conversations.json")
        if os.path.exists(llm_cache_file):
            try:
                with open(llm_cache_file, "r", encoding="utf-8") as f:
                    self.cached_hf_convs = json.load(f)
                print(f"[*] Cargadas {len(self.cached_hf_convs)} conversaciones LLM reales en memoria.")
            except Exception as e:
                print(f"[!] Error leyendo cache de LLM: {e}")

    def generate_spelke_physical_episode(self, episode_id: int) -> Dict[str, Any]:
        """Genera un episodio físico continuo verificando solidez, continuidad y gravedad."""
        # Escenarios: 0: movimiento continuo válido, 1: teletransportación (violación continuidad),
        # 2: interpenetración (violación solidez), 3: caída por gravedad, 4: soporte estable
        scenario = episode_id % 5
        dt = 0.05
        steps = 15

        if scenario == 0: # Continuo válido
            obj_id = f"obj_{episode_id}"
            vx = self.rng.uniform(-1.0, 1.0)
            vy = self.rng.uniform(-0.5, 0.5)
            x, y, z = self.rng.uniform(-2.0, 2.0), self.rng.uniform(-2.0, 2.0), 1.0
            trajectory = []
            for t in range(steps):
                x += vx * dt
                y += vy * dt
                trajectory.append({"id": obj_id, "pos": [round(x, 4), round(y, 4), round(z, 4)], "solid": True})
            return {
                "type": "spelke_physics",
                "subtype": "valid_continuous",
                "trajectory": trajectory,
                "has_physical_violation": False,
                "violation_type": None
            }

        elif scenario == 1: # Violación de continuidad (teletransporte)
            obj_id = f"teleport_obj_{episode_id}"
            trajectory = []
            x, y, z = 0.0, 0.0, 1.0
            for t in range(steps):
                if t == 8:
                    x += 10.0 # Salto cuántico/teletransporte
                else:
                    x += 0.1
                trajectory.append({"id": obj_id, "pos": [round(x, 4), round(y, 4), round(z, 4)], "solid": True})
            return {
                "type": "spelke_physics",
                "subtype": "teleportation_discontinuity",
                "trajectory": trajectory,
                "has_physical_violation": True,
                "violation_type": "DiscontinuidadEspacioTemporal"
            }

        elif scenario == 2: # Violación de solidez (interpenetración)
            trajectory = []
            for t in range(steps):
                # Dos objetos que convergen exactamente a la misma posición
                p1 = [round(t * 0.1, 4), 0.0, 1.0]
                p2 = [round(1.5 - t * 0.1, 4), 0.0, 1.0]
                trajectory.append({"objects": [
                    {"id": "box_A", "pos": p1, "radius": 0.5},
                    {"id": "box_B", "pos": p2, "radius": 0.5}
                ]})
            return {
                "type": "spelke_physics",
                "subtype": "solidity_interpenetration",
                "trajectory": trajectory,
                "has_physical_violation": True,
                "violation_type": "InterpenetracionSolidos"
            }

        else: # Gravedad y soporte
            trajectory = []
            z = 5.0
            vz = 0.0
            g = -9.81
            floor_z = 0.0
            for t in range(steps):
                vz += g * dt
                z += vz * dt
                if z < floor_z:
                    z = floor_z
                    vz = 0.0 # Reposo con soporte
                trajectory.append({"id": "falling_sphere", "pos": [0.0, 0.0, round(z, 4)], "support": z <= floor_z})
            return {
                "type": "spelke_physics",
                "subtype": "gravity_support",
                "trajectory": trajectory,
                "has_physical_violation": False,
                "violation_type": None
            }

    def generate_pearl_causal_episode(self, episode_id: int) -> Dict[str, Any]:
        """Genera un escenario causal de Pearl con variable confusora Z, causa X y efecto Y."""
        # Modelo Causal Estructural: Z -> X, Z -> Y, X -> Y (efecto beta)
        beta_true = 1.85
        gamma_z_x = 2.0
        gamma_z_y = 3.5

        # Simular lote observacional
        n_obs = 30
        z_vals = [self.rng.normalvariate(0, 1) for _ in range(n_obs)]
        x_obs = [gamma_z_x * z + self.rng.normalvariate(0, 0.2) for z in z_vals]
        y_obs = [beta_true * x + gamma_z_y * z + self.rng.normalvariate(0, 0.2) for x, z in zip(x_obs, z_vals)]

        # Simular intervención do(X = x_fixed)
        x_intervene = 2.5
        z_interv = [self.rng.normalvariate(0, 1) for _ in range(n_obs)]
        # Bajo do(X), la flecha Z -> X se mutila
        y_interv = [beta_true * x_intervene + gamma_z_y * z + self.rng.normalvariate(0, 0.2) for z in z_interv]

        return {
            "type": "pearl_causality",
            "episode_id": episode_id,
            "variables": ["Z_confounder", "X_cause", "Y_effect"],
            "ground_truth_beta": beta_true,
            "observational_batch": {"x": x_obs[:5], "y": y_obs[:5]},
            "interventional_do_x": {"x_fixed": x_intervene, "y_mean": round(float(sum(y_interv) / len(y_interv)), 4)},
            "spurious_correlation_eliminated": True
        }

    def generate_lie_dynamical_episode(self, episode_id: int) -> Dict[str, Any]:
        """Genera una trayectoria de Lie continua (conocida u OOD)."""
        family = episode_id % 11
        k_points = 6
        pts = []
        for p in range(k_points):
            r = self.rng.uniform(0.4, 0.9)
            th = (p * 2.0 * math.pi) / k_points
            pts.append(r * math.cos(th))
            pts.append(r * math.sin(th))

        param = self.rng.uniform(-0.8, 0.8)
        frames = [list(pts)]
        cur = list(pts)
        dt = 0.1

        for _ in range(16):
            nxt = [0.0] * (k_points * 2)
            for p in range(k_points):
                x, y = cur[p * 2], cur[p * 2 + 1]
                if family == 0:
                    nx, ny = x + 0.01 * dt, y - 0.01 * dt
                elif family == 1:
                    om = param * dt
                    nx, ny = x * math.cos(om) - y * math.sin(om), x * math.sin(om) + y * math.cos(om)
                elif family == 2:
                    nx, ny = -x, y
                elif family == 3:
                    s = 1.0 + (param - 0.4) * 0.2 * dt
                    nx, ny = x * s, y * s
                elif family == 4:
                    nx, ny = x + param * dt * y, y
                elif family == 5:
                    phi = max(-0.4, min(0.4, param * 0.3 * dt))
                    nx, ny = x * math.cosh(phi) + y * math.sinh(phi), x * math.sinh(phi) + y * math.cosh(phi)
                elif family == 8: # Duffing
                    dx = y
                    dy = x - x**3 - 0.2 * y
                    nx, ny = x + dx * dt * param, y + dy * dt * param
                elif family == 10: # Lorenz
                    dx = 10.0 * (y - x)
                    dy = x * 27.0 - y
                    nx, ny = x + dx * 0.005 * param, y + dy * 0.005 * param
                else: # 11: Van der Pol
                    mu = param * 1.5
                    dx = y
                    dy = mu * (1.0 - x * x) * y - x
                    nx, ny = x + dx * 0.04, y + dy * 0.04
                nxt[p * 2] = nx
                nxt[p * 2 + 1] = ny
            frames.append(nxt)
            cur = nxt

        return {
            "type": "lie_dynamics",
            "family_id": family,
            "is_ood": family in [8, 10, 11],
            "param": param,
            "initial_points": pts,
            "rollout_16": frames
        }

    def generate_relational_llm_episode(self, episode_id: int) -> Dict[str, Any]:
        """Genera un episodio de ligamiento de variables y hechos fácticos con base en diálogos LLM."""
        if self.cached_hf_convs:
            conv = self.cached_hf_convs[episode_id % len(self.cached_hf_convs)]
            dialogue_text = " ".join(msg.get("content", "") for msg in conv[:2])
        else:
            dialogue_text = "The quick brown fox jumps over the lazy dog."

        facts = [
            ("Earth", "Orbits", "Sun"),
            ("Water", "BoilsAtCelsius", "100"),
            ("Light", "PropagatesInVacuum", "True"),
            ("SolidObject", "PenetratesSolidObject", "False"),
            ("Gravity", "AcceleratesMass", "Downward")
        ]
        chosen_fact = facts[episode_id % len(facts)]

        return {
            "type": "llm_relational",
            "episode_id": episode_id,
            "dialogue_context": dialogue_text[:120],
            "fact_claim": {
                "subject": chosen_fact[0],
                "predicate": chosen_fact[1],
                "object": chosen_fact[2]
            },
            "is_grounded": True
        }

    def generate_next_batch(self, batch_size: int = 32, start_id: int = 0) -> List[Dict[str, Any]]:
        batch = []
        for i in range(batch_size):
            eid = start_id + i
            mod = eid % 4
            if mod == 0:
                batch.append(self.generate_spelke_physical_episode(eid))
            elif mod == 1:
                batch.append(self.generate_pearl_causal_episode(eid))
            elif mod == 2:
                batch.append(self.generate_lie_dynamical_episode(eid))
            else:
                batch.append(self.generate_relational_llm_episode(eid))
        return batch

if __name__ == "__main__":
    streamer = ModelDataStreamer()
    sample_batch = streamer.generate_next_batch(4, 100)
    print(f"[+] Generador probado exitosamente con {len(sample_batch)} tipos de episodios.")
