//! v4 E43/E44 — linguistic transfer and LLM swap (plan §17–§18,
//! preregistro §11). Frozen LLM periphery → PCA (fit on Spanish TRAIN only)
//! → Dφ (residual MLP on the whole vector, context = relation one-hot)
//! → closed-vocabulary cosine decoder (candidates = all word forms of the
//! evaluated language; reported as `decoder_lookup`, never as clean memory).

#![allow(clippy::needless_range_loop)]

use crate::v4_controls::{Adam, Mlp};
use crate::v4_metrics as m;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::HashMap;

pub const LANGS: [&str; 4] = ["es", "en", "fr", "ja"];
/// relation 0 = plural, 1 = feminine
pub const RELS: [&str; 2] = ["plural", "feminine"];

/// (concept, es, en, fr, ja) pairs `src|dst`.
pub const PLURAL: [(&str, [&str; 4]); 16] = [
    ("dog", ["perro|perros", "dog|dogs", "chien|chiens", ""]),
    ("cat", ["gato|gatos", "cat|cats", "chat|chats", ""]),
    (
        "house",
        ["casa|casas", "house|houses", "maison|maisons", "家|家々"],
    ),
    ("book", ["libro|libros", "book|books", "livre|livres", ""]),
    (
        "tree",
        ["árbol|árboles", "tree|trees", "arbre|arbres", "木|木々"],
    ),
    ("car", ["coche|coches", "car|cars", "voiture|voitures", ""]),
    (
        "flower",
        ["flor|flores", "flower|flowers", "fleur|fleurs", "花|花々"],
    ),
    (
        "bird",
        ["pájaro|pájaros", "bird|birds", "oiseau|oiseaux", ""],
    ),
    ("table", ["mesa|mesas", "table|tables", "table|tables", ""]),
    (
        "city",
        ["ciudad|ciudades", "city|cities", "ville|villes", ""],
    ),
    ("door", ["puerta|puertas", "door|doors", "porte|portes", ""]),
    (
        "window",
        ["ventana|ventanas", "window|windows", "fenêtre|fenêtres", ""],
    ),
    (
        "mountain",
        [
            "montaña|montañas",
            "mountain|mountains",
            "montagne|montagnes",
            "山|山々",
        ],
    ),
    (
        "star",
        [
            "estrella|estrellas",
            "star|stars",
            "étoile|étoiles",
            "星|星々",
        ],
    ),
    (
        "person",
        [
            "persona|personas",
            "person|people",
            "personne|personnes",
            "人|人々",
        ],
    ),
    (
        "country",
        ["país|países", "country|countries", "pays|pays", "国|国々"],
    ),
];

pub const FEMININE: [(&str, [&str; 4]); 14] = [
    (
        "brother",
        ["hermano|hermana", "brother|sister", "frère|sœur", "兄|姉"],
    ),
    (
        "son",
        ["hijo|hija", "son|daughter", "fils|fille", "息子|娘"],
    ),
    (
        "uncle",
        ["tío|tía", "uncle|aunt", "oncle|tante", "叔父|叔母"],
    ),
    (
        "grandfather",
        [
            "abuelo|abuela",
            "grandfather|grandmother",
            "grand-père|grand-mère",
            "祖父|祖母",
        ],
    ),
    ("king", ["rey|reina", "king|queen", "roi|reine", "王|女王"]),
    ("man", ["hombre|mujer", "man|woman", "homme|femme", "男|女"]),
    (
        "boy",
        ["niño|niña", "boy|girl", "garçon|fillette", "少年|少女"],
    ),
    (
        "actor",
        [
            "actor|actriz",
            "actor|actress",
            "acteur|actrice",
            "俳優|女優",
        ],
    ),
    (
        "waiter",
        [
            "camarero|camarera",
            "waiter|waitress",
            "serveur|serveuse",
            "ウェイター|ウェイトレス",
        ],
    ),
    (
        "prince",
        [
            "príncipe|princesa",
            "prince|princess",
            "prince|princesse",
            "王子|王女",
        ],
    ),
    (
        "husband",
        ["esposo|esposa", "husband|wife", "mari|épouse", "夫|妻"],
    ),
    (
        "father",
        ["padre|madre", "father|mother", "père|mère", "父|母"],
    ),
    (
        "nephew",
        ["sobrino|sobrina", "nephew|niece", "neveu|nièce", "甥|姪"],
    ),
    (
        "lion",
        [
            "león|leona",
            "lion|lioness",
            "lion|lionne",
            "雄ライオン|雌ライオン",
        ],
    ),
];

/// Anchor vocabulary for the E44 alignment (disjoint from task words).
pub const ANCHORS: [&str; 96] = [
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
    "diez",
    "rojo",
    "azul",
    "verde",
    "amarillo",
    "negro",
    "blanco",
    "gris",
    "naranja",
    "comer",
    "beber",
    "correr",
    "saltar",
    "dormir",
    "hablar",
    "escribir",
    "leer",
    "cantar",
    "bailar",
    "nadar",
    "volar",
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "agua",
    "fuego",
    "tierra",
    "aire",
    "sol",
    "luna",
    "mar",
    "playa",
    "nieve",
    "lluvia",
    "viento",
    "pan",
    "leche",
    "queso",
    "carne",
    "arroz",
    "sal",
    "azúcar",
    "café",
    "té",
    "vino",
    "grande",
    "pequeño",
    "rápido",
    "lento",
    "alto",
    "bajo",
    "nuevo",
    "viejo",
    "bueno",
    "malo",
    "feliz",
    "triste",
    "caliente",
    "frío",
    "hoy",
    "mañana",
    "ayer",
    "siempre",
    "nunca",
    "aquí",
    "allí",
    "dinero",
    "trabajo",
    "escuela",
    "música",
    "tiempo",
    "mundo",
    "vida",
    "amor",
    "guerra",
];

#[derive(Clone, Debug)]
pub struct Pair {
    pub concept: &'static str,
    pub rel: usize,
    pub lang: usize,
    pub src: String,
    pub dst: String,
}

pub fn all_pairs() -> Vec<Pair> {
    let mut v = vec![];
    for (rel, table) in [(0usize, &PLURAL[..]), (1, &FEMININE[..])] {
        for (c, forms) in table {
            for (l, f) in forms.iter().enumerate() {
                if let Some((a, b)) = f.split_once('|') {
                    if a != b {
                        v.push(Pair {
                            concept: c,
                            rel,
                            lang: l,
                            src: a.into(),
                            dst: b.into(),
                        });
                    }
                }
            }
        }
    }
    v
}

pub fn all_texts() -> Vec<String> {
    let mut t: Vec<String> = all_pairs()
        .iter()
        .flat_map(|p| [p.src.clone(), p.dst.clone()])
        .collect();
    t.extend(ANCHORS.iter().map(|s| s.to_string()));
    t.sort();
    t.dedup();
    t
}

/// Deterministic frozen random encoder (char-trigram hashing → Gaussian projection).
pub fn random_encoder(text: &str, dim: usize) -> Vec<f64> {
    let chars: Vec<char> = format!("^{text}$").chars().collect();
    let mut v = vec![0.0; dim];
    for w in chars.windows(3) {
        let mut h: u64 = 0xcbf29ce484222325;
        for c in w {
            h ^= *c as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let mut r = Xoshiro256StarStar::seed_from_u64(h);
        for x in v.iter_mut() {
            *x += r.gen_range(-1.0..1.0);
        }
    }
    v
}

pub type Emb = HashMap<String, Vec<f64>>;

pub fn load_emb(path: &str) -> Option<Emb> {
    let t = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&t).ok()
}

/// PCA by power iteration with deflation (top-d components), centred.
pub struct Pca {
    pub mean: Vec<f64>,
    pub comps: Vec<Vec<f64>>,
}

impl Pca {
    pub fn fit(xs: &[Vec<f64>], d: usize) -> Self {
        let n = xs[0].len();
        let mean: Vec<f64> = (0..n)
            .map(|j| xs.iter().map(|x| x[j]).sum::<f64>() / xs.len() as f64)
            .collect();
        let c: Vec<Vec<f64>> = xs
            .iter()
            .map(|x| x.iter().zip(&mean).map(|(a, b)| a - b).collect())
            .collect();
        let mut comps: Vec<Vec<f64>> = vec![];
        for k in 0..d.min(xs.len() - 1) {
            let mut v: Vec<f64> = (0..n)
                .map(|j| ((j * 7 + k * 13) % 17) as f64 - 8.0)
                .collect();
            for _ in 0..80 {
                // w = Cᵀ C v
                let cv: Vec<f64> = c
                    .iter()
                    .map(|r| r.iter().zip(&v).map(|(a, b)| a * b).sum())
                    .collect();
                let mut w = vec![0.0; n];
                for (r, s) in c.iter().zip(&cv) {
                    for j in 0..n {
                        w[j] += r[j] * s;
                    }
                }
                for p in &comps {
                    let d: f64 = w.iter().zip(p).map(|(a, b)| a * b).sum();
                    for j in 0..n {
                        w[j] -= d * p[j];
                    }
                }
                let nn = m::norm(&w).max(1e-12);
                v = w.iter().map(|x| x / nn).collect();
            }
            comps.push(v);
        }
        Self { mean, comps }
    }

    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        self.comps
            .iter()
            .map(|p| {
                p.iter()
                    .zip(x)
                    .zip(&self.mean)
                    .map(|((a, b), c)| a * (b - c))
                    .sum()
            })
            .collect()
    }
}

/// Whole-vector residual Dφ in PCA space, context = relation one-hot.
pub struct LangDphi {
    pub net: Mlp,
    pub d: usize,
}

impl LangDphi {
    pub fn new(d: usize, seed: u64) -> Self {
        Self {
            net: Mlp::new(d + 2, 64, d, 0.1, seed),
            d,
        }
    }
    fn inp(z: &[f64], rel: usize) -> Vec<f64> {
        let mut v = z.to_vec();
        v.push((rel == 0) as u8 as f64);
        v.push((rel == 1) as u8 as f64);
        v
    }
    pub fn step(&self, z: &[f64], rel: usize) -> Vec<f64> {
        let (_, o) = self.net.forward(&Self::inp(z, rel));
        z.iter().zip(&o).map(|(a, b)| a + b).collect()
    }
    pub fn train(&mut self, data: &[(Vec<f64>, usize, Vec<f64>)], steps: usize, seed: u64) {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let n = self.net.n_params();
        let mut opt = Adam::new(n, 3e-3);
        let mut p = self.net.flat();
        for _ in 0..steps {
            let mut g = vec![0.0; n];
            for _ in 0..8 {
                let (x, r, y) = &data[rng.gen_range(0..data.len())];
                let inp = Self::inp(x, *r);
                let (h, o) = self.net.forward(&inp);
                let d: Vec<f64> = (0..self.d)
                    .map(|i| 2.0 * (x[i] + o[i] - y[i]) / (8 * self.d) as f64)
                    .collect();
                self.net.backward(&inp, &h, &d, &mut g);
            }
            // weight decay (small data)
            for i in 0..n {
                g[i] += 1e-4 * p[i];
            }
            opt.step(&mut p, &g);
            self.net.set_flat(&p);
        }
    }
}

pub struct LangOut {
    pub rows: Vec<String>, // csv: seed,exp,encoder,lang,subset,method,n,acc
    pub e43_pass: bool,
    pub e44_pass: bool,
    pub note43: String,
    pub note44: String,
}

fn nearest(z: &[f64], cands: &[(String, Vec<f64>)]) -> String {
    cands
        .iter()
        .map(|(w, v)| (w, m::cosine(z, v)))
        .fold(
            (&cands[0].0, f64::MIN),
            |a, b| if b.1 > a.1 { b } else { a },
        )
        .0
        .clone()
}

/// Ridge map from raw encoder space to the A-PCA space, fit on anchors.
fn anchor_map(src: &Emb, pca_a: &Pca, emb_a: &Emb, scale: f64) -> impl Fn(&[f64]) -> Vec<f64> {
    let xs: Vec<Vec<f64>> = ANCHORS.iter().map(|w| src[*w].clone()).collect();
    let n = xs[0].len();
    let mu: Vec<f64> = (0..n)
        .map(|j| xs.iter().map(|x| x[j]).sum::<f64>() / xs.len() as f64)
        .collect();
    let xc: Vec<Vec<f64>> = xs
        .iter()
        .map(|x| x.iter().zip(&mu).map(|(a, b)| a - b).collect())
        .collect();
    let ys: Vec<Vec<f64>> = ANCHORS
        .iter()
        .map(|w| {
            pca_a
                .project(&emb_a[*w])
                .iter()
                .map(|v| v / scale)
                .collect()
        })
        .collect();
    // dual ridge: alpha = (K + λI)^-1 Y, pred = k(x)ᵀ alpha
    let k: Vec<Vec<f64>> = xc
        .iter()
        .map(|a| {
            xc.iter()
                .map(|b| a.iter().zip(b).map(|(p, q)| p * q).sum())
                .collect()
        })
        .collect();
    let tr: f64 = (0..k.len()).map(|i| k[i][i]).sum::<f64>() / k.len() as f64;
    let lam = 1e-3 * tr;
    let kk: Vec<Vec<f64>> = (0..k.len())
        .map(|i| {
            (0..k.len())
                .map(|j| k[i][j] + if i == j { lam } else { 0.0 })
                .collect()
        })
        .collect();
    // solve with identity features: ridge(features=K+λI rows, ys, 0) == solve
    let alpha = crate::v4_controls::ridge(&kk, &ys, 1e-12);
    move |x: &[f64]| {
        let xcx: Vec<f64> = x.iter().zip(&mu).map(|(a, b)| a - b).collect();
        let kx: Vec<f64> = xc
            .iter()
            .map(|b| xcx.iter().zip(b).map(|(p, q)| p * q).sum())
            .collect();
        (0..alpha[0].len())
            .map(|d| kx.iter().zip(&alpha).map(|(a, r)| a * r[d]).sum())
            .collect()
    }
}

pub fn run_seed(seed: u64, ga: &Emb, gb: &Emb) -> LangOut {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x43);
    let pairs = all_pairs();
    let mut concepts: Vec<&str> = PLURAL
        .iter()
        .map(|c| c.0)
        .chain(FEMININE.iter().map(|c| c.0))
        .collect();
    for i in (1..concepts.len()).rev() {
        concepts.swap(i, rng.gen_range(0..=i));
    }
    let ntr = (concepts.len() as f64 * 0.7).round() as usize;
    let train_c: Vec<&str> = concepts[..ntr].to_vec();
    let train: Vec<&Pair> = pairs
        .iter()
        .filter(|p| p.lang == 0 && train_c.contains(&p.concept))
        .collect();
    let novel: Vec<&Pair> = pairs
        .iter()
        .filter(|p| !train_c.contains(&p.concept))
        .collect();
    let rand_emb: Emb = all_texts()
        .into_iter()
        .map(|t| {
            let v = random_encoder(&t, 512);
            (t, v)
        })
        .collect();
    let mut rows = vec![];
    // returns (acc per lang for novel, pooled, pca, scale, dphi) for an encoder
    struct Fitted {
        pca: Pca,
        scale: f64,
        dphi: LangDphi,
        offset: [Vec<f64>; 2],
    }
    let fit = |emb: &Emb, s: u64| -> Fitted {
        let words: Vec<Vec<f64>> = train
            .iter()
            .flat_map(|p| [emb[&p.src].clone(), emb[&p.dst].clone()])
            .collect();
        let pca = Pca::fit(&words, 16);
        let proj: Vec<Vec<f64>> = words.iter().map(|w| pca.project(w)).collect();
        let scale = (proj.iter().flatten().map(|v| v * v).sum::<f64>()
            / proj.iter().flatten().count() as f64)
            .sqrt()
            .max(1e-12);
        let z = |w: &str| -> Vec<f64> { pca.project(&emb[w]).iter().map(|v| v / scale).collect() };
        let data: Vec<(Vec<f64>, usize, Vec<f64>)> = train
            .iter()
            .map(|p| (z(&p.src), p.rel, z(&p.dst)))
            .collect();
        let mut dphi = LangDphi::new(pca.comps.len(), s ^ 0xD0);
        dphi.train(&data, 3000, s ^ 0x7A);
        let mut offset = [vec![0.0; pca.comps.len()], vec![0.0; pca.comps.len()]];
        for r in 0..2 {
            let ds: Vec<&(Vec<f64>, usize, Vec<f64>)> = data.iter().filter(|d| d.1 == r).collect();
            for d in &ds {
                for i in 0..offset[r].len() {
                    offset[r][i] += (d.2[i] - d.0[i]) / ds.len() as f64;
                }
            }
        }
        Fitted {
            pca,
            scale,
            dphi,
            offset,
        }
    };
    let fa = fit(ga, seed);
    let fr = fit(&rand_emb, seed);
    // evaluation helper: encoder map → per-lang accuracies
    let eval = |f: &Fitted,
                enc: &dyn Fn(&str) -> Vec<f64>,
                tag: &str,
                exp: &str,
                rows: &mut Vec<String>|
     -> (Vec<f64>, f64) {
        let mut per = vec![];
        let (mut tot_ok, mut tot_n) = (0usize, 0usize);
        for l in 0..4 {
            let lp: Vec<&&Pair> = novel.iter().filter(|p| p.lang == l).collect();
            let cands: Vec<(String, Vec<f64>)> = {
                let mut ws: Vec<String> = pairs
                    .iter()
                    .filter(|p| p.lang == l)
                    .flat_map(|p| [p.src.clone(), p.dst.clone()])
                    .collect();
                ws.sort();
                ws.dedup();
                ws.into_iter()
                    .map(|w| {
                        let v = enc(&w);
                        (w, v)
                    })
                    .collect()
            };
            let (mut ok_d, mut ok_o, mut ok_s) = (0, 0, 0);
            for p in &lp {
                let z = enc(&p.src);
                ok_d += (nearest(&f.dphi.step(&z, p.rel), &cands) == p.dst) as usize;
                let zo: Vec<f64> = z.iter().zip(&f.offset[p.rel]).map(|(a, b)| a + b).collect();
                ok_o += (nearest(&zo, &cands) == p.dst) as usize;
                ok_s += (nearest(&z, &cands) == p.dst) as usize;
            }
            let n = lp.len().max(1) as f64;
            for (meth, ok) in [("dphi", ok_d), ("offset", ok_o), ("static", ok_s)] {
                rows.push(format!(
                    "0x{seed:X},{exp},{tag},{},novel,{meth},{},{:.4}",
                    LANGS[l],
                    lp.len(),
                    ok as f64 / n
                ));
            }
            per.push(ok_d as f64 / n);
            tot_ok += ok_d;
            tot_n += lp.len();
        }
        (per, tot_ok as f64 / tot_n.max(1) as f64)
    };
    let enc_a = |w: &str| -> Vec<f64> {
        fa.pca
            .project(&ga[w])
            .iter()
            .map(|v| v / fa.scale)
            .collect()
    };
    let enc_r = |w: &str| -> Vec<f64> {
        fr.pca
            .project(&rand_emb[w])
            .iter()
            .map(|v| v / fr.scale)
            .collect()
    };
    let (acc_a, pooled_a) = eval(&fa, &enc_a, "gemma", "E43", &mut rows);
    let (acc_r, _) = eval(&fr, &enc_r, "random", "E43", &mut rows);
    // static per language for gate
    let stat: Vec<f64> = (0..4)
        .map(|l| {
            rows.iter()
                .find(|r| {
                    r.contains(",E43,gemma,") && r.contains(&format!(",{},novel,static,", LANGS[l]))
                })
                .and_then(|r| r.rsplit(',').next().and_then(|v| v.parse().ok()))
                .unwrap_or(0.0)
        })
        .collect();
    let wins = (1..4)
        .filter(|&l| acc_a[l] >= stat[l].max(acc_r[l]) + 0.05)
        .count();
    let note43 = format!(
        "gemma dphi es/en/fr/ja={:.2}/{:.2}/{:.2}/{:.2} random={:.2}/{:.2}/{:.2}/{:.2} static={:.2}/{:.2}/{:.2}/{:.2} wins={wins}/3 decoder_lookup=closed_vocab",
        acc_a[0], acc_a[1], acc_a[2], acc_a[3], acc_r[0], acc_r[1], acc_r[2], acc_r[3], stat[0], stat[1], stat[2], stat[3]
    );
    // E44: Dφ trained with A; test inputs via B / R aligned to A-PCA by anchors
    let map_b = anchor_map(gb, &fa.pca, ga, fa.scale);
    let map_r = anchor_map(&rand_emb, &fa.pca, ga, fa.scale);
    let map_a = anchor_map(ga, &fa.pca, ga, fa.scale);
    let enc_b = |w: &str| map_b(&gb[w]);
    let enc_rr = |w: &str| map_r(&rand_emb[w]);
    let enc_aa = |w: &str| map_a(&ga[w]);
    let (_, pb) = eval(&fa, &enc_b, "qwen_aligned", "E44", &mut rows);
    let (_, pr) = eval(&fa, &enc_rr, "random_aligned", "E44", &mut rows);
    let (_, paa) = eval(&fa, &enc_aa, "gemma_aligned", "E44", &mut rows);
    let e44 = pb >= pr + 0.05 && pb >= 0.5 * pooled_a;
    let note44 = format!("pooled novel acc: gemma={pooled_a:.3} gemma_anchor_aligned={paa:.3} qwen_aligned={pb:.3} random_aligned={pr:.3}");
    LangOut {
        rows,
        e43_pass: wins >= 2,
        e44_pass: e44,
        note43,
        note44,
    }
}
