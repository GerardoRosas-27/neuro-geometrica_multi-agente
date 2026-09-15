//! Dataset general enorme: frases en español de temas variados, generadas
//! al vuelo por índice. No se materializan 100k cadenas en RAM.

use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

pub const N_TOPICS: usize = 16;

const TOPICS: [&str; N_TOPICS] = [
    "clima", "fuego", "animales", "comida", "familia", "ciudad", "cuerpo", "tiempo", "oficios",
    "emocion", "mar", "plantas", "escuela", "deporte", "hogar", "viaje",
];

const NOUNS: [&[&str]; N_TOPICS] = [
    &[
        "el frío",
        "el hielo",
        "la nieve",
        "el viento",
        "la lluvia",
        "el granizo",
        "la escarcha",
        "la tormenta",
        "la niebla",
        "el relámpago",
    ],
    &[
        "el fuego",
        "la llama",
        "la brasa",
        "el horno",
        "el sol",
        "la hoguera",
        "el asfalto",
        "el desierto",
        "el vapor",
        "el carbón",
    ],
    &[
        "el perro",
        "el gato",
        "el caballo",
        "la vaca",
        "el lobo",
        "la oveja",
        "el zorro",
        "el águila",
        "el ciervo",
        "el pájaro",
    ],
    &[
        "el pan",
        "la sopa",
        "el arroz",
        "la fruta",
        "el queso",
        "el pescado",
        "la miel",
        "el maíz",
        "la leche",
        "el trigo",
    ],
    &[
        "la madre",
        "el padre",
        "el niño",
        "la niña",
        "el abuelo",
        "la hermana",
        "el vecino",
        "la familia",
        "el bebé",
        "el amigo",
    ],
    &[
        "la plaza",
        "la calle",
        "el mercado",
        "el puente",
        "el barrio",
        "la estación",
        "el parque",
        "el edificio",
        "la esquina",
        "el puerto",
    ],
    &[
        "la mano",
        "el ojo",
        "el corazón",
        "la espalda",
        "el pie",
        "la cabeza",
        "el brazo",
        "la voz",
        "el oído",
        "la piel",
    ],
    &[
        "la mañana",
        "la noche",
        "el invierno",
        "el verano",
        "el lunes",
        "el alba",
        "el ocaso",
        "la hora",
        "el año",
        "el instante",
    ],
    &[
        "el martillo",
        "la aguja",
        "el arado",
        "la cuerda",
        "el banco",
        "la sierra",
        "el clavo",
        "la rueda",
        "el cesto",
        "la pala",
    ],
    &[
        "la alegría",
        "el miedo",
        "la calma",
        "la rabia",
        "la pena",
        "el asombro",
        "la risa",
        "el silencio",
        "la esperanza",
        "el tedio",
    ],
    &[
        "el mar",
        "la ola",
        "el río",
        "la orilla",
        "el barco",
        "la marea",
        "el pez",
        "el puerto",
        "la bahía",
        "el coral",
    ],
    &[
        "el árbol",
        "la flor",
        "el trigo",
        "la semilla",
        "el bosque",
        "la raíz",
        "la hoja",
        "el jardín",
        "la rama",
        "el musgo",
    ],
    &[
        "el libro",
        "la tiza",
        "el aula",
        "la pregunta",
        "el mapa",
        "la lección",
        "el lápiz",
        "la pizarra",
        "el examen",
        "el receso",
    ],
    &[
        "el balón",
        "la carrera",
        "el salto",
        "la meta",
        "el equipo",
        "la pista",
        "el gol",
        "la red",
        "el silbato",
        "el marcador",
    ],
    &[
        "la mesa",
        "la silla",
        "la ventana",
        "la lámpara",
        "el patio",
        "la cocina",
        "la puerta",
        "el techo",
        "la cama",
        "el patio interior",
    ],
    &[
        "el camino",
        "el tren",
        "la maleta",
        "el mapa",
        "el hotel",
        "el sendero",
        "el paso",
        "la frontera",
        "el valle",
        "la cumbre",
    ],
];

const VERBS: [&[&str]; N_TOPICS] = [
    &[
        "cubre",
        "golpea",
        "envuelve",
        "cae sobre",
        "atraviesa",
        "silba en",
        "esconde",
        "sacude",
    ],
    &[
        "quema", "calienta", "alumbra", "arde en", "tuesta", "enciende", "ase", "hierve",
    ],
    &[
        "corre por",
        "duerme en",
        "cruza",
        "mira desde",
        "pastorea en",
        "vuela sobre",
        "bebe en",
        "caza junto a",
    ],
    &[
        "se sirve en",
        "se cuece en",
        "se parte en",
        "se guarda en",
        "se vende en",
        "se comparte en",
        "se hornea en",
        "se sazona en",
    ],
    &[
        "espera en",
        "abraza en",
        "camina por",
        "cuenta en",
        "cuida en",
        "visita",
        "recuerda en",
        "saluda en",
    ],
    &[
        "se llena en",
        "se cruza por",
        "se abre hacia",
        "se ilumina en",
        "se vacía al",
        "se escucha en",
        "se recorre por",
        "se construye junto a",
    ],
    &[
        "duele en",
        "descansa sobre",
        "se mueve en",
        "se calienta en",
        "se enfría en",
        "sostiene",
        "señala hacia",
        "tiembla en",
    ],
    &[
        "llega a",
        "pasa por",
        "marca",
        "cierra",
        "abre",
        "recorre",
        "se detiene en",
        "empieza en",
    ],
    &[
        "golpea", "cose", "labra", "sostiene", "corta", "une", "mide", "carga",
    ],
    &[
        "crece en",
        "se calma en",
        "estalla en",
        "se esconde en",
        "llena",
        "acompaña",
        "cambia en",
        "pesa en",
    ],
    &[
        "rompe en",
        "moja",
        "arrastra",
        "refleja",
        "golpea",
        "envuelve",
        "se retira de",
        "canta en",
    ],
    &[
        "crece en",
        "florece en",
        "sombra",
        "alimenta",
        "se seca en",
        "se dobla con",
        "cubre",
        "nace en",
    ],
    &[
        "explica",
        "se abre en",
        "se copia en",
        "se discute en",
        "se guarda en",
        "se olvida en",
        "se aprende en",
        "se señala en",
    ],
    &[
        "rebota en",
        "cruza",
        "gana en",
        "cierra en",
        "entrena en",
        "se disputa en",
        "cae en",
        "se celebra en",
    ],
    &[
        "se limpia en",
        "se abre hacia",
        "sostiene",
        "se cierra en",
        "se comparte en",
        "se repara en",
        "se ordena en",
        "se ilumina en",
    ],
    &[
        "parte hacia",
        "atraviesa",
        "se detiene en",
        "sigue",
        "descubre",
        "deja atrás",
        "une",
        "se pierde en",
    ],
];

const PLACES: [&[&str]; N_TOPICS] = [
    &[
        "la montaña",
        "el valle",
        "el lago",
        "el pueblo",
        "la llanura",
        "el tejado",
        "el bosque alto",
        "la cumbre",
        "el camino rural",
        "la ventana",
    ],
    &[
        "la panadería",
        "la chimenea",
        "la plaza",
        "el campamento",
        "la cocina",
        "el taller",
        "la estufa",
        "el crisol",
        "la terraza",
        "el horno de barro",
    ],
    &[
        "el patio",
        "el prado",
        "el río",
        "el matorral",
        "el risco",
        "el establo",
        "la silla",
        "el claro",
        "la orilla",
        "el gallinero",
    ],
    &[
        "la mesa",
        "el mercado",
        "la olla",
        "el cesto",
        "la despensa",
        "el comedor",
        "la feria",
        "el huerto",
        "la posada",
        "el almacén",
    ],
    &[
        "la casa",
        "el jardín",
        "la cocina",
        "el umbral",
        "la sala",
        "el pueblo",
        "la escuela",
        "el banco de la plaza",
        "el corredor",
        "la ventana",
    ],
    &[
        "el centro",
        "el muelle",
        "la avenida",
        "el mercado",
        "la estación",
        "el barrio viejo",
        "el parque",
        "el puente",
        "la plaza mayor",
        "el callejón",
    ],
    &[
        "la cama",
        "el banco",
        "el camino",
        "el agua fría",
        "el sol",
        "la sombra",
        "el trabajo",
        "la silla",
        "el suelo",
        "el aire",
    ],
    &[
        "el calendario",
        "la estación",
        "el reloj",
        "el invierno",
        "el pueblo",
        "la costa",
        "el aula",
        "el campo",
        "la ciudad",
        "el umbral",
    ],
    &[
        "el taller",
        "el campo",
        "el banco de trabajo",
        "la obra",
        "el granero",
        "la tejeduría",
        "el muelle",
        "la cantera",
        "el huerto",
        "el cobertizo",
    ],
    &[
        "el pecho",
        "la mirada",
        "la carta",
        "la noche",
        "la espera",
        "el reencuentro",
        "la fiesta",
        "el adiós",
        "el silencio",
        "la lluvia",
    ],
    &[
        "la costa",
        "el acantilado",
        "el puerto",
        "la playa",
        "el canal",
        "la red",
        "el muelle",
        "la isla",
        "el estuario",
        "la caleta",
    ],
    &[
        "el huerto",
        "la ladera",
        "el jardín",
        "el semillero",
        "el bosque",
        "la vega",
        "el maceta",
        "el cercado",
        "la ribera",
        "el invernadero",
    ],
    &[
        "el aula",
        "la biblioteca",
        "el patio escolar",
        "el pupitre",
        "la pizarra",
        "el pasillo",
        "el laboratorio",
        "el recreo",
        "la fila",
        "el cuaderno",
    ],
    &[
        "la cancha",
        "el estadio",
        "la pista",
        "el patio",
        "el gimnasio",
        "la meta",
        "el banquillo",
        "el campo",
        "la grada",
        "el vestuario",
    ],
    &[
        "la cocina",
        "el comedor",
        "el patio",
        "la recámara",
        "el umbral",
        "la azotea",
        "el pasillo",
        "la despensa",
        "el patio de luces",
        "la sala",
    ],
    &[
        "la estación",
        "el puerto",
        "el sendero",
        "la frontera",
        "el valle",
        "el desierto",
        "la posada",
        "el paso de montaña",
        "el andén",
        "el cruce",
    ],
];

const ADJS: [&[&str]; N_TOPICS] = [
    &[
        "gélido",
        "húmedo",
        "gris",
        "bruto",
        "fino",
        "constante",
        "seco",
        "blanco",
    ],
    &[
        "vivo", "rojo", "lento", "seco", "alto", "breve", "denso", "claro",
    ],
    &[
        "negro",
        "quieto",
        "joven",
        "ágil",
        "viejo",
        "hambreado",
        "libre",
        "manso",
    ],
    &[
        "caliente", "dulce", "salado", "fresco", "seco", "amarillo", "suave", "amargo",
    ],
    &[
        "cansado", "alegre", "pequeño", "serio", "ruidoso", "paciente", "nuevo", "lejano",
    ],
    &[
        "estrecho", "lleno", "viejo", "limpio", "oscuro", "ancho", "ruidoso", "quieto",
    ],
    &[
        "fuerte",
        "frío",
        "cansado",
        "firme",
        "tembloroso",
        "caliente",
        "lento",
        "hábil",
    ],
    &[
        "largo", "breve", "oscuro", "claro", "lento", "temprano", "tardío", "seco",
    ],
    &[
        "pesado", "afilado", "útil", "viejo", "preciso", "tosco", "nuevo", "firme",
    ],
    &[
        "honda", "breve", "sorda", "clara", "lenta", "súbita", "tenue", "grande",
    ],
    &[
        "verde", "salado", "bravo", "calmo", "oscuro", "claro", "hondo", "lejano",
    ],
    &[
        "verde",
        "seco",
        "alto",
        "joven",
        "espeso",
        "fino",
        "salvaje",
        "cultivado",
    ],
    &[
        "abierto", "difícil", "claro", "largo", "breve", "nuevo", "antiguo", "preciso",
    ],
    &[
        "rápido", "lento", "limpio", "duro", "justo", "corto", "alto", "cerrado",
    ],
    &[
        "limpio", "estrecho", "cálido", "viejo", "ordenado", "oscuro", "amplio", "simple",
    ],
    &[
        "largo",
        "incierto",
        "polvoriento",
        "claro",
        "lejano",
        "estrecho",
        "nuevo",
        "alto",
    ],
];

fn mix(seed: u64, index: u64) -> u64 {
    let mut h = seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h
}

pub fn topic_name(cluster: usize) -> &'static str {
    TOPICS[cluster % N_TOPICS]
}

/// Una frase determinista. El espacio de combinaciones supera 400 mil ítems.
pub fn example_at(index: u64, seed: u64) -> (String, usize) {
    let h = mix(seed, index);
    let topic = (h as usize) % N_TOPICS;
    let nouns = NOUNS[topic];
    let verbs = VERBS[topic];
    let places = PLACES[topic];
    let adjs = ADJS[topic];
    let noun = nouns[(h as usize >> 4) % nouns.len()];
    let verb = verbs[(h as usize >> 8) % verbs.len()];
    let place = places[(h as usize >> 12) % places.len()];
    let adj = adjs[(h as usize >> 16) % adjs.len()];
    let frame = (h >> 20) % 4;
    let text = match frame {
        0 => format!("{noun} {adj} {verb} {place}"),
        1 => format!("en {place}, {noun} {verb} el aire {adj}"),
        2 => format!("{noun} {verb} {place} con un gesto {adj}"),
        _ => format!("se ve {noun} {adj} cuando {verb} {place}"),
    };
    (text, topic)
}

pub fn is_holdout(index: u64) -> bool {
    index % 10 == 0
}

pub fn sample_indices(total: usize, count: usize, seed: u64, holdout: bool) -> Vec<u64> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let mut out = Vec::with_capacity(count);
    let mut guard = 0usize;
    while out.len() < count && guard < count.saturating_mul(40) + 8 {
        guard += 1;
        if total == 0 {
            break;
        }
        let idx = rng.gen_range(0..total as u64);
        if is_holdout(idx) != holdout {
            continue;
        }
        out.push(idx);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn general_corpus_covers_all_topics_and_is_huge() {
        let mut topics = HashSet::new();
        let mut texts = HashSet::new();
        for i in 0..4000u64 {
            let (t, c) = example_at(i, 0xC0B0);
            topics.insert(c);
            texts.insert(t);
        }
        assert_eq!(topics.len(), N_TOPICS);
        assert!(texts.len() > 3500, "unique={}", texts.len());
        let (a, _) = example_at(7, 0xC0B0);
        let (b, _) = example_at(7, 0xC0B0);
        assert_eq!(a, b);
    }

    #[test]
    fn holdout_split_is_one_in_ten() {
        let h = (0..1000u64).filter(|&i| is_holdout(i)).count();
        assert_eq!(h, 100);
    }
}
