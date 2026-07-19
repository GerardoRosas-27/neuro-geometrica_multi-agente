//! Motor termodinámico paginado experimental.
//!
//! Cada peso GGUF desquantizado se conserva como una arista implícita:
//! tensor + índice lineal determinan sus extremos; el shard guarda su peso f32.
//! El proceso es reanudable y usa memoria acotada.

use std::env;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const DEFAULT_OUTPUT: &str = "data/native_gemma2_paged_thermo";

#[derive(Clone, Debug)]
struct TensorInfo {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    offset: u64,
}

#[derive(Debug)]
struct GgufIndex {
    version: u32,
    alignment: u64,
    data_start: u64,
    file_len: u64,
    tensors: Vec<TensorInfo>,
    tokenizer_tokens: Vec<String>,
}

#[derive(Clone, Copy)]
struct QuantLayout {
    weights_per_block: u64,
    bytes_per_block: u64,
    label: &'static str,
}

fn main() -> Result<(), String> {
    let model = arg_value("--model").unwrap_or_else(|| "gemma2:2b".to_string());
    let output = PathBuf::from(arg_value("--output").unwrap_or_else(|| DEFAULT_OUTPUT.to_string()));
    let batch_mib = arg_usize("--batch-mib", 8).max(1);
    let shard_mib = arg_usize("--shard-mib", 64).max(1);
    let initialize_only = has_flag("--initialize-only");
    let lazy = has_flag("--lazy");
    let max_tensors = arg_value("--max-tensors").and_then(|value| value.parse::<usize>().ok());
    let max_shards = arg_value("--max-shards").and_then(|value| value.parse::<u64>().ok());
    let source = resolve_ollama_gguf(&model)?;
    let mut file = File::open(&source).map_err(|error| format!("abrir GGUF: {error}"))?;
    let index = read_gguf_index(&mut file)?;
    let (total_weights, logical_nodes) = model_capacity(&index)?;
    let expanded_bytes = total_weights
        .checked_mul(4)
        .ok_or("tamaño expandido excede u64")?;

    initialize_engine(
        &output,
        &model,
        &source,
        &index,
        total_weights,
        logical_nodes,
        expanded_bytes,
        if lazy { "gguf_lazy" } else { "f32_shards" },
    )?;
    println!("Native paged thermodynamic edge engine");
    println!(
        "status={} model={} gguf={} tensors={} logical_nodes={} logical_edges={} expanded_gib={:.2} output={} batch_mib={} shard_mib={}",
        if initialize_only { "empty_initialized" } else { "streaming" },
        model,
        source.display(),
        index.tensors.len(),
        logical_nodes,
        total_weights,
        expanded_bytes as f64 / 1024_f64.powi(3),
        output.display(),
        batch_mib,
        shard_mib,
    );
    println!(
        "edge_semantics=weight_strength sign_as_phase endpoints=implicit_tensor_axes memory_policy=flush_drop_resume"
    );
    if initialize_only {
        return Ok(());
    }
    if lazy {
        write_status(
            &output,
            "complete_lazy",
            index.tensors.len(),
            total_weights,
            total_weights,
        )?;
        if let (Some(tensor_id), Some(weight_index)) = (
            arg_value("--probe-tensor").and_then(|value| value.parse::<usize>().ok()),
            arg_value("--probe-weight").and_then(|value| value.parse::<u64>().ok()),
        ) {
            let (weight, source_node, target_node) =
                read_lazy_edge(&mut file, &index, tensor_id, weight_index)?;
            println!(
                "lazy_edge tensor={} weight_index={} source_node={} target_node={} weight={:.8} phase={:.5} energy={:.8}",
                tensor_id,
                weight_index,
                source_node,
                target_node,
                weight,
                if weight < 0.0 { std::f32::consts::PI } else { 0.0 },
                0.5 * weight * weight,
            );
        }
        println!(
            "done=true mode=gguf_lazy physical_copy_bytes=0 logical_edges={} backing_store={} output={}",
            total_weights,
            source.display(),
            output.display(),
        );
        return Ok(());
    }

    write_status(&output, "loading", 0, 0, total_weights)?;
    let start = Instant::now();
    let mut completed_weights = 0u64;
    let mut completed_shards = 0u64;
    let mut completed_tensors = 0usize;
    let mut stopped_early = false;
    let tensor_limit = max_tensors
        .unwrap_or(index.tensors.len())
        .min(index.tensors.len());
    'tensor_loop: for (tensor_id, tensor) in index.tensors.iter().take(tensor_limit).enumerate() {
        let layout = quant_layout(tensor.ggml_type)?;
        let weights = tensor_weight_count(tensor)?;
        let shard_target_weights =
            ((shard_mib as u64 * 1024 * 1024) / 4).max(layout.weights_per_block);
        let shard_weights =
            (shard_target_weights / layout.weights_per_block) * layout.weights_per_block;
        let tensor_dir = output.join("edges").join(format!("{tensor_id:04}"));
        fs::create_dir_all(&tensor_dir)
            .map_err(|error| format!("crear {}: {error}", tensor_dir.display()))?;

        let mut weight_start = 0u64;
        let mut shard_id = 0u64;
        while weight_start < weights {
            let count = (weights - weight_start).min(shard_weights);
            let shard_path = tensor_dir.join(format!("{shard_id:06}.edges.f32"));
            let expected_size = count * 4;
            let complete = shard_path
                .metadata()
                .map(|metadata| metadata.len() == expected_size)
                .unwrap_or(false);
            if !complete {
                stream_shard(
                    &mut file,
                    &index,
                    tensor,
                    layout,
                    weight_start,
                    count,
                    &shard_path,
                    batch_mib * 1024 * 1024,
                )?;
            }
            completed_weights += count;
            completed_shards += 1;
            weight_start += count;
            shard_id += 1;
            write_status(
                &output,
                "loading",
                tensor_id + 1,
                completed_weights,
                total_weights,
            )?;
            println!(
                "tensor={}/{} name={} quant={} shard={} weights={} progress={:.4}% reused={} elapsed_s={:.1}",
                tensor_id + 1,
                tensor_limit,
                tensor.name,
                layout.label,
                shard_id,
                count,
                completed_weights as f64 / total_weights.max(1) as f64 * 100.0,
                complete,
                start.elapsed().as_secs_f64(),
            );
            if max_shards.is_some_and(|limit| completed_shards >= limit) {
                stopped_early = true;
                break 'tensor_loop;
            }
        }
        completed_tensors = tensor_id + 1;
    }
    let complete = !stopped_early && tensor_limit == index.tensors.len();
    write_status(
        &output,
        if complete { "complete" } else { "partial_test" },
        completed_tensors,
        completed_weights,
        total_weights,
    )?;
    println!(
        "done={} tensors={} shards={} weights={} total_weights={} elapsed_s={:.1} output={}",
        complete,
        completed_tensors,
        completed_shards,
        completed_weights,
        total_weights,
        start.elapsed().as_secs_f64(),
        output.display(),
    );
    Ok(())
}

fn initialize_engine(
    output: &Path,
    model: &str,
    source: &Path,
    index: &GgufIndex,
    total_weights: u64,
    logical_nodes: u64,
    expanded_bytes: u64,
    storage: &str,
) -> Result<(), String> {
    fs::create_dir_all(output.join("edges"))
        .map_err(|error| format!("crear motor paginado: {error}"))?;
    let manifest = format!(
        "NATIVE_THERMO_PAGED_EDGES_V1\nmodel={model}\nsource={}\nstorage={storage}\ngguf_version={}\ngguf_alignment={}\ngguf_file_bytes={}\ngguf_data_start={}\ntensors={}\nlogical_nodes={logical_nodes}\nlogical_edges={total_weights}\nexpanded_f32_bytes={expanded_bytes}\nedge_record=implicit_endpoints_plus_weight\nphase_rule=weight_negative_pi_else_zero\nenergy_rule=0.5_weight_squared\n",
        source.display(),
        index.version,
        index.alignment,
        index.file_len,
        index.data_start,
        index.tensors.len(),
    );
    fs::write(output.join("manifest.txt"), manifest)
        .map_err(|error| format!("guardar manifest: {error}"))?;
    let mut catalog = String::from(
        "tensor_id\tname\tggml_type\tquant\tdimensions\tweights\tlogical_source_nodes\tlogical_target_nodes\tgguf_relative_offset\tweights_per_block\tbytes_per_block\n",
    );
    for (tensor_id, tensor) in index.tensors.iter().enumerate() {
        let layout = quant_layout(tensor.ggml_type)?;
        let weights = tensor_weight_count(tensor)?;
        let source_nodes = tensor.dimensions.first().copied().unwrap_or(1);
        let target_nodes = if tensor.dimensions.len() > 1 {
            tensor.dimensions[1..].iter().product()
        } else {
            source_nodes
        };
        catalog.push_str(&format!(
            "{tensor_id}\t{}\t{}\t{}\t{}\t{weights}\t{source_nodes}\t{target_nodes}\t{}\t{}\t{}\n",
            tensor.name,
            tensor.ggml_type,
            layout.label,
            tensor
                .dimensions
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join("x"),
            tensor.offset,
            layout.weights_per_block,
            layout.bytes_per_block,
        ));
    }
    fs::write(output.join("tensor_catalog.tsv"), catalog)
        .map_err(|error| format!("guardar catálogo: {error}"))?;
    if !index.tokenizer_tokens.is_empty() {
        let mut vocabulary = String::from("token_id\thex_utf8\n");
        for (token_id, token) in index.tokenizer_tokens.iter().enumerate() {
            vocabulary.push_str(&format!("{token_id}\t{}\n", hex_encode(token.as_bytes())));
        }
        fs::write(output.join("tokenizer_vocab.hex.tsv"), vocabulary)
            .map_err(|error| format!("guardar vocabulario: {error}"))?;
    }
    Ok(())
}

fn stream_shard(
    file: &mut File,
    index: &GgufIndex,
    tensor: &TensorInfo,
    layout: QuantLayout,
    weight_start: u64,
    weight_count: u64,
    output: &Path,
    batch_bytes: usize,
) -> Result<(), String> {
    if weight_start % layout.weights_per_block != 0 {
        return Err("inicio de shard no alineado a bloque cuantizado".to_string());
    }
    let block_start = weight_start / layout.weights_per_block;
    let blocks = weight_count.div_ceil(layout.weights_per_block);
    let encoded_start = index.data_start + tensor.offset + block_start * layout.bytes_per_block;
    file.seek(SeekFrom::Start(encoded_start))
        .map_err(|error| format!("seek {}: {error}", tensor.name))?;
    let part = output.with_extension("f32.part");
    if part.exists() {
        fs::remove_file(&part).map_err(|error| format!("limpiar shard parcial: {error}"))?;
    }
    let part_file =
        File::create(&part).map_err(|error| format!("crear {}: {error}", part.display()))?;
    let mut writer = BufWriter::with_capacity(batch_bytes, part_file);
    let mut reader = BufReader::with_capacity(batch_bytes, file);
    let mut remaining = weight_count;
    let mut encoded = vec![0u8; layout.bytes_per_block as usize];
    let mut decoded = [0f32; 256];
    for _ in 0..blocks {
        reader
            .read_exact(&mut encoded)
            .map_err(|error| format!("leer bloque {}: {error}", tensor.name))?;
        let produced = decode_exact_block(tensor.ggml_type, &encoded, &mut decoded)?;
        let take = remaining.min(produced as u64) as usize;
        for value in &decoded[..take] {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|error| format!("escribir arista: {error}"))?;
        }
        remaining -= take as u64;
    }
    writer
        .flush()
        .map_err(|error| format!("flush shard: {error}"))?;
    drop(writer);
    let actual = part
        .metadata()
        .map_err(|error| format!("metadata shard: {error}"))?
        .len();
    if actual != weight_count * 4 {
        return Err(format!(
            "shard incompleto: esperado={} actual={actual}",
            weight_count * 4
        ));
    }
    fs::rename(&part, output).map_err(|error| format!("confirmar shard: {error}"))
}

fn read_lazy_edge(
    file: &mut File,
    index: &GgufIndex,
    tensor_id: usize,
    weight_index: u64,
) -> Result<(f32, u64, u64), String> {
    let tensor = index
        .tensors
        .get(tensor_id)
        .ok_or_else(|| format!("tensor fuera de rango: {tensor_id}"))?;
    let weights = tensor_weight_count(tensor)?;
    if weight_index >= weights {
        return Err(format!(
            "peso fuera de rango: {weight_index}, tensor_weights={weights}"
        ));
    }
    let layout = quant_layout(tensor.ggml_type)?;
    let block_index = weight_index / layout.weights_per_block;
    let within_block = (weight_index % layout.weights_per_block) as usize;
    let encoded_start = index.data_start + tensor.offset + block_index * layout.bytes_per_block;
    file.seek(SeekFrom::Start(encoded_start))
        .map_err(|error| format!("seek lazy: {error}"))?;
    let mut encoded = vec![0u8; layout.bytes_per_block as usize];
    file.read_exact(&mut encoded)
        .map_err(|error| format!("leer bloque lazy: {error}"))?;
    let mut decoded = [0f32; 256];
    let produced = decode_exact_block(tensor.ggml_type, &encoded, &mut decoded)?;
    if within_block >= produced {
        return Err("índice interno inválido".to_string());
    }
    let source_nodes = tensor.dimensions.first().copied().unwrap_or(1).max(1);
    Ok((
        decoded[within_block],
        weight_index % source_nodes,
        weight_index / source_nodes,
    ))
}

fn decode_exact_block(
    ggml_type: u32,
    block: &[u8],
    output: &mut [f32; 256],
) -> Result<usize, String> {
    match ggml_type {
        0 => {
            output[0] = f32::from_le_bytes(block[0..4].try_into().unwrap());
            Ok(1)
        }
        1 => {
            output[0] = f16_to_f32(u16::from_le_bytes(block[0..2].try_into().unwrap()));
            Ok(1)
        }
        2 => {
            let scale = f16_to_f32(u16::from_le_bytes(block[0..2].try_into().unwrap()));
            for index in 0..16 {
                let quants = block[2 + index];
                output[index] = scale * ((quants & 0x0f) as i8 - 8) as f32;
                output[index + 16] = scale * ((quants >> 4) as i8 - 8) as f32;
            }
            Ok(32)
        }
        14 => {
            let ql = &block[0..128];
            let qh = &block[128..192];
            let scales = &block[192..208];
            let scale = f16_to_f32(u16::from_le_bytes(block[208..210].try_into().unwrap()));
            for ip in 0..2 {
                for il in 0..32 {
                    let scale_index = 8 * ip + il / 16;
                    let ql_index = 64 * ip + il;
                    let high = qh[32 * ip + il];
                    let base = 128 * ip + il;
                    let signed_scale = |offset: usize| scales[scale_index + offset] as i8 as f32;
                    output[base] = scale
                        * signed_scale(0)
                        * ((((ql[ql_index] & 0x0f) | ((high & 0x03) << 4)) as i8 - 32) as f32);
                    output[base + 32] = scale
                        * signed_scale(2)
                        * ((((ql[ql_index + 32] & 0x0f) | (((high >> 2) & 0x03) << 4)) as i8 - 32)
                            as f32);
                    output[base + 64] = scale
                        * signed_scale(4)
                        * ((((ql[ql_index] >> 4) | (((high >> 4) & 0x03) << 4)) as i8 - 32) as f32);
                    output[base + 96] = scale
                        * signed_scale(6)
                        * ((((ql[ql_index + 32] >> 4) | (((high >> 6) & 0x03) << 4)) as i8 - 32)
                            as f32);
                }
            }
            Ok(256)
        }
        _ => Err(format!(
            "GGML type {ggml_type} no tiene desquantización exacta implementada"
        )),
    }
}

fn quant_layout(ggml_type: u32) -> Result<QuantLayout, String> {
    match ggml_type {
        0 => Ok(QuantLayout {
            weights_per_block: 1,
            bytes_per_block: 4,
            label: "F32",
        }),
        1 => Ok(QuantLayout {
            weights_per_block: 1,
            bytes_per_block: 2,
            label: "F16",
        }),
        2 => Ok(QuantLayout {
            weights_per_block: 32,
            bytes_per_block: 18,
            label: "Q4_0",
        }),
        14 => Ok(QuantLayout {
            weights_per_block: 256,
            bytes_per_block: 210,
            label: "Q6_K",
        }),
        _ => Err(format!("tipo GGML no soportado exactamente: {ggml_type}")),
    }
}

fn model_capacity(index: &GgufIndex) -> Result<(u64, u64), String> {
    let mut edges = 0u64;
    let mut nodes = 0u64;
    for tensor in &index.tensors {
        quant_layout(tensor.ggml_type)?;
        edges = edges
            .checked_add(tensor_weight_count(tensor)?)
            .ok_or("demasiadas aristas")?;
        let source = tensor.dimensions.first().copied().unwrap_or(1);
        let target = if tensor.dimensions.len() > 1 {
            checked_product(&tensor.dimensions[1..])?
        } else {
            source
        };
        nodes = nodes
            .checked_add(source)
            .and_then(|value| value.checked_add(target))
            .ok_or("demasiados nodos lógicos")?;
    }
    Ok((edges, nodes))
}

fn tensor_weight_count(tensor: &TensorInfo) -> Result<u64, String> {
    checked_product(&tensor.dimensions)
}

fn checked_product(values: &[u64]) -> Result<u64, String> {
    values.iter().try_fold(1u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or("dimensiones exceden u64".to_string())
    })
}

fn write_status(
    output: &Path,
    status: &str,
    tensors: usize,
    weights: u64,
    total: u64,
) -> Result<(), String> {
    let body = format!(
        "status={status}\ntensors_completed={tensors}\nweights_completed={weights}\ntotal_weights={total}\nprogress={:.8}\n",
        weights as f64 / total.max(1) as f64
    );
    let temporary = output.join("progress.tmp");
    fs::write(&temporary, body).map_err(|error| format!("guardar progreso: {error}"))?;
    let destination = output.join("progress.txt");
    if destination.exists() {
        fs::remove_file(&destination).map_err(|error| format!("reemplazar progreso: {error}"))?;
    }
    fs::rename(temporary, destination).map_err(|error| format!("confirmar progreso: {error}"))
}

fn read_gguf_index(file: &mut File) -> Result<GgufIndex, String> {
    let file_len = file
        .metadata()
        .map_err(|error| format!("metadata GGUF: {error}"))?
        .len();
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|error| format!("magic GGUF: {error}"))?;
    if &magic != b"GGUF" {
        return Err("el blob de Ollama no es GGUF".to_string());
    }
    let version = read_u32(file)?;
    let tensor_count = read_u64(file)?;
    let metadata_count = read_u64(file)?;
    let mut alignment = 32u64;
    let mut tokenizer_tokens = Vec::new();
    for _ in 0..metadata_count {
        let key = read_string(file)?;
        let value_type = read_u32(file)?;
        if key == "general.alignment" && value_type == 4 {
            alignment = read_u32(file)? as u64;
        } else if key == "tokenizer.ggml.tokens" && value_type == 9 {
            tokenizer_tokens = read_string_array(file)?;
        } else {
            skip_value(file, value_type)?;
        }
    }
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(file)?;
        let dimensions_count = read_u32(file)? as usize;
        let mut dimensions = Vec::with_capacity(dimensions_count);
        for _ in 0..dimensions_count {
            dimensions.push(read_u64(file)?);
        }
        tensors.push(TensorInfo {
            name,
            dimensions,
            ggml_type: read_u32(file)?,
            offset: read_u64(file)?,
        });
    }
    let position = file
        .stream_position()
        .map_err(|error| format!("posición GGUF: {error}"))?;
    Ok(GgufIndex {
        version,
        alignment,
        data_start: align_up(position, alignment.max(1)),
        file_len,
        tensors,
        tokenizer_tokens,
    })
}

fn read_string_array(file: &mut File) -> Result<Vec<String>, String> {
    let element_type = read_u32(file)?;
    if element_type != 8 {
        return Err(format!(
            "tokenizer.ggml.tokens no contiene strings: type={element_type}"
        ));
    }
    let count = read_u64(file)?;
    let mut values = Vec::with_capacity(count.min(usize::MAX as u64) as usize);
    for _ in 0..count {
        values.push(read_string(file)?);
    }
    Ok(values)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn skip_value(file: &mut File, value_type: u32) -> Result<(), String> {
    if let Some(bytes) = primitive_size(value_type) {
        file.seek(SeekFrom::Current(bytes as i64))
            .map_err(|error| format!("saltar metadata: {error}"))?;
        return Ok(());
    }
    match value_type {
        8 => {
            let length = read_u64(file)?;
            file.seek(SeekFrom::Current(length as i64))
                .map_err(|error| format!("saltar string: {error}"))?;
            Ok(())
        }
        9 => {
            let element_type = read_u32(file)?;
            let count = read_u64(file)?;
            if let Some(size) = primitive_size(element_type) {
                let bytes = count
                    .checked_mul(size)
                    .ok_or("array GGUF demasiado grande")?;
                file.seek(SeekFrom::Current(bytes as i64))
                    .map_err(|error| format!("saltar array: {error}"))?;
            } else {
                for _ in 0..count {
                    skip_value(file, element_type)?;
                }
            }
            Ok(())
        }
        _ => Err(format!("tipo metadata GGUF no soportado: {value_type}")),
    }
}

fn primitive_size(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4 | 5 | 6 => Some(4),
        10 | 11 | 12 => Some(8),
        _ => None,
    }
}

fn resolve_ollama_gguf(model: &str) -> Result<PathBuf, String> {
    if let Ok(path) = env::var("OLLAMA_GGUF_PATH") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
    }
    let output = Command::new("ollama")
        .args(["show", model, "--modelfile"])
        .output()
        .map_err(|error| format!("ejecutar ollama: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let from = stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix("FROM "))
        .ok_or("ollama no devolvió ruta FROM")?;
    let path = PathBuf::from(from.trim_matches('"'));
    path.exists()
        .then_some(path)
        .ok_or("FROM no apunta a un blob local".to_string())
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x03ff) as u32;
    let value = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut mantissa = mantissa;
            let mut exponent = 113u32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((mantissa & 0x03ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(value)
}

fn read_u32(reader: &mut File) -> Result<u32, String> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("leer u32: {error}"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut File) -> Result<u64, String> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("leer u64: {error}"))?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_string(reader: &mut File) -> Result<String, String> {
    let length = read_u64(reader)?;
    if length > 16 * 1024 * 1024 {
        return Err("string GGUF demasiado grande".to_string());
    }
    let mut bytes = vec![0u8; length as usize];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("leer string: {error}"))?;
    String::from_utf8(bytes).map_err(|error| format!("UTF-8 inválido: {error}"))
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

fn arg_usize(name: &str, fallback: usize) -> usize {
    arg_value(name)
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn arg_value(name: &str) -> Option<String> {
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == name {
            return args.next();
        }
    }
    None
}

fn has_flag(name: &str) -> bool {
    env::args().any(|argument| argument == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_q4_0_block_exactly() {
        let mut block = [0xf0u8; 18];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        let mut output = [0.0; 256];
        assert_eq!(decode_exact_block(2, &block, &mut output).unwrap(), 32);
        assert!(output[..16].iter().all(|value| *value == -8.0));
        assert!(output[16..32].iter().all(|value| *value == 7.0));
    }

    #[test]
    fn decodes_q6_k_block_exactly() {
        let mut block = [0u8; 210];
        block[192..208].fill(1);
        block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
        let mut output = [0.0; 256];
        assert_eq!(decode_exact_block(14, &block, &mut output).unwrap(), 256);
        assert!(output.iter().all(|value| *value == -32.0));
    }
}
