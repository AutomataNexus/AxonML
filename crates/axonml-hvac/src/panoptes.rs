//! Panoptes — Facility-Wide Anomaly Detection Model
//!
//! # File
//! `crates/axonml-hvac/src/panoptes.rs`
//!
//! # Description
//! All-seeing facility monitor that ingests heterogeneous equipment sensor data
//! from an entire building simultaneously. Uses per-type feature encoders,
//! equipment embeddings, cross-equipment transformer attention, and temporal
//! LSTM to detect anomalies by learning normal operating patterns and
//! cross-equipment correlations.
//!
//! Named after Argus Panoptes — the hundred-eyed giant of Greek mythology.
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 9, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_nn::Module;
use axonml_nn::layers::{Embedding, LSTM, LayerNorm, Linear, TransformerEncoder};
use axonml_nn::parameter::Parameter;
use axonml_tensor::Tensor;

// =============================================================================
// Equipment Type Constants
// =============================================================================

/// Equipment type ID: Air Handler Unit.
pub const EQUIP_AHU: usize = 0;
/// Equipment type ID: Dedicated Outdoor Air System.
pub const EQUIP_DOAS: usize = 1;
/// Equipment type ID: Steam Boiler.
pub const EQUIP_BOILER: usize = 2;
/// Equipment type ID: Steam Bundle (heat exchanger).
pub const EQUIP_STEAM_BUNDLE: usize = 3;
/// Equipment type ID: Fan Coil Unit.
pub const EQUIP_FAN_COIL: usize = 4;
/// Equipment type ID: Pump (HW, CW, CHW).
pub const EQUIP_PUMP: usize = 5;
/// Equipment type ID: Chiller.
pub const EQUIP_CHILLER: usize = 6;
/// Total number of equipment types.
pub const NUM_EQUIP_TYPES: usize = 7;

/// Number of sensor channels for AHU equipment.
pub const AHU_SENSORS: usize = 12;
/// Number of sensor channels for DOAS equipment.
pub const DOAS_SENSORS: usize = 6;
/// Number of sensor channels for Boiler equipment.
pub const BOILER_SENSORS: usize = 7;
/// Number of sensor channels for Steam Bundle equipment.
pub const STEAM_BUNDLE_SENSORS: usize = 5;
/// Number of sensor channels for Fan Coil equipment.
pub const FAN_COIL_SENSORS: usize = 9;
/// Number of sensor channels for Pump equipment.
pub const PUMP_SENSORS: usize = 7;
/// Number of sensor channels for Chiller equipment.
pub const CHILLER_SENSORS: usize = 9;

/// Maximum sensor count across all types (for padding).
pub const MAX_SENSORS: usize = AHU_SENSORS; // 12

/// Common embedding dimension for all equipment after encoding.
pub const EMBED_DIM: usize = 32;

// =============================================================================
// FacilitySnapshot — Input Data Structure
// =============================================================================

/// A single point-in-time snapshot of all equipment in a facility.
///
/// Each equipment's sensors are normalized to `MAX_SENSORS` width with
/// zero-padding for types with fewer sensors. A mask tensor indicates
/// which values are real readings vs missing/padded.
///
/// Shape: features `[num_equipment, MAX_SENSORS]`, mask `[num_equipment, MAX_SENSORS]`
#[derive(Clone)]
pub struct FacilitySnapshot {
    /// Equipment sensor values, zero-filled for missing. Shape: [num_equip, MAX_SENSORS]
    pub features: Vec<f32>,
    /// Binary mask: 1.0 = real reading, 0.0 = missing/padded. Shape: [num_equip, MAX_SENSORS]
    pub mask: Vec<f32>,
    /// Equipment type ID for each row. Length: num_equip
    pub equip_types: Vec<usize>,
    /// Equipment string IDs (e.g., "warren-ahu-6"). Length: num_equip
    pub equip_ids: Vec<String>,
    /// Number of equipment in this snapshot.
    pub num_equipment: usize,
}

impl FacilitySnapshot {
    /// Creates a new empty snapshot for the given number of equipment.
    pub fn new(num_equipment: usize) -> Self {
        Self {
            features: vec![0.0; num_equipment * MAX_SENSORS],
            mask: vec![0.0; num_equipment * MAX_SENSORS],
            equip_types: vec![0; num_equipment],
            equip_ids: vec![String::new(); num_equipment],
            num_equipment,
        }
    }

    /// Sets sensor readings for a specific equipment slot.
    ///
    /// `values` contains the raw readings; `None` entries are treated as missing.
    pub fn set_equipment(
        &mut self,
        slot: usize,
        equip_id: &str,
        equip_type: usize,
        values: &[Option<f32>],
    ) {
        assert!(slot < self.num_equipment);
        self.equip_ids[slot] = equip_id.to_string();
        self.equip_types[slot] = equip_type;

        let base = slot * MAX_SENSORS;
        for (i, val) in values.iter().enumerate() {
            if i >= MAX_SENSORS {
                break;
            }
            match val {
                Some(v) => {
                    self.features[base + i] = *v;
                    self.mask[base + i] = 1.0;
                }
                None => {
                    self.features[base + i] = 0.0;
                    self.mask[base + i] = 0.0;
                }
            }
        }
    }
}

// =============================================================================
// FacilityConfig — Describes a specific facility's equipment layout
// =============================================================================

/// Configuration describing which equipment a facility has.
///
/// Used to create the equipment ID embedding and to parse incoming data
/// into the correct slots.
pub struct FacilityConfig {
    /// Ordered list of (equipment_id, equipment_type) pairs.
    pub equipment: Vec<(String, usize)>,
    /// Lookup from equipment ID string to slot index.
    pub id_to_slot: HashMap<String, usize>,
}

impl FacilityConfig {
    /// Creates a facility config from a list of (id, type) pairs.
    pub fn new(equipment: Vec<(String, usize)>) -> Self {
        let id_to_slot: HashMap<String, usize> = equipment
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (id.clone(), i))
            .collect();
        Self {
            equipment,
            id_to_slot,
        }
    }

    /// Number of equipment in this facility.
    pub fn num_equipment(&self) -> usize {
        self.equipment.len()
    }

    /// Creates the Warren facility config from the live equipment list.
    pub fn warren() -> Self {
        let mut equipment = Vec::new();

        // Air Handlers (6)
        for id in &[
            "warren-ahu-6",
            "warren-ahu-1",
            "warren-ahu-4",
            "warren-ahu-2",
            "warren-ahu-5",
            "warren-ahu-7",
        ] {
            equipment.push(((*id).to_string(), EQUIP_AHU));
        }

        // DOAS (1)
        equipment.push(("warren-fahl-doas".to_string(), EQUIP_DOAS));

        // Steam Boilers (3)
        for id in &["warren-boiler-1", "warren-boiler-2", "warren-boiler-3"] {
            equipment.push(((*id).to_string(), EQUIP_BOILER));
        }

        // Steam Bundles (9)
        for id in &[
            "warren-steambundle-1",
            "warren-steambundle-5",
            "warren-steambundle-4",
            "warren-steambundle-fahl",
            "warren-steambundle-6",
            "warren-steambundle-7",
            "warren-steambundle-3",
            "warren-steambundle-8",
            "warren-steambundle-2",
        ] {
            equipment.push(((*id).to_string(), EQUIP_STEAM_BUNDLE));
        }

        // Fan Coils (18)
        for i in 1..=18 {
            equipment.push((format!("warren-fancoil-{i}"), EQUIP_FAN_COIL));
        }

        // Pumps (20)
        for id in &[
            "warren-cwbooster-1",
            "warren-cwbooster-2",
            "warren-cwpump-5",
            "warren-cwpump-6",
            "warren-hwpump-7",
            "warren-hwpump-8",
            "warren-hwpump-9",
            "warren-hwpump-10",
            "warren-hwpump-11",
            "warren-hwpump-12",
            "warren-chwpump-3",
            "warren-chwpump-4",
            "warren-cwpump-3",
            "warren-cwpump-4",
            "warren-hwpump-5",
            "warren-hwpump-6",
            "warren-hwpump-1",
            "warren-hwpump-2",
            "warren-hwpump-3",
            "warren-hwpump-4",
        ] {
            equipment.push(((*id).to_string(), EQUIP_PUMP));
        }

        // Chillers (2)
        for id in &["warren-chiller-2", "warren-chiller-1"] {
            equipment.push(((*id).to_string(), EQUIP_CHILLER));
        }

        Self::new(equipment)
    }
}

// =============================================================================
// EquipTypeEncoder — Per-type feature encoder
// =============================================================================

/// Encodes raw sensor readings for one equipment type to EMBED_DIM.
///
/// Each equipment type has its own encoder because sensor counts and
/// semantics differ (AHU has 12 sensors, SteamBundle has 5, etc.).
struct EquipTypeEncoder {
    /// Linear projection: sensor_count → EMBED_DIM
    linear: Linear,
    /// LayerNorm on output
    norm: LayerNorm,
}

impl EquipTypeEncoder {
    fn new(sensor_count: usize) -> Self {
        Self {
            linear: Linear::new(sensor_count, EMBED_DIM),
            norm: LayerNorm::new(vec![EMBED_DIM]),
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        self.norm.forward(&self.linear.forward(x).relu())
    }

    fn parameters(&self) -> Vec<Parameter> {
        [self.linear.parameters(), self.norm.parameters()].concat()
    }
}

// =============================================================================
// Panoptes — The Facility-Wide Anomaly Detection Model
// =============================================================================

/// Facility-wide anomaly detection model.
///
/// # Architecture
///
/// ```text
/// Raw Sensors ──→ Per-Type Encoders ──→ [num_equip, 32]
///                                          + Type Embed (7 → 32)
///                                          + ID Embed (max_equip → 32)
///                                          ↓
///                                   Cross-Equipment Transformer (2 layers, 4 heads)
///                                          ↓  [num_equip, 32]
///                                       Temporal LSTM (hidden=64)
///                                          ↓  [num_equip, 64]
///                                   ┌──────┴──────┐
///                              Per-Equip Head   Facility Head
///                              Linear(64→1)    Pool → Linear(64→1)
///                                   ↓               ↓
///                            Anomaly Scores    Health Score
/// ```
///
/// # Parameters
/// ~250K total — deployable on edge devices.
pub struct Panoptes {
    /// Per-type feature encoders (one per equipment type).
    type_encoders: Vec<EquipTypeEncoder>,
    /// Equipment type embedding: NUM_EQUIP_TYPES → EMBED_DIM.
    type_embed: Embedding,
    /// Equipment ID embedding: max_equipment → EMBED_DIM.
    id_embed: Embedding,
    /// Missing value embedding: learned vector added when sensor is missing.
    missing_embed: Parameter,
    /// Cross-equipment transformer encoder (captures inter-equipment correlations).
    cross_attn: TransformerEncoder,
    /// Temporal LSTM: processes sequence of facility snapshots.
    temporal_lstm: LSTM,
    /// Per-equipment anomaly scoring head (from EMBED_DIM for snapshot, from 64 for temporal).
    equip_head_snapshot: Linear,
    /// Per-equipment anomaly scoring head (from LSTM hidden).
    equip_head_temporal: Linear,
    /// Facility-level health scoring head (from EMBED_DIM for snapshot).
    facility_head_snapshot: Linear,
    /// Facility-level health scoring head (from LSTM hidden).
    facility_head_temporal: Linear,
    /// Number of equipment slots this model was configured for.
    num_equipment: usize,
    /// Sensor counts per equipment type.
    sensor_counts: [usize; NUM_EQUIP_TYPES],
}

impl Panoptes {
    /// Creates a new Panoptes model for the given number of equipment slots.
    ///
    /// # Arguments
    /// * `num_equipment` - Total equipment count in the facility (e.g., 59 for Warren)
    pub fn new(num_equipment: usize) -> Self {
        let sensor_counts = [
            AHU_SENSORS,          // 12
            DOAS_SENSORS,         // 6
            BOILER_SENSORS,       // 7
            STEAM_BUNDLE_SENSORS, // 5
            FAN_COIL_SENSORS,     // 9
            PUMP_SENSORS,         // 7
            CHILLER_SENSORS,      // 9
        ];

        let type_encoders: Vec<EquipTypeEncoder> = sensor_counts
            .iter()
            .map(|&count| EquipTypeEncoder::new(count))
            .collect();

        // Missing value embedding: a learnable vector of EMBED_DIM
        let missing_data = axonml_nn::init::normal(&[1, EMBED_DIM], 0.0, 0.02);
        let missing_embed = Parameter::named("missing_embed", missing_data, true);

        Self {
            type_encoders,
            type_embed: Embedding::new(NUM_EQUIP_TYPES, EMBED_DIM),
            id_embed: Embedding::new(num_equipment, EMBED_DIM),
            missing_embed,
            // 2 layers, 4 heads, feedforward dim 64
            cross_attn: TransformerEncoder::new(EMBED_DIM, 4, 64, 2),
            // LSTM: input=EMBED_DIM, hidden=64, 1 layer
            temporal_lstm: LSTM::new(EMBED_DIM, 64, 1),
            equip_head_snapshot: Linear::new(EMBED_DIM, 1),
            equip_head_temporal: Linear::new(64, 1),
            facility_head_snapshot: Linear::new(EMBED_DIM, 1),
            facility_head_temporal: Linear::new(64, 1),
            num_equipment,
            sensor_counts,
        }
    }

    /// Returns total parameter count.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Encodes a single facility snapshot into per-equipment embeddings.
    ///
    /// # Arguments
    /// * `snapshot` - Raw facility sensor data
    ///
    /// # Returns
    /// Variable of shape `[1, num_equipment, EMBED_DIM]` — one embedding per equipment.
    pub fn encode_snapshot(&self, snapshot: &FacilitySnapshot) -> Variable {
        let n = snapshot.num_equipment;
        assert_eq!(n, self.num_equipment);

        // Encode each equipment through its type-specific encoder
        let mut encoded_vecs: Vec<f32> = Vec::with_capacity(n * EMBED_DIM);

        for slot in 0..n {
            let equip_type = snapshot.equip_types[slot];
            let sensor_count = self.sensor_counts[equip_type];
            let base = slot * MAX_SENSORS;

            // Extract this equipment's sensors (only sensor_count values)
            let sensor_values: Vec<f32> = (0..sensor_count)
                .map(|i| snapshot.features[base + i])
                .collect();

            // Check if all sensors are missing
            let all_missing = (0..sensor_count).all(|i| snapshot.mask[base + i] == 0.0);

            if all_missing {
                // Use the missing embedding directly
                let missing = self.missing_embed.data().to_vec();
                encoded_vecs.extend_from_slice(&missing);
            } else {
                // Apply mask: zero out missing values (they're already 0, but be explicit)
                let masked_values: Vec<f32> = (0..sensor_count)
                    .map(|i| sensor_values[i] * snapshot.mask[base + i])
                    .collect();

                let input_tensor = Tensor::from_vec(masked_values, &[1, sensor_count]).unwrap();
                let input_var = Variable::new(input_tensor, false);

                let encoded = self.type_encoders[equip_type].forward(&input_var);
                let enc_data = encoded.data().to_vec();
                encoded_vecs.extend_from_slice(&enc_data);
            }
        }

        // Shape: [1, num_equipment, EMBED_DIM]
        let encoded_tensor = Tensor::from_vec(encoded_vecs, &[1, n, EMBED_DIM]).unwrap();
        let mut result = Variable::new(encoded_tensor, true);

        // Add type embeddings
        let type_indices: Vec<f32> = snapshot.equip_types.iter().map(|&t| t as f32).collect();
        let type_idx_tensor = Tensor::from_vec(type_indices, &[1, n]).unwrap();
        let type_idx_var = Variable::new(type_idx_tensor, false);
        let type_emb = self.type_embed.forward(&type_idx_var); // [1, n, EMBED_DIM]
        result = result.add_var(&type_emb);

        // Add ID embeddings
        let id_indices: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let id_idx_tensor = Tensor::from_vec(id_indices, &[1, n]).unwrap();
        let id_idx_var = Variable::new(id_idx_tensor, false);
        let id_emb = self.id_embed.forward(&id_idx_var); // [1, n, EMBED_DIM]
        result = result.add_var(&id_emb);

        result
    }

    /// Forward pass on a single snapshot (no temporal context).
    ///
    /// Returns per-equipment anomaly scores and a facility health score.
    ///
    /// # Returns
    /// `(equip_scores, facility_score)` where:
    /// - `equip_scores`: `[1, num_equipment]` — per-equipment anomaly score (higher = more anomalous)
    /// - `facility_score`: `[1, 1]` — overall facility health (higher = more anomalous)
    pub fn forward_snapshot(&self, snapshot: &FacilitySnapshot) -> (Variable, Variable) {
        // Encode: [1, num_equip, EMBED_DIM]
        let encoded = self.encode_snapshot(snapshot);

        // Cross-equipment attention: [1, num_equip, EMBED_DIM]
        let attended = self.cross_attn.forward(&encoded);

        // Per-equipment anomaly scores
        // Reshape to [num_equip, EMBED_DIM] for the linear head
        let flat = attended.reshape(&[self.num_equipment, EMBED_DIM]);
        let equip_scores = self.equip_head_snapshot.forward(&flat); // [num_equip, 1]
        let equip_scores = equip_scores.reshape(&[1, self.num_equipment]);

        // Facility-level: mean pool across equipment
        let pooled = attended.mean_dim(1, false); // [1, EMBED_DIM]
        let facility_score = self.facility_head_snapshot.forward(&pooled); // [1, 1]

        (equip_scores, facility_score)
    }

    /// Forward pass on a temporal sequence of snapshots.
    ///
    /// Processes a window of snapshots through the cross-equipment transformer
    /// first, then feeds the sequence through the temporal LSTM.
    ///
    /// # Arguments
    /// * `snapshots` - Sequence of facility snapshots (oldest first)
    ///
    /// # Returns
    /// `(equip_scores, facility_score)` based on the final timestep.
    pub fn forward_temporal(&self, snapshots: &[FacilitySnapshot]) -> (Variable, Variable) {
        let seq_len = snapshots.len();
        let n = self.num_equipment;

        // Encode each snapshot and apply cross-equipment attention
        let mut all_attended: Vec<f32> = Vec::with_capacity(seq_len * n * EMBED_DIM);
        for snap in snapshots {
            let encoded = self.encode_snapshot(snap);
            let attended = self.cross_attn.forward(&encoded); // [1, n, EMBED_DIM]
            all_attended.extend_from_slice(&attended.data().to_vec());
        }

        // For each equipment, run LSTM over the temporal sequence
        // Reshape: for each equipment slot, we have [seq_len, EMBED_DIM]
        // Process as batch of equipment: [num_equip, seq_len, EMBED_DIM]
        let mut lstm_input_data: Vec<f32> = vec![0.0; n * seq_len * EMBED_DIM];
        for t in 0..seq_len {
            for e in 0..n {
                for d in 0..EMBED_DIM {
                    let src = t * n * EMBED_DIM + e * EMBED_DIM + d;
                    let dst = e * seq_len * EMBED_DIM + t * EMBED_DIM + d;
                    lstm_input_data[dst] = all_attended[src];
                }
            }
        }

        let lstm_input = Variable::new(
            Tensor::from_vec(lstm_input_data, &[n, seq_len, EMBED_DIM]).unwrap(),
            true,
        );

        // LSTM forward: [num_equip, seq_len, 64]
        let lstm_out = self.temporal_lstm.forward(&lstm_input);

        // Take the last timestep: [num_equip, 64]
        let last = lstm_out.narrow(1, seq_len - 1, 1).reshape(&[n, 64]);

        // Per-equipment anomaly scores: [num_equip, 1] → [1, num_equip]
        let equip_scores = self.equip_head_temporal.forward(&last).reshape(&[1, n]);

        // Facility score: mean pool → [1, 64] → [1, 1]
        let pooled = last.reshape(&[1, n, 64]).mean_dim(1, false);
        let facility_score = self.facility_head_temporal.forward(&pooled);

        (equip_scores, facility_score)
    }

    /// Collects all trainable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();

        // Type encoders
        for encoder in &self.type_encoders {
            params.extend(encoder.parameters());
        }

        // Embeddings
        params.extend(self.type_embed.parameters());
        params.extend(self.id_embed.parameters());
        params.push(self.missing_embed.clone());

        // Cross-attention transformer
        params.extend(self.cross_attn.parameters());

        // Temporal LSTM
        params.extend(self.temporal_lstm.parameters());

        // Output heads
        params.extend(self.equip_head_snapshot.parameters());
        params.extend(self.equip_head_temporal.parameters());
        params.extend(self.facility_head_snapshot.parameters());
        params.extend(self.facility_head_temporal.parameters());

        params
    }
}

impl std::fmt::Debug for Panoptes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Panoptes")
            .field("num_equipment", &self.num_equipment)
            .field("embed_dim", &EMBED_DIM)
            .field("num_parameters", &self.num_parameters())
            .finish()
    }
}

// =============================================================================
// PanoptesOutput — Structured inference output
// =============================================================================

/// Structured output from Panoptes inference.
#[derive(Debug)]
pub struct PanoptesOutput {
    /// Per-equipment anomaly scores (0.0 = normal, higher = more anomalous).
    pub equipment_scores: Vec<(String, f32)>,
    /// Overall facility health score (0.0 = normal).
    pub facility_score: f32,
    /// Equipment flagged as anomalous (score > threshold).
    pub alerts: Vec<PanoptesAlert>,
}

/// An alert generated by Panoptes when equipment exceeds anomaly threshold.
#[derive(Debug)]
pub struct PanoptesAlert {
    /// Equipment ID string.
    pub equipment_id: String,
    /// Equipment type.
    pub equipment_type: usize,
    /// Anomaly score.
    pub score: f32,
    /// Severity: "low", "medium", "high", "critical"
    pub severity: &'static str,
}

impl PanoptesOutput {
    /// Creates a PanoptesOutput from raw model outputs and a facility config.
    pub fn from_scores(
        equip_scores: &[f32],
        facility_score: f32,
        config: &FacilityConfig,
        threshold: f32,
    ) -> Self {
        let mut equipment_scores = Vec::new();
        let mut alerts = Vec::new();

        for (i, &score) in equip_scores.iter().enumerate() {
            let (id, equip_type) = &config.equipment[i];
            equipment_scores.push((id.clone(), score));

            if score > threshold {
                let severity = if score > threshold * 4.0 {
                    "critical"
                } else if score > threshold * 2.5 {
                    "high"
                } else if score > threshold * 1.5 {
                    "medium"
                } else {
                    "low"
                };

                alerts.push(PanoptesAlert {
                    equipment_id: id.clone(),
                    equipment_type: *equip_type,
                    score,
                    severity,
                });
            }
        }

        // Sort alerts by score descending
        alerts.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Self {
            equipment_scores,
            facility_score,
            alerts,
        }
    }

    /// Pretty-print the output.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Panoptes Facility Health: {:.4} (0=normal)",
            self.facility_score
        ));

        if self.alerts.is_empty() {
            lines.push("  No anomalies detected.".to_string());
        } else {
            lines.push(format!("  {} alert(s):", self.alerts.len()));
            for alert in &self.alerts {
                let type_name = match alert.equipment_type {
                    EQUIP_AHU => "AHU",
                    EQUIP_DOAS => "DOAS",
                    EQUIP_BOILER => "Boiler",
                    EQUIP_STEAM_BUNDLE => "SteamBundle",
                    EQUIP_FAN_COIL => "FanCoil",
                    EQUIP_PUMP => "Pump",
                    EQUIP_CHILLER => "Chiller",
                    _ => "Unknown",
                };
                lines.push(format!(
                    "    [{:8}] {} — score: {:.4} ({})",
                    type_name, alert.equipment_id, alert.score, alert.severity
                ));
            }
        }

        lines.join("\n")
    }
}

// =============================================================================
// Helper: Parse live readings into FacilitySnapshot
// =============================================================================

/// Convenience function to create a Warren snapshot from individual readings.
///
/// Each equipment's readings are passed as `Option<f32>` slices matching
/// the sensor order defined for each type.
///
/// # AHU sensors (12):
/// setpoint, space_temp, supply_temp, mixed_temp, oa_damper_pct, ret_damper_pct,
/// hw_valve_pct, cw_valve_pct, amps, status(1=run/0=idle), fan_speed_pct, discharge_temp
///
/// # Boiler sensors (7):
/// supply_temp, flash_tank_temp, header_psi, status(1=run/0=idle),
/// lead_lag(0=lead/1=lag), runtime_hrs, safeties(1=ok/0=fault)
///
/// # SteamBundle sensors (5):
/// setpoint, supply_temp, return_temp, valve1_pct, valve2_pct
///
/// # FanCoil sensors (9):
/// setpoint, space_temp, supply_temp, hw_valve_pct, cw_valve_pct,
/// oa_damper_pct, amps, status(1=run/0=idle), fan_speed_pct
///
/// # Pump sensors (7):
/// speed_pct, amps, psi_setpoint, discharge_psi, runtime_hrs,
/// status(1=run/0=idle), flow_gpm
///
/// # Chiller sensors (9):
/// setpoint, supply_temp, return_temp, pressure_psi, amps,
/// status(1=run/0=idle), runtime_hrs, interlocks(1=ok/0=fault), enabled(1/0)
///
/// # DOAS sensors (6):
/// setpoint, supply_temp, space_temp, outdoor_temp, heat_status(1/0), cool_status(1/0)
impl FacilitySnapshot {
    /// Creates a Warren facility snapshot from the config.
    pub fn for_warren(config: &FacilityConfig) -> Self {
        Self::new(config.num_equipment())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_snapshot(n: usize) -> FacilitySnapshot {
        let mut snap = FacilitySnapshot::new(n);
        // Fill with some dummy data
        for i in 0..n {
            let equip_type = i % NUM_EQUIP_TYPES;
            let sensor_count = [
                AHU_SENSORS,
                DOAS_SENSORS,
                BOILER_SENSORS,
                STEAM_BUNDLE_SENSORS,
                FAN_COIL_SENSORS,
                PUMP_SENSORS,
                CHILLER_SENSORS,
            ][equip_type];

            let values: Vec<Option<f32>> = (0..sensor_count)
                .map(|j| Some((i as f32 * 0.1 + j as f32 * 0.3).sin() * 50.0 + 70.0))
                .collect();
            snap.set_equipment(i, &format!("test-equip-{i}"), equip_type, &values);
        }
        snap
    }

    #[test]
    fn test_panoptes_creation() {
        let model = Panoptes::new(59);
        println!("Panoptes parameters: {}", model.num_parameters());
        assert!(
            model.num_parameters() > 40_000,
            "got {}",
            model.num_parameters()
        );
        assert!(model.num_parameters() < 500_000);
    }

    #[test]
    fn test_facility_snapshot() {
        let config = FacilityConfig::warren();
        assert_eq!(config.num_equipment(), 59);
        assert_eq!(config.id_to_slot["warren-ahu-6"], 0);
        assert_eq!(config.id_to_slot["warren-chiller-1"], 58);
    }

    #[test]
    fn test_snapshot_encoding() {
        let n = 7; // One of each type
        let model = Panoptes::new(n);
        let snap = make_test_snapshot(n);
        let encoded = model.encode_snapshot(&snap);
        assert_eq!(encoded.shape(), vec![1, n, EMBED_DIM]);
    }

    #[test]
    fn test_forward_snapshot() {
        let n = 7;
        let model = Panoptes::new(n);
        let snap = make_test_snapshot(n);
        let (equip_scores, facility_score) = model.forward_snapshot(&snap);
        assert_eq!(equip_scores.shape(), vec![1, n]);
        assert_eq!(facility_score.shape(), vec![1, 1]);
    }

    #[test]
    fn test_forward_temporal() {
        let n = 7;
        let model = Panoptes::new(n);

        // Create a sequence of 5 snapshots
        let snapshots: Vec<FacilitySnapshot> = (0..5).map(|_| make_test_snapshot(n)).collect();

        let (equip_scores, facility_score) = model.forward_temporal(&snapshots);
        assert_eq!(equip_scores.shape(), vec![1, n]);
        assert_eq!(facility_score.shape(), vec![1, 1]);
    }

    #[test]
    fn test_missing_values() {
        let n = 3;
        let model = Panoptes::new(n);

        let mut snap = FacilitySnapshot::new(n);
        // Equipment with all missing values
        snap.set_equipment(0, "test-missing", EQUIP_AHU, &vec![None; AHU_SENSORS]);
        // Equipment with partial values
        snap.set_equipment(
            1,
            "test-partial",
            EQUIP_BOILER,
            &[
                Some(127.0),
                None,
                Some(88.5),
                Some(1.0),
                None,
                Some(17.0),
                Some(1.0),
            ],
        );
        // Equipment with all values
        snap.set_equipment(
            2,
            "test-full",
            EQUIP_FAN_COIL,
            &[
                Some(73.5),
                Some(74.5),
                Some(71.4),
                Some(0.0),
                Some(0.0),
                Some(100.0),
                Some(2.2),
                Some(1.0),
                Some(50.0),
            ],
        );

        let (equip_scores, _) = model.forward_snapshot(&snap);
        assert_eq!(equip_scores.shape(), vec![1, n]);
    }

    #[test]
    fn test_warren_config() {
        let config = FacilityConfig::warren();

        // Verify counts
        let ahu_count = config
            .equipment
            .iter()
            .filter(|(_, t)| *t == EQUIP_AHU)
            .count();
        let boiler_count = config
            .equipment
            .iter()
            .filter(|(_, t)| *t == EQUIP_BOILER)
            .count();
        let bundle_count = config
            .equipment
            .iter()
            .filter(|(_, t)| *t == EQUIP_STEAM_BUNDLE)
            .count();
        let fc_count = config
            .equipment
            .iter()
            .filter(|(_, t)| *t == EQUIP_FAN_COIL)
            .count();
        let pump_count = config
            .equipment
            .iter()
            .filter(|(_, t)| *t == EQUIP_PUMP)
            .count();
        let chiller_count = config
            .equipment
            .iter()
            .filter(|(_, t)| *t == EQUIP_CHILLER)
            .count();
        let doas_count = config
            .equipment
            .iter()
            .filter(|(_, t)| *t == EQUIP_DOAS)
            .count();

        assert_eq!(ahu_count, 6);
        assert_eq!(boiler_count, 3);
        assert_eq!(bundle_count, 9);
        assert_eq!(fc_count, 18);
        assert_eq!(pump_count, 20);
        assert_eq!(chiller_count, 2);
        assert_eq!(doas_count, 1);
        assert_eq!(config.num_equipment(), 59);
    }

    #[test]
    fn test_panoptes_output() {
        let config = FacilityConfig::warren();
        let scores = vec![0.1; 59];
        let output = PanoptesOutput::from_scores(&scores, 0.05, &config, 0.5);
        assert!(output.alerts.is_empty());
        assert_eq!(output.equipment_scores.len(), 59);

        // Test with anomalies
        let mut scores2 = vec![0.1; 59];
        scores2[0] = 2.5; // AHU-6 anomaly
        scores2[10] = 1.8; // Steam bundle anomaly
        let output2 = PanoptesOutput::from_scores(&scores2, 1.2, &config, 0.5);
        assert_eq!(output2.alerts.len(), 2);
        assert_eq!(output2.alerts[0].equipment_id, "warren-ahu-6");
    }

    #[test]
    fn test_panoptes_gradient_flow() {
        use axonml_optim::Optimizer;
        let n = 7;
        let model = Panoptes::new(n);
        let snap = make_test_snapshot(n);

        let mse = axonml_nn::MSELoss::new();
        // Target: all zeros (normal = 0 anomaly score)
        let target = Variable::new(Tensor::from_vec(vec![0.0; n], &[1, n]).unwrap(), false);

        let params = model.parameters();
        let mut optimizer = axonml_optim::Adam::new(params, 1e-3);

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for step in 0..10 {
            optimizer.zero_grad();
            let (equip_scores, _) = model.forward_snapshot(&snap);
            let loss = mse.compute(&equip_scores, &target);
            let loss_val = loss.data().to_vec()[0];

            if step == 0 {
                first_loss = loss_val;
            }
            last_loss = loss_val;

            loss.backward();
            optimizer.step();
        }

        println!("Panoptes training: {:.6} → {:.6}", first_loss, last_loss);
        assert!(
            last_loss < first_loss,
            "Loss did not decrease: {} → {}",
            first_loss,
            last_loss
        );
    }
}
