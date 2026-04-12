//! HVAC Sensor Data Pipeline
//!
//! # File
//! `crates/axonml-hvac/src/data.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_autograd::Variable;
use axonml_tensor::Tensor;
use rand::Rng;

// =============================================================================
// Sensor Data Structures
// =============================================================================

/// Raw sensor readings from HVAC equipment.
///
/// Each field contains a batch of time-series data for a specific subsystem.
/// Sensor channels per subsystem (7 each, matching boardconfig):
/// - Electrical: voltage_a/b/c, current_a/b/c, power_factor
/// - Refrigeration: suction_pressure, discharge_pressure, suction_temp, discharge_temp, subcool, superheat, flow_rate
/// - Water: supply_temp, return_temp, flow_rate, pressure_diff, valve_pos, pump_current, ph
/// - Mechanical: vibration_x/y/z, bearing_temp, motor_current, speed_rpm, torque
/// - Airflow: supply_temp, return_temp, outdoor_temp, mixed_temp, fan_amps, oa_damper, mixed_damper
pub struct HvacSensorData {
    /// Electrical subsystem: (batch, 64, 7)
    pub electrical: Variable,
    /// Refrigeration subsystem: (batch, 80, 7)
    pub refrigeration: Variable,
    /// Water subsystem: (batch, 64, 7)
    pub water: Variable,
    /// Mechanical subsystem: (batch, 96, 7)
    pub mechanical: Variable,
    /// Airflow subsystem: (batch, 72, 7)
    pub airflow: Variable,
}

/// Labels for HVAC fault diagnosis.
pub struct HvacLabels {
    // Specialist labels
    /// Electrical fault class (0-12)
    pub electrical_fault: Variable,
    /// Electrical severity (0-4)
    pub electrical_severity: Variable,
    /// Refrigeration fault class (0-15)
    pub refrigeration_fault: Variable,
    /// Water system fault class (0-10)
    pub water_fault: Variable,
    /// Mechanical fault class (0-14)
    pub mechanical_fault: Variable,
    /// Airflow fault class (0-11)
    pub airflow_fault: Variable,
    // System-level labels
    /// Overall system fault class (0-23)
    pub system_fault: Variable,
    /// Safety validation state (0-4)
    pub safety_state: Variable,
    /// Master diagnosis class (0-11)
    pub diagnosis: Variable,
}

/// Output from a single specialist model.
pub struct SpecialistOutput {
    /// Primary fault classification logits
    pub fault_logits: Variable,
    /// Additional head outputs (severity, health scores, etc.)
    pub aux_outputs: Vec<Variable>,
    /// Feature embedding for downstream aggregators
    pub embedding: Variable,
}

/// Combined output from all specialists.
pub struct PipelineOutput {
    /// Specialist embeddings concatenated
    pub specialist_features: Variable,
    /// Colossus aggregator output
    pub aggregator_output: Variable,
    /// Gaia safety validation output
    pub safety_output: Variable,
    /// Apollo final diagnosis
    pub diagnosis: Variable,
    /// Per-specialist fault classifications
    pub specialist_faults: Vec<Variable>,
}

// =============================================================================
// Sensor Ranges (from boardconfig.json - AHU-6 Heritage Warren A-Wing)
// =============================================================================

/// Temperature sensor ranges (10K NTC thermistors, °F)
pub const TEMP_MIN: f32 = -40.0;
/// Maximum temperature sensor range (°F).
pub const TEMP_MAX: f32 = 250.0;
/// Normal operating temperatures
pub const SUPPLY_TEMP_NORMAL: (f32, f32) = (52.0, 58.0);
/// Return air temperature normal range (°F).
pub const RETURN_TEMP_NORMAL: (f32, f32) = (70.0, 78.0);
/// Outdoor temperature operating range (°F).
pub const OUTDOOR_TEMP_RANGE: (f32, f32) = (-10.0, 110.0);
/// Mixed air temperature normal range (°F).
pub const MIXED_TEMP_NORMAL: (f32, f32) = (55.0, 72.0);

/// Analog output ranges (0-10V)
pub const VALVE_MIN: f32 = 0.0;
/// Maximum valve position (V).
pub const VALVE_MAX: f32 = 10.0;
/// Maximum fan current draw (A).
pub const FAN_AMPS_MAX: f32 = 50.0;
/// Maximum pump current draw (A).
pub const PUMP_CURRENT_MAX: f32 = 50.0;

// =============================================================================
// Fault Types
// =============================================================================

/// Electrical fault categories.
pub const ELECTRICAL_FAULTS: [&str; 13] = [
    "normal",
    "phase_imbalance",
    "overvoltage",
    "undervoltage",
    "overcurrent",
    "ground_fault",
    "harmonic_distortion",
    "power_factor_low",
    "single_phasing",
    "voltage_sag",
    "voltage_swell",
    "transient",
    "insulation_breakdown",
];

/// Refrigeration fault categories.
pub const REFRIGERATION_FAULTS: [&str; 16] = [
    "normal",
    "low_charge",
    "overcharge",
    "compressor_valve_leak",
    "condenser_fouling",
    "evaporator_fouling",
    "txv_malfunction",
    "liquid_line_restriction",
    "non_condensable_gas",
    "oil_logging",
    "compressor_mechanical",
    "high_head_pressure",
    "low_suction",
    "superheat_high",
    "subcool_low",
    "refrigerant_migration",
];

/// Water system fault categories.
pub const WATER_FAULTS: [&str; 11] = [
    "normal",
    "air_in_system",
    "pump_cavitation",
    "valve_stuck",
    "flow_restriction",
    "leak",
    "scale_buildup",
    "glycol_degradation",
    "pump_bearing_wear",
    "strainer_clog",
    "heat_exchanger_fouling",
];

/// Mechanical fault categories.
pub const MECHANICAL_FAULTS: [&str; 15] = [
    "normal",
    "bearing_inner_race",
    "bearing_outer_race",
    "bearing_ball",
    "misalignment_angular",
    "misalignment_parallel",
    "imbalance",
    "looseness_structural",
    "looseness_rotating",
    "belt_wear",
    "belt_misalignment",
    "coupling_wear",
    "gear_mesh",
    "resonance",
    "shaft_crack",
];

/// Airflow fault categories.
pub const AIRFLOW_FAULTS: [&str; 12] = [
    "normal",
    "filter_loading",
    "damper_stuck_open",
    "damper_stuck_closed",
    "duct_leak",
    "fan_belt_slip",
    "fan_bearing_wear",
    "coil_blockage",
    "sensor_drift",
    "economizer_malfunction",
    "vav_hunting",
    "static_pressure_high",
];

// =============================================================================
// Synthetic Data Generation
// =============================================================================

/// Generates synthetic HVAC sensor data for training and testing.
pub struct SyntheticHvacGenerator {
    _rng_seed: u64,
}

impl SyntheticHvacGenerator {
    /// Creates a new generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { _rng_seed: seed }
    }

    /// Generates a batch of normal operating data.
    pub fn generate_normal(&self, batch_size: usize) -> (HvacSensorData, HvacLabels) {
        let mut rng = rand::thread_rng();

        let data = HvacSensorData {
            electrical: self.gen_electrical_normal(&mut rng, batch_size),
            refrigeration: self.gen_refrigeration_normal(&mut rng, batch_size),
            water: self.gen_water_normal(&mut rng, batch_size),
            mechanical: self.gen_mechanical_normal(&mut rng, batch_size),
            airflow: self.gen_airflow_normal(&mut rng, batch_size),
        };

        let labels = HvacLabels {
            electrical_fault: Variable::new(
                Tensor::from_vec(vec![0.0; batch_size], &[batch_size]).unwrap(),
                false,
            ),
            electrical_severity: Variable::new(
                Tensor::from_vec(vec![0.0; batch_size], &[batch_size]).unwrap(),
                false,
            ),
            refrigeration_fault: Variable::new(
                Tensor::from_vec(vec![0.0; batch_size], &[batch_size]).unwrap(),
                false,
            ),
            water_fault: Variable::new(
                Tensor::from_vec(vec![0.0; batch_size], &[batch_size]).unwrap(),
                false,
            ),
            mechanical_fault: Variable::new(
                Tensor::from_vec(vec![0.0; batch_size], &[batch_size]).unwrap(),
                false,
            ),
            airflow_fault: Variable::new(
                Tensor::from_vec(vec![0.0; batch_size], &[batch_size]).unwrap(),
                false,
            ),
            system_fault: Variable::new(
                Tensor::from_vec(vec![0.0; batch_size], &[batch_size]).unwrap(),
                false,
            ),
            safety_state: Variable::new(
                Tensor::from_vec(vec![0.0; batch_size], &[batch_size]).unwrap(),
                false,
            ),
            diagnosis: Variable::new(
                Tensor::from_vec(vec![0.0; batch_size], &[batch_size]).unwrap(),
                false,
            ),
        };

        (data, labels)
    }

    /// Generates a batch with injected faults for training.
    pub fn generate_with_faults(&self, batch_size: usize) -> (HvacSensorData, HvacLabels) {
        let mut rng = rand::thread_rng();

        // Start from normal data
        let mut electrical = self.gen_electrical_normal_raw(&mut rng, batch_size);
        let mut refrigeration = self.gen_refrigeration_normal_raw(&mut rng, batch_size);
        let mut water = self.gen_water_normal_raw(&mut rng, batch_size);
        let mut mechanical = self.gen_mechanical_normal_raw(&mut rng, batch_size);
        let mut airflow = self.gen_airflow_normal_raw(&mut rng, batch_size);

        let mut elec_faults = vec![0.0f32; batch_size];
        let mut elec_severity = vec![0.0f32; batch_size];
        let mut refrig_faults = vec![0.0f32; batch_size];
        let mut water_faults = vec![0.0f32; batch_size];
        let mut mech_faults = vec![0.0f32; batch_size];
        let mut air_faults = vec![0.0f32; batch_size];
        let mut sys_faults = vec![0.0f32; batch_size];
        let mut safety = vec![0.0f32; batch_size];
        let mut diag = vec![0.0f32; batch_size];

        // Inject faults into ~60% of samples
        for b in 0..batch_size {
            if rng.r#gen::<f32>() < 0.6 {
                let fault_type = rng.gen_range(0..5);
                match fault_type {
                    0 => {
                        // Electrical fault
                        let fault_id = rng.gen_range(1..ELECTRICAL_FAULTS.len());
                        self.inject_electrical_fault(&mut rng, &mut electrical, b, fault_id);
                        elec_faults[b] = fault_id as f32;
                        elec_severity[b] = rng.gen_range(1..5) as f32;
                        sys_faults[b] = fault_id as f32;
                        diag[b] = 1.0;
                    }
                    1 => {
                        // Refrigeration fault
                        let fault_id = rng.gen_range(1..REFRIGERATION_FAULTS.len());
                        self.inject_refrigeration_fault(&mut rng, &mut refrigeration, b, fault_id);
                        refrig_faults[b] = fault_id as f32;
                        sys_faults[b] = (13 + fault_id % 11) as f32;
                        diag[b] = 2.0;
                    }
                    2 => {
                        // Water fault
                        let fault_id = rng.gen_range(1..WATER_FAULTS.len());
                        self.inject_water_fault(&mut rng, &mut water, b, fault_id);
                        water_faults[b] = fault_id as f32;
                        diag[b] = 3.0;
                    }
                    3 => {
                        // Mechanical fault
                        let fault_id = rng.gen_range(1..MECHANICAL_FAULTS.len());
                        self.inject_mechanical_fault(&mut rng, &mut mechanical, b, fault_id);
                        mech_faults[b] = fault_id as f32;
                        diag[b] = 4.0;
                    }
                    4 => {
                        // Airflow fault
                        let fault_id = rng.gen_range(1..AIRFLOW_FAULTS.len());
                        self.inject_airflow_fault(&mut rng, &mut airflow, b, fault_id);
                        air_faults[b] = fault_id as f32;
                        diag[b] = 5.0;
                    }
                    _ => {}
                }

                // Safety escalation for severe faults
                if elec_severity[b] >= 4.0 || mech_faults[b] >= 10.0 {
                    safety[b] = rng.gen_range(2..5) as f32;
                }
            }
        }

        let data = HvacSensorData {
            electrical: Variable::new(
                Tensor::from_vec(electrical, &[batch_size, 64, 7]).unwrap(),
                false,
            ),
            refrigeration: Variable::new(
                Tensor::from_vec(refrigeration, &[batch_size, 80, 7]).unwrap(),
                false,
            ),
            water: Variable::new(
                Tensor::from_vec(water, &[batch_size, 64, 7]).unwrap(),
                false,
            ),
            mechanical: Variable::new(
                Tensor::from_vec(mechanical, &[batch_size, 96, 7]).unwrap(),
                false,
            ),
            airflow: Variable::new(
                Tensor::from_vec(airflow, &[batch_size, 72, 7]).unwrap(),
                false,
            ),
        };

        let labels = HvacLabels {
            electrical_fault: Variable::new(
                Tensor::from_vec(elec_faults, &[batch_size]).unwrap(),
                false,
            ),
            electrical_severity: Variable::new(
                Tensor::from_vec(elec_severity, &[batch_size]).unwrap(),
                false,
            ),
            refrigeration_fault: Variable::new(
                Tensor::from_vec(refrig_faults, &[batch_size]).unwrap(),
                false,
            ),
            water_fault: Variable::new(
                Tensor::from_vec(water_faults, &[batch_size]).unwrap(),
                false,
            ),
            mechanical_fault: Variable::new(
                Tensor::from_vec(mech_faults, &[batch_size]).unwrap(),
                false,
            ),
            airflow_fault: Variable::new(
                Tensor::from_vec(air_faults, &[batch_size]).unwrap(),
                false,
            ),
            system_fault: Variable::new(
                Tensor::from_vec(sys_faults, &[batch_size]).unwrap(),
                false,
            ),
            safety_state: Variable::new(Tensor::from_vec(safety, &[batch_size]).unwrap(), false),
            diagnosis: Variable::new(Tensor::from_vec(diag, &[batch_size]).unwrap(), false),
        };

        (data, labels)
    }

    // =========================================================================
    // Normal Data Generation (raw Vec<f32>)
    // =========================================================================

    fn gen_electrical_normal(&self, rng: &mut impl Rng, batch: usize) -> Variable {
        let data = self.gen_electrical_normal_raw(rng, batch);
        Variable::new(Tensor::from_vec(data, &[batch, 64, 7]).unwrap(), false)
    }

    fn gen_electrical_normal_raw(&self, rng: &mut impl Rng, batch: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(batch * 64 * 7);
        for _ in 0..batch {
            for t in 0..64 {
                let phase = t as f32 * 0.1;
                // voltage_a/b/c (around 480V, 120° phase shifted)
                data.push(480.0 + 5.0 * (phase).sin() + rng.r#gen::<f32>() * 2.0);
                data.push(480.0 + 5.0 * (phase + 2.094).sin() + rng.r#gen::<f32>() * 2.0);
                data.push(480.0 + 5.0 * (phase + 4.189).sin() + rng.r#gen::<f32>() * 2.0);
                // current_a/b/c (balanced ~15A)
                data.push(15.0 + rng.r#gen::<f32>() * 1.0);
                data.push(15.0 + rng.r#gen::<f32>() * 1.0);
                data.push(15.0 + rng.r#gen::<f32>() * 1.0);
                // power_factor (0.85-0.95)
                data.push(0.90 + rng.r#gen::<f32>() * 0.05);
            }
        }
        data
    }

    fn gen_refrigeration_normal(&self, rng: &mut impl Rng, batch: usize) -> Variable {
        let data = self.gen_refrigeration_normal_raw(rng, batch);
        Variable::new(Tensor::from_vec(data, &[batch, 80, 7]).unwrap(), false)
    }

    fn gen_refrigeration_normal_raw(&self, rng: &mut impl Rng, batch: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(batch * 80 * 7);
        for _ in 0..batch {
            for _ in 0..80 {
                data.push(65.0 + rng.r#gen::<f32>() * 5.0); // suction_pressure (psig)
                data.push(220.0 + rng.r#gen::<f32>() * 10.0); // discharge_pressure
                data.push(40.0 + rng.r#gen::<f32>() * 3.0); // suction_temp (°F)
                data.push(160.0 + rng.r#gen::<f32>() * 5.0); // discharge_temp
                data.push(10.0 + rng.r#gen::<f32>() * 3.0); // subcool
                data.push(12.0 + rng.r#gen::<f32>() * 3.0); // superheat
                data.push(8.0 + rng.r#gen::<f32>() * 1.0); // flow_rate (GPM)
            }
        }
        data
    }

    fn gen_water_normal(&self, rng: &mut impl Rng, batch: usize) -> Variable {
        let data = self.gen_water_normal_raw(rng, batch);
        Variable::new(Tensor::from_vec(data, &[batch, 64, 7]).unwrap(), false)
    }

    fn gen_water_normal_raw(&self, rng: &mut impl Rng, batch: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(batch * 64 * 7);
        for _ in 0..batch {
            for _ in 0..64 {
                data.push(44.0 + rng.r#gen::<f32>() * 2.0); // supply_temp
                data.push(54.0 + rng.r#gen::<f32>() * 2.0); // return_temp
                data.push(120.0 + rng.r#gen::<f32>() * 10.0); // flow_rate (GPM)
                data.push(12.0 + rng.r#gen::<f32>() * 2.0); // pressure_diff (psi)
                data.push(5.0 + rng.r#gen::<f32>() * 2.0); // valve_pos (V)
                data.push(8.0 + rng.r#gen::<f32>() * 1.0); // pump_current (A)
                data.push(7.0 + rng.r#gen::<f32>() * 0.5); // pH
            }
        }
        data
    }

    fn gen_mechanical_normal(&self, rng: &mut impl Rng, batch: usize) -> Variable {
        let data = self.gen_mechanical_normal_raw(rng, batch);
        Variable::new(Tensor::from_vec(data, &[batch, 96, 7]).unwrap(), false)
    }

    fn gen_mechanical_normal_raw(&self, rng: &mut impl Rng, batch: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(batch * 96 * 7);
        for _ in 0..batch {
            for t in 0..96 {
                let freq = t as f32 * 0.5;
                // vibration_x/y/z (low amplitude, mm/s)
                data.push(0.5 * freq.sin() + rng.r#gen::<f32>() * 0.2);
                data.push(0.3 * (freq * 1.1).sin() + rng.r#gen::<f32>() * 0.2);
                data.push(0.2 * (freq * 0.9).cos() + rng.r#gen::<f32>() * 0.1);
                // bearing_temp (°F, normal 120-150)
                data.push(135.0 + rng.r#gen::<f32>() * 10.0);
                // motor_current (A)
                data.push(12.0 + rng.r#gen::<f32>() * 1.0);
                // speed_rpm
                data.push(1770.0 + rng.r#gen::<f32>() * 10.0);
                // torque (Nm)
                data.push(45.0 + rng.r#gen::<f32>() * 3.0);
            }
        }
        data
    }

    fn gen_airflow_normal(&self, rng: &mut impl Rng, batch: usize) -> Variable {
        let data = self.gen_airflow_normal_raw(rng, batch);
        Variable::new(Tensor::from_vec(data, &[batch, 72, 7]).unwrap(), false)
    }

    fn gen_airflow_normal_raw(&self, rng: &mut impl Rng, batch: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(batch * 72 * 7);
        for _ in 0..batch {
            let outdoor = 75.0 + rng.r#gen::<f32>() * 20.0 - 10.0;
            for _ in 0..72 {
                data.push(55.0 + rng.r#gen::<f32>() * 2.0); // supply_temp
                data.push(74.0 + rng.r#gen::<f32>() * 2.0); // return_temp
                data.push(outdoor + rng.r#gen::<f32>() * 1.0); // outdoor_temp
                data.push(65.0 + rng.r#gen::<f32>() * 3.0); // mixed_temp
                data.push(18.0 + rng.r#gen::<f32>() * 2.0); // fan_amps
                data.push(3.0 + rng.r#gen::<f32>() * 2.0); // oa_damper (V)
                data.push(5.0 + rng.r#gen::<f32>() * 2.0); // mixed_damper (V)
            }
        }
        data
    }

    // =========================================================================
    // Fault Injection
    // =========================================================================

    fn inject_electrical_fault(
        &self,
        rng: &mut impl Rng,
        data: &mut [f32],
        batch_idx: usize,
        fault_id: usize,
    ) {
        let offset = batch_idx * 64 * 7;
        let severity = rng.r#gen::<f32>() * 0.5 + 0.5; // 0.5-1.0
        for t in 0..64 {
            let idx = offset + t * 7;
            match fault_id {
                1 => {
                    // phase_imbalance
                    data[idx] *= 1.0 + severity * 0.15;
                    data[idx + 1] *= 1.0 - severity * 0.10;
                }
                2 => data[idx] *= 1.0 + severity * 0.12, // overvoltage
                3 => data[idx] *= 1.0 - severity * 0.15, // undervoltage
                4 => data[idx + 3] *= 1.0 + severity * 0.5, // overcurrent
                5 => {
                    // ground_fault - current spike on one phase
                    data[idx + 3] += severity * 20.0;
                }
                6 => {
                    // harmonic_distortion
                    let harmonic = (t as f32 * 0.3 * 3.0).sin() * severity * 15.0;
                    data[idx] += harmonic;
                }
                7 => data[idx + 6] -= severity * 0.15, // power_factor_low
                _ => {
                    // generic fault: add noise
                    for c in 0..7 {
                        data[idx + c] += rng.r#gen::<f32>() * severity * 10.0;
                    }
                }
            }
        }
    }

    fn inject_refrigeration_fault(
        &self,
        rng: &mut impl Rng,
        data: &mut [f32],
        batch_idx: usize,
        fault_id: usize,
    ) {
        let offset = batch_idx * 80 * 7;
        let severity = rng.r#gen::<f32>() * 0.5 + 0.5;
        for t in 0..80 {
            let idx = offset + t * 7;
            match fault_id {
                1 => {
                    // low_charge
                    data[idx] -= severity * 15.0; // low suction pressure
                    data[idx + 5] += severity * 8.0; // high superheat
                }
                2 => {
                    // overcharge
                    data[idx] += severity * 10.0;
                    data[idx + 4] += severity * 5.0; // high subcool
                }
                3 => {
                    // compressor_valve_leak
                    data[idx + 1] -= severity * 20.0; // low discharge pressure
                }
                _ => {
                    for c in 0..7 {
                        data[idx + c] += rng.r#gen::<f32>() * severity * 5.0;
                    }
                }
            }
        }
    }

    fn inject_water_fault(
        &self,
        rng: &mut impl Rng,
        data: &mut [f32],
        batch_idx: usize,
        fault_id: usize,
    ) {
        let offset = batch_idx * 64 * 7;
        let severity = rng.r#gen::<f32>() * 0.5 + 0.5;
        for t in 0..64 {
            let idx = offset + t * 7;
            match fault_id {
                1 => data[idx + 3] += severity * 5.0, // air_in_system - pressure fluctuation
                2 => {
                    // pump_cavitation
                    data[idx + 2] -= severity * 30.0; // reduced flow
                    data[idx + 5] += severity * 3.0; // pump current spike
                }
                3 => data[idx + 4] = if severity > 0.7 { 0.0 } else { 10.0 }, // valve_stuck
                _ => {
                    for c in 0..7 {
                        data[idx + c] += rng.r#gen::<f32>() * severity * 3.0;
                    }
                }
            }
        }
    }

    fn inject_mechanical_fault(
        &self,
        rng: &mut impl Rng,
        data: &mut [f32],
        batch_idx: usize,
        fault_id: usize,
    ) {
        let offset = batch_idx * 96 * 7;
        let severity = rng.r#gen::<f32>() * 0.5 + 0.5;
        for t in 0..96 {
            let idx = offset + t * 7;
            match fault_id {
                1..=3 => {
                    // bearing faults - increased vibration
                    data[idx] += severity * 3.0 * (t as f32 * 0.8).sin();
                    data[idx + 1] += severity * 2.0;
                    data[idx + 3] += severity * 20.0; // bearing temp rise
                }
                4 | 5 => {
                    // misalignment
                    data[idx] += severity * 2.0 * (t as f32 * 1.0).sin();
                    data[idx + 2] += severity * 1.5;
                }
                6 => {
                    // imbalance
                    let amp = severity * 4.0;
                    data[idx] += amp * (t as f32 * 0.5).sin();
                    data[idx + 1] += amp * (t as f32 * 0.5).cos();
                }
                _ => {
                    for c in 0..3 {
                        data[idx + c] += rng.r#gen::<f32>() * severity * 2.0;
                    }
                }
            }
        }
    }

    fn inject_airflow_fault(
        &self,
        rng: &mut impl Rng,
        data: &mut [f32],
        batch_idx: usize,
        fault_id: usize,
    ) {
        let offset = batch_idx * 72 * 7;
        let severity = rng.r#gen::<f32>() * 0.5 + 0.5;
        for t in 0..72 {
            let idx = offset + t * 7;
            match fault_id {
                1 => {
                    // filter_loading
                    data[idx + 4] += severity * 5.0; // fan amps increase
                    data[idx] += severity * 3.0; // supply temp drift
                }
                2 => data[idx + 5] = VALVE_MAX, // damper_stuck_open
                3 => data[idx + 5] = VALVE_MIN, // damper_stuck_closed
                4 => {
                    // duct_leak
                    data[idx] += severity * 4.0; // supply temp rise
                    data[idx + 4] += severity * 2.0;
                }
                _ => {
                    for c in 0..7 {
                        data[idx + c] += rng.r#gen::<f32>() * severity * 2.0;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_normal_shapes() {
        let generator = SyntheticHvacGenerator::new(42);
        let (data, labels) = generator.generate_normal(4);

        assert_eq!(data.electrical.shape(), vec![4, 64, 7]);
        assert_eq!(data.refrigeration.shape(), vec![4, 80, 7]);
        assert_eq!(data.water.shape(), vec![4, 64, 7]);
        assert_eq!(data.mechanical.shape(), vec![4, 96, 7]);
        assert_eq!(data.airflow.shape(), vec![4, 72, 7]);

        assert_eq!(labels.electrical_fault.shape(), vec![4]);
        assert_eq!(labels.diagnosis.shape(), vec![4]);
    }

    #[test]
    fn test_generate_normal_all_zeros_labels() {
        let generator = SyntheticHvacGenerator::new(42);
        let (_, labels) = generator.generate_normal(8);

        // Normal data should have all-zero fault labels
        let faults = labels.electrical_fault.data().to_vec();
        assert!(faults.iter().all(|&f| f == 0.0));
    }

    #[test]
    fn test_generate_with_faults_shapes() {
        let generator = SyntheticHvacGenerator::new(42);
        let (data, labels) = generator.generate_with_faults(16);

        assert_eq!(data.electrical.shape(), vec![16, 64, 7]);
        assert_eq!(data.refrigeration.shape(), vec![16, 80, 7]);
        assert_eq!(data.water.shape(), vec![16, 64, 7]);
        assert_eq!(data.mechanical.shape(), vec![16, 96, 7]);
        assert_eq!(data.airflow.shape(), vec![16, 72, 7]);

        assert_eq!(labels.system_fault.shape(), vec![16]);
    }

    #[test]
    fn test_generate_with_faults_has_faults() {
        let generator = SyntheticHvacGenerator::new(42);
        let (_, labels) = generator.generate_with_faults(100);

        // With 100 samples and 60% fault rate, we should have some faults
        let diag = labels.diagnosis.data().to_vec();
        let num_faults = diag.iter().filter(|&&d| d > 0.0).count();
        assert!(
            num_faults > 20,
            "Expected >20 faults in 100 samples, got {}",
            num_faults
        );
    }

    #[test]
    fn test_sensor_data_values_reasonable() {
        let generator = SyntheticHvacGenerator::new(42);
        let (data, _) = generator.generate_normal(2);

        // Check electrical voltages are around 480V
        let elec = data.electrical.data().to_vec();
        let first_voltage = elec[0]; // First sample, first timestep, voltage_a
        assert!(
            first_voltage > 400.0 && first_voltage < 600.0,
            "Voltage {} out of expected range",
            first_voltage
        );
    }

    #[test]
    fn test_fault_categories() {
        assert_eq!(ELECTRICAL_FAULTS.len(), 13);
        assert_eq!(REFRIGERATION_FAULTS.len(), 16);
        assert_eq!(WATER_FAULTS.len(), 11);
        assert_eq!(MECHANICAL_FAULTS.len(), 15);
        assert_eq!(AIRFLOW_FAULTS.len(), 12);
    }
}
