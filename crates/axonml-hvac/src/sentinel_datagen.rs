//! Sentinel Synthetic Data Generator — Per-Equipment-Type HVAC Fault Simulation
//!
//! Generates training data for Sentinel models. Each equipment type has its own
//! sensor profile, normal operating envelope, and fault injection patterns.
//! Output includes anomaly labels and predictive failure labels at 6 horizons.
//!
//! Copyright (c) 2026 Andrew Jewell Sr. / AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060

use rand::Rng;
use rand::SeedableRng;
use super::sentinel::{EquipmentType, HORIZON_NAMES, TIMESTEPS};
use super::sentinel_datagen_ext;

/// A single training sample: 8 timesteps of sensor readings + labels.
pub struct SentinelSample {
    pub data: Vec<f32>,       // (TIMESTEPS × n_features) flattened
    pub labels: [f32; 7],     // [anomaly, fail_30m, fail_4h, fail_12h, fail_3d, fail_7d, fail_14d]
    pub fault_name: String,
}

/// Fault scenario with degradation timeline.
#[derive(Clone)]
pub struct FaultScenario {
    pub name: String,
    pub affected_sensors: Vec<usize>,
    pub severity: f32,
    pub onset_hours: f32,
}

pub struct SentinelDatagen {
    equipment_type: EquipmentType,
    seed: u64,
}

impl SentinelDatagen {
    pub fn new(equipment_type: EquipmentType, seed: u64) -> Self {
        Self { equipment_type, seed }
    }

    fn normal_ranges(&self) -> Vec<(f32, f32)> {
        match self.equipment_type {
            EquipmentType::Boiler => vec![
                (100.0, 200.0),  // loop_supply_t
                (80.0, 180.0),   // loop_return_t
                (100.0, 200.0),  // setpoint
                (0.0, 1.0),      // enable_signal (0=off, 1=enabled)
                (0.0, 15.0),     // amps
                (0.0, 1.0),      // flow_switch (0=no flow, 1=flow proven)
                (0.0, 1.0),      // flame_proven (0=no flame, 1=flame)
                (0.0, 1.0),      // ignition_status (0=fail, 1=ok)
                (-10.0, 100.0),  // oat
                (0.0, 300.0),    // run_timer_sec (time since last state change)
            ],
            EquipmentType::Chiller => vec![
                (50.0, 56.0),    // evap_in
                (42.0, 46.0),    // evap_out
                (80.0, 95.0),    // cond_in
                (90.0, 105.0),   // cond_out
                (30.0, 120.0),   // comp_amps
                (100.0, 600.0),  // cw_flow
                (100.0, 500.0),  // chw_flow
                (55.0, 75.0),    // suction_p
                (180.0, 300.0),  // discharge_p
                (8.0, 15.0),     // subcool
                (8.0, 15.0),     // superheat
                (30.0, 60.0),    // oil_p
            ],
            EquipmentType::Pump => vec![
                (2.0, 15.0),     // amps
                (1750.0, 1800.0),// speed_rpm
                (15.0, 45.0),    // discharge_p
                (5.0, 15.0),     // suction_p
                (0.0, 0.3),      // vibration_ips
                (100.0, 160.0),  // bearing_t
                (20.0, 100.0),   // flow_gpm
            ],
            EquipmentType::Ahu => vec![
                (0.0, 100.0),    // oa_damper %
                (0.0, 100.0),    // return_damper %
                (0.0, 1.0),      // freeze_stat (0=ok, 1=tripped)
                (0.0, 1.0),      // fan_enable
                (0.0, 30.0),     // fan_amps
                (0.0, 100.0),    // cw_valve %
                (0.0, 100.0),    // hw_valve %
                (35.0, 160.0),   // coil_temp
                (0.0, 40.0),     // coil_delta_t
                (45.0, 100.0),   // supply_t
                (65.0, 82.0),    // return_t
                (45.0, 85.0),    // mixed_t
                (0.0, 3.0),      // duct_static_inwc
                (0.0, 1.0),      // occupancy
                (68.0, 76.0),    // space_setpoint
                (0.0, 0.1),      // bldg_pressure_sp_inwc
                (-0.1, 0.2),     // bldg_pressure_inwc
            ],
            EquipmentType::ZoneReheat => vec![
                (50.0, 100.0),   // incoming_air_t
                (0.0, 100.0),    // damper %
                (0.0, 100.0),    // hw_valve %
                (50.0, 110.0),   // supply_t (discharge)
                (65.0, 82.0),    // space_t
                (0.0, 5.0),      // fan_amps
                (0.0, 1.0),      // eht1_enable
                (0.0, 15.0),     // eht1_amps
                (0.0, 1.0),      // eht2_enable
                (0.0, 15.0),     // eht2_amps
                (0.0, 1.0),      // eht3_enable
                (0.0, 15.0),     // eht3_amps
                (0.0, 1.0),      // eht4_enable
                (0.0, 15.0),     // eht4_amps
            ],
            EquipmentType::BoosterPump => vec![
                (0.0, 120.0),    // system_pressure
                (0.0, 120.0),    // pump1_pressure
                (0.0, 120.0),    // pump2_pressure
                (0.0, 120.0),    // ess_pressure
                (0.0, 10.0),     // pump1_speed_cmd
                (0.0, 10.0),     // pump2_speed_cmd
                (0.0, 20.0),     // pump1_current
                (0.0, 20.0),     // pump2_current
                (0.0, 1.0),      // pump1_enable
                (0.0, 1.0),      // pump2_enable
                (0.0, 120.0),    // tank_pressure
                (0.0, 600.0),    // sleep_wake_timer
            ],
            EquipmentType::HeatPump => vec![
                (40.0, 130.0),   // supply_t
                (60.0, 85.0),    // return_t
                (-10.0, 115.0),  // oat
                (5.0, 80.0),     // comp_amps
                (0.0, 1.0),      // rev_valve (0=heat, 1=cool)
                (40.0, 90.0),    // suction_p (410a: ~118psi@40F sat)
                (150.0, 450.0),  // discharge_p (410a: ~418psi@120F sat)
                (5.0, 25.0),     // subcool
                (5.0, 30.0),     // superheat
                (60.0, 120.0),   // liquid_line_t
                (120.0, 250.0),  // discharge_line_t
                (0.0, 1.0),      // defrost_status
                (0.0, 1.0),      // aux_heat_enable
                (0.0, 15.0),     // condenser_fan_amps
            ],
            _ => sentinel_datagen_ext::extended_normal_ranges(&self.equipment_type)
                .unwrap_or_else(|| vec![(0.0, 1.0); self.equipment_type.sensor_count()]),
        }
    }

    fn fault_scenarios(&self) -> Vec<FaultScenario> {
        match self.equipment_type {
            EquipmentType::Boiler => vec![
                FaultScenario { name: "ignition_failure".into(), affected_sensors: vec![7, 6, 0], severity: 1.0, onset_hours: 0.0 },
                FaultScenario { name: "flame_failure".into(), affected_sensors: vec![6, 0, 1], severity: 1.0, onset_hours: 0.0 },
                FaultScenario { name: "overtemp".into(), affected_sensors: vec![0, 1, 3], severity: 0.9, onset_hours: 0.5 },
                FaultScenario { name: "undertemp".into(), affected_sensors: vec![0, 1, 6], severity: 0.7, onset_hours: 2.0 },
                FaultScenario { name: "low_water".into(), affected_sensors: vec![5, 0, 4], severity: 1.0, onset_hours: 0.0 },
                FaultScenario { name: "short_cycling".into(), affected_sensors: vec![3, 9, 0], severity: 0.8, onset_hours: 1.0 },
                FaultScenario { name: "pump_failure".into(), affected_sensors: vec![4, 5, 0, 1], severity: 1.0, onset_hours: 0.0 },
                FaultScenario { name: "sensor_drift".into(), affected_sensors: vec![0, 1], severity: 0.5, onset_hours: 72.0 },
            ],
            EquipmentType::Chiller => vec![
                FaultScenario { name: "low_refrigerant".into(), affected_sensors: vec![7, 9, 10], severity: 0.7, onset_hours: 48.0 },
                FaultScenario { name: "condenser_fouling".into(), affected_sensors: vec![2, 3, 8], severity: 0.6, onset_hours: 168.0 },
                FaultScenario { name: "comp_valve_leak".into(), affected_sensors: vec![4, 7, 8], severity: 0.8, onset_hours: 24.0 },
                FaultScenario { name: "evap_fouling".into(), affected_sensors: vec![0, 1, 6], severity: 0.6, onset_hours: 168.0 },
                FaultScenario { name: "oil_pressure_low".into(), affected_sensors: vec![11, 4], severity: 0.9, onset_hours: 1.0 },
                FaultScenario { name: "cw_flow_loss".into(), affected_sensors: vec![5, 2, 3], severity: 1.0, onset_hours: 0.0 },
            ],
            EquipmentType::Pump => vec![
                FaultScenario { name: "cavitation".into(), affected_sensors: vec![3, 4, 6], severity: 0.7, onset_hours: 1.0 },
                FaultScenario { name: "bearing_wear".into(), affected_sensors: vec![4, 5], severity: 0.5, onset_hours: 336.0 },
                FaultScenario { name: "impeller_damage".into(), affected_sensors: vec![0, 2, 6], severity: 0.8, onset_hours: 48.0 },
                FaultScenario { name: "seal_leak".into(), affected_sensors: vec![6, 2], severity: 0.6, onset_hours: 72.0 },
                FaultScenario { name: "motor_overload".into(), affected_sensors: vec![0, 5], severity: 0.9, onset_hours: 0.5 },
                FaultScenario { name: "dead_head".into(), affected_sensors: vec![2, 6, 0, 5], severity: 1.0, onset_hours: 0.0 },
            ],
            // sensors: 0=oa_damper 1=return_damper 2=freeze_stat 3=fan_enable 4=fan_amps 5=cw_valve 6=hw_valve 7=coil_temp 8=coil_delta 9=supply_t 10=return_t 11=mixed_t 12=duct_static 13=occupancy 14=space_sp 15=bldg_p_sp 16=bldg_p
            EquipmentType::Ahu => vec![
                FaultScenario { name: "frozen_coil".into(), affected_sensors: vec![2, 7, 9, 8, 6], severity: 1.0, onset_hours: 0.5 },
                FaultScenario { name: "oa_damper_stuck".into(), affected_sensors: vec![0, 11, 9], severity: 0.8, onset_hours: 0.0 },
                FaultScenario { name: "return_damper_stuck".into(), affected_sensors: vec![1, 11, 16], severity: 0.8, onset_hours: 0.0 },
                FaultScenario { name: "fan_failure".into(), affected_sensors: vec![3, 4, 12, 9], severity: 1.0, onset_hours: 0.0 },
                FaultScenario { name: "cw_valve_stuck".into(), affected_sensors: vec![5, 9, 7, 8], severity: 0.8, onset_hours: 0.0 },
                FaultScenario { name: "hw_valve_stuck".into(), affected_sensors: vec![6, 9, 7, 8], severity: 0.8, onset_hours: 0.0 },
                FaultScenario { name: "economizer_fail".into(), affected_sensors: vec![0, 1, 11, 9], severity: 0.7, onset_hours: 0.0 },
                FaultScenario { name: "duct_static_high".into(), affected_sensors: vec![12, 4], severity: 0.6, onset_hours: 4.0 },
                FaultScenario { name: "supply_temp_drift".into(), affected_sensors: vec![9, 7, 8], severity: 0.5, onset_hours: 72.0 },
                FaultScenario { name: "bldg_pressure_loss".into(), affected_sensors: vec![16, 0, 1], severity: 0.7, onset_hours: 1.0 },
            ],
            // sensors: 0=incoming_air 1=damper 2=hw_valve 3=supply_t 4=space_t 5=fan_amps 6=eht1_en 7=eht1_a 8=eht2_en 9=eht2_a 10=eht3_en 11=eht3_a 12=eht4_en 13=eht4_a
            EquipmentType::ZoneReheat => vec![
                FaultScenario { name: "damper_stuck_closed".into(), affected_sensors: vec![1, 4, 3, 0], severity: 0.9, onset_hours: 0.0 },
                FaultScenario { name: "damper_stuck_open".into(), affected_sensors: vec![1, 4, 3], severity: 0.7, onset_hours: 0.0 },
                FaultScenario { name: "hw_valve_stuck".into(), affected_sensors: vec![2, 3, 4], severity: 0.8, onset_hours: 0.0 },
                FaultScenario { name: "eht_stage_stuck_on".into(), affected_sensors: vec![6, 7, 3, 4], severity: 0.8, onset_hours: 0.0 },
                FaultScenario { name: "eht_failure".into(), affected_sensors: vec![7, 9, 11, 13, 4], severity: 0.7, onset_hours: 0.0 },
                FaultScenario { name: "overcool".into(), affected_sensors: vec![4, 3, 1, 0], severity: 0.7, onset_hours: 0.5 },
                FaultScenario { name: "overheat".into(), affected_sensors: vec![4, 3, 6, 7], severity: 0.7, onset_hours: 0.5 },
                FaultScenario { name: "supply_temp_drift".into(), affected_sensors: vec![3, 0], severity: 0.5, onset_hours: 72.0 },
            ],
            // Booster pump pack FDD
            // sensors: 0=sys_p 1=p1_p 2=p2_p 3=ess_p 4=p1_spd 5=p2_spd 6=p1_amps 7=p2_amps 8=p1_en 9=p2_en 10=tank_p 11=sleep_timer
            EquipmentType::BoosterPump => vec![
                FaultScenario { name: "tank_bladder_fail".into(), affected_sensors: vec![0, 10, 11], severity: 1.0, onset_hours: 0.0 },
                FaultScenario { name: "check_valve_leak_p2".into(), affected_sensors: vec![2, 0, 7], severity: 0.8, onset_hours: 0.0 },
                FaultScenario { name: "pump1_dry_run".into(), affected_sensors: vec![4, 6, 0], severity: 1.0, onset_hours: 0.0 },
                FaultScenario { name: "pump2_dry_run".into(), affected_sensors: vec![5, 7, 0], severity: 1.0, onset_hours: 0.0 },
                FaultScenario { name: "transducer_drift".into(), affected_sensors: vec![0, 1, 2, 3], severity: 0.5, onset_hours: 168.0 },
                FaultScenario { name: "vfd1_fail_to_start".into(), affected_sensors: vec![8, 6, 0], severity: 1.0, onset_hours: 0.0 },
                FaultScenario { name: "vfd2_fail_to_start".into(), affected_sensors: vec![9, 7, 0], severity: 1.0, onset_hours: 0.0 },
                FaultScenario { name: "short_cycling".into(), affected_sensors: vec![11, 0, 10, 4], severity: 0.8, onset_hours: 1.0 },
            ],
            // 410a refrigerant FDD per DOE/Payne methodology
            // sensors: 0=supply_t 1=return_t 2=oat 3=comp_amps 4=rev_valve 5=suction_p 6=discharge_p 7=subcool 8=superheat 9=liquid_line_t 10=discharge_line_t 11=defrost 12=aux_heat 13=cond_fan_amps
            EquipmentType::HeatPump => vec![
                FaultScenario { name: "low_charge".into(), affected_sensors: vec![5, 6, 7, 8, 9], severity: 0.7, onset_hours: 48.0 },
                FaultScenario { name: "overcharge".into(), affected_sensors: vec![5, 6, 7, 8, 3], severity: 0.6, onset_hours: 48.0 },
                FaultScenario { name: "compressor_valve_leak".into(), affected_sensors: vec![3, 5, 6, 8, 10], severity: 0.8, onset_hours: 24.0 },
                FaultScenario { name: "condenser_fouling".into(), affected_sensors: vec![6, 7, 10, 3, 13], severity: 0.6, onset_hours: 336.0 },
                FaultScenario { name: "evaporator_fouling".into(), affected_sensors: vec![5, 8, 0, 1], severity: 0.6, onset_hours: 336.0 },
                FaultScenario { name: "txv_restriction".into(), affected_sensors: vec![5, 8, 7, 9], severity: 0.7, onset_hours: 12.0 },
                FaultScenario { name: "liquid_line_restriction".into(), affected_sensors: vec![7, 9, 5, 8], severity: 0.8, onset_hours: 4.0 },
                FaultScenario { name: "rev_valve_leak".into(), affected_sensors: vec![0, 1, 5, 6, 3], severity: 0.7, onset_hours: 0.0 },
                FaultScenario { name: "defrost_fail".into(), affected_sensors: vec![0, 8, 5, 11], severity: 0.8, onset_hours: 1.0 },
                FaultScenario { name: "comp_failure".into(), affected_sensors: vec![3, 5, 6, 0, 1], severity: 1.0, onset_hours: 0.0 },
            ],
            _ => sentinel_datagen_ext::extended_fault_scenarios(&self.equipment_type)
                .unwrap_or_default(),
        }
    }

    /// Generate a complete dataset: normal + faulted samples with labels.
    /// Uses equipment-type-specific generation for types that need it.
    pub fn generate(&self, normal_count: usize, fault_count_per_type: usize) -> Vec<SentinelSample> {
        match self.equipment_type {
            EquipmentType::Boiler => self.generate_boiler(normal_count, fault_count_per_type),
            EquipmentType::Ahu => self.generate_ahu(normal_count, fault_count_per_type),
            EquipmentType::ZoneReheat => self.generate_zone_reheat(normal_count, fault_count_per_type),
            EquipmentType::HeatPump => self.generate_heat_pump(normal_count, fault_count_per_type),
            EquipmentType::Pump => self.generate_pump(normal_count, fault_count_per_type),
            EquipmentType::BoosterPump => self.generate_booster_pump(normal_count, fault_count_per_type),
            _ => {
                // Try extended per-type generator first, fall back to generic
                if let Some(samples) = sentinel_datagen_ext::extended_generate(&self.equipment_type, self.seed, normal_count, fault_count_per_type) {
                    samples
                } else {
                    self.generate_generic(normal_count, fault_count_per_type)
                }
            },
        }
    }

    fn generate_generic(&self, normal_count: usize, fault_count_per_type: usize) -> Vec<SentinelSample> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let ranges = self.normal_ranges();
        let n_feat = self.equipment_type.sensor_count();
        let faults = self.fault_scenarios();
        let mut samples = Vec::new();

        for _ in 0..normal_count {
            let data = self.generate_normal_window(&mut rng, &ranges, n_feat);
            samples.push(SentinelSample { data, labels: [0.0; 7], fault_name: "normal".into() });
        }
        for fault in &faults {
            for fi in 0..fault_count_per_type {
                let progress = fi as f32 / fault_count_per_type as f32;
                let (data, labels) = self.generate_fault_window(&mut rng, &ranges, n_feat, fault, progress);
                samples.push(SentinelSample { data, labels, fault_name: fault.name.clone() });
            }
        }
        samples
    }

    /// Boiler-specific datagen: 5° hysteresis, min run time, enable/flame/flow interlocks
    /// sensors: 0=loop_supply 1=loop_return 2=setpoint 3=enable 4=amps 5=flow_switch 6=flame 7=ignition 8=oat 9=run_timer
    fn generate_boiler(&self, normal_count: usize, fault_count_per_type: usize) -> Vec<SentinelSample> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let n = self.equipment_type.sensor_count();
        let faults = self.fault_scenarios();
        let ranges = self.normal_ranges();
        let mut samples = Vec::new();
        let hysteresis = 5.0;
        let min_run_sec = 120.0;

        for _ in 0..normal_count {
            let oat = rng.gen_range(-10.0f32..90.0);
            let setpoint = if oat < 30.0 { 170.0 } else if oat < 55.0 { 155.0 } else { 140.0 };
            let cycle_pos = rng.gen_range(0.0f32..1.0);
            let firing = cycle_pos < 0.6;
            let supply_base = if firing {
                setpoint - hysteresis + cycle_pos / 0.6 * (hysteresis * 2.0)
            } else {
                setpoint + hysteresis - (cycle_pos - 0.6) / 0.4 * (hysteresis * 2.0)
            };
            let delta_t = 15.0 + rng.gen_range(0.0..10.0);
            let ret_base = supply_base - delta_t;
            let run_timer0 = if firing { rng.gen_range(min_run_sec..600.0) } else { rng.gen_range(30.0..300.0) };
            let transitioning = rng.gen_bool(0.2);
            let trans_step = rng.gen_range(2usize..6);
            let fire_rate = rng.gen_range(1.0f32..3.0);
            let cool_rate = rng.gen_range(0.5f32..1.5);

            let mut data = Vec::with_capacity(TIMESTEPS * n);
            let mut cum_temp = 0.0f32;
            for t in 0..TIMESTEPS {
                let is_firing = if transitioning {
                    if firing { t < trans_step } else { t >= trans_step }
                } else { firing };
                if is_firing { cum_temp += fire_rate * 0.5; } else { cum_temp -= cool_rate * 0.5; }
                let s = supply_base + cum_temp + rng.gen_range(-1.5..1.5);
                let r = ret_base + cum_temp * 0.6 + rng.gen_range(-1.5..1.5);
                let cur_amps = if is_firing { 5.0 + rng.gen_range(0.0..3.0) } else { 0.3 + rng.gen_range(0.0..0.2) };
                let vals = [
                    s, r, setpoint, 1.0, cur_amps, 1.0,
                    if is_firing { 1.0 } else { 0.0 }, 1.0,
                    oat + rng.gen_range(-0.5..0.5),
                    run_timer0 + t as f32 * 30.0,
                ];
                for (i, &v) in vals.iter().enumerate() {
                    data.push(((v - ranges[i].0) / (ranges[i].1 - ranges[i].0 + 1e-7)).clamp(-0.5, 1.5));
                }
            }
            samples.push(SentinelSample { data, labels: [0.0; 7], fault_name: "normal".into() });
        }

        for fault in &faults {
            for fi in 0..fault_count_per_type {
                let progress = fi as f32 / fault_count_per_type as f32;
                let deg = progress * fault.severity;
                let oat = rng.gen_range(-10.0f32..90.0);
                let setpoint = 160.0;
                let supply_base = setpoint - 3.0 + rng.gen_range(-2.0..2.0);
                let ret_base = supply_base - 18.0;

                let mut data = Vec::with_capacity(TIMESTEPS * n);
                for t in 0..TIMESTEPS {
                    let tf = t as f32 / TIMESTEPS as f32;
                    let fs = deg * (0.3 + 0.7 * tf);
                    let (supply, ret, enable, amps, flow, flame, ign, timer) = match fault.name.as_str() {
                        "ignition_failure" => {
                            // Enable on, no flame, no amps, temp dropping
                            (supply_base - fs * 30.0, ret_base - fs * 20.0, 1.0, 0.3, 1.0, 0.0, 0.0, rng.gen_range(0.0..30.0))
                        },
                        "flame_failure" => {
                            // Was running, flame drops mid-window, temp drops
                            let flame = if tf < 0.4 { 1.0 } else { 0.0 };
                            let a = if tf < 0.4 { 6.0 } else { 0.3 };
                            (supply_base - fs * 25.0 * tf, ret_base - fs * 15.0 * tf, 1.0, a, 1.0, flame, 1.0, 200.0 + t as f32 * 10.0)
                        },
                        "overtemp" => {
                            // Supply climbing past setpoint + hysteresis, boiler should have shut off but didn't
                            (setpoint + hysteresis + fs * 20.0, ret_base + fs * 10.0, 1.0, 6.0, 1.0, 1.0, 1.0, 300.0 + t as f32 * 10.0)
                        },
                        "undertemp" => {
                            // Supply can't reach setpoint despite firing
                            (setpoint - 15.0 - fs * 20.0, ret_base - fs * 15.0, 1.0, 5.0, 1.0, 1.0, 1.0, 400.0)
                        },
                        "low_water" => {
                            // Flow switch drops, boiler should interlock off
                            (supply_base - fs * 10.0, ret_base, 1.0, if fs > 0.5 { 0.3 } else { 5.0 }, 0.0, 0.0, 1.0, 10.0)
                        },
                        "short_cycling" => {
                            // On/off faster than min run time
                            let cycle = ((tf * 24.0).sin() > 0.0) as u8 as f32;
                            let timer_val = (tf * 24.0).sin().abs() * 40.0; // always < min_run
                            (supply_base + (cycle - 0.5) * fs * 8.0, ret_base, 1.0,
                             cycle * 5.0 + 0.3, 1.0, cycle, 1.0, timer_val)
                        },
                        "pump_failure" => {
                            // Amps drop, flow switch drops, supply spikes, return drops
                            (supply_base + fs * 25.0, ret_base - fs * 10.0, 1.0, 0.3 * fs + 5.0 * (1.0 - fs), 1.0 - fs.round(), 1.0, 1.0, 300.0)
                        },
                        "sensor_drift" => {
                            // Supply reads higher than actual, delta-T looks wrong
                            (supply_base + fs * 15.0, ret_base, 1.0, 5.0, 1.0, 1.0, 1.0, 200.0)
                        },
                        _ => (supply_base, ret_base, 1.0, 5.0, 1.0, 1.0, 1.0, 200.0),
                    };
                    let vals = [supply, ret, setpoint, enable, amps, flow, flame, ign, oat + rng.gen_range(-0.5..0.5), timer];
                    for (i, &v) in vals.iter().enumerate() {
                        let noise = rng.gen_range(-0.005..0.005) * (ranges[i].1 - ranges[i].0);
                        data.push(((v + noise - ranges[i].0) / (ranges[i].1 - ranges[i].0 + 1e-7)).clamp(-0.5, 1.5));
                    }
                }
                let horizons: [f32; 6] = [0.5, 4.0, 12.0, 72.0, 168.0, 336.0];
                let mut labels = [0.0f32; 7];
                labels[0] = if deg > 0.15 { 1.0 } else { 0.0 };
                for (hi, &h) in horizons.iter().enumerate() {
                    let hrs = fault.onset_hours * (1.0 - progress);
                    labels[hi + 1] = if hrs <= h { 1.0 } else { (1.0 - (hrs - h) / h).max(0.0) };
                }
                samples.push(SentinelSample { data, labels, fault_name: fault.name.clone() });
            }
        }
        samples
    }

    /// AHU-specific: mode-dependent operation (heating/cooling/economizer),
    /// supply_t and mixed_t correlate with valve positions and dampers
    /// AHU: mode-dependent (heating/cooling/econ), damper/valve/temp interlocks
    /// 0=oa_damper 1=return_damper 2=freeze_stat 3=fan_enable 4=fan_amps 5=cw_valve 6=hw_valve 7=coil_temp 8=coil_delta 9=supply_t 10=return_t 11=mixed_t 12=duct_static 13=occupancy 14=space_sp 15=bldg_p_sp 16=bldg_p
    fn generate_ahu(&self, normal_count: usize, fault_count_per_type: usize) -> Vec<SentinelSample> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let n = 17;
        let faults = self.fault_scenarios();
        let ranges = self.normal_ranges();
        let mut samples = Vec::new();

        for _ in 0..normal_count {
            let oat = rng.gen_range(-10.0f32..100.0);
            let ret_t = rng.gen_range(71.0..77.0);
            let cooling = oat > 65.0;
            let econ = oat > 50.0 && oat < 72.0;
            let occ = if rng.gen_bool(0.7) { 1.0 } else { 0.0 };
            let sp = rng.gen_range(70.0..74.0);
            let oa_pct0 = if econ { rng.gen_range(0.5..0.9) } else { rng.gen_range(0.1..0.3) };
            let cw0 = if cooling { rng.gen_range(30.0..90.0) } else { 0.0 };
            let hw0 = if !cooling && oat < 55.0 { rng.gen_range(20.0..80.0) } else { 0.0 };
            let fan_a0 = 8.0 + rng.gen_range(2.0..12.0);
            let static0 = 1.0 + rng.gen_range(-0.2..0.2);
            let bldg_p_sp = 0.05;
            let load_drift = rng.gen_range(-0.4f32..0.4);
            let static_phase = rng.gen_range(0.0f32..6.28);

            let mut data = Vec::with_capacity(TIMESTEPS * n);
            for t in 0..TIMESTEPS {
                let tf = t as f32 / TIMESTEPS as f32;
                let oa_pct = (oa_pct0 + load_drift * 0.12 * tf).clamp(0.05, 0.95);
                let mixed = oat * oa_pct + ret_t * (1.0 - oa_pct);
                let cw = if cooling { (cw0 + load_drift * 25.0 * tf).clamp(0.0, 100.0) } else { 0.0 };
                let hw = if !cooling && oat < 55.0 { (hw0 + load_drift * 25.0 * tf).clamp(0.0, 100.0) } else { 0.0 };
                let supply = if cooling { 52.0 - (cw - cw0) * 0.1 + rng.gen_range(-0.8..0.8) }
                             else if hw > 0.0 { 80.0 + (hw - hw0) * 0.15 + rng.gen_range(-0.8..0.8) }
                             else { mixed + rng.gen_range(-1.5..1.5) };
                let coil_t = if cooling { 42.0 - (cw - cw0) * 0.08 } else { 130.0 + (hw - hw0) * 0.12 };
                let coil_d = (ret_t - supply).abs().max(0.0);
                let static_p = static0 + 0.1 * (static_phase + tf * 5.0).sin();
                let fan_a = fan_a0 + (static_p - static0) * 4.0 + rng.gen_range(-0.4..0.4);
                let bldg_p = bldg_p_sp + rng.gen_range(-0.015..0.015) + (oa_pct - oa_pct0) * 0.04;
                let vals = [oa_pct*100.0, (1.0-oa_pct)*100.0, 0.0, 1.0, fan_a, cw, hw, coil_t, coil_d,
                            supply, ret_t + rng.gen_range(-0.3..0.3), mixed, static_p, occ, sp, bldg_p_sp, bldg_p];
                for (i, &v) in vals.iter().enumerate() {
                    let noise = rng.gen_range(-0.008..0.008) * (ranges[i].1 - ranges[i].0);
                    data.push(((v + noise - ranges[i].0) / (ranges[i].1 - ranges[i].0 + 1e-7)).clamp(-0.5, 1.5));
                }
            }
            samples.push(SentinelSample { data, labels: [0.0; 7], fault_name: "normal".into() });
        }

        for fault in &faults {
            for fi in 0..fault_count_per_type {
                let progress = fi as f32 / fault_count_per_type as f32;
                let deg = progress * fault.severity;
                let oat = rng.gen_range(-10.0f32..100.0);
                let ret_t = rng.gen_range(71.0..77.0);
                let cooling = oat > 65.0;
                let oa_pct = rng.gen_range(0.15..0.5);
                let mixed_n = oat * oa_pct + ret_t * (1.0 - oa_pct);
                let supply_n = if cooling { 53.0 } else { 82.0 };
                let mut data = Vec::with_capacity(TIMESTEPS * n);
                for t in 0..TIMESTEPS {
                    let tf = t as f32 / TIMESTEPS as f32;
                    let fs = deg * (0.3 + 0.7 * tf);
                    let (oa_d, ra_d, freeze, fan_en, fan_a, cw, hw, coil_t, coil_d, supply, mixed, static_p, bldg_p) = match fault.name.as_str() {
                        "frozen_coil" => (oa_pct*100.0, (1.0-oa_pct)*100.0, fs.round(), 1.0, 12.0, 0.0, 80.0, 35.0-fs*20.0, 5.0+fs*15.0, supply_n-fs*15.0, mixed_n, 1.0, 0.05),
                        "oa_damper_stuck" => (15.0, 85.0, 0.0, 1.0, 12.0, if cooling{50.0}else{0.0}, if !cooling{50.0}else{0.0}, 80.0, 10.0, supply_n+fs*12.0, oat*0.15+ret_t*0.85, 1.0, 0.05),
                        "return_damper_stuck" => (oa_pct*100.0, 10.0, 0.0, 1.0, 12.0, 50.0, 30.0, 80.0, 8.0, supply_n+fs*5.0, mixed_n+fs*8.0, 1.0, 0.05-fs*0.06),
                        "fan_failure" => (oa_pct*100.0, (1.0-oa_pct)*100.0, 0.0, 1.0-fs.round(), 12.0*(1.0-fs), 50.0, 30.0, 80.0, 2.0*(1.0-fs), ret_t, mixed_n, 0.1*(1.0-fs), 0.05-fs*0.04),
                        "cw_valve_stuck" => (oa_pct*100.0, (1.0-oa_pct)*100.0, 0.0, 1.0, 12.0, 50.0, 0.0, 50.0, 5.0-fs*3.0, supply_n+fs*15.0, mixed_n, 1.0, 0.05),
                        "hw_valve_stuck" => (oa_pct*100.0, (1.0-oa_pct)*100.0, 0.0, 1.0, 12.0, 0.0, 80.0, 140.0+fs*20.0, 20.0+fs*10.0, supply_n+fs*20.0, mixed_n, 1.0, 0.05),
                        "economizer_fail" => (fs*100.0, (1.0-fs)*100.0, 0.0, 1.0, 12.0, 50.0, 30.0, 80.0, 8.0, supply_n+fs*10.0, oat*fs+ret_t*(1.0-fs), 1.0, 0.05),
                        "duct_static_high" => (oa_pct*100.0, (1.0-oa_pct)*100.0, 0.0, 1.0, 12.0+fs*8.0, 50.0, 30.0, 80.0, 8.0, supply_n, mixed_n, 1.0+fs*1.5, 0.05+fs*0.03),
                        "bldg_pressure_loss" => (oa_pct*100.0+fs*30.0, (1.0-oa_pct)*100.0-fs*30.0, 0.0, 1.0, 12.0, 50.0, 30.0, 80.0, 8.0, supply_n, mixed_n-fs*5.0, 1.0, 0.05-fs*0.1),
                        _ => (oa_pct*100.0, (1.0-oa_pct)*100.0, 0.0, 1.0, 12.0, 50.0, 30.0, 80.0, 8.0, supply_n, mixed_n, 1.0, 0.05),
                    };
                    let vals = [oa_d, ra_d, freeze, fan_en, fan_a, cw, hw, coil_t, coil_d, supply, ret_t, mixed, static_p, 1.0, 72.0, 0.05, bldg_p];
                    for (i, &v) in vals.iter().enumerate() {
                        let noise = rng.gen_range(-0.005..0.005) * (ranges[i].1 - ranges[i].0);
                        data.push(((v + noise - ranges[i].0) / (ranges[i].1 - ranges[i].0 + 1e-7)).clamp(-0.5, 1.5));
                    }
                }
                let horizons: [f32; 6] = [0.5, 4.0, 12.0, 72.0, 168.0, 336.0];
                let mut labels = [0.0f32; 7];
                labels[0] = if deg > 0.15 { 1.0 } else { 0.0 };
                for (hi, &h) in horizons.iter().enumerate() { let hrs = fault.onset_hours * (1.0 - progress); labels[hi+1] = if hrs <= h { 1.0 } else { (1.0 - (hrs-h)/h).max(0.0) }; }
                samples.push(SentinelSample { data, labels, fault_name: fault.name.clone() });
            }
        }
        samples
    }

    /// Zone reheat: incoming air + damper + HW valve + electric heat stages
    /// 0=incoming_air 1=damper 2=hw_valve 3=supply_t 4=space_t 5=fan_amps 6=eht1_en 7=eht1_a 8=eht2_en 9=eht2_a 10=eht3_en 11=eht3_a 12=eht4_en 13=eht4_a
    fn generate_zone_reheat(&self, normal_count: usize, fault_count_per_type: usize) -> Vec<SentinelSample> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let n = 14;
        let faults = self.fault_scenarios();
        let ranges = self.normal_ranges();
        let mut samples = Vec::new();

        for _ in 0..normal_count {
            let incoming = rng.gen_range(52.0f32..65.0);
            let sp = rng.gen_range(70.0..74.0);
            let space0 = sp + rng.gen_range(-2.5..2.5);
            let space_drift = rng.gen_range(-1.5f32..1.5);
            let incoming_drift = rng.gen_range(-2.0f32..2.0);

            let mut data = Vec::with_capacity(TIMESTEPS * n);
            for t in 0..TIMESTEPS {
                let tf = t as f32 / TIMESTEPS as f32;
                let space = space0 + space_drift * tf;
                let inc = incoming + incoming_drift * tf + rng.gen_range(-0.5..0.5);
                let err = sp - space;
                let damper = (50.0f32 + err * 15.0).clamp(15.0, 100.0);
                let hw_v: f32 = if err > 1.0 { (err * 20.0).clamp(0.0, 100.0) } else { 0.0 };
                let eht_demand = (err - 2.0).max(0.0f32) / 6.0;
                let eht1_en = if eht_demand > 0.0 { 1.0 } else { 0.0 };
                let eht1_a = eht1_en * (8.0 + rng.gen_range(-0.5..0.5));
                let eht2_en = if eht_demand > 0.25 { 1.0 } else { 0.0 };
                let eht2_a = eht2_en * (8.0 + rng.gen_range(-0.5..0.5));
                let eht3_en = if eht_demand > 0.5 { 1.0 } else { 0.0 };
                let eht3_a = eht3_en * (8.0 + rng.gen_range(-0.5..0.5));
                let eht4_en = if eht_demand > 0.75 { 1.0 } else { 0.0 };
                let eht4_a = eht4_en * (8.0 + rng.gen_range(-0.5..0.5));
                let supply = (inc + hw_v * 0.3 + eht1_a * 1.5 + eht2_a * 1.5 + eht3_a * 1.5 + eht4_a * 1.5).clamp(50.0, 110.0);
                let fan_a = 1.0 + damper * 0.02 + rng.gen_range(-0.15..0.15);
                let vals = [inc, damper, hw_v, supply, space, fan_a, eht1_en, eht1_a, eht2_en, eht2_a, eht3_en, eht3_a, eht4_en, eht4_a];
                for (i, &v) in vals.iter().enumerate() {
                    let noise = rng.gen_range(-0.008..0.008) * (ranges[i].1 - ranges[i].0);
                    data.push(((v + noise - ranges[i].0) / (ranges[i].1 - ranges[i].0 + 1e-7)).clamp(-0.5, 1.5));
                }
            }
            samples.push(SentinelSample { data, labels: [0.0; 7], fault_name: "normal".into() });
        }

        for fault in &faults {
            for fi in 0..fault_count_per_type {
                let progress = fi as f32 / fault_count_per_type as f32;
                let deg = progress * fault.severity;
                let incoming = rng.gen_range(53.0f32..62.0);
                let sp = 72.0;
                let mut data = Vec::with_capacity(TIMESTEPS * n);
                for t in 0..TIMESTEPS {
                    let tf = t as f32 / TIMESTEPS as f32;
                    let fs = deg * (0.3 + 0.7 * tf);
                    let (space, damper, hw_v, supply, fan_a, e1_en, e1_a, e2_en, e2_a, e3_en, e3_a, e4_en, e4_a) = match fault.name.as_str() {
                        "damper_stuck_closed" => (sp + fs*12.0, 5.0, fs*80.0, incoming+fs*25.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        "damper_stuck_open" => (sp - fs*10.0, 100.0, 0.0, incoming-fs*5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        "hw_valve_stuck" => (sp + fs*10.0, 70.0, 100.0, incoming+30.0+fs*10.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        "eht_stage_stuck_on" => (sp + fs*15.0, 60.0, 0.0, incoming+fs*45.0, 1.5, 1.0, 12.0, 1.0, 12.0, 0.0, 0.0, 0.0, 0.0),
                        "eht_failure" => (sp - fs*12.0, 50.0, fs*60.0, incoming+fs*3.0, 1.5, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1),
                        "overcool" => (sp - fs*15.0, 100.0, 0.0, incoming-fs*8.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        "overheat" => (sp + fs*15.0, 20.0, fs*100.0, incoming+fs*40.0, 0.8, 1.0, 12.0*fs, 1.0*fs.round(), 12.0*fs, 0.0, 0.0, 0.0, 0.0),
                        _ => (sp, 50.0, 20.0, incoming+10.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    };
                    let vals = [incoming, damper, hw_v, supply, space, fan_a, e1_en, e1_a, e2_en, e2_a, e3_en, e3_a, e4_en, e4_a];
                    for (i, &v) in vals.iter().enumerate() {
                        let noise = rng.gen_range(-0.005..0.005) * (ranges[i].1 - ranges[i].0);
                        data.push(((v + noise - ranges[i].0) / (ranges[i].1 - ranges[i].0 + 1e-7)).clamp(-0.5, 1.5));
                    }
                }
                let horizons: [f32; 6] = [0.5, 4.0, 12.0, 72.0, 168.0, 336.0];
                let mut labels = [0.0f32; 7];
                labels[0] = if deg > 0.15 { 1.0 } else { 0.0 };
                for (hi, &h) in horizons.iter().enumerate() { let hrs = fault.onset_hours * (1.0 - progress); labels[hi+1] = if hrs <= h { 1.0 } else { (1.0 - (hrs-h)/h).max(0.0) }; }
                samples.push(SentinelSample { data, labels, fault_name: fault.name.clone() });
            }
        }
        samples
    }

    /// Heat pump: 410a refrigerant cycle physics per DOE FDD methodology
    /// 0=supply_t 1=return_t 2=oat 3=comp_amps 4=rev_valve 5=suction_p 6=discharge_p 7=subcool 8=superheat 9=liquid_line_t 10=discharge_line_t 11=defrost 12=aux_heat 13=cond_fan_amps
    fn generate_heat_pump(&self, normal_count: usize, fault_count_per_type: usize) -> Vec<SentinelSample> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let n = 14;
        let faults = self.fault_scenarios();
        let ranges = self.normal_ranges();
        let mut samples = Vec::new();

        for _ in 0..normal_count {
            let oat = rng.gen_range(-10.0f32..110.0);
            let heating = oat < 60.0;
            let load0 = rng.gen_range(0.3f32..1.0);
            let load_drift = rng.gen_range(-0.15f32..0.15);
            let ret = if heating { 68.0 + rng.gen_range(-2.0..2.0) } else { 76.0 + rng.gen_range(-2.0..2.0) };
            let rev = if heating { 0.0 } else { 1.0 };
            let defrost = if oat < 35.0 && oat > 15.0 && rng.gen_bool(0.2) { 1.0 } else { 0.0 };
            let aux = if oat < 20.0 && heating { 1.0 } else { 0.0 };

            let mut data = Vec::with_capacity(TIMESTEPS * n);
            for t in 0..TIMESTEPS {
                let tf = t as f32 / TIMESTEPS as f32;
                let load = (load0 + load_drift * tf).clamp(0.3, 1.0);
                let suct = (if heating { 55.0 + oat * 0.3 } else { 65.0 + load * 10.0 }) + rng.gen_range(-2.0..2.0);
                let disc = (if heating { 200.0 + load * 80.0 } else { 250.0 + load * 100.0 + oat * 0.5 }) + rng.gen_range(-5.0..5.0);
                let subcool = 10.0 + rng.gen_range(-1.5..1.5);
                let superheat = 10.0 + rng.gen_range(-1.5..1.5);
                let supply = (if heating { 90.0 + load * 25.0 } else { 48.0 - load * 5.0 }) + rng.gen_range(-1.5..1.5);
                let comp = 15.0 + load * 40.0 + rng.gen_range(-1.5..1.5);
                let liq_t = 80.0 + load * 5.0 + rng.gen_range(-3.0..3.0);
                let disc_t = 150.0 + load * 50.0 + rng.gen_range(-3.0..3.0);
                let cond_fan = 3.0 + load * 5.0 + rng.gen_range(-0.3..0.3);
                let vals = [supply, ret + rng.gen_range(-0.3..0.3), oat + rng.gen_range(-0.3..0.3),
                            comp, rev, suct, disc, subcool, superheat, liq_t, disc_t, defrost, aux, cond_fan];
                for (i, &v) in vals.iter().enumerate() {
                    let noise = rng.gen_range(-0.005..0.005) * (ranges[i].1 - ranges[i].0);
                    data.push(((v + noise - ranges[i].0) / (ranges[i].1 - ranges[i].0 + 1e-7)).clamp(-0.5, 1.5));
                }
            }
            samples.push(SentinelSample { data, labels: [0.0; 7], fault_name: "normal".into() });
        }

        for fault in &faults {
            for fi in 0..fault_count_per_type {
                let progress = fi as f32 / fault_count_per_type as f32;
                let deg = progress * fault.severity;
                let oat = rng.gen_range(-10.0f32..100.0);
                let heating = oat < 60.0;
                let load = rng.gen_range(0.4f32..0.9);
                let b_supply = if heating { 100.0 + load * 15.0 } else { 47.0 };
                let b_ret = if heating { 70.0 } else { 76.0 };
                let b_comp = 20.0 + load * 30.0;
                let b_suct = 65.0;
                let b_disc = 250.0 + load * 50.0;
                let mut data = Vec::with_capacity(TIMESTEPS * n);
                for t in 0..TIMESTEPS {
                    let tf = t as f32 / TIMESTEPS as f32;
                    let fs = deg * (0.3 + 0.7 * tf);
                    // 410a FDD signatures per DOE methodology
                    let (supply, ret, comp, suct, disc, subcool, superheat, liq_t, disc_t, cond_fan) = match fault.name.as_str() {
                        "low_charge" => (b_supply-fs*15.0, b_ret, b_comp+fs*5.0, b_suct-fs*15.0, b_disc-fs*30.0, 10.0-fs*8.0, 10.0+fs*12.0, 80.0-fs*10.0, 150.0+fs*20.0, 5.0),
                        "overcharge" => (b_supply+fs*5.0, b_ret, b_comp+fs*8.0, b_suct+fs*10.0, b_disc+fs*25.0, 10.0+fs*10.0, 10.0-fs*5.0, 80.0+fs*8.0, 150.0+fs*15.0, 5.0),
                        "compressor_valve_leak" => (b_supply-fs*20.0, b_ret, b_comp-fs*10.0, b_suct+fs*8.0, b_disc-fs*40.0, 10.0, 10.0+fs*8.0, 80.0, 150.0-fs*30.0, 5.0),
                        "condenser_fouling" => (b_supply-fs*8.0, b_ret, b_comp+fs*10.0, b_suct, b_disc+fs*50.0, 10.0-fs*4.0, 10.0, 80.0+fs*10.0, 150.0+fs*40.0, 5.0+fs*3.0),
                        "evaporator_fouling" => (b_supply+fs*8.0, b_ret+fs*3.0, b_comp-fs*5.0, b_suct-fs*10.0, b_disc, 10.0, 10.0+fs*8.0, 80.0, 150.0, 5.0),
                        "txv_restriction" => (b_supply-fs*10.0, b_ret, b_comp, b_suct-fs*12.0, b_disc, 10.0+fs*8.0, 10.0+fs*15.0, 80.0-fs*5.0, 150.0, 5.0),
                        "liquid_line_restriction" => (b_supply-fs*12.0, b_ret, b_comp, b_suct-fs*10.0, b_disc, 10.0+fs*12.0, 10.0+fs*10.0, 80.0-fs*15.0, 150.0, 5.0),
                        "rev_valve_leak" => (b_supply-fs*15.0, b_ret+fs*5.0, b_comp+fs*5.0, b_suct+fs*5.0, b_disc-fs*15.0, 10.0, 10.0+fs*5.0, 80.0, 150.0, 5.0),
                        "defrost_fail" => (b_supply-fs*25.0, b_ret, b_comp, b_suct-fs*15.0, b_disc, 10.0, 10.0+fs*8.0, 80.0-fs*10.0, 150.0, 5.0-fs*2.0),
                        "comp_failure" => (b_supply-fs*35.0, b_ret, b_comp*(1.0-fs), b_suct+fs*10.0, b_disc-fs*100.0, 10.0, 10.0, 80.0, 150.0*(1.0-fs*0.5), 5.0),
                        _ => (b_supply, b_ret, b_comp, b_suct, b_disc, 10.0, 10.0, 80.0, 150.0, 5.0),
                    };
                    let rev = if heating { 0.0 } else { 1.0 };
                    let defrost = if fault.name == "defrost_fail" { 1.0 } else { 0.0 };
                    let aux = if oat < 20.0 && heating { 1.0 } else { 0.0 };
                    let vals = [supply, ret, oat, comp, rev, suct, disc, subcool, superheat, liq_t, disc_t, defrost, aux, cond_fan];
                    for (i, &v) in vals.iter().enumerate() {
                        let noise = rng.gen_range(-0.005..0.005) * (ranges[i].1 - ranges[i].0);
                        data.push(((v + noise - ranges[i].0) / (ranges[i].1 - ranges[i].0 + 1e-7)).clamp(-0.5, 1.5));
                    }
                }
                let horizons: [f32; 6] = [0.5, 4.0, 12.0, 72.0, 168.0, 336.0];
                let mut labels = [0.0f32; 7];
                labels[0] = if deg > 0.15 { 1.0 } else { 0.0 };
                for (hi, &h) in horizons.iter().enumerate() { let hrs = fault.onset_hours * (1.0 - progress); labels[hi+1] = if hrs <= h { 1.0 } else { (1.0 - (hrs-h)/h).max(0.0) }; }
                samples.push(SentinelSample { data, labels, fault_name: fault.name.clone() });
            }
        }
        samples
    }

    fn generate_normal_window(&self, rng: &mut impl Rng, ranges: &[(f32, f32)], n_feat: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(TIMESTEPS * n_feat);
        // Operating point varies across the full range (not just midpoint)
        let load_pct = rng.gen_range(0.1f32..1.0); // 10-100% load
        let oat = rng.gen_range(-10.0f32..100.0);
        // Base operating point per sensor varies with load
        let base: Vec<f32> = ranges.iter().enumerate().map(|(i, &(lo, hi))| {
            let span = hi - lo;
            match self.equipment_type {
                EquipmentType::Boiler => {
                    // Supply temp rises with load, valve follows demand
                    if i == 0 { lo + span * (0.4 + 0.5 * load_pct) }      // supply_t
                    else if i == 1 { lo + span * (0.3 + 0.4 * load_pct) }  // return_t
                    else if i == 2 { oat.clamp(lo, hi) }                    // oat
                    else if i == 4 { lo + span * load_pct }                 // valve_pos
                    else if i == 5 { lo + span * (0.2 + 0.6 * load_pct) }  // pump_amps
                    else { lo + span * (0.3 + rng.gen_range(0.0..0.4)) }
                },
                EquipmentType::Ahu => {
                    // Mode-dependent: heating vs cooling vs economizer
                    let cooling = oat > 65.0;
                    let econ = oat > 55.0 && oat < 75.0;
                    if i == 0 { if cooling { lo + span * 0.15 } else { lo + span * 0.6 } } // supply_t
                    else if i == 2 { oat.clamp(lo, hi) }     // oat
                    else if i == 5 { if econ { lo + span * 0.8 } else { lo + span * 0.15 } } // oa_damper
                    else if i == 8 { if cooling { lo + span * load_pct } else { 0.0 } }     // chw_valve
                    else if i == 9 { if !cooling { lo + span * load_pct } else { 0.0 } }    // hw_valve
                    else { lo + span * (0.3 + rng.gen_range(0.0..0.4)) }
                },
                EquipmentType::ZoneReheat => {
                    if i == 0 { lo + span * (0.3 + 0.4 * rng.gen_range(0.0..1.0)) } // zone_t varies
                    else if i == 1 { lo + span * 0.5 }  // setpoint fixed
                    else if i == 2 { lo + span * (0.3 + 0.5 * load_pct) } // damper
                    else if i == 3 { lo + span * (1.0 - load_pct).max(0.0) } // reheat inversely
                    else { lo + span * (0.3 + rng.gen_range(0.0..0.4)) }
                },
                _ => lo + span * (0.2 + 0.6 * load_pct + rng.gen_range(-0.1..0.1)),
            }
        }).collect();

        for t in 0..TIMESTEPS {
            let t_frac = t as f32 / TIMESTEPS as f32;
            for (i, &(lo, hi)) in ranges.iter().enumerate() {
                let noise = rng.gen_range(-0.05..0.05) * (hi - lo);
                let drift = (t_frac - 0.5) * (hi - lo) * 0.02;
                let val = base[i] + noise + drift;
                data.push((val - lo) / (hi - lo + 1e-7));
            }
        }
        data
    }

    fn generate_fault_window(
        &self, rng: &mut impl Rng, ranges: &[(f32, f32)], n_feat: usize,
        fault: &FaultScenario, progress: f32,
    ) -> (Vec<f32>, [f32; 7]) {
        let mut data = Vec::with_capacity(TIMESTEPS * n_feat);
        let degradation = progress * fault.severity;

        // Generate normal baseline first
        let load = rng.gen_range(0.3f32..0.9);
        let base: Vec<f32> = ranges.iter().map(|&(lo, hi)| {
            lo + (hi - lo) * (0.2 + 0.6 * load + rng.gen_range(-0.05..0.05))
        }).collect();

        // Fault temporal pattern type (what makes faults detectable across time)
        let pattern: u8 = rng.gen_range(0..4);

        for t in 0..TIMESTEPS {
            let t_frac = t as f32 / TIMESTEPS as f32;

            for (i, &(lo, hi)) in ranges.iter().enumerate() {
                let span = hi - lo;
                let noise = rng.gen_range(-0.03..0.03) * span;

                let val = if fault.affected_sensors.contains(&i) {
                    match pattern {
                        0 => {
                            // STUCK: sensor locks at a fixed value from timestep 3 onward
                            let stuck_val = base[i] + span * degradation * 0.8;
                            if t >= 3 { stuck_val } else { base[i] + noise }
                        },
                        1 => {
                            // DRIFT: monotonic trend across all 8 timesteps
                            let drift = t_frac * span * degradation * 1.5;
                            base[i] + drift + noise
                        },
                        2 => {
                            // OSCILLATION: rapid cycling (short cycling pattern)
                            let osc = (t as f32 * 3.14159).sin() * span * degradation * 0.6;
                            base[i] + osc + noise
                        },
                        _ => {
                            // STEP CHANGE: sudden jump at timestep 4
                            let step = if t >= 4 { span * degradation * 1.2 } else { 0.0 };
                            base[i] + step + noise
                        },
                    }
                } else {
                    base[i] + noise
                };

                data.push(((val - lo) / (span + 1e-7)).clamp(-0.5, 1.5));
            }
        }

        // Labels: anomaly + predictive horizons
        let horizons_hours: [f32; 6] = [0.5, 4.0, 12.0, 72.0, 168.0, 336.0];
        let mut labels = [0.0f32; 7];
        labels[0] = if degradation > 0.15 { 1.0 } else { 0.0 };
        for (hi, &horizon) in horizons_hours.iter().enumerate() {
            let hours_until_fail = fault.onset_hours * (1.0 - progress);
            labels[hi + 1] = if hours_until_fail <= horizon { 1.0 } else {
                (1.0 - (hours_until_fail - horizon) / horizon).max(0.0)
            };
        }

        (data, labels)
    }

    /// Pump-specific: amps/pressure/flow/vibration/bearing physics
    /// sensors: 0=amps 1=speed_rpm 2=discharge_p 3=suction_p 4=vibration 5=bearing_t 6=flow
    fn generate_pump(&self, normal_count: usize, fault_count_per_type: usize) -> Vec<SentinelSample> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let n = self.equipment_type.sensor_count();
        let faults = self.fault_scenarios();
        let ranges = self.normal_ranges();
        let mut samples = Vec::new();

        for _ in 0..normal_count {
            let load = rng.gen_range(0.3f32..1.0);
            let amps = 4.0 + load * 8.0 + rng.gen_range(-0.3..0.3);
            let rpm = 1770.0 + rng.gen_range(-10.0..10.0);
            let disch_p = 20.0 + load * 20.0 + rng.gen_range(-1.0..1.0);
            let suct_p = 8.0 + rng.gen_range(-1.0..1.0);
            let vib = 0.05 + load * 0.1 + rng.gen_range(-0.02..0.02);
            let bearing = 110.0 + load * 20.0 + rng.gen_range(-3.0..3.0);
            let flow = 30.0 + load * 60.0 + rng.gen_range(-2.0..2.0);
            let base = [amps, rpm, disch_p, suct_p, vib, bearing, flow];
            let mut data = Vec::with_capacity(TIMESTEPS * n);
            for t in 0..TIMESTEPS {
                for (i, &v) in base.iter().enumerate() {
                    let noise = rng.gen_range(-0.01..0.01) * (ranges[i].1 - ranges[i].0);
                    data.push((v + noise - ranges[i].0) / (ranges[i].1 - ranges[i].0 + 1e-7));
                }
            }
            samples.push(SentinelSample { data, labels: [0.0; 7], fault_name: "normal".into() });
        }

        for fault in &faults {
            for fi in 0..fault_count_per_type {
                let progress = fi as f32 / fault_count_per_type as f32;
                let deg = progress * fault.severity;
                let load = rng.gen_range(0.4f32..0.9);
                let base_amps = 5.0 + load * 7.0;
                let base_flow = 30.0 + load * 55.0;
                let base_disch = 22.0 + load * 18.0;
                let base_suct = 8.0;
                let mut data = Vec::with_capacity(TIMESTEPS * n);
                for t in 0..TIMESTEPS {
                    let tf = t as f32 / TIMESTEPS as f32;
                    let fs = deg * (0.3 + 0.7 * tf);
                    let (amps, rpm, disch, suct, vib, bearing, flow) = match fault.name.as_str() {
                        "dead_head" => {
                            // Closed valve: pressure spikes, flow zero, amps climb, motor heats
                            (base_amps + fs * 6.0, 1770.0, base_disch + fs * 25.0, base_suct - fs * 3.0, 0.1 + fs * 0.15, 120.0 + fs * 40.0, base_flow * (1.0 - fs).max(0.0))
                        },
                        "cavitation" => {
                            // Low suction pressure, noise/vibration, flow drops, erratic amps
                            let osc = (tf * 15.0).sin() * fs * 0.1;
                            (base_amps + osc * 3.0, 1770.0, base_disch - fs * 8.0, base_suct - fs * 5.0, 0.08 + fs * 0.25, 115.0 + fs * 10.0, base_flow * (1.0 - fs * 0.4))
                        },
                        "bearing_wear" => {
                            // Vibration trends up slowly, bearing temp climbs, amps slight increase
                            (base_amps + fs * 1.5, 1770.0 - fs * 5.0, base_disch, base_suct, 0.06 + fs * 0.25, 115.0 + fs * 35.0, base_flow - fs * 5.0)
                        },
                        "impeller_damage" => {
                            // Flow drops, discharge pressure drops, vibration up, amps change
                            (base_amps - fs * 2.0, 1770.0, base_disch - fs * 12.0, base_suct, 0.08 + fs * 0.2, 120.0 + fs * 10.0, base_flow * (1.0 - fs * 0.5))
                        },
                        "seal_leak" => {
                            // Flow drops gradually, pressure drops, bearing temp may rise
                            (base_amps - fs * 1.0, 1770.0, base_disch - fs * 6.0, base_suct - fs * 2.0, 0.07 + fs * 0.05, 115.0 + fs * 8.0, base_flow * (1.0 - fs * 0.4))
                        },
                        "motor_overload" => {
                            // Amps spike, bearing temp spike, vibration up
                            (base_amps + fs * 8.0, 1770.0 - fs * 15.0, base_disch, base_suct, 0.08 + fs * 0.15, 120.0 + fs * 40.0, base_flow - fs * 10.0)
                        },
                        _ => (base_amps, 1770.0, base_disch, base_suct, 0.08, 115.0, base_flow),
                    };
                    let vals = [amps, rpm, disch, suct, vib, bearing, flow];
                    for (i, &v) in vals.iter().enumerate() {
                        let noise = rng.gen_range(-0.005..0.005) * (ranges[i].1 - ranges[i].0);
                        data.push(((v + noise - ranges[i].0) / (ranges[i].1 - ranges[i].0 + 1e-7)).clamp(-0.5, 1.5));
                    }
                }
                let horizons: [f32; 6] = [0.5, 4.0, 12.0, 72.0, 168.0, 336.0];
                let mut labels = [0.0f32; 7];
                labels[0] = if deg > 0.15 { 1.0 } else { 0.0 };
                for (hi, &h) in horizons.iter().enumerate() {
                    let hrs = fault.onset_hours * (1.0 - progress);
                    labels[hi + 1] = if hrs <= h { 1.0 } else { (1.0 - (hrs - h) / h).max(0.0) };
                }
                samples.push(SentinelSample { data, labels, fault_name: fault.name.clone() });
            }
        }
        samples
    }

    /// Booster pump pack: hydro-pneumatic tank, dual VFD pumps, pressure control
    /// 0=sys_p 1=p1_p 2=p2_p 3=ess_p 4=p1_spd 5=p2_spd 6=p1_amps 7=p2_amps 8=p1_en 9=p2_en 10=tank_p 11=sleep_timer
    fn generate_booster_pump(&self, normal_count: usize, fault_count_per_type: usize) -> Vec<SentinelSample> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let n = 12;
        let faults = self.fault_scenarios();
        let ranges = self.normal_ranges();
        let mut samples = Vec::new();
        let sp_high = 89.0f32; // sleep at 89 PSI
        let sp_low = 78.0f32;  // wake at 78 PSI

        for _ in 0..normal_count {
            let demand0 = rng.gen_range(0.0f32..1.0);
            let demand_drift = rng.gen_range(-0.15f32..0.15);
            let transitioning = rng.gen_bool(0.25);
            let trans_step = rng.gen_range(2usize..6);
            let p_decay = rng.gen_range(0.5f32..2.0);

            let mut data = Vec::with_capacity(TIMESTEPS * n);
            for t in 0..TIMESTEPS {
                let tf = t as f32 / TIMESTEPS as f32;
                let demand = (demand0 + demand_drift * tf).clamp(0.0, 1.0);
                let sleeping = if transitioning {
                    if demand0 < 0.2 { t >= trans_step } else { t < trans_step && demand0 < 0.3 }
                } else { demand < 0.2 };
                let p1_on = !sleeping;
                let p2_on = demand > 0.6;
                let base_p = if sleeping { sp_high - p_decay * t as f32 } else { sp_low + demand * (sp_high - sp_low) };
                let sys_p = base_p + rng.gen_range(-1.5..1.5);
                let p1_spd = if p1_on { (3.0 + demand * 6.0).max(0.0) } else { 0.0 };
                let p2_spd = if p2_on { (3.0 + (demand - 0.6) * 10.0).max(0.0) } else { 0.0 };
                let p1_amps = if p1_on { 5.0 + demand * 8.0 + rng.gen_range(-0.3..0.3) } else { 0.0 };
                let p2_amps = if p2_on { 5.0 + (demand - 0.6) * 12.0 + rng.gen_range(-0.3..0.3) } else { 0.0 };
                let p1_p = if p1_on { sys_p + rng.gen_range(-1.0..2.0) } else { sys_p };
                let p2_p = if p2_on { sys_p + rng.gen_range(-1.0..2.0) } else { sys_p };
                let ess_p = sys_p + rng.gen_range(-0.5..0.5);
                let tank_p = sys_p + rng.gen_range(-1.0..1.0);
                let sleep_timer = if sleeping { 60.0 + t as f32 * 30.0 } else { t as f32 * 5.0 };
                let vals = [sys_p, p1_p, p2_p, ess_p, p1_spd, p2_spd, p1_amps, p2_amps,
                            if p1_on {1.0} else {0.0}, if p2_on {1.0} else {0.0}, tank_p, sleep_timer];
                for (i, &v) in vals.iter().enumerate() {
                    let noise = rng.gen_range(-0.005..0.005) * (ranges[i].1 - ranges[i].0);
                    data.push(((v + noise - ranges[i].0) / (ranges[i].1 - ranges[i].0 + 1e-7)).clamp(-0.5, 1.5));
                }
            }
            samples.push(SentinelSample { data, labels: [0.0; 7], fault_name: "normal".into() });
        }

        for fault in &faults {
            for fi in 0..fault_count_per_type {
                let progress = fi as f32 / fault_count_per_type as f32;
                let deg = progress * fault.severity;
                let mut data = Vec::with_capacity(TIMESTEPS * n);
                for t in 0..TIMESTEPS {
                    let tf = t as f32 / TIMESTEPS as f32;
                    let fs = deg * (0.3 + 0.7 * tf);
                    let (sys_p, p1_p, p2_p, ess_p, p1_spd, p2_spd, p1_a, p2_a, p1_en, p2_en, tank_p, timer) = match fault.name.as_str() {
                        "tank_bladder_fail" => {
                            // Tank waterlogged: sleep→wake in <45s, rapid cycling
                            let wake_fast = 15.0 + (1.0-fs) * 50.0; // gets faster as fault progresses
                            (82.0 - fs*8.0, 82.0, 82.0, 82.0, 5.0*fs, 0.0, 6.0*fs, 0.0, 1.0, 0.0, 82.0-fs*5.0, wake_fast)
                        },
                        "check_valve_leak_p2" => {
                            // P2 off but P2 pressure matches system (backflow through stuck check valve)
                            (85.0, 85.0, 85.0, 85.0, 5.0, 0.0, 7.0, 0.0, 1.0, 0.0, 85.0, 100.0)
                        },
                        "pump1_dry_run" => {
                            // P1 at 100% speed, near-zero amps, pressure dropping
                            (78.0-fs*15.0, 78.0-fs*15.0, 78.0, 78.0, 10.0, 0.0, 0.3+rng.gen_range(0.0..0.2), 0.0, 1.0, 0.0, 78.0-fs*10.0, 5.0)
                        },
                        "pump2_dry_run" => {
                            (78.0-fs*15.0, 78.0, 78.0-fs*15.0, 78.0, 0.0, 10.0, 0.0, 0.3+rng.gen_range(0.0..0.2), 0.0, 1.0, 78.0-fs*10.0, 5.0)
                        },
                        "transducer_drift" => {
                            // During sleep, sensors should all match. One drifts.
                            let drift = fs * 8.0;
                            (85.0, 85.0+drift, 85.0, 85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 85.0, 300.0)
                        },
                        "vfd1_fail_to_start" => {
                            // Enable on, speed commanded, zero amps, pressure dropping
                            (78.0-fs*10.0, 78.0-fs*10.0, 78.0, 78.0, 8.0, 0.0, 0.0, 0.0, 1.0, 0.0, 78.0-fs*8.0, 3.0)
                        },
                        "vfd2_fail_to_start" => {
                            (78.0-fs*10.0, 78.0, 78.0-fs*10.0, 78.0, 0.0, 8.0, 0.0, 0.0, 0.0, 1.0, 78.0-fs*8.0, 3.0)
                        },
                        "short_cycling" => {
                            // On/off faster than normal, timer always short
                            let cyc = ((tf * 20.0).sin() > 0.0) as u8 as f32;
                            (78.0 + cyc*11.0*fs, 78.0+cyc*11.0*fs, 78.0, 78.0, cyc*7.0*fs, 0.0, cyc*8.0*fs, 0.0, cyc, 0.0, 78.0+cyc*8.0*fs, 20.0*fs)
                        },
                        _ => (83.0, 83.0, 83.0, 83.0, 5.0, 0.0, 7.0, 0.0, 1.0, 0.0, 83.0, 100.0),
                    };
                    let vals = [sys_p, p1_p, p2_p, ess_p, p1_spd, p2_spd, p1_a, p2_a, p1_en, p2_en, tank_p, timer];
                    for (i, &v) in vals.iter().enumerate() {
                        let noise = rng.gen_range(-0.005..0.005) * (ranges[i].1 - ranges[i].0);
                        data.push(((v + noise - ranges[i].0) / (ranges[i].1 - ranges[i].0 + 1e-7)).clamp(-0.5, 1.5));
                    }
                }
                let horizons: [f32; 6] = [0.5, 4.0, 12.0, 72.0, 168.0, 336.0];
                let mut labels = [0.0f32; 7];
                labels[0] = if deg > 0.15 { 1.0 } else { 0.0 };
                for (hi, &h) in horizons.iter().enumerate() { let hrs = fault.onset_hours * (1.0 - progress); labels[hi+1] = if hrs <= h { 1.0 } else { (1.0 - (hrs-h)/h).max(0.0) }; }
                samples.push(SentinelSample { data, labels, fault_name: fault.name.clone() });
            }
        }
        samples
    }

    pub fn sensor_names(&self) -> Vec<&'static str> {
        match self.equipment_type {
            EquipmentType::Boiler => vec!["loop_supply_t", "loop_return_t", "setpoint", "enable", "amps", "flow_switch", "flame_proven", "ignition_status", "oat", "run_timer"],
            EquipmentType::Chiller => vec!["evap_in", "evap_out", "cond_in", "cond_out", "comp_amps", "cw_flow", "chw_flow", "suction_p", "discharge_p", "subcool", "superheat", "oil_p"],
            EquipmentType::Pump => vec!["amps", "speed_rpm", "discharge_p", "suction_p", "vibration", "bearing_t", "flow"],
            EquipmentType::Ahu => vec!["oa_damper", "return_damper", "freeze_stat", "fan_enable", "fan_amps", "cw_valve", "hw_valve", "coil_temp", "coil_delta", "supply_t", "return_t", "mixed_t", "duct_static", "occupancy", "space_setpoint", "bldg_pressure_sp", "bldg_pressure"],
            EquipmentType::ZoneReheat => vec!["incoming_air", "damper", "hw_valve", "supply_t", "space_t", "fan_amps", "eht1_enable", "eht1_amps", "eht2_enable", "eht2_amps", "eht3_enable", "eht3_amps", "eht4_enable", "eht4_amps"],
            EquipmentType::BoosterPump => vec!["system_pressure", "pump1_pressure", "pump2_pressure", "ess_pressure", "pump1_speed_cmd", "pump2_speed_cmd", "pump1_current", "pump2_current", "pump1_enable", "pump2_enable", "tank_pressure", "sleep_wake_timer"],
            EquipmentType::HeatPump => vec!["supply_t", "return_t", "oat", "comp_amps", "rev_valve", "suction_p", "discharge_p", "subcool", "superheat", "liquid_line_t", "discharge_line_t", "defrost_status", "aux_heat_enable", "condenser_fan_amps"],
            _ => sentinel_datagen_ext::extended_sensor_names(&self.equipment_type)
                .unwrap_or_else(|| vec!["unknown"; self.equipment_type.sensor_count()]),
        }
    }
}
