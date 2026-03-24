//! Panoptes Synthetic Data Generator — Physics-Informed Facility Simulation
//!
//! # File
//! `crates/axonml/src/hvac/panoptes_datagen.rs`
//!
//! # Description
//! Generates realistic training data for the Panoptes facility-wide anomaly
//! detection model. Instead of random noise, this generator simulates actual
//! BAS control logic from Heritage Pointe of Warren — TMC valve modulation,
//! OA reset curves, freeze protection, lead/lag pump rotation, economizer
//! control, and mode transitions based on outdoor air temperature.
//!
//! Control logic derived from `/opt/LocationLogic/Warren/` BAS files.
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

use rand::Rng;
use rand::SeedableRng;

use super::panoptes::{
    FacilityConfig, FacilitySnapshot, EQUIP_AHU, EQUIP_BOILER, EQUIP_CHILLER, EQUIP_DOAS,
    EQUIP_FAN_COIL, EQUIP_PUMP, EQUIP_STEAM_BUNDLE,
};

// =============================================================================
// Operating Mode
// =============================================================================

/// HVAC operating mode based on outdoor air temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HvacMode {
    /// OAT < 40°F — freeze protection active, OA damper closed, HW 100%
    FreezeProtect,
    /// 40°F ≤ OAT < 55°F — heavy heating, minimal OA
    Heating,
    /// 55°F ≤ OAT < 65°F — mild heating, economizer may engage
    Mild,
    /// 65°F ≤ OAT < 75°F — economizer/free cooling
    Economizer,
    /// OAT ≥ 75°F — mechanical cooling, chiller active
    Cooling,
}

impl HvacMode {
    /// Determine mode from outdoor air temperature.
    pub fn from_oat(oat: f32) -> Self {
        if oat < 40.0 {
            HvacMode::FreezeProtect
        } else if oat < 55.0 {
            HvacMode::Heating
        } else if oat < 65.0 {
            HvacMode::Mild
        } else if oat < 75.0 {
            HvacMode::Economizer
        } else {
            HvacMode::Cooling
        }
    }
}

// =============================================================================
// TMC Simulator — Thermal Momentum Control
// =============================================================================

/// Simplified TMC output for valve position given current/target temps.
fn tmc_valve_output(current: f32, setpoint: f32, thermal_mass: f32, sensitivity: f32) -> f32 {
    let error = setpoint - current;
    let output = error * sensitivity * thermal_mass * 10.0;
    output.clamp(0.0, 100.0)
}

/// OA Reset: linearly interpolate setpoint between two outdoor temp bounds.
fn oa_reset(oat: f32, min_oat: f32, max_oat: f32, max_sp: f32, min_sp: f32) -> f32 {
    if oat <= min_oat {
        max_sp
    } else if oat >= max_oat {
        min_sp
    } else {
        max_sp - (oat - min_oat) / (max_oat - min_oat) * (max_sp - min_sp)
    }
}

// =============================================================================
// Fault Types
// =============================================================================

/// Types of faults that can be injected into facility data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaultType {
    /// No fault — normal operation.
    Normal,
    /// Valve stuck at a fixed position despite demand change.
    StuckValve {
        /// Which equipment has the stuck valve.
        equipment_slot: usize,
        /// Fixed valve position (0.0-1.0).
        position: f32,
    },
    /// Sensor reading drifting from true value.
    SensorDrift {
        /// Which equipment has the drifting sensor.
        equipment_slot: usize,
        /// Index of the drifting sensor within the equipment.
        sensor_idx: usize,
        /// Magnitude of the drift applied to readings.
        drift: f32,
    },
    /// Pump failure — amps drop to 0 while status shows running.
    PumpFailure {
        /// Which equipment has the failed pump.
        equipment_slot: usize,
    },
    /// Freeze protection not activating despite low temps.
    FreezeProtectFailure {
        /// Which equipment has the freeze protection failure.
        equipment_slot: usize,
    },
    /// Short cycling — rapid on/off toggling.
    ShortCycling {
        /// Which equipment is short cycling.
        equipment_slot: usize,
    },
    /// Cross-equipment: boiler drops but bundles don't respond.
    BoilerCascadeFailure,
    /// Cross-equipment: chiller supply rises but CW pump doesn't speed up.
    ChillerPumpMismatch,
    /// Coil fouling — low delta-T despite high demand.
    CoilFouling {
        /// Which equipment has the fouled coil.
        equipment_slot: usize,
    },
    /// Damper stuck — position doesn't match command.
    DamperStuck {
        /// Which equipment has the stuck damper.
        equipment_slot: usize,
        /// Fixed damper position (0.0-1.0).
        position: f32,
    },
}

// =============================================================================
// WarrenSimulator — Full Facility Simulation
// =============================================================================

/// Simulates Heritage Pointe of Warren facility sensor readings.
///
/// Generates realistic `FacilitySnapshot` data based on actual BAS control
/// logic, outdoor conditions, and time of day.
pub struct WarrenSimulator {
    config: FacilityConfig,
    rng_seed: u64,
}

impl WarrenSimulator {
    /// Creates a new Warren simulator.
    pub fn new(seed: u64) -> Self {
        Self {
            config: FacilityConfig::warren(),
            rng_seed: seed,
        }
    }

    /// Returns the facility config.
    pub fn config(&self) -> &FacilityConfig {
        &self.config
    }

    /// Generates a batch of normal operation snapshots across varied conditions.
    ///
    /// Simulates different outdoor temperatures (seasons), time of day,
    /// and occupancy patterns to cover the full operating envelope.
    pub fn generate_normal(&self, count: usize) -> Vec<FacilitySnapshot> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_seed);
        let mut snapshots = Vec::with_capacity(count);

        for i in 0..count {
            // Vary outdoor temp: -10°F to 100°F across samples
            let oat = -10.0 + (i as f32 / count as f32) * 110.0 + rng.gen_range(-5.0..5.0);
            let oat = oat.clamp(-10.0, 105.0);

            // Time of day affects occupancy (0-23)
            let hour = (i % 24) as f32;
            let occupied = (6.0..=22.0).contains(&hour);

            let snap = self.simulate_snapshot(&mut rng, oat, occupied, &FaultType::Normal);
            snapshots.push(snap);
        }

        snapshots
    }

    /// Generates snapshots with injected faults.
    ///
    /// Returns `(snapshot, fault_type, affected_slots)` for each sample.
    pub fn generate_with_faults(
        &self,
        count: usize,
        fault_ratio: f32,
    ) -> Vec<(FacilitySnapshot, FaultType, Vec<usize>)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_seed + 1000);
        let mut samples = Vec::with_capacity(count);

        for _i in 0..count {
            let oat = -10.0 + rng.gen_range(0.0..110.0);
            let hour = rng.gen_range(0.0..24.0);
            let occupied = (6.0..=22.0).contains(&hour);

            let (fault, affected) = if rng.gen::<f32>() < fault_ratio {
                self.random_fault(&mut rng)
            } else {
                (FaultType::Normal, vec![])
            };

            let snap = self.simulate_snapshot(&mut rng, oat, occupied, &fault);
            samples.push((snap, fault, affected));
        }

        samples
    }

    /// Generates a temporal sequence of snapshots (simulating time progression).
    ///
    /// OAT changes slowly, equipment responds according to control logic.
    pub fn generate_temporal_sequence(
        &self,
        length: usize,
        start_oat: f32,
        oat_drift_per_step: f32,
    ) -> Vec<FacilitySnapshot> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_seed + 2000);
        let mut snapshots = Vec::with_capacity(length);
        let mut oat = start_oat;

        for step in 0..length {
            let hour = ((step % 288) as f32 / 12.0) % 24.0; // 5-min intervals
            let occupied = (6.0..=22.0).contains(&hour);

            let snap = self.simulate_snapshot(&mut rng, oat, occupied, &FaultType::Normal);
            snapshots.push(snap);

            // OAT drifts slowly
            oat += oat_drift_per_step + rng.gen_range(-0.3..0.3);
            oat = oat.clamp(-15.0, 110.0);
        }

        snapshots
    }

    /// Generates temporal sequences with a fault injected partway through.
    ///
    /// Returns `(sequence, fault_onset_step, fault_type, affected_slots)`.
    /// The first `fault_onset` steps are normal, then the fault is active
    /// for the remaining steps. This simulates equipment degradation or
    /// sudden failures that temporal training should detect.
    pub fn generate_temporal_with_fault(
        &self,
        length: usize,
        start_oat: f32,
        oat_drift_per_step: f32,
        seed_offset: u64,
    ) -> (Vec<FacilitySnapshot>, usize, FaultType, Vec<usize>) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_seed + 3000 + seed_offset);
        let mut snapshots = Vec::with_capacity(length);
        let mut oat = start_oat;

        // Fault starts between 30-70% through the sequence
        let fault_onset = rng.gen_range(length * 3 / 10..length * 7 / 10);
        let (fault, affected) = self.random_fault(&mut rng);

        for step in 0..length {
            let hour = ((step % 288) as f32 / 12.0) % 24.0;
            let occupied = (6.0..=22.0).contains(&hour);

            let current_fault = if step >= fault_onset {
                &fault
            } else {
                &FaultType::Normal
            };

            let snap = self.simulate_snapshot(&mut rng, oat, occupied, current_fault);
            snapshots.push(snap);

            oat += oat_drift_per_step + rng.gen_range(-0.3..0.3);
            oat = oat.clamp(-15.0, 110.0);
        }

        (snapshots, fault_onset, fault, affected)
    }

    // =========================================================================
    // Core simulation
    // =========================================================================

    fn simulate_snapshot(
        &self,
        rng: &mut impl Rng,
        oat: f32,
        occupied: bool,
        fault: &FaultType,
    ) -> FacilitySnapshot {
        let n = self.config.num_equipment();
        let mut snap = FacilitySnapshot::new(n);
        let mode = HvacMode::from_oat(oat);

        // --- AHUs (slots 0-5) ---
        self.sim_ahus(rng, &mut snap, oat, mode, occupied, fault);

        // --- DOAS (slot 6) ---
        self.sim_doas(rng, &mut snap, oat, mode);

        // --- Boilers (slots 7-9) ---
        self.sim_boilers(rng, &mut snap, oat, mode, fault);

        // --- Steam Bundles (slots 10-18) ---
        self.sim_steam_bundles(rng, &mut snap, oat, mode, fault);

        // --- Fan Coils (slots 19-36) ---
        self.sim_fan_coils(rng, &mut snap, oat, mode, occupied, fault);

        // --- Pumps (slots 37-56) ---
        self.sim_pumps(rng, &mut snap, oat, mode, fault);

        // --- Chillers (slots 57-58) ---
        self.sim_chillers(rng, &mut snap, oat, mode, fault);

        snap
    }

    // =========================================================================
    // AHU simulation (6 units, slots 0-5)
    // =========================================================================

    fn sim_ahus(
        &self,
        rng: &mut impl Rng,
        snap: &mut FacilitySnapshot,
        oat: f32,
        mode: HvacMode,
        occupied: bool,
        fault: &FaultType,
    ) {
        // AHU setpoints vary by zone
        let ahu_configs: [(usize, &str, f32); 6] = [
            (0, "warren-ahu-6", 65.0), // A Wing Basement
            (1, "warren-ahu-1", 70.0), // Fahl AHU-1
            (2, "warren-ahu-4", 70.5), // Fahl Beauty Salon
            (3, "warren-ahu-2", 74.0), // Fahl Dining Room
            (4, "warren-ahu-5", 72.5), // McAllister Laundry
            (5, "warren-ahu-7", 83.5), // Natatorium (pool - high setpoint)
        ];

        for &(slot, id, setpoint) in &ahu_configs {
            let running = occupied || slot == 0 || slot == 5; // basement + pool always run
            let status = if running { 1.0 } else { 0.0 };

            // Space temp: hovers near setpoint ± noise
            let space_temp = if running {
                setpoint + rng.gen_range(-2.0..3.0)
            } else {
                setpoint + rng.gen_range(-4.0..6.0) // drifts more when off
            };

            // Supply temp: TMC-controlled
            let sat_target = if mode == HvacMode::FreezeProtect {
                65.0 // warm up
            } else {
                setpoint - rng.gen_range(0.0..5.0) // typically below setpoint
            };
            let supply_temp = sat_target + rng.gen_range(-1.0..1.0);

            // Mixed air: blend of OAT and return
            let return_temp = space_temp - rng.gen_range(0.0..2.0);
            let oa_pct = match mode {
                HvacMode::FreezeProtect => 0.0,
                HvacMode::Heating => 20.0 + rng.gen_range(0.0..10.0),
                HvacMode::Mild => 40.0 + rng.gen_range(0.0..20.0),
                HvacMode::Economizer => 80.0 + rng.gen_range(0.0..20.0),
                HvacMode::Cooling => 20.0 + rng.gen_range(0.0..15.0),
            };
            let mixed_temp = return_temp * (1.0 - oa_pct / 100.0)
                + oat * (oa_pct / 100.0)
                + rng.gen_range(-1.0..1.0);

            // Return damper: inverse of OA
            let ret_damper = (100.0 - oa_pct).max(0.0);

            // HW valve: TMC-modulated heating
            let hw_valve = match mode {
                HvacMode::FreezeProtect => 100.0,
                HvacMode::Heating => tmc_valve_output(supply_temp, sat_target, 2.0, 0.4),
                HvacMode::Mild => tmc_valve_output(supply_temp, sat_target, 2.0, 0.3),
                _ => 0.0,
            };

            // CW valve: cooling demand
            let cw_valve = match mode {
                HvacMode::Cooling => tmc_valve_output(sat_target, supply_temp, 1.5, 0.5),
                _ => 0.0,
            };

            // Amps: proportional to fan speed + valve work
            let amps = if running {
                3.0 + rng.gen_range(0.0..4.0) + (hw_valve + cw_valve) * 0.05
            } else {
                rng.gen_range(0.0..0.1)
            };

            // Apply fault if targeting this slot
            let mut values: Vec<Option<f32>> = vec![
                Some(setpoint),
                Some(space_temp),
                Some(supply_temp),
                Some(mixed_temp),
                Some(oa_pct),
                Some(ret_damper),
                Some(hw_valve),
                Some(cw_valve),
                Some(amps),
                Some(status),
                None,
                None,
            ];

            // Inject faults
            match fault {
                FaultType::StuckValve {
                    equipment_slot,
                    position,
                } if *equipment_slot == slot => {
                    values[6] = Some(*position); // HW valve stuck
                }
                FaultType::SensorDrift {
                    equipment_slot,
                    sensor_idx,
                    drift,
                } if *equipment_slot == slot => {
                    if let Some(Some(v)) = values.get(*sensor_idx) {
                        values[*sensor_idx] = Some(v + drift);
                    }
                }
                FaultType::DamperStuck {
                    equipment_slot,
                    position,
                } if *equipment_slot == slot => {
                    values[4] = Some(*position); // OA damper stuck
                }
                FaultType::FreezeProtectFailure { equipment_slot }
                    if *equipment_slot == slot && mode == HvacMode::FreezeProtect =>
                {
                    // Freeze protect should be active but isn't
                    values[4] = Some(80.0); // OA damper still open
                    values[6] = Some(0.0); // HW valve still closed
                }
                _ => {}
            }

            // Some sensors occasionally missing
            if rng.gen::<f32>() < 0.1 {
                values[1] = None;
            } // space temp sometimes missing
            if rng.gen::<f32>() < 0.15 {
                values[3] = None;
            } // mixed temp sometimes missing

            snap.set_equipment(slot, id, EQUIP_AHU, &values);
        }
    }

    // =========================================================================
    // DOAS simulation (1 unit, slot 6)
    // =========================================================================

    fn sim_doas(&self, rng: &mut impl Rng, snap: &mut FacilitySnapshot, oat: f32, mode: HvacMode) {
        let setpoint = 55.0;
        let supply = setpoint + rng.gen_range(-2.0..2.0);
        let heat_on = matches!(mode, HvacMode::Heating | HvacMode::FreezeProtect);
        let cool_on = matches!(mode, HvacMode::Cooling);

        snap.set_equipment(
            6,
            "warren-fahl-doas",
            EQUIP_DOAS,
            &[
                Some(setpoint),
                Some(supply),
                None, // space temp typically not wired
                Some(oat + rng.gen_range(-1.0..1.0)),
                Some(if heat_on { 1.0 } else { 0.0 }),
                Some(if cool_on { 1.0 } else { 0.0 }),
            ],
        );
    }

    // =========================================================================
    // Boiler simulation (3 units, slots 7-9)
    // =========================================================================

    fn sim_boilers(
        &self,
        rng: &mut impl Rng,
        snap: &mut FacilitySnapshot,
        _oat: f32,
        mode: HvacMode,
        fault: &FaultType,
    ) {
        let heating_demand = matches!(
            mode,
            HvacMode::FreezeProtect | HvacMode::Heating | HvacMode::Mild
        );

        // Header pressure: maintained by running boiler
        let mut header_psi = if heating_demand {
            85.0 + rng.gen_range(0.0..8.0)
        } else {
            80.0 + rng.gen_range(0.0..5.0)
        };

        // Flash tank temp
        let flash_tank = 108.0 + rng.gen_range(0.0..6.0);

        // Lead/lag based on simulated week number
        let lead_boiler = 0; // Simplified: boiler-1 leads

        for i in 0..3 {
            let slot = 7 + i;
            let is_lead = i == lead_boiler;
            let running = heating_demand && is_lead;

            let supply_temp = if running {
                120.0 + rng.gen_range(0.0..15.0)
            } else {
                85.0 + rng.gen_range(0.0..10.0)
            };

            let runtime = if running {
                rng.gen_range(5.0..200.0)
            } else {
                0.0
            };

            let mut values: Vec<Option<f32>> = vec![
                Some(supply_temp),
                Some(flash_tank),
                Some(header_psi),
                Some(if running { 1.0 } else { 0.0 }),
                Some(if is_lead { 0.0 } else { 1.0 }), // lead=0, lag=1
                Some(runtime),
                Some(0.0), // safeties idle
            ];

            // Cascade failure: boiler drops but pressure stays (impossible)
            if matches!(fault, FaultType::BoilerCascadeFailure) && is_lead {
                values[0] = Some(75.0); // supply drops
                                        // but header_psi stays artificially high (sensor stuck)
                header_psi = 90.0;
            }

            let id = match i {
                0 => "warren-boiler-1",
                1 => "warren-boiler-2",
                _ => "warren-boiler-3",
            };
            snap.set_equipment(slot, id, EQUIP_BOILER, &values);
        }
    }

    // =========================================================================
    // Steam Bundle simulation (9 units, slots 10-18)
    // =========================================================================

    fn sim_steam_bundles(
        &self,
        rng: &mut impl Rng,
        snap: &mut FacilitySnapshot,
        oat: f32,
        mode: HvacMode,
        fault: &FaultType,
    ) {
        let bundle_configs: [(usize, &str, f32, f32); 9] = [
            (10, "warren-steambundle-1", 135.5, 145.0), // A Wing Basement
            (11, "warren-steambundle-5", 118.5, 125.0), // A Wing Domestic
            (12, "warren-steambundle-4", 145.0, 155.0), // Chompson
            (13, "warren-steambundle-fahl", 135.5, 155.0), // Fahl Wing
            (14, "warren-steambundle-6", 135.5, 145.0), // Innis Wing
            (15, "warren-steambundle-7", 115.0, 135.0), // McAllister Domestic
            (16, "warren-steambundle-3", 135.5, 145.0), // McAllister Laundry
            (17, "warren-steambundle-8", 145.0, 150.0), // McAllister Residents
            (18, "warren-steambundle-2", 135.0, 150.0), // Souder Wing
        ];

        for &(slot, id, base_sp, max_supply) in &bundle_configs {
            // OA Reset: adjust setpoint based on outdoor temp
            let setpoint = oa_reset(oat, 20.0, 60.0, base_sp + 15.0, base_sp);

            // Supply temp: tracks setpoint via TMC valve control
            let demand = match mode {
                HvacMode::FreezeProtect | HvacMode::Heating => 0.6 + rng.gen_range(0.0..0.4),
                HvacMode::Mild => 0.2 + rng.gen_range(0.0..0.3),
                _ => rng.gen_range(0.0..0.1),
            };

            let supply_temp = setpoint + rng.gen_range(-5.0..8.0) * demand;
            let supply_temp = supply_temp.min(max_supply);

            let return_temp = if demand > 0.1 {
                supply_temp - rng.gen_range(3.0..12.0) // normal delta-T
            } else {
                supply_temp - rng.gen_range(0.0..3.0)
            };

            // Valve 1: TMC-controlled
            let valve1 = tmc_valve_output(supply_temp, setpoint, 4.0, 0.2) * demand;

            // Valve 2: stages in when valve1 > 85%
            let valve2 = if valve1 > 85.0 {
                (valve1 - 85.0) * 6.67 // 0-100% over remaining range
            } else {
                0.0
            };

            let mut values: Vec<Option<f32>> = vec![
                Some(setpoint),
                Some(supply_temp),
                if rng.gen::<f32>() < 0.3 {
                    None
                } else {
                    Some(return_temp)
                },
                Some(valve1),
                if valve2 > 0.0 { Some(valve2) } else { None },
            ];

            // Cascade failure: boiler dropped but bundle supply stays high
            if matches!(fault, FaultType::BoilerCascadeFailure) {
                values[1] = Some(setpoint + 10.0); // supply unrealistically high
                values[3] = Some(0.0); // valve closed — contradicts high supply
            }

            // Coil fouling: low delta-T despite high demand
            if matches!(fault, FaultType::CoilFouling { equipment_slot } if *equipment_slot == slot)
            {
                let bad_return = supply_temp - rng.gen_range(0.5..1.5); // tiny delta-T
                values[2] = Some(bad_return);
                values[3] = Some(95.0); // valve wide open
            }

            snap.set_equipment(slot, id, EQUIP_STEAM_BUNDLE, &values);
        }
    }

    // =========================================================================
    // Fan Coil simulation (18 units, slots 19-36)
    // =========================================================================

    fn sim_fan_coils(
        &self,
        rng: &mut impl Rng,
        snap: &mut FacilitySnapshot,
        _oat: f32,
        mode: HvacMode,
        occupied: bool,
        fault: &FaultType,
    ) {
        let fc_configs: [(usize, &str, f32); 18] = [
            (19, "warren-fancoil-12", 73.5), // Activity Room 1
            (20, "warren-fancoil-13", 73.5),
            (21, "warren-fancoil-14", 74.0), // Activity Room 2
            (22, "warren-fancoil-15", 74.0),
            (23, "warren-fancoil-7", 72.5), // Chapel North
            (24, "warren-fancoil-8", 72.5),
            (25, "warren-fancoil-9", 72.5), // Chapel South
            (26, "warren-fancoil-10", 72.5),
            (27, "warren-fancoil-5", 70.0), // Cove Kitchen
            (28, "warren-fancoil-3", 71.0), // Cove North
            (29, "warren-fancoil-4", 71.0),
            (30, "warren-fancoil-1", 71.0), // Cove South
            (31, "warren-fancoil-2", 71.0),
            (32, "warren-fancoil-16", 70.5), // Executive 1-2
            (33, "warren-fancoil-17", 71.0),
            (34, "warren-fancoil-18", 71.0), // Executive 3
            (35, "warren-fancoil-6", 70.5),  // Mail Room
            (36, "warren-fancoil-11", 70.0), // Vestibule
        ];

        for &(slot, id, setpoint) in &fc_configs {
            // Determine if this FC is running
            let space_temp = setpoint + rng.gen_range(-2.0..4.0);
            let needs_heating = space_temp < setpoint - 0.5;
            let needs_cooling = space_temp > setpoint + 0.5;
            let running = occupied && (needs_heating || needs_cooling);

            let status = if running { 1.0 } else { 0.0 };

            // Supply temp
            let supply_temp = if running && needs_heating {
                space_temp + rng.gen_range(5.0..20.0) // heating: supply > space
            } else if running && needs_cooling {
                space_temp - rng.gen_range(5.0..15.0) // cooling: supply < space
            } else {
                space_temp + rng.gen_range(-3.0..3.0) // idle: near space temp
            };

            // Valve positions
            let hw_valve = if needs_heating && running {
                tmc_valve_output(space_temp, setpoint, 2.0, 0.5)
            } else {
                0.0
            };

            let cw_valve = if needs_cooling
                && running
                && matches!(mode, HvacMode::Cooling | HvacMode::Economizer)
            {
                tmc_valve_output(setpoint, space_temp, 1.5, 0.5)
            } else {
                0.0
            };

            // OA damper: some FCs have OA, some don't
            let has_oa = slot <= 26 || slot >= 35; // chapel + activity rooms + vestibule
            let oa_damper = if has_oa && running {
                match mode {
                    HvacMode::FreezeProtect => 0.0,
                    HvacMode::Economizer => 100.0,
                    _ => {
                        if occupied {
                            100.0
                        } else {
                            0.0
                        }
                    }
                }
            } else if has_oa {
                0.0
            } else {
                -1.0 // sentinel for "no OA damper"
            };

            // Amps
            let amps = if running {
                2.0 + rng.gen_range(0.0..4.0)
            } else {
                rng.gen_range(0.0..0.1)
            };

            let mut values: Vec<Option<f32>> = vec![
                Some(setpoint),
                Some(space_temp),
                Some(supply_temp),
                Some(hw_valve),
                Some(cw_valve),
                if oa_damper >= 0.0 {
                    Some(oa_damper)
                } else {
                    None
                },
                Some(amps),
                Some(status),
                None, // fan speed
            ];

            // Short cycling fault
            if matches!(fault, FaultType::ShortCycling { equipment_slot } if *equipment_slot == slot)
            {
                // Rapidly toggling: show contradictory state
                values[7] = Some(if rng.gen::<f32>() > 0.5 { 1.0 } else { 0.0 });
                values[6] = Some(rng.gen_range(0.0..8.0)); // erratic amps
            }

            // Stuck valve
            if matches!(fault, FaultType::StuckValve { equipment_slot, position } if *equipment_slot == slot)
            {
                if let FaultType::StuckValve { position, .. } = fault {
                    values[3] = Some(*position);
                }
            }

            snap.set_equipment(slot, id, EQUIP_FAN_COIL, &values);
        }
    }

    // =========================================================================
    // Pump simulation (20 units, slots 37-56)
    // =========================================================================

    fn sim_pumps(
        &self,
        rng: &mut impl Rng,
        snap: &mut FacilitySnapshot,
        _oat: f32,
        mode: HvacMode,
        fault: &FaultType,
    ) {
        let pump_configs: [(usize, &str, bool, bool); 20] = [
            // (slot, id, is_hw, is_lead)
            (37, "warren-cwbooster-1", false, true),
            (38, "warren-cwbooster-2", false, false),
            (39, "warren-cwpump-5", false, true),
            (40, "warren-cwpump-6", false, false),
            (41, "warren-hwpump-7", true, true),
            (42, "warren-hwpump-8", true, false),
            (43, "warren-hwpump-9", true, false),
            (44, "warren-hwpump-10", true, true),
            (45, "warren-hwpump-11", true, true),
            (46, "warren-hwpump-12", true, true),
            (47, "warren-chwpump-3", false, true),
            (48, "warren-chwpump-4", false, false),
            (49, "warren-cwpump-3", false, true),
            (50, "warren-cwpump-4", false, false),
            (51, "warren-hwpump-5", true, true),
            (52, "warren-hwpump-6", true, false),
            (53, "warren-hwpump-1", true, true),
            (54, "warren-hwpump-2", true, true),
            (55, "warren-hwpump-3", true, false),
            (56, "warren-hwpump-4", true, true),
        ];

        let heating = matches!(
            mode,
            HvacMode::FreezeProtect | HvacMode::Heating | HvacMode::Mild
        );
        let cooling = matches!(mode, HvacMode::Cooling | HvacMode::Economizer);

        for &(slot, id, is_hw, is_lead) in &pump_configs {
            // Pump runs if: it's the lead AND there's demand for its type
            let should_run = is_lead && ((is_hw && heating) || (!is_hw && cooling));
            let running = should_run;

            let speed = if running {
                30.0 + rng.gen_range(0.0..70.0)
            } else {
                0.0
            };

            let amps = if running {
                1.5 + rng.gen_range(0.0..12.0) + speed * 0.1
            } else {
                rng.gen_range(0.0..0.15)
            };

            let psi_sp = 15.0;
            let discharge_psi = if running {
                psi_sp - rng.gen_range(0.0..3.0)
            } else {
                psi_sp
            };

            let runtime = if running {
                rng.gen_range(10.0..250.0)
            } else {
                0.0
            };

            let mut values: Vec<Option<f32>> = vec![
                Some(speed),
                Some(amps),
                Some(psi_sp),
                if running { Some(discharge_psi) } else { None },
                Some(runtime),
                Some(if running { 1.0 } else { 0.0 }),
                None, // flow
            ];

            // Pump failure fault
            if matches!(fault, FaultType::PumpFailure { equipment_slot } if *equipment_slot == slot)
            {
                values[0] = Some(speed); // speed command still showing
                values[1] = Some(0.0); // but amps = 0 (motor not turning)
                values[5] = Some(1.0); // status still shows running
            }

            // Chiller-pump mismatch
            if matches!(fault, FaultType::ChillerPumpMismatch) && !is_hw && is_lead {
                values[0] = Some(10.0); // pump barely running
                values[1] = Some(0.5); // low amps
            }

            snap.set_equipment(slot, id, EQUIP_PUMP, &values);
        }
    }

    // =========================================================================
    // Chiller simulation (2 units, slots 57-58)
    // =========================================================================

    fn sim_chillers(
        &self,
        rng: &mut impl Rng,
        snap: &mut FacilitySnapshot,
        oat: f32,
        mode: HvacMode,
        fault: &FaultType,
    ) {
        let chiller_configs: [(usize, &str, f32); 2] = [
            (57, "warren-chiller-2", 47.5), // Innis Wing
            (58, "warren-chiller-1", 44.0), // Souder Wing
        ];

        let chiller_allowed = oat >= 65.0; // Lockout below 65°F OAT

        for &(slot, id, setpoint) in &chiller_configs {
            let running = chiller_allowed && matches!(mode, HvacMode::Cooling);
            let enabled = chiller_allowed;

            let supply_temp = if running {
                setpoint + rng.gen_range(-2.0..5.0)
            } else {
                80.0 + rng.gen_range(0.0..20.0) // warm when idle
            };

            let return_temp = if running {
                supply_temp + rng.gen_range(5.0..15.0)
            } else {
                supply_temp + rng.gen_range(0.0..5.0)
            };

            let pressure = if running {
                15.0 + rng.gen_range(-1.0..2.0)
            } else {
                rng.gen_range(0.0..5.0)
            };

            let amps = if running {
                rng.gen_range(5.0..25.0)
            } else {
                0.0
            };

            let runtime = if running {
                rng.gen_range(10.0..100.0)
            } else {
                0.0
            };

            let mut values: Vec<Option<f32>> = vec![
                Some(setpoint),
                Some(supply_temp),
                if running { Some(return_temp) } else { None },
                Some(pressure),
                if running { Some(amps) } else { None },
                Some(if running { 1.0 } else { 0.0 }),
                Some(runtime),
                Some(1.0), // interlocks OK
                Some(if enabled { 1.0 } else { 0.0 }),
            ];

            // Chiller-pump mismatch: chiller supply rising but nobody cares
            if matches!(fault, FaultType::ChillerPumpMismatch) && running {
                values[1] = Some(setpoint + 15.0); // supply way above setpoint
            }

            snap.set_equipment(slot, id, EQUIP_CHILLER, &values);
        }
    }

    // =========================================================================
    // Random fault generation
    // =========================================================================

    fn random_fault(&self, rng: &mut impl Rng) -> (FaultType, Vec<usize>) {
        let fault_type = rng.gen_range(0..10);
        match fault_type {
            0 => {
                // Stuck valve on random AHU
                let slot = rng.gen_range(0..6);
                let pos = rng.gen_range(0.0..100.0);
                (
                    FaultType::StuckValve {
                        equipment_slot: slot,
                        position: pos,
                    },
                    vec![slot],
                )
            }
            1 => {
                // Sensor drift on random equipment
                let slot = rng.gen_range(0..59);
                let sensor = rng.gen_range(0..3);
                let drift = rng.gen_range(5.0..25.0) * if rng.gen::<bool>() { 1.0 } else { -1.0 };
                (
                    FaultType::SensorDrift {
                        equipment_slot: slot,
                        sensor_idx: sensor,
                        drift,
                    },
                    vec![slot],
                )
            }
            2 => {
                // Pump failure
                let slot = rng.gen_range(37..57);
                (
                    FaultType::PumpFailure {
                        equipment_slot: slot,
                    },
                    vec![slot],
                )
            }
            3 => {
                // Freeze protect failure on AHU
                let slot = rng.gen_range(0..6);
                (
                    FaultType::FreezeProtectFailure {
                        equipment_slot: slot,
                    },
                    vec![slot],
                )
            }
            4 => {
                // Short cycling on fan coil
                let slot = rng.gen_range(19..37);
                (
                    FaultType::ShortCycling {
                        equipment_slot: slot,
                    },
                    vec![slot],
                )
            }
            5 => {
                // Boiler cascade failure — affects boilers + all steam bundles
                let mut affected: Vec<usize> = (7..19).collect();
                affected.push(7); // lead boiler
                (FaultType::BoilerCascadeFailure, affected)
            }
            6 => {
                // Chiller-pump mismatch — affects chillers + CW pumps
                let affected = vec![47, 48, 49, 50, 57, 58];
                (FaultType::ChillerPumpMismatch, affected)
            }
            7 => {
                // Coil fouling on steam bundle
                let slot = rng.gen_range(10..19);
                (
                    FaultType::CoilFouling {
                        equipment_slot: slot,
                    },
                    vec![slot],
                )
            }
            8 => {
                // Damper stuck on AHU
                let slot = rng.gen_range(0..6);
                let pos = if rng.gen::<bool>() { 0.0 } else { 100.0 };
                (
                    FaultType::DamperStuck {
                        equipment_slot: slot,
                        position: pos,
                    },
                    vec![slot],
                )
            }
            _ => {
                // Stuck valve on fan coil
                let slot = rng.gen_range(19..37);
                let pos = rng.gen_range(0.0..100.0);
                (
                    FaultType::StuckValve {
                        equipment_slot: slot,
                        position: pos,
                    },
                    vec![slot],
                )
            }
        }
    }
}

// =============================================================================
// Training Data Builder
// =============================================================================

/// Builds training datasets for Panoptes from the Warren simulator.
pub struct PanoptesTrainingData {
    /// Normal snapshots (target: all zeros).
    pub normal: Vec<FacilitySnapshot>,
    /// Fault snapshots with labels.
    pub faults: Vec<(FacilitySnapshot, FaultType, Vec<usize>)>,
}

impl PanoptesTrainingData {
    /// Generates a complete training dataset.
    ///
    /// # Arguments
    /// * `normal_count` - Number of normal operation samples
    /// * `fault_count` - Number of fault samples
    /// * `seed` - Random seed for reproducibility
    pub fn generate(normal_count: usize, fault_count: usize, seed: u64) -> Self {
        let sim = WarrenSimulator::new(seed);
        let normal = sim.generate_normal(normal_count);
        let faults = sim.generate_with_faults(fault_count, 1.0); // 100% fault rate

        Self { normal, faults }
    }

    /// Total sample count.
    pub fn len(&self) -> usize {
        self.normal.len() + self.faults.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.normal.is_empty() && self.faults.is_empty()
    }

    /// Returns target anomaly scores for a fault sample.
    ///
    /// Normal equipment gets 0.0, affected equipment gets 1.0.
    pub fn fault_target(num_equipment: usize, affected_slots: &[usize]) -> Vec<f32> {
        let mut target = vec![0.0; num_equipment];
        for &slot in affected_slots {
            if slot < num_equipment {
                target[slot] = 1.0;
            }
        }
        target
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hvac::panoptes::MAX_SENSORS;

    #[test]
    fn test_hvac_mode() {
        assert_eq!(HvacMode::from_oat(30.0), HvacMode::FreezeProtect);
        assert_eq!(HvacMode::from_oat(50.0), HvacMode::Heating);
        assert_eq!(HvacMode::from_oat(60.0), HvacMode::Mild);
        assert_eq!(HvacMode::from_oat(70.0), HvacMode::Economizer);
        assert_eq!(HvacMode::from_oat(80.0), HvacMode::Cooling);
    }

    #[test]
    fn test_oa_reset() {
        let sp = oa_reset(20.0, 20.0, 60.0, 150.0, 135.0);
        assert!((sp - 150.0).abs() < 0.01);
        let sp = oa_reset(60.0, 20.0, 60.0, 150.0, 135.0);
        assert!((sp - 135.0).abs() < 0.01);
        let sp = oa_reset(40.0, 20.0, 60.0, 150.0, 135.0);
        assert!((sp - 142.5).abs() < 0.01);
    }

    #[test]
    fn test_warren_simulator_normal() {
        let sim = WarrenSimulator::new(42);
        let snaps = sim.generate_normal(100);
        assert_eq!(snaps.len(), 100);
        assert_eq!(snaps[0].num_equipment, 59);

        // Verify equipment types are set
        assert_eq!(snaps[0].equip_types[0], EQUIP_AHU);
        assert_eq!(snaps[0].equip_types[6], EQUIP_DOAS);
        assert_eq!(snaps[0].equip_types[7], EQUIP_BOILER);
        assert_eq!(snaps[0].equip_types[10], EQUIP_STEAM_BUNDLE);
        assert_eq!(snaps[0].equip_types[19], EQUIP_FAN_COIL);
        assert_eq!(snaps[0].equip_types[37], EQUIP_PUMP);
        assert_eq!(snaps[0].equip_types[57], EQUIP_CHILLER);
    }

    #[test]
    fn test_warren_simulator_faults() {
        let sim = WarrenSimulator::new(42);
        let samples = sim.generate_with_faults(50, 0.5);
        assert_eq!(samples.len(), 50);

        let fault_count = samples
            .iter()
            .filter(|(_, f, _)| *f != FaultType::Normal)
            .count();
        // With 50% fault ratio, expect roughly 25 faults (±10)
        assert!(fault_count > 10, "Too few faults: {fault_count}");
        assert!(fault_count < 45, "Too many faults: {fault_count}");
    }

    #[test]
    fn test_temporal_sequence() {
        let sim = WarrenSimulator::new(42);
        let seq = sim.generate_temporal_sequence(12, 70.0, -0.5);
        assert_eq!(seq.len(), 12);
        assert_eq!(seq[0].num_equipment, 59);
    }

    #[test]
    fn test_training_data() {
        let data = PanoptesTrainingData::generate(100, 50, 42);
        assert_eq!(data.normal.len(), 100);
        assert_eq!(data.faults.len(), 50);
        assert_eq!(data.len(), 150);

        let target = PanoptesTrainingData::fault_target(59, &[0, 10, 11]);
        assert_eq!(target[0], 1.0);
        assert_eq!(target[10], 1.0);
        assert_eq!(target[11], 1.0);
        assert_eq!(target[5], 0.0);
    }

    #[test]
    fn test_freeze_protect_mode() {
        let sim = WarrenSimulator::new(42);
        // Generate snapshot at very cold OAT
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let snap = sim.simulate_snapshot(&mut rng, 10.0, true, &FaultType::Normal);

        // AHU-6 (slot 0): OA damper should be ~0%, HW valve should be high
        let base = 0 * MAX_SENSORS;
        let oa_damper = snap.features[base + 4];
        let hw_valve = snap.features[base + 6];
        assert!(
            oa_damper < 5.0,
            "OA damper should be ~0 in freeze protect, got {oa_damper}"
        );
        assert!(
            hw_valve > 50.0,
            "HW valve should be high in freeze protect, got {hw_valve}"
        );
    }
}
