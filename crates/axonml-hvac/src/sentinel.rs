//! Sentinel — Temporal Conv1D Anomaly Detector + Predictive Failure Model
//!
//! Replaces LSTM (Nehebkau) and GRU (Medjed) with an H8-native architecture
//! that compiles to single-context HEF at 3K+ FPS.
//!
//! Architecture: Conv1d temporal feature extraction → Dense classification
//!   Input:  (batch, timesteps=8, n_features)
//!   Conv1d layers capture temporal patterns (trends, rate-of-change, oscillations)
//!   Dense head produces anomaly flag + 6 predictive failure horizons
//!   Output: (batch, 7) → [anomaly, fail_30m, fail_4h, fail_12h, fail_3d, fail_7d, fail_14d]
//!
//! One Sentinel instance per equipment type, each with type-specific sensor count.
//!
//! Copyright (c) 2026 Andrew Jewell Sr. / AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060

use axonml_autograd::Variable;
use axonml_nn::layers::conv::Conv1d;
use axonml_nn::layers::dropout::Dropout;
use axonml_nn::layers::linear::Linear;
use axonml_nn::layers::norm::BatchNorm1d;
use axonml_nn::parameter::Parameter;
use axonml_nn::Module;

pub const TIMESTEPS: usize = 8;
pub const NUM_OUTPUTS: usize = 7; // anomaly + 6 prediction horizons

/// Equipment types supported by Sentinel models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EquipmentType {
    Boiler,
    Chiller,
    Pump,
    Ahu,
    ZoneReheat,
    HeatPump,
    BoosterPump,
    Rtu,
    Doas,
    CascadeBoiler,
    WaterCooledChiller,
    CoolingTower,
    ResiFurnace,
    ResiHeatPump,
    ResiBoiler,
    VfdPumpPack,
}

impl EquipmentType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Boiler => "boiler",
            Self::Chiller => "chiller",
            Self::Pump => "pump",
            Self::Ahu => "ahu",
            Self::ZoneReheat => "zone_reheat",
            Self::HeatPump => "heat_pump",
            Self::BoosterPump => "booster_pump",
            Self::Rtu => "rtu",
            Self::Doas => "doas",
            Self::CascadeBoiler => "cascade_boiler",
            Self::WaterCooledChiller => "water_cooled_chiller",
            Self::CoolingTower => "cooling_tower",
            Self::ResiFurnace => "resi_furnace",
            Self::ResiHeatPump => "resi_heat_pump",
            Self::ResiBoiler => "resi_boiler",
            Self::VfdPumpPack => "vfd_pump_pack",
        }
    }

    pub fn sensor_count(&self) -> usize {
        match self {
            Self::Boiler => 10,     // loop_supply_t, loop_return_t, setpoint, enable, amps, flow_switch, flame_proven, ignition_status, oat, run_timer
            Self::Chiller => 12,    // evap_in/out, cond_in/out, comp_amps, cw_flow, chw_flow, suction_p, discharge_p, subcool, superheat, oil_p
            Self::Pump => 7,        // amps, speed_rpm, discharge_p, suction_p, vibration, bearing_t, flow
            Self::Ahu => 17,        // oa_damper, return_damper, freeze_stat, fan_enable, fan_amps, cw_valve, hw_valve, coil_temp, coil_delta, supply_t, return_t, mixed_t, duct_static, occupancy, space_setpoint, bldg_pressure_sp, bldg_pressure
            Self::ZoneReheat => 14, // incoming_air, damper, hw_valve, supply_t, space_t, fan_amps, eht1_enable, eht1_amps, eht2_enable, eht2_amps, eht3_enable, eht3_amps, eht4_enable, eht4_amps
            Self::HeatPump => 14,   // supply_t, return_t, oat, comp_amps, rev_valve, suction_p, discharge_p, subcool, superheat, liquid_line_t, discharge_line_t, defrost_status, aux_heat_enable, condenser_fan_amps
            Self::BoosterPump => 12, // system_pressure, pump1_pressure, pump2_pressure, ess_pressure, pump1_speed_cmd, pump2_speed_cmd, pump1_current, pump2_current, pump1_enable, pump2_enable, tank_pressure, sleep_wake_timer
            Self::Rtu => 10,         // supply_t, mixed_t, oat, fan_amps, dx_stage1, dx_stage2, gas_heat, burner_amps, filter_dp, return_t
            Self::Doas => 10,        // supply_t, oat, fan_amps, gas_stage1, gas_stage2, dx_stage1, dx_stage2, dx1_amps, dx2_amps, filter_dp
            Self::CascadeBoiler => 12, // common_supply, common_return, oat, b1_supply, b2_entering, b1_mod_v, b2_mod_v, b1_enable, b2_enable, pump1_amps, pump2_amps, system_dp
            Self::WaterCooledChiller => 12, // ecew, lcw, ecw_tower, lcw_tower, sat_cond, sat_suct, evap_dp, comp1_amps, comp2_amps, tower_fan_v, chw_pump_amps, cw_pump_amps
            Self::CoolingTower => 8, // ecw, lcw, oat_wetbulb, fan_speed_v, fan_amps, cw_pump_amps, basin_level, vibration
            Self::ResiFurnace => 8,  // supply_t, return_t, furnace_amps, fan_enable, heat_call, cool_call, filter_loading_ratio, inducer_run_sec
            Self::ResiHeatPump => 8, // supply_t, handler_amps, comp_heating, aux_heat_cmd, fan_active, oat, defrost_active, cool_mode
            Self::ResiBoiler => 8,   // supply_t, return_t, burner_active, system_pressure, flow_switch, zone_call, burner_run_sec, temp_rise_rate
            Self::VfdPumpPack => 12, // suction_p, p1_disch_p, p2_disch_p, plant_dp, loop_p, loop_sp, p1_speed_v, p2_speed_v, p1_amps, p2_amps, p1_enable, p2_enable
        }
    }

    pub fn all() -> &'static [EquipmentType] {
        &[
            Self::Boiler, Self::Chiller, Self::Pump,
            Self::Ahu, Self::ZoneReheat, Self::HeatPump,
            Self::BoosterPump, Self::Rtu, Self::Doas,
            Self::CascadeBoiler, Self::WaterCooledChiller,
            Self::CoolingTower, Self::ResiFurnace,
            Self::ResiHeatPump, Self::ResiBoiler, Self::VfdPumpPack,
        ]
    }
}

/// Prediction horizon labels.
pub const HORIZON_NAMES: [&str; 6] = ["30min", "4hr", "12hr", "3day", "7day", "14day"];

/// Sentinel model — temporal Conv1D architecture used by Nehebkau (anomaly detector)
/// and Medjed (predictive failure detector). Exported as nehebkau_{type}.hef / medjed_{type}.hef.
pub struct Sentinel {
    pub equipment_type: EquipmentType,
    // Temporal feature extraction (Conv1d: in_channels = n_features, operates across timesteps)
    pub conv1: Conv1d,
    pub bn1: BatchNorm1d,
    pub conv2: Conv1d,
    pub bn2: BatchNorm1d,
    pub conv3: Conv1d,
    // Classification head
    pub drop1: Dropout,
    pub fc1: Linear,
    pub drop2: Dropout,
    pub fc2: Linear,
}

impl Sentinel {
    pub fn new(equipment_type: EquipmentType) -> Self {
        let n_feat = equipment_type.sensor_count();
        let conv1 = Conv1d::new(n_feat, 64, 3);
        let bn1 = BatchNorm1d::new(64);
        let conv2 = Conv1d::new(64, 128, 3);
        let bn2 = BatchNorm1d::new(128);
        let conv3 = Conv1d::new(128, 64, 3);
        // After 3 conv layers with kernel=3, no padding: length shrinks 8→6→4→2
        // Flatten: 64 channels × 2 timesteps = 128
        let drop1 = Dropout::new(0.3);
        let fc1 = Linear::new(128, 64);
        let drop2 = Dropout::new(0.2);
        let fc2 = Linear::new(64, NUM_OUTPUTS);

        Self { equipment_type, conv1, bn1, conv2, bn2, conv3, drop1, fc1, drop2, fc2 }
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        // x: (batch, timesteps=8, n_features)
        let x = x.transpose(1, 2);

        let x = self.conv1.forward(&x).relu();
        let x = self.bn1.forward(&x);
        let x = self.conv2.forward(&x).relu();
        let x = self.bn2.forward(&x);
        let x = self.conv3.forward(&x).relu();

        let batch = x.shape()[0];
        let flat_size = x.shape()[1] * x.shape()[2];
        let x = x.reshape(&[batch, flat_size]);

        let x = self.drop1.forward(&x);
        let x = self.fc1.forward(&x).relu();
        let x = self.drop2.forward(&x);
        let x = self.fc2.forward(&x).sigmoid();

        x
    }

    pub fn param_count(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

impl Module for Sentinel {
    fn forward(&self, input: &Variable) -> Variable {
        Sentinel::forward(self, input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.conv3.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }

    fn named_parameters(&self) -> std::collections::HashMap<String, Parameter> {
        let mut map = std::collections::HashMap::new();
        map.insert("conv1.weight".into(), self.conv1.weight.clone());
        if let Some(ref b) = self.conv1.bias { map.insert("conv1.bias".into(), b.clone()); }
        map.insert("bn1.weight".into(), self.bn1.weight.clone());
        map.insert("bn1.bias".into(), self.bn1.bias.clone());
        map.insert("conv2.weight".into(), self.conv2.weight.clone());
        if let Some(ref b) = self.conv2.bias { map.insert("conv2.bias".into(), b.clone()); }
        map.insert("bn2.weight".into(), self.bn2.weight.clone());
        map.insert("bn2.bias".into(), self.bn2.bias.clone());
        map.insert("conv3.weight".into(), self.conv3.weight.clone());
        if let Some(ref b) = self.conv3.bias { map.insert("conv3.bias".into(), b.clone()); }
        map.insert("fc1.weight".into(), self.fc1.weight.clone());
        if let Some(ref b) = self.fc1.bias { map.insert("fc1.bias".into(), b.clone()); }
        map.insert("fc2.weight".into(), self.fc2.weight.clone());
        if let Some(ref b) = self.fc2.bias { map.insert("fc2.bias".into(), b.clone()); }
        map
    }
}
