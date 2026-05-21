//! Extended equipment type datagen for RTU, DOAS, CascadeBoiler, WaterCooledChiller,
//! CoolingTower, ResiFurnace, ResiHeatPump, ResiBoiler, VfdPumpPack.
//!
//! Each type uses the generic generate path with physics-informed fault scenarios.
//! Per-type generators to be added as accuracy requires.

use super::sentinel::{EquipmentType, TIMESTEPS, NUM_OUTPUTS};
use super::sentinel_datagen::{FaultScenario, SentinelSample};
use rand::Rng;
use rand::SeedableRng;

pub fn extended_normal_ranges(eq: &EquipmentType) -> Option<Vec<(f32, f32)>> {
    match eq {
        EquipmentType::Rtu => Some(vec![
            (45.0, 110.0), (45.0, 85.0), (-10.0, 115.0), (0.0, 25.0), (0.0, 1.0),
            (0.0, 1.0), (0.0, 1.0), (0.0, 10.0), (0.1, 2.0), (65.0, 82.0),
        ]),
        EquipmentType::Doas => Some(vec![
            (45.0, 140.0), (-10.0, 115.0), (0.0, 30.0), (0.0, 1.0), (0.0, 1.0),
            (0.0, 1.0), (0.0, 1.0), (0.0, 30.0), (0.0, 30.0), (0.1, 2.0),
        ]),
        EquipmentType::CascadeBoiler => Some(vec![
            (100.0, 200.0), (80.0, 180.0), (-10.0, 100.0), (100.0, 200.0), (80.0, 180.0),
            (0.0, 10.0), (0.0, 10.0), (0.0, 1.0), (0.0, 1.0), (0.0, 20.0),
            (0.0, 20.0), (0.0, 50.0),
        ]),
        EquipmentType::WaterCooledChiller => Some(vec![
            (38.0, 60.0), (38.0, 55.0), (65.0, 100.0), (75.0, 110.0), (80.0, 140.0),
            (30.0, 50.0), (5.0, 30.0), (0.0, 120.0), (0.0, 120.0), (0.0, 10.0),
            (0.0, 30.0), (0.0, 30.0),
        ]),
        EquipmentType::CoolingTower => Some(vec![
            (60.0, 100.0), (70.0, 110.0), (50.0, 85.0), (0.0, 10.0), (0.0, 20.0),
            (0.0, 30.0), (0.0, 1.0), (0.0, 1.0),
        ]),
        EquipmentType::ResiFurnace => Some(vec![
            (50.0, 160.0), (65.0, 82.0), (0.0, 15.0), (0.0, 1.0), (0.0, 1.0),
            (0.0, 1.0), (0.5, 2.0), (0.0, 120.0),
        ]),
        EquipmentType::ResiHeatPump => Some(vec![
            (45.0, 130.0), (0.0, 30.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
            (-10.0, 115.0), (0.0, 1.0), (0.0, 1.0),
        ]),
        EquipmentType::ResiBoiler => Some(vec![
            (100.0, 210.0), (80.0, 190.0), (0.0, 1.0), (5.0, 30.0), (0.0, 1.0),
            (0.0, 1.0), (0.0, 600.0), (0.0, 5.0),
        ]),
        EquipmentType::VfdPumpPack => Some(vec![
            (0.0, 100.0), (0.0, 100.0), (0.0, 100.0), (0.0, 50.0), (0.0, 100.0),
            (0.0, 100.0), (0.0, 10.0), (0.0, 10.0), (0.0, 30.0), (0.0, 30.0),
            (0.0, 1.0), (0.0, 1.0),
        ]),
        _ => None,
    }
}

pub fn extended_fault_scenarios(eq: &EquipmentType) -> Option<Vec<FaultScenario>> {
    match eq {
        EquipmentType::Rtu => Some(vec![
            FaultScenario { name: "dx_coil_freeze".into(), affected_sensors: vec![0, 4, 3], severity: 0.9, onset_hours: 0.5 },
            FaultScenario { name: "gas_heat_ignition_fail".into(), affected_sensors: vec![0, 1, 6, 7], severity: 1.0, onset_hours: 0.0 },
            FaultScenario { name: "high_limit_cycling".into(), affected_sensors: vec![0, 7, 6], severity: 0.8, onset_hours: 1.0 },
            FaultScenario { name: "comp_short_cycle".into(), affected_sensors: vec![4, 5, 0], severity: 0.8, onset_hours: 0.0 },
            FaultScenario { name: "staging_inversion".into(), affected_sensors: vec![4, 5, 0], severity: 0.9, onset_hours: 0.0 },
            FaultScenario { name: "filter_clog".into(), affected_sensors: vec![8, 3, 0], severity: 0.5, onset_hours: 720.0 },
        ]),
        EquipmentType::Doas => Some(vec![
            FaultScenario { name: "dx_latent_fail".into(), affected_sensors: vec![0, 1, 5, 6], severity: 0.7, onset_hours: 4.0 },
            FaultScenario { name: "wind_blowout".into(), affected_sensors: vec![0, 3, 4], severity: 1.0, onset_hours: 0.0 },
            FaultScenario { name: "staging_inversion".into(), affected_sensors: vec![5, 6, 7, 8], severity: 0.9, onset_hours: 0.0 },
            FaultScenario { name: "supply_overheat".into(), affected_sensors: vec![0, 3, 4], severity: 1.0, onset_hours: 0.0 },
            FaultScenario { name: "filter_clog".into(), affected_sensors: vec![9, 2, 0], severity: 0.5, onset_hours: 720.0 },
        ]),
        EquipmentType::CascadeBoiler => Some(vec![
            FaultScenario { name: "check_valve_leak".into(), affected_sensors: vec![4, 1, 0], severity: 0.7, onset_hours: 0.0 },
            FaultScenario { name: "pump_coupling_shear".into(), affected_sensors: vec![9, 10, 11], severity: 1.0, onset_hours: 0.0 },
            FaultScenario { name: "flue_degradation".into(), affected_sensors: vec![3, 5, 0], severity: 0.5, onset_hours: 336.0 },
            FaultScenario { name: "return_condensing_corrosion".into(), affected_sensors: vec![1, 0], severity: 0.6, onset_hours: 72.0 },
            FaultScenario { name: "lead_boiler_fail".into(), affected_sensors: vec![0, 7, 5], severity: 1.0, onset_hours: 0.0 },
        ]),
        EquipmentType::WaterCooledChiller => Some(vec![
            FaultScenario { name: "condenser_tube_fouling".into(), affected_sensors: vec![4, 3, 7], severity: 0.6, onset_hours: 336.0 },
            FaultScenario { name: "tower_fan_stall".into(), affected_sensors: vec![2, 9, 3], severity: 0.8, onset_hours: 0.0 },
            FaultScenario { name: "low_load_cycling".into(), affected_sensors: vec![7, 0, 1], severity: 0.7, onset_hours: 1.0 },
            FaultScenario { name: "evap_approach_high".into(), affected_sensors: vec![1, 5, 6], severity: 0.7, onset_hours: 168.0 },
            FaultScenario { name: "barrel_freeze".into(), affected_sensors: vec![1, 5, 6, 10], severity: 1.0, onset_hours: 0.0 },
        ]),
        EquipmentType::CoolingTower => Some(vec![
            FaultScenario { name: "resonance_vibration".into(), affected_sensors: vec![3, 4, 7], severity: 0.8, onset_hours: 0.0 },
            FaultScenario { name: "thermal_stratification".into(), affected_sensors: vec![0, 1, 5], severity: 0.7, onset_hours: 1.0 },
            FaultScenario { name: "windmill_reverse".into(), affected_sensors: vec![4, 3], severity: 0.9, onset_hours: 0.0 },
            FaultScenario { name: "basin_low_water".into(), affected_sensors: vec![6, 0], severity: 1.0, onset_hours: 0.0 },
            FaultScenario { name: "fan_belt_slip".into(), affected_sensors: vec![4, 3, 0], severity: 0.6, onset_hours: 48.0 },
        ]),
        EquipmentType::ResiFurnace => Some(vec![
            FaultScenario { name: "filter_restriction".into(), affected_sensors: vec![2, 6, 0], severity: 0.5, onset_hours: 720.0 },
            FaultScenario { name: "condensate_blockage".into(), affected_sensors: vec![2, 7, 0], severity: 0.8, onset_hours: 4.0 },
            FaultScenario { name: "acoil_freeze".into(), affected_sensors: vec![0, 5, 2], severity: 0.9, onset_hours: 0.5 },
            FaultScenario { name: "ignition_fail".into(), affected_sensors: vec![0, 2, 4, 7], severity: 1.0, onset_hours: 0.0 },
            FaultScenario { name: "plenum_overheat".into(), affected_sensors: vec![0, 4, 2], severity: 1.0, onset_hours: 0.0 },
        ]),
        EquipmentType::ResiHeatPump => Some(vec![
            FaultScenario { name: "defrost_fail".into(), affected_sensors: vec![0, 6, 5], severity: 0.8, onset_hours: 1.0 },
            FaultScenario { name: "aux_heat_stuck_on".into(), affected_sensors: vec![1, 3, 0], severity: 1.0, onset_hours: 0.0 },
            FaultScenario { name: "simultaneous_heat_cool".into(), affected_sensors: vec![3, 7, 1], severity: 1.0, onset_hours: 0.0 },
            FaultScenario { name: "comp_degradation".into(), affected_sensors: vec![0, 2, 5], severity: 0.6, onset_hours: 168.0 },
            FaultScenario { name: "blower_fail".into(), affected_sensors: vec![4, 0, 1], severity: 1.0, onset_hours: 0.0 },
        ]),
        EquipmentType::ResiBoiler => Some(vec![
            FaultScenario { name: "expansion_tank_fail".into(), affected_sensors: vec![3, 7, 0], severity: 0.9, onset_hours: 0.0 },
            FaultScenario { name: "airlock".into(), affected_sensors: vec![0, 1, 6], severity: 0.8, onset_hours: 0.0 },
            FaultScenario { name: "flue_condensation".into(), affected_sensors: vec![1, 0, 6], severity: 0.6, onset_hours: 72.0 },
            FaultScenario { name: "low_water_cutoff".into(), affected_sensors: vec![4, 2, 3], severity: 1.0, onset_hours: 0.0 },
            FaultScenario { name: "flame_rollout".into(), affected_sensors: vec![0, 2, 7], severity: 1.0, onset_hours: 0.0 },
        ]),
        EquipmentType::VfdPumpPack => Some(vec![
            FaultScenario { name: "sheared_coupling".into(), affected_sensors: vec![8, 9, 3, 4], severity: 1.0, onset_hours: 0.0 },
            FaultScenario { name: "locked_rotor".into(), affected_sensors: vec![8, 9, 6, 7], severity: 1.0, onset_hours: 0.0 },
            FaultScenario { name: "check_valve_leak".into(), affected_sensors: vec![1, 2, 0], severity: 0.8, onset_hours: 0.0 },
            FaultScenario { name: "dead_head".into(), affected_sensors: vec![1, 3, 8, 6], severity: 1.0, onset_hours: 0.0 },
            FaultScenario { name: "cavitation".into(), affected_sensors: vec![0, 8, 9, 4], severity: 0.9, onset_hours: 0.5 },
            FaultScenario { name: "suction_low".into(), affected_sensors: vec![0, 4, 8], severity: 1.0, onset_hours: 0.0 },
        ]),
        _ => None,
    }
}

/// Try to generate data using an extended per-type generator.
pub fn extended_generate(eq: &EquipmentType, seed: u64, normal_count: usize, fault_per: usize) -> Option<Vec<SentinelSample>> {
    match eq {
        EquipmentType::Doas => Some(gen_doas(seed, normal_count, fault_per)),
        EquipmentType::CoolingTower => Some(gen_cooling_tower(seed, normal_count, fault_per)),
        EquipmentType::ResiFurnace => Some(gen_resi_furnace(seed, normal_count, fault_per)),
        EquipmentType::ResiHeatPump => Some(gen_resi_heat_pump(seed, normal_count, fault_per)),
        EquipmentType::VfdPumpPack => Some(gen_vfd_pump_pack(seed, normal_count, fault_per)),
        EquipmentType::CascadeBoiler => Some(gen_cascade_boiler(seed, normal_count, fault_per)),
        _ => None,
    }
}

fn norm(val: f32, lo: f32, hi: f32) -> f32 { ((val - lo) / (hi - lo + 1e-7)).clamp(-0.5, 1.5) }

fn make_labels(deg: f32, onset_hours: f32, progress: f32) -> [f32; 7] {
    let horizons: [f32; 6] = [0.5, 4.0, 12.0, 72.0, 168.0, 336.0];
    let mut labels = [0.0f32; 7];
    labels[0] = if deg > 0.15 { 1.0 } else { 0.0 };
    for (hi, &h) in horizons.iter().enumerate() {
        let hrs = onset_hours * (1.0 - progress);
        labels[hi + 1] = if hrs <= h { 1.0 } else { (1.0 - (hrs - h) / h).max(0.0) };
    }
    labels
}

fn gen_doas(seed: u64, nc: usize, fpc: usize) -> Vec<SentinelSample> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut s = Vec::new();
    let r = [(45.0,140.0),(-10.0,115.0),(0.0,30.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,30.0),(0.0,30.0),(0.1,2.0)];
    for _ in 0..nc {
        let oat0 = rng.gen_range(-10.0f32..110.0);
        let oat_drift = rng.gen_range(-3.0f32..3.0);
        let fan0 = 10.0+rng.gen_range(0.0..8.0);
        let load_drift = rng.gen_range(-0.2f32..0.2);
        let mut data = Vec::with_capacity(TIMESTEPS*10);
        for t in 0..TIMESTEPS {
            let tf = t as f32 / TIMESTEPS as f32;
            let oat = oat0 + oat_drift * tf;
            let cooling = oat > 65.0; let heating = oat < 45.0;
            let g1 = if heating{1.0}else{0.0}; let g2 = if oat<25.0{1.0}else{0.0};
            let d1 = if cooling{1.0}else{0.0}; let d2 = if oat>85.0{1.0}else{0.0};
            let d1a = d1*(12.0+rng.gen_range(-0.8..0.8)); let d2a = d2*(12.0+rng.gen_range(-0.8..0.8));
            let supply = if cooling{53.0-d1a*0.15+rng.gen_range(-1.0..1.0)} else if heating{90.0+rng.gen_range(-3.0..3.0)} else{oat+rng.gen_range(-2.0..2.0)};
            let fan = fan0 + load_drift * 8.0 * tf + rng.gen_range(-0.3..0.3);
            let fdp = 0.3 + tf * 0.05 + rng.gen_range(-0.03..0.03);
            let vals=[supply,oat,fan,g1,g2,d1,d2,d1a,d2a,fdp];
            for (i,&v) in vals.iter().enumerate(){let n=rng.gen_range(-0.005..0.005)*(r[i].1-r[i].0);data.push(norm(v+n,r[i].0,r[i].1));}
        }
        s.push(SentinelSample{data, labels:[0.0;7], fault_name:"normal".into()});
    }
    let faults = extended_fault_scenarios(&EquipmentType::Doas).unwrap();
    for f in &faults { for fi in 0..fpc {
        let p=fi as f32/fpc as f32; let deg=p*f.severity; let oat=rng.gen_range(-10.0f32..110.0);
        let mut data = Vec::with_capacity(TIMESTEPS*10);
        for t in 0..TIMESTEPS { let tf=t as f32/TIMESTEPS as f32; let fs=deg*(0.3+0.7*tf);
            let (sup,fan,g1,g2,d1,d2,d1a,d2a,fdp) = match f.name.as_str() {
                "dx_latent_fail" => (oat-fs*3.0,12.0,0.0,0.0,1.0,1.0,12.0,12.0,0.5),
                "wind_blowout" => {let drop=if tf>0.3{fs*40.0}else{0.0};(90.0-drop,12.0,1.0,1.0,0.0,0.0,0.0,0.0,0.5)},
                "staging_inversion" => (55.0+fs*10.0,12.0,0.0,0.0,0.0,1.0,0.0,12.0*fs,0.5),
                "supply_overheat" => (90.0+fs*50.0,12.0,1.0,1.0,0.0,0.0,0.0,0.0,0.5),
                "filter_clog" => (55.0+fs*5.0,12.0+fs*5.0,0.0,0.0,1.0,0.0,10.0,0.0,0.5+fs*1.5),
                _ => (70.0,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5),
            };
            let vals=[sup,oat,fan,g1,g2,d1,d2,d1a,d2a,fdp];
            for (i,&v) in vals.iter().enumerate() { let n=rng.gen_range(-0.005..0.005)*(r[i].1-r[i].0); data.push(norm(v+n,r[i].0,r[i].1)); }
        }
        s.push(SentinelSample{data,labels:make_labels(deg,f.onset_hours,p),fault_name:f.name.clone()});
    }}
    s
}

fn gen_cooling_tower(seed: u64, nc: usize, fpc: usize) -> Vec<SentinelSample> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut s = Vec::new();
    let r = [(60.0,100.0),(70.0,110.0),(50.0,85.0),(0.0,10.0),(0.0,20.0),(0.0,30.0),(0.0,1.0),(0.0,1.0)];
    for _ in 0..nc {
        let wb=rng.gen_range(55.0f32..80.0); let load=rng.gen_range(0.3f32..1.0);
        let ecw=wb+5.0+load*8.0+rng.gen_range(-1.0..1.0); let lcw=ecw+load*8.0+rng.gen_range(-1.0..1.0);
        let base=[ecw,lcw,wb,3.0+load*6.0,3.0+load*10.0,8.0+load*8.0,0.7+rng.gen_range(-0.1..0.1),0.05+rng.gen_range(-0.02..0.02)];
        let mut data = Vec::with_capacity(TIMESTEPS*8);
        for _ in 0..TIMESTEPS { for (i,&v) in base.iter().enumerate() { let n=rng.gen_range(-0.01..0.01)*(r[i].1-r[i].0); data.push(norm(v+n,r[i].0,r[i].1)); }}
        s.push(SentinelSample{data,labels:[0.0;7],fault_name:"normal".into()});
    }
    let faults = extended_fault_scenarios(&EquipmentType::CoolingTower).unwrap();
    for f in &faults { for fi in 0..fpc {
        let p=fi as f32/fpc as f32; let deg=p*f.severity; let wb=rng.gen_range(58.0f32..75.0);
        let mut data = Vec::with_capacity(TIMESTEPS*8);
        for t in 0..TIMESTEPS { let tf=t as f32/TIMESTEPS as f32; let fs=deg*(0.3+0.7*tf);
            let (ecw,lcw,fv,fa,pa,basin,vib) = match f.name.as_str() {
                "resonance_vibration" => (75.0,85.0,7.0,8.0,12.0,0.7,0.05+fs*0.8),
                "thermal_stratification" => (75.0+fs*15.0,85.0+fs*20.0,8.0,10.0,12.0*(1.0-fs*0.6),0.7,0.05),
                "windmill_reverse" => (75.0,85.0,0.0,fs*3.0,0.0,0.7,fs*0.3),
                "basin_low_water" => (75.0+fs*10.0,85.0+fs*12.0,8.0,10.0,12.0,0.7-fs*0.7,0.05),
                "fan_belt_slip" => (75.0+fs*8.0,85.0+fs*10.0,8.0,10.0*(1.0-fs*0.6),12.0,0.7,0.05+fs*0.1),
                _ => (75.0,85.0,5.0,8.0,12.0,0.7,0.05),
            };
            let vals=[ecw,lcw,wb,fv,fa,pa,basin,vib];
            for (i,&v) in vals.iter().enumerate() { let n=rng.gen_range(-0.005..0.005)*(r[i].1-r[i].0); data.push(norm(v+n,r[i].0,r[i].1)); }
        }
        s.push(SentinelSample{data,labels:make_labels(deg,f.onset_hours,p),fault_name:f.name.clone()});
    }}
    s
}

fn gen_resi_furnace(seed: u64, nc: usize, fpc: usize) -> Vec<SentinelSample> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut s = Vec::new();
    let r = [(50.0,160.0),(65.0,82.0),(0.0,15.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.5,2.0),(0.0,120.0)];
    for _ in 0..nc {
        let ret = rng.gen_range(69.0f32..77.0);
        let mode = rng.gen_range(0u8..5);
        let cycle_pos = rng.gen_range(0.0f32..1.0);
        let rise_rate = rng.gen_range(3.0f32..8.0);
        let mut data = Vec::with_capacity(TIMESTEPS*8);
        for t in 0..TIMESTEPS {
            let tf = t as f32 / TIMESTEPS as f32;
            let (sup, amps, fan, heat, cool, filt, ind) = match mode {
                0 => {
                    let sup = ret + 20.0 + rise_rate * (cycle_pos + tf * 0.3) + rng.gen_range(-1.5..1.5);
                    (sup, 6.0+rng.gen_range(-0.5..0.5), 1.0, 1.0, 0.0, 1.0+rng.gen_range(-0.05..0.05),
                     60.0 + (cycle_pos + tf * 0.3) * 80.0)
                },
                1 => {
                    let cooldown = cycle_pos * 10.0 + tf * 5.0;
                    (ret + 15.0 - cooldown + rng.gen_range(-1.0..1.0), 2.0+rng.gen_range(-0.2..0.2),
                     if tf < 0.7 { 1.0 } else { 0.0 }, 0.0, 0.0, 1.0, 0.0)
                },
                2 => {
                    (ret - 15.0 - tf * 3.0 + rng.gen_range(-1.0..1.0), 4.0+rng.gen_range(-0.3..0.3),
                     1.0, 0.0, 1.0, 1.0+rng.gen_range(-0.05..0.05), 0.0)
                },
                3 => {
                    let trans = (cycle_pos * 6.0) as usize;
                    let firing = t >= trans;
                    let sup = if firing { ret + 10.0 + rise_rate * (t - trans) as f32 * 0.5 }
                              else { ret + rng.gen_range(-1.0..1.0) };
                    let ind = if firing { 30.0 + (t - trans) as f32 * 15.0 } else { 0.0 };
                    (sup + rng.gen_range(-1.0..1.0), if firing{6.0}else{2.0}+rng.gen_range(-0.3..0.3),
                     1.0, if firing{1.0}else{0.0}, 0.0, 1.0, ind)
                },
                _ => {
                    let trans = 2 + (cycle_pos * 4.0) as usize;
                    let firing = t < trans;
                    let sup = if firing { ret + 30.0 - rise_rate * 0.3 * tf }
                              else { ret + 30.0 - rise_rate * (t - trans + 1) as f32 * 0.8 };
                    (sup + rng.gen_range(-1.0..1.0), if firing{6.0}else{2.0}+rng.gen_range(-0.3..0.3),
                     1.0, if firing{1.0}else{0.0}, 0.0, 1.0, if firing{60.0+t as f32*10.0}else{0.0})
                },
            };
            let vals = [sup, ret+rng.gen_range(-0.3..0.3), amps, fan, heat, cool, filt, ind];
            for (i,&v) in vals.iter().enumerate() {
                let n = rng.gen_range(-0.005..0.005)*(r[i].1-r[i].0);
                data.push(norm(v+n, r[i].0, r[i].1));
            }
        }
        s.push(SentinelSample{data,labels:[0.0;7],fault_name:"normal".into()});
    }
    let faults = extended_fault_scenarios(&EquipmentType::ResiFurnace).unwrap();
    for f in &faults { for fi in 0..fpc {
        let p=fi as f32/fpc as f32; let deg=p*f.severity; let ret=rng.gen_range(70.0f32..76.0);
        let mut data = Vec::with_capacity(TIMESTEPS*8);
        for t in 0..TIMESTEPS { let tf=t as f32/TIMESTEPS as f32; let fs=deg*(0.3+0.7*tf);
            let (sup,amps,fan,heat,cool,filt,ind) = match f.name.as_str() {
                "filter_restriction" => (ret+25.0,6.0+fs*3.0,1.0,1.0,0.0,1.0+fs*0.8,60.0),
                "condensate_blockage" => (ret+5.0*(1.0-fs),2.0+fs*1.0,1.0,1.0,0.0,1.0,60.0+fs*50.0),
                "acoil_freeze" => (ret-fs*25.0,4.0,1.0,0.0,1.0,1.0,0.0),
                "ignition_fail" => (ret-fs*5.0,1.5,1.0,1.0,0.0,1.0,90.0+fs*30.0),
                "plenum_overheat" => (ret+50.0+fs*60.0,7.0+fs*3.0,1.0,1.0,0.0,1.0,60.0),
                _ => (ret+20.0,5.0,1.0,1.0,0.0,1.0,60.0),
            };
            let vals=[sup,ret,amps,fan,heat,cool,filt,ind];
            for (i,&v) in vals.iter().enumerate() { let n=rng.gen_range(-0.005..0.005)*(r[i].1-r[i].0); data.push(norm(v+n,r[i].0,r[i].1)); }
        }
        s.push(SentinelSample{data,labels:make_labels(deg,f.onset_hours,p),fault_name:f.name.clone()});
    }}
    s
}

fn gen_resi_heat_pump(seed: u64, nc: usize, fpc: usize) -> Vec<SentinelSample> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut s = Vec::new();
    let r = [(45.0,130.0),(0.0,30.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(-10.0,115.0),(0.0,1.0),(0.0,1.0)];
    for _ in 0..nc {
        let oat=rng.gen_range(-10.0f32..110.0); let heating=oat<60.0;
        let sup=if heating{90.0+rng.gen_range(-5.0..10.0)}else{52.0+rng.gen_range(-3.0..3.0)};
        let amps=if heating{5.0+rng.gen_range(0.0..3.0)}else{4.0+rng.gen_range(0.0..2.0)};
        let base=[sup,amps,if heating{1.0}else{0.0},if oat<20.0{1.0}else{0.0},1.0,oat,if oat>15.0&&oat<35.0&&rng.gen_bool(0.15){1.0}else{0.0},if !heating{1.0}else{0.0}];
        let mut data = Vec::with_capacity(TIMESTEPS*8);
        for _ in 0..TIMESTEPS { for (i,&v) in base.iter().enumerate() { let n=rng.gen_range(-0.01..0.01)*(r[i].1-r[i].0); data.push(norm(v+n,r[i].0,r[i].1)); }}
        s.push(SentinelSample{data,labels:[0.0;7],fault_name:"normal".into()});
    }
    let faults = extended_fault_scenarios(&EquipmentType::ResiHeatPump).unwrap();
    for f in &faults { for fi in 0..fpc {
        let p=fi as f32/fpc as f32; let deg=p*f.severity; let oat=rng.gen_range(-10.0f32..100.0);
        let mut data = Vec::with_capacity(TIMESTEPS*8);
        for t in 0..TIMESTEPS { let tf=t as f32/TIMESTEPS as f32; let fs=deg*(0.3+0.7*tf);
            let (sup,amps,comp,aux,fan,def,cool) = match f.name.as_str() {
                "defrost_fail" => (90.0-fs*35.0,5.0,1.0,0.0,1.0,1.0,0.0),
                "aux_heat_stuck_on" => (90.0+fs*30.0,5.0+fs*20.0,0.0,0.0,1.0,0.0,0.0),
                "simultaneous_heat_cool" => (70.0,5.0+fs*15.0,0.0,fs.round(),1.0,0.0,1.0),
                "comp_degradation" => (90.0-fs*20.0,5.0-fs*2.0,1.0,0.0,1.0,0.0,0.0),
                "blower_fail" => (70.0+fs*30.0,0.3,1.0,0.0,0.0,0.0,0.0),
                _ => (85.0,5.0,1.0,0.0,1.0,0.0,0.0),
            };
            let vals=[sup,amps,comp,aux,fan,oat,def,cool];
            for (i,&v) in vals.iter().enumerate() { let n=rng.gen_range(-0.005..0.005)*(r[i].1-r[i].0); data.push(norm(v+n,r[i].0,r[i].1)); }
        }
        s.push(SentinelSample{data,labels:make_labels(deg,f.onset_hours,p),fault_name:f.name.clone()});
    }}
    s
}

fn gen_vfd_pump_pack(seed: u64, nc: usize, fpc: usize) -> Vec<SentinelSample> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut s = Vec::new();
    let r = [(0.0,100.0),(0.0,100.0),(0.0,100.0),(0.0,50.0),(0.0,100.0),(0.0,100.0),(0.0,10.0),(0.0,10.0),(0.0,30.0),(0.0,30.0),(0.0,1.0),(0.0,1.0)];
    for _ in 0..nc {
        let load=rng.gen_range(0.3f32..1.0); let sp=45.0+rng.gen_range(-2.0..2.0); let suct=35.0+rng.gen_range(-2.0..2.0);
        let p2_on=load>0.7; let spd1=3.0+load*5.0; let spd2=if p2_on{3.0+(load-0.7)*10.0}else{0.0};
        let a1=5.0+load*10.0; let a2=if p2_on{5.0+(load-0.7)*15.0}else{0.0};
        let d1=suct+15.0+load*15.0; let d2=if p2_on{d1+rng.gen_range(-1.0..1.0)}else{suct};
        let base=[suct,d1,d2,d1-suct,sp+rng.gen_range(-1.0..1.0),sp,spd1,spd2,a1,a2,1.0,if p2_on{1.0}else{0.0}];
        let mut data = Vec::with_capacity(TIMESTEPS*12);
        for _ in 0..TIMESTEPS { for (i,&v) in base.iter().enumerate() { let n=rng.gen_range(-0.008..0.008)*(r[i].1-r[i].0); data.push(norm(v+n,r[i].0,r[i].1)); }}
        s.push(SentinelSample{data,labels:[0.0;7],fault_name:"normal".into()});
    }
    let faults = extended_fault_scenarios(&EquipmentType::VfdPumpPack).unwrap();
    for f in &faults { for fi in 0..fpc {
        let p=fi as f32/fpc as f32; let deg=p*f.severity; let sp=45.0; let suct=35.0;
        let mut data = Vec::with_capacity(TIMESTEPS*12);
        for t in 0..TIMESTEPS { let tf=t as f32/TIMESTEPS as f32; let fs=deg*(0.3+0.7*tf);
            let (su,d1,d2,dp,lp,s1,s2,a1,a2,e1,e2) = match f.name.as_str() {
                "sheared_coupling" => (suct,suct+5.0,suct,2.0,sp-fs*15.0,9.0,0.0,2.0,0.0,1.0,0.0),
                "locked_rotor" => (suct,suct+20.0,suct,15.0,sp-fs*10.0,5.0,0.0,17.0+fs*10.0,0.0,1.0,0.0),
                "check_valve_leak" => (suct,suct+25.0,suct+25.0*fs,20.0,sp,6.0,0.0,10.0,0.0,1.0,0.0),
                "dead_head" => (suct,suct+40.0+fs*30.0,suct,0.5,sp-fs*20.0,9.0,0.0,12.0+fs*5.0,0.0,1.0,0.0),
                "cavitation" => (suct-fs*20.0,suct+10.0-fs*10.0,suct,5.0-fs*4.0,sp-fs*10.0,8.0,0.0,8.0,0.0,1.0,0.0),
                "suction_low" => (suct-fs*25.0,suct+5.0,suct,3.0,sp-fs*15.0,8.0,0.0,10.0,0.0,1.0,0.0),
                _ => (suct,suct+20.0,suct,15.0,sp,6.0,0.0,10.0,0.0,1.0,0.0),
            };
            let vals=[su,d1,d2,dp,lp,sp,s1,s2,a1,a2,e1,e2];
            for (i,&v) in vals.iter().enumerate() { let n=rng.gen_range(-0.005..0.005)*(r[i].1-r[i].0); data.push(norm(v+n,r[i].0,r[i].1)); }
        }
        s.push(SentinelSample{data,labels:make_labels(deg,f.onset_hours,p),fault_name:f.name.clone()});
    }}
    s
}

fn gen_cascade_boiler(seed: u64, nc: usize, fpc: usize) -> Vec<SentinelSample> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut s = Vec::new();
    let r = [(100.0,200.0),(80.0,180.0),(-10.0,100.0),(100.0,200.0),(80.0,180.0),(0.0,10.0),(0.0,10.0),(0.0,1.0),(0.0,1.0),(0.0,20.0),(0.0,20.0),(0.0,50.0)];
    for _ in 0..nc {
        let oat=rng.gen_range(-10.0f32..90.0); let oat_c=oat.clamp(0.0,60.0);
        let sp=180.0-(oat_c/60.0)*50.0;
        if oat>=65.0 { let base=[sp,sp-5.0,oat,sp,sp-5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
            let mut data=Vec::with_capacity(TIMESTEPS*12); for _ in 0..TIMESTEPS{for(i,&v)in base.iter().enumerate(){data.push(norm(v,r[i].0,r[i].1));}}
            s.push(SentinelSample{data,labels:[0.0;7],fault_name:"normal".into()}); continue;
        }
        let load=rng.gen_range(0.3f32..1.0);
        let sup=sp-5.0+load*8.0+rng.gen_range(-2.0..2.0); let ret=sup-15.0-load*5.0+rng.gen_range(-2.0..2.0);
        let b2_on=load>0.7;
        let base=[sup,ret,oat,sup+rng.gen_range(-1.0..1.0),ret+rng.gen_range(-1.0..1.0),load*8.0,if b2_on{3.0+(load-0.7)*15.0}else{0.0},1.0,if b2_on{1.0}else{0.0},5.0+load*8.0,0.0,15.0+load*15.0];
        let mut data=Vec::with_capacity(TIMESTEPS*12);
        for _ in 0..TIMESTEPS{for(i,&v)in base.iter().enumerate(){let n=rng.gen_range(-0.008..0.008)*(r[i].1-r[i].0);data.push(norm(v+n,r[i].0,r[i].1));}}
        s.push(SentinelSample{data,labels:[0.0;7],fault_name:"normal".into()});
    }
    let faults = extended_fault_scenarios(&EquipmentType::CascadeBoiler).unwrap();
    for f in &faults { for fi in 0..fpc {
        let p=fi as f32/fpc as f32; let deg=p*f.severity; let oat=rng.gen_range(-10.0f32..55.0); let sp=160.0;
        let mut data=Vec::with_capacity(TIMESTEPS*12);
        for t in 0..TIMESTEPS{let tf=t as f32/TIMESTEPS as f32;let fs=deg*(0.3+0.7*tf);
            let(sup,ret,b1s,b2e,b1m,b2m,b1e,b2e_en,pa1,pa2,dp) = match f.name.as_str() {
                "check_valve_leak" => (sp-5.0,sp-20.0,sp-3.0,sp-20.0+fs*15.0,7.0,0.0,1.0,0.0,8.0,0.0,20.0),
                "pump_coupling_shear" => (sp-fs*15.0,sp-20.0,sp-5.0,sp-20.0,7.0,0.0,1.0,0.0,1.5,0.0,3.0),
                "flue_degradation" => (sp-10.0-fs*10.0,sp-25.0,sp-8.0-fs*8.0,sp-25.0,9.5,0.0,1.0,0.0,8.0,0.0,20.0),
                "return_condensing_corrosion" => (sp-5.0,100.0+fs*10.0,sp-3.0,100.0+fs*10.0,7.0,0.0,1.0,0.0,8.0,0.0,20.0),
                "lead_boiler_fail" => (sp-fs*30.0,sp-20.0-fs*10.0,sp-fs*30.0,sp-20.0,0.0,0.0,1.0,0.0,8.0,0.0,20.0),
                _ => (sp-5.0,sp-20.0,sp-3.0,sp-20.0,7.0,0.0,1.0,0.0,8.0,0.0,20.0),
            };
            let vals=[sup,ret,oat,b1s,b2e,b1m,b2m,b1e,b2e_en,pa1,pa2,dp];
            for(i,&v)in vals.iter().enumerate(){let n=rng.gen_range(-0.005..0.005)*(r[i].1-r[i].0);data.push(norm(v+n,r[i].0,r[i].1));}
        }
        s.push(SentinelSample{data,labels:make_labels(deg,f.onset_hours,p),fault_name:f.name.clone()});
    }}
    s
}

pub fn extended_sensor_names(eq: &EquipmentType) -> Option<Vec<&'static str>> {
    match eq {
        EquipmentType::Rtu => Some(vec!["supply_t", "mixed_t", "oat", "fan_amps", "dx_stage1", "dx_stage2", "gas_heat", "burner_amps", "filter_dp", "return_t"]),
        EquipmentType::Doas => Some(vec!["supply_t", "oat", "fan_amps", "gas_stage1", "gas_stage2", "dx_stage1", "dx_stage2", "dx1_amps", "dx2_amps", "filter_dp"]),
        EquipmentType::CascadeBoiler => Some(vec!["common_supply", "common_return", "oat", "b1_supply", "b2_entering", "b1_mod_v", "b2_mod_v", "b1_enable", "b2_enable", "pump1_amps", "pump2_amps", "system_dp"]),
        EquipmentType::WaterCooledChiller => Some(vec!["ecew", "lcw", "ecw_tower", "lcw_tower", "sat_cond", "sat_suct", "evap_dp", "comp1_amps", "comp2_amps", "tower_fan_v", "chw_pump_amps", "cw_pump_amps"]),
        EquipmentType::CoolingTower => Some(vec!["ecw", "lcw", "oat_wetbulb", "fan_speed_v", "fan_amps", "cw_pump_amps", "basin_level", "vibration"]),
        EquipmentType::ResiFurnace => Some(vec!["supply_t", "return_t", "furnace_amps", "fan_enable", "heat_call", "cool_call", "filter_ratio", "inducer_sec"]),
        EquipmentType::ResiHeatPump => Some(vec!["supply_t", "handler_amps", "comp_heating", "aux_heat_cmd", "fan_active", "oat", "defrost", "cool_mode"]),
        EquipmentType::ResiBoiler => Some(vec!["supply_t", "return_t", "burner_active", "system_press", "flow_switch", "zone_call", "burner_run_sec", "temp_rise_rate"]),
        EquipmentType::VfdPumpPack => Some(vec!["suction_p", "p1_disch_p", "p2_disch_p", "plant_dp", "loop_p", "loop_sp", "p1_speed_v", "p2_speed_v", "p1_amps", "p2_amps", "p1_enable", "p2_enable"]),
        _ => None,
    }
}
