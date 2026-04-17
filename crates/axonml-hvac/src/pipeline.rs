//! HVAC 4-Stage Inference Pipeline
//!
//! # File
//! `crates/axonml-hvac/src/pipeline.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_autograd::Variable;

use super::{
    apollo::Apollo,
    aquilo::Aquilo,
    boreas::Boreas,
    colossus::Colossus,
    data::{HvacSensorData, PipelineOutput},
    gaia::Gaia,
    naiad::Naiad,
    vulcan::Vulcan,
    zephyrus::Zephyrus,
};

// =============================================================================
// HvacPipeline
// =============================================================================

/// Full 4-stage HVAC diagnostic pipeline.
///
/// Runs all 8 models in the correct dependency order and produces
/// a comprehensive diagnostic output.
pub struct HvacPipeline {
    /// Stage 1: Electrical systems specialist.
    pub aquilo: Aquilo,
    /// Stage 1: Refrigeration systems specialist.
    pub boreas: Boreas,
    /// Stage 1: Water systems specialist.
    pub naiad: Naiad,
    /// Stage 1: Mechanical systems specialist.
    pub vulcan: Vulcan,
    /// Stage 1: Airflow systems specialist.
    pub zephyrus: Zephyrus,
    /// Stage 2: Cross-specialist aggregator.
    pub colossus: Colossus,
    /// Stage 3: Safety validator.
    pub gaia: Gaia,
    /// Stage 4: Master coordinator.
    pub apollo: Apollo,
}

impl Default for HvacPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl HvacPipeline {
    /// Creates a new pipeline with all models initialized.
    pub fn new() -> Self {
        Self {
            aquilo: Aquilo::new(),
            boreas: Boreas::new(),
            naiad: Naiad::new(),
            vulcan: Vulcan::new(),
            zephyrus: Zephyrus::new(),
            colossus: Colossus::new(),
            gaia: Gaia::new(),
            apollo: Apollo::new(),
        }
    }

    /// Run the full diagnostic pipeline.
    ///
    /// # Arguments
    /// * `sensor_data` - Raw HVAC sensor readings from all subsystems
    ///
    /// # Returns
    /// Complete pipeline output including all model results
    pub fn diagnose(&self, sensor_data: &HvacSensorData) -> PipelineOutput {
        let batch = sensor_data.electrical.shape()[0];

        // =====================================================================
        // Stage 1: Specialist Models (can run in parallel)
        // =====================================================================

        // Aquilo: flatten (batch, 64, 7) → (batch, 448), take first 168
        let elec_flat = flatten_sensor(&sensor_data.electrical, batch, 64, 7, 168);
        let (aquilo_fault, _, _, _, aquilo_emb) = self.aquilo.forward_all(&elec_flat);

        // Boreas: (batch, 80, 7) as-is
        let (boreas_fault, _, _, _, boreas_emb) =
            self.boreas.forward_all(&sensor_data.refrigeration);

        // Naiad: transpose (batch, 64, 7) → (batch, 7, 64) for Conv1d
        let water_t = transpose_last_two(&sensor_data.water, batch, 64, 7);
        let (naiad_fault, _, _, _, naiad_emb) = self.naiad.forward_all(&water_t);

        // Vulcan: flatten (batch, 96, 7) → (batch, 672)
        let mech_flat = flatten_sensor(&sensor_data.mechanical, batch, 96, 7, 672);
        let (vulcan_fault, _, _, _, vulcan_emb) = self.vulcan.forward_all(&mech_flat);

        // Zephyrus: transpose (batch, 72, 7) → (batch, 7, 72) for Conv1d/GCN
        let air_t = transpose_last_two(&sensor_data.airflow, batch, 72, 7);
        let (zephyrus_fault, _, _, _, zephyrus_emb) = self.zephyrus.forward_all(&air_t);

        // Concatenate specialist embeddings
        let specialist_features = super::aquilo::concat_variables(
            &[
                &aquilo_emb,
                &boreas_emb,
                &naiad_emb,
                &vulcan_emb,
                &zephyrus_emb,
            ],
            batch,
        );

        // =====================================================================
        // Stage 2: Colossus Aggregator
        // =====================================================================

        let (_, _, _, _, colossus_emb) = self.colossus.forward_specialists(
            &aquilo_emb,
            &boreas_emb,
            &naiad_emb,
            &vulcan_emb,
            &zephyrus_emb,
        );

        // =====================================================================
        // Stage 3: Gaia Safety Validator
        // =====================================================================

        let (_, safety_score, _, _, gaia_emb) =
            self.gaia.forward_parts(&specialist_features, &colossus_emb);

        // =====================================================================
        // Stage 4: Apollo Master Coordinator
        // =====================================================================

        // Summarize raw sensors: mean across time for each channel
        let raw_sensor_summary = summarize_sensors(sensor_data, batch);

        let model_embs = [
            &aquilo_emb,
            &boreas_emb,
            &naiad_emb,
            &vulcan_emb,
            &zephyrus_emb,
            &colossus_emb,
            &gaia_emb,
        ];
        let model_refs: Vec<&Variable> = model_embs.to_vec();

        let (diagnosis, _, _, _, _) = self.apollo.forward_parts(&model_refs, &raw_sensor_summary);

        // =====================================================================
        // Collect outputs
        // =====================================================================

        PipelineOutput {
            specialist_features,
            aggregator_output: colossus_emb,
            safety_output: safety_score,
            diagnosis,
            specialist_faults: vec![
                aquilo_fault,
                boreas_fault,
                naiad_fault,
                vulcan_fault,
                zephyrus_fault,
            ],
        }
    }

    /// Returns total parameter count across all 8 models.
    pub fn total_parameters(&self) -> usize {
        use axonml_nn::Module;
        self.aquilo.num_parameters()
            + self.boreas.num_parameters()
            + self.naiad.num_parameters()
            + self.vulcan.num_parameters()
            + self.zephyrus.num_parameters()
            + self.colossus.num_parameters()
            + self.gaia.num_parameters()
            + self.apollo.num_parameters()
    }

    /// Sets all models to eval mode.
    pub fn eval(&mut self) {
        use axonml_nn::Module;
        self.aquilo.set_training(false);
        self.boreas.set_training(false);
        self.naiad.set_training(false);
        self.vulcan.set_training(false);
        self.zephyrus.set_training(false);
        self.colossus.set_training(false);
        self.gaia.set_training(false);
        self.apollo.set_training(false);
    }

    /// Sets all models to train mode.
    pub fn train(&mut self) {
        use axonml_nn::Module;
        self.aquilo.set_training(true);
        self.boreas.set_training(true);
        self.naiad.set_training(true);
        self.vulcan.set_training(true);
        self.zephyrus.set_training(true);
        self.colossus.set_training(true);
        self.gaia.set_training(true);
        self.apollo.set_training(true);
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Flatten sensor data from (batch, time, channels) to (batch, flat_dim).
fn flatten_sensor(
    input: &Variable,
    batch: usize,
    time: usize,
    channels: usize,
    target_dim: usize,
) -> Variable {
    let full_dim = time * channels;
    let dim = target_dim.min(full_dim);

    let flat = input.reshape(&[batch, full_dim]);
    if dim < full_dim {
        flat.narrow(1, 0, dim)
    } else {
        flat
    }
}

/// Transpose last two dimensions: (batch, T, C) → (batch, C, T).
fn transpose_last_two(input: &Variable, _batch: usize, _time: usize, _channels: usize) -> Variable {
    input.transpose(1, 2)
}

/// Compute mean sensor values across time for raw sensor summary.
fn summarize_sensors(data: &HvacSensorData, _batch: usize) -> Variable {
    // Mean over time dimension (dim=1) for each sensor: (batch, T, 7) → (batch, 7)
    let elec_mean = data.electrical.mean_dim(1, false); // (batch, 7)
    let refrig_mean = data.refrigeration.mean_dim(1, false); // (batch, 7)
    let water_mean = data.water.mean_dim(1, false); // (batch, 7)
    let mech_mean = data.mechanical.mean_dim(1, false); // (batch, 7)
    let air_mean = data.airflow.mean_dim(1, false); // (batch, 7)

    // Concat along last dim: (batch, 35)
    Variable::cat(
        &[&elec_mean, &refrig_mean, &water_mean, &mech_mean, &air_mean],
        1,
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::data::SyntheticHvacGenerator;
    use super::*;
    use axonml_nn::Module;
    use axonml_tensor::Tensor;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = HvacPipeline::new();
        let total = pipeline.total_parameters();
        // 8-model pipeline total
        assert!(
            total > 3_000_000 && total < 15_000_000,
            "Total pipeline has {} params",
            total
        );
    }

    #[test]
    fn test_pipeline_end_to_end() {
        let pipeline = HvacPipeline::new();
        let generator = SyntheticHvacGenerator::new(42);
        let (sensor_data, _labels) = generator.generate_normal(2);

        let output = pipeline.diagnose(&sensor_data);

        // Verify output shapes
        assert_eq!(output.diagnosis.shape(), vec![2, 12]);
        assert_eq!(output.specialist_faults.len(), 5);
        assert_eq!(output.specialist_faults[0].shape(), vec![2, 13]); // Aquilo
        assert_eq!(output.specialist_faults[1].shape(), vec![2, 16]); // Boreas
        assert_eq!(output.safety_output.shape(), vec![2, 1]);
    }

    #[test]
    fn test_pipeline_eval_mode() {
        let mut pipeline = HvacPipeline::new();
        pipeline.eval();
        assert!(!pipeline.aquilo.is_training());
        assert!(!pipeline.boreas.is_training());

        pipeline.train();
        assert!(pipeline.aquilo.is_training());
    }

    #[test]
    fn test_transpose_last_two() {
        let input = Variable::new(
            Tensor::from_vec(
                vec![
                    1.0, 2.0, 3.0, // t=0: [c0, c1, c2]
                    4.0, 5.0, 6.0, // t=1
                ],
                &[1, 2, 3],
            )
            .unwrap(),
            false,
        );

        let output = transpose_last_two(&input, 1, 2, 3);
        let data = output.data().to_vec();
        assert_eq!(output.shape(), vec![1, 3, 2]);
        // c0: [1, 4], c1: [2, 5], c2: [3, 6]
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_summarize_sensors() {
        let generator = SyntheticHvacGenerator::new(42);
        let (sensor_data, _) = generator.generate_normal(2);

        let summary = summarize_sensors(&sensor_data, 2);
        assert_eq!(summary.shape(), vec![2, 35]);

        // Values should be reasonable (not NaN or extreme)
        let data = summary.data().to_vec();
        for val in &data {
            assert!(val.is_finite(), "Sensor summary contains non-finite value");
        }
    }
}
