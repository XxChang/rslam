use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct FramePointGeneratorConfig {
    number_of_detectors_vertical: usize,
    number_of_detectors_horizontal: usize,
}

impl Default for FramePointGeneratorConfig {
    fn default() -> Self {
        FramePointGeneratorConfig {
            number_of_detectors_vertical: 1,
            number_of_detectors_horizontal: 1,
        }
    }
}
