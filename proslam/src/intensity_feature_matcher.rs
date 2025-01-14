use anyhow::{bail, Result};
use opencv::core::{KeyPoint, MatTraitConst, Vector};

pub struct IntensityFeatureMatcher {}

impl IntensityFeatureMatcher {
    pub fn set_fatures(
        &mut self,
        keypoints: &Vector<KeyPoint>,
        descriptors: &opencv::prelude::Mat,
    ) -> Result<()> {
        if keypoints.len() != descriptors.rows() as usize {
            bail!("Number of keypoints and descriptors do not match");
        }
        unimplemented!()
    }
}
