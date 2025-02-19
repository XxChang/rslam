use std::{collections::BTreeSet, num::NonZeroUsize, rc::Rc};

use anyhow::{bail, Result};
use opencv::core::{KeyPoint, KeyPointTraitConst, Mat, MatTraitConst, Vector};

pub struct IntensityFeatureMatcher {
    pub number_of_rows: i32,
    pub number_of_cols: i32,
    pub feature_lattice: Vec<Vec<Option<Rc<IntensityFeature>>>>,
    pub feature_vector: Vec<Rc<IntensityFeature>>,
}

impl Default for IntensityFeatureMatcher {
    fn default() -> Self {
        Self {
            number_of_rows: 0,
            number_of_cols: 0,
            feature_lattice: Vec::new(),
            feature_vector: Vec::new(),
        }
    }
}

impl IntensityFeatureMatcher {
    pub fn configure(&mut self, rows: NonZeroUsize, cols: NonZeroUsize) {
        log::debug!("configuring");
        if !self.feature_lattice.is_empty() {
            panic!("Feature lattice is not empty");
        }

        self.feature_lattice
            .resize(rows.into(), vec![None; cols.into()]);

        self.number_of_rows = rows.get() as i32;
        self.number_of_cols = cols.get() as i32;
    }
}

impl IntensityFeatureMatcher {
    pub fn set_fatures(
        &mut self,
        keypoints: &Vector<KeyPoint>,
        descriptors: &opencv::prelude::Mat,
    ) -> Result<()> {
        if keypoints.len() != descriptors.rows() as usize {
            bail!("Number of keypoints and descriptors do not match");
        }

        for row in 0..self.number_of_rows {
            for col in 0..self.number_of_cols {
                self.feature_lattice[row as usize][col as usize] = None;
            }
        }

        self.feature_vector.clear();

        for (index, keypoint) in keypoints.iter().enumerate() {
            let feature = Rc::new(IntensityFeature::new(
                &keypoint,
                &descriptors.row(index as i32).unwrap(),
                index,
            ));
            self.feature_vector.push(feature.clone());
            let row = feature.row;
            let col = feature.col;
            self.feature_lattice[row as usize][col as usize] = Some(feature);
        }

        Ok(())
    }

    pub fn sort_feature_vector(&mut self) {
        self.feature_vector.sort_by(|a, b| {
            if a.row == b.row {
                a.col.cmp(&b.col)
            } else {
                a.row.cmp(&b.row)
            }
        });
    }

    pub fn prune(&mut self, matched_indices: &BTreeSet<usize>) {
        let mut number_of_unmatched_elements = 0;
        for index in 0..self.feature_vector.len() {
            //ds if we haven't matched this index yet
            if !matched_indices.contains(&index) {
                // keep the element (this operation is not problemenatic since we do not loop reversely here)
                self.feature_vector[number_of_unmatched_elements] =
                    self.feature_vector[index].clone();
                number_of_unmatched_elements += 1;
            }
        }
        self.feature_vector.truncate(number_of_unmatched_elements);
    }
}

/// @struct container holding spatial and appearance information (used in findStereoKeypoints)
pub struct IntensityFeature {
    // geometric: feature location in 2D
    pub keypoint: KeyPoint,
    // appearance: feature descriptor
    pub descriptor: Mat,
    // pixel column coordinate (v)
    pub row: i32,
    // pixel row coordinate (u)
    pub col: i32,
    // inverted index for vector containing this
    index_in_vector: usize,
}

impl IntensityFeature {
    pub fn new(keypoint: &KeyPoint, descriptor: &Mat, index_in_vector: usize) -> Self {
        Self {
            keypoint: keypoint.clone(),
            descriptor: descriptor.clone(),
            row: keypoint.pt().y as i32,
            col: keypoint.pt().x as i32,
            index_in_vector,
        }
    }
}
