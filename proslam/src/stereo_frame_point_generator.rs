use std::{
    collections::BTreeSet,
    num::NonZeroUsize,
};

use anyhow::Result;
use opencv::{
    core::{
        KeyPoint, KeyPointTrait, KeyPointTraitConst, Mat, MatTraitConst, Point2f, Ptr, Rect,
        Rect2i, NORM_HAMMING,
    },
    features2d::{
        self, DescriptorExtractor, FastFeatureDetector, FastFeatureDetector_DetectorType, ORB,
    },
    prelude::{FastFeatureDetectorTrait, FastFeatureDetectorTraitConst, Feature2DTrait},
};
use serde::Deserialize;

use crate::{
    intensity_feature_matcher::IntensityFeatureMatcher, stereo_framepoint::IntensityFeature,
};

#[derive(Debug, Deserialize)]
pub struct StereoFramePointGeneratorCfg {
    number_of_detectors_vertical: usize,
    number_of_detectors_horizontal: usize,
    target_number_of_keypoints_tolerance: f32,
    detector_threshold_maximum_change: f32,
    detector_threshold_initial: usize,
    detector_threshold_minimum: usize,
    detector_threshold_maximum: usize,

    bin_size_pixels: usize,
    maximum_matching_distance_triangulation: i32,
    descriptor_type: String,

    minimum_disparity_pixels: f32,
    maximum_epipolar_search_offset_pixels: i32,
}

impl Default for StereoFramePointGeneratorCfg {
    fn default() -> Self {
        Self {
            number_of_detectors_vertical: 1,
            number_of_detectors_horizontal: 1,
            target_number_of_keypoints_tolerance: 0.1,
            detector_threshold_maximum_change: 0.5,
            detector_threshold_initial: 15,
            detector_threshold_minimum: 15,
            detector_threshold_maximum: 100,
            bin_size_pixels: 25,

            maximum_matching_distance_triangulation: (0.2 * 256f32) as i32,
            descriptor_type: String::from("ORB-256"),

            minimum_disparity_pixels: 1.0,
            maximum_epipolar_search_offset_pixels: 0,
        }
    }
}

impl StereoFramePointGeneratorCfg {
    pub fn finalize(self, width: usize, height: usize) -> Result<StereoFramePointGenerator> {
        let mut detectors = vec![];
        let mut detector_regions = vec![];
        let mut detector_thresholds: Vec<Vec<f32>> = vec![];

        let number_of_rows_image = height;
        let number_of_cols_image = width;

        let pixel_rows_per_detector =
            number_of_rows_image as f32 / self.number_of_detectors_vertical as f32;
        let pixel_cols_per_detector =
            number_of_cols_image as f32 / self.number_of_detectors_horizontal as f32;

        for r in 0..self.number_of_detectors_vertical {
            let mut sub_detectors = vec![];
            let mut sub_detector_regions = vec![];
            let mut sub_detector_thresholds = vec![];
            for c in 0..self.number_of_detectors_horizontal {
                sub_detectors.push(FastFeatureDetector::create(
                    self.detector_threshold_minimum as _,
                    true,
                    FastFeatureDetector_DetectorType::TYPE_9_16,
                )?);
                let rect = Rect::new(
                    (c as f32 * pixel_cols_per_detector).round() as _,
                    (r as f32 * pixel_rows_per_detector).round() as _,
                    pixel_cols_per_detector as _,
                    pixel_rows_per_detector as _,
                );
                sub_detector_regions.push(rect);
                sub_detector_thresholds.push(self.detector_threshold_minimum as f32);
            }
            detectors.push(sub_detectors);
            detector_regions.push(sub_detector_regions);
            detector_thresholds.push(sub_detector_thresholds);
        }

        let detectors = detectors
            .into_iter()
            .zip(detector_regions)
            .zip(detector_thresholds)
            .into_iter()
            .map(|((detectors, regions), thresholds)| {
                detectors
                    .into_iter()
                    .zip(regions)
                    .zip(thresholds)
                    .map(|((detector, region), threshold)| (detector, region, threshold))
                    .collect()
            })
            .collect();

        let number_of_detectors =
            self.number_of_detectors_horizontal * self.number_of_detectors_vertical;

        let number_of_cols_bin = (width as f32 / self.bin_size_pixels as f32).floor() + 1f32;
        let number_of_rows_bin = (height as f32 / self.bin_size_pixels as f32).floor() + 1f32;

        let target_number_of_keypoints = number_of_cols_bin * number_of_rows_bin;
        log::info!(
            "current target number of points: {}",
            target_number_of_keypoints
        );

        let target_number_of_keypoints_per_detector =
            target_number_of_keypoints / number_of_detectors as f32;
        log::info!("current target number of points per image region: {target_number_of_keypoints_per_detector}");

        let descriptor_extract = match self.descriptor_type.as_str() {
            "ORB-256" => ORB::create_def()?.try_into()?,
            _ => {
                unimplemented!()
            }
        };

        let mut feature_matcher_left = IntensityFeatureMatcher::default();
        let mut feature_matcher_right = IntensityFeatureMatcher::default();
        feature_matcher_left.configure(
            NonZeroUsize::new(number_of_rows_image).unwrap(),
            NonZeroUsize::new(number_of_cols_image).unwrap(),
        );
        feature_matcher_right.configure(
            NonZeroUsize::new(number_of_rows_image).unwrap(),
            NonZeroUsize::new(number_of_cols_image).unwrap(),
        );

        let mut epipolar_search_offset_pixels = vec![0];
        for i in 1..self.maximum_epipolar_search_offset_pixels {
            epipolar_search_offset_pixels.push(i);
            epipolar_search_offset_pixels.push(-i);
        }

        log::info!(
            "number of epipolar lines considered for stereo matching: {}",
            epipolar_search_offset_pixels.len()
        );
        log::info!("configured");

        Ok(StereoFramePointGenerator {
            detectors,
            target_number_of_keypoints,
            target_number_of_keypoints_per_detector,
            target_number_of_keypoints_tolerance: self.target_number_of_keypoints_tolerance,
            detector_threshold_maximum_change: self.detector_threshold_maximum_change,
            detector_threshold_minimum: self.detector_threshold_minimum as f32,

            number_of_detected_keypoints: 0,
            descriptor_extractor: descriptor_extract,

            current_maximum_descriptor_distance_triangulation: 0.1 * 256f32,
            epipolar_search_distance: epipolar_search_offset_pixels,

            minimum_disparity_pixels: self.minimum_disparity_pixels,
            feature_matcher_left,
            feature_matcher_right,
        })
    }
}

pub struct StereoFramePointGenerator {
    detectors: Vec<Vec<(Ptr<features2d::FastFeatureDetector>, Rect2i, f32)>>,
    target_number_of_keypoints: f32,
    target_number_of_keypoints_per_detector: f32,
    target_number_of_keypoints_tolerance: f32,
    detector_threshold_maximum_change: f32,
    detector_threshold_minimum: f32,

    number_of_detected_keypoints: usize,
    descriptor_extractor: Ptr<DescriptorExtractor>,

    current_maximum_descriptor_distance_triangulation: f32,
    minimum_disparity_pixels: f32,

    epipolar_search_distance: Vec<i32>,

    feature_matcher_left: IntensityFeatureMatcher,
    feature_matcher_right: IntensityFeatureMatcher,
}

impl StereoFramePointGenerator {
    pub fn initialize(&mut self, frame: &mut Frame, extract_features: bool) -> Result<()> {
        if extract_features {
            frame.keypoints_left = self.detect_keypoints(&frame.intensity_image_left)?;
            frame.keypoints_right = self.detect_keypoints(&frame.intensity_image_right)?;

            self.adjust_detector_thresholds()?;

            self.number_of_detected_keypoints =
                (frame.keypoints_left.len() + frame.keypoints_right.len()) / 2;
            frame.number_of_detected_keypoints = self.number_of_detected_keypoints;

            self.compute_descriptors(
                &frame.intensity_image_left,
                &mut frame.keypoints_left,
                &mut frame.descriptors_left,
            )?;
            self.compute_descriptors(
                &frame.intensity_image_right,
                &mut frame.keypoints_right,
                &mut frame.descriptors_right,
            )?;
            log::debug!(
                "extracted features L: {} R: {}",
                frame.keypoints_left.len(),
                frame.keypoints_right.len()
            );

            if frame.status == FrameStatus::Localizing {
                self.current_maximum_descriptor_distance_triangulation = 0.1 * 256f32;
            } else {
                let ratio_available_points = (self.number_of_detected_keypoints as f32
                    / self.target_number_of_keypoints)
                    .min(1.0);
                self.current_maximum_descriptor_distance_triangulation = (ratio_available_points
                    * self.current_maximum_descriptor_distance_triangulation)
                    .max(0.1 * 256f32);
            }
        }

        self.feature_matcher_left
            .set_fatures(&frame.keypoints_left, &frame.descriptors_left)?;
        self.feature_matcher_right
            .set_fatures(&frame.keypoints_right, &frame.descriptors_right)?;
        Ok(())
    }

    pub fn get_epipolar_matches(&mut self, frame: &mut Frame, epipolar_offset: i32) -> Result<(Vec<KeyPoint>, Vec<KeyPoint>)> {
        self.feature_matcher_left.sort_feature_vector();
        self.feature_matcher_right.sort_feature_vector();

        let features_left = &self.feature_matcher_left.feature_vector;
        let features_right = &self.feature_matcher_right.feature_vector;

        let mut matched_indices_left = BTreeSet::<usize>::new();
        let mut matched_indices_right = BTreeSet::<usize>::new();

        let mut index_r = 0;
        let mut index_l = 0;

        'out: while index_l < features_left.len() {
            if index_r == features_right.len() {
                break;
            }

            while features_left[index_l].row < features_right[index_r].row + epipolar_offset {
                index_l += 1;
                if index_l == features_left.len() {
                    break 'out;
                }
            }

            let feature_left = features_left[index_l].clone();

            while feature_left.row > features_right[index_r].row + epipolar_offset {
                index_r += 1;
                if index_r == features_right.len() {
                    break 'out;
                }
            }

            let mut index_search_r = index_r;
            let mut descriptor_distance_best =
                self.current_maximum_descriptor_distance_triangulation;
            let mut index_best_r = 0;

            while feature_left.row == features_right[index_search_r].row + epipolar_offset {
                if feature_left.col - features_right[index_search_r].col < 0 {
                    break;
                }

                let descriptor_distance = opencv::core::norm2(
                    &feature_left.descriptor,
                    &features_right[index_search_r].descriptor,
                    NORM_HAMMING,
                    &Mat::default(),
                )?;
                if descriptor_distance < descriptor_distance_best as f64 {
                    descriptor_distance_best = descriptor_distance as f32;
                    index_best_r = index_search_r;
                }
                index_search_r += 1;
                if index_search_r == features_right.len() {
                    break;
                }
            }

            if descriptor_distance_best < self.current_maximum_descriptor_distance_triangulation {
                matched_indices_left.insert(index_l);
                matched_indices_right.insert(index_best_r);

                index_r = index_best_r + 1;
            }
            index_l += 1;
        }

        let left_key_points: Vec<_> =  matched_indices_left.iter().map(|x| {
            self.feature_matcher_left.feature_vector[*x].keypoint.clone()
        }).collect();

        let right_key_points: Vec<_> =  matched_indices_right.iter().map(|x| {
            self.feature_matcher_right.feature_vector[*x].keypoint.clone()
        }).collect();

        self.feature_matcher_left.prune(&matched_indices_left);
        self.feature_matcher_right.prune(&matched_indices_right);

        Ok((left_key_points, right_key_points))
    }

    pub fn compute_frame_point(&mut self, frame: &mut Frame) -> Result<()> {
        for epipolar_offset in self.epipolar_search_distance.clone() {
            let (features_left, features_right) = self.get_epipolar_matches(frame, epipolar_offset)?;

            // 跳过视差太小的两点
            for (feature_left, feature_right) in features_left.iter().zip(features_right.iter()) {
                let disparity = feature_left.pt().x - feature_right.pt().x;
                if disparity < self.minimum_disparity_pixels {
                    continue;
                }

                
            }
        }

        Ok(())
    }

    fn detect_keypoints(
        &mut self,
        intensity_image: &opencv::core::Mat,
    ) -> anyhow::Result<opencv::core::Vector<KeyPoint>> {
        let mut key_points = opencv::core::Vector::<KeyPoint>::new();
        for sub_detectors in self.detectors.iter_mut() {
            for (detector, region, threshold) in sub_detectors.iter_mut() {
                let mut keypoints_per_detector = opencv::core::Vector::<KeyPoint>::new();
                detector.detect_def(
                    &intensity_image.apply_1(*region)?,
                    &mut keypoints_per_detector,
                )?;
                let mut detector_threshold = detector.get_threshold()? as f32;

                let delta = (keypoints_per_detector.len() as f32
                    - self.target_number_of_keypoints_per_detector)
                    / self.target_number_of_keypoints_per_detector;

                //检查target points是否大量缺失
                if delta < -self.target_number_of_keypoints_tolerance {
                    let change = delta.max(self.detector_threshold_maximum_change);

                    detector_threshold += (change * detector_threshold as f32).min(-1.0);

                    if detector_threshold < self.detector_threshold_minimum {
                        detector_threshold = self.detector_threshold_minimum;
                    }
                } else if delta > self.target_number_of_keypoints_tolerance {
                    let change = delta.min(self.detector_threshold_maximum_change);

                    detector_threshold += (change * detector_threshold as f32).max(1.0);
                }

                *threshold = detector_threshold;

                let offset = region.tl();
                let offset = Point2f::new(offset.x as f32, offset.y as f32);
                let keypoints_per_detector: opencv::core::Vector<_> = keypoints_per_detector
                    .iter()
                    .map(|mut x| {
                        x.set_pt(x.pt() + offset);
                        x
                    })
                    .collect();

                key_points.extend(keypoints_per_detector.iter());
            }
        }
        Ok(key_points)
    }

    pub fn adjust_detector_thresholds(&mut self) -> anyhow::Result<()> {
        for sub_detectors in self.detectors.iter_mut() {
            for (detector, _, threshold) in sub_detectors.iter_mut() {
                detector.set_threshold(*threshold as _)?;
            }
        }
        Ok(())
    }

    pub fn compute_descriptors(
        &mut self,
        intensity_image: &opencv::core::Mat,
        keypoints: &mut opencv::core::Vector<KeyPoint>,
        descriptors: &mut Mat,
    ) -> anyhow::Result<()> {
        self.descriptor_extractor
            .compute(intensity_image, keypoints, descriptors)?;
        Ok(())
    }
}

pub struct Frame {
    pub keypoints_left: opencv::core::Vector<KeyPoint>,
    pub keypoints_right: opencv::core::Vector<KeyPoint>,

    pub descriptors_left: Mat,
    pub descriptors_right: Mat,

    pub intensity_image_left: opencv::core::Mat,
    pub intensity_image_right: opencv::core::Mat,

    pub number_of_detected_keypoints: usize,

    pub status: FrameStatus,
}

#[derive(PartialEq, Eq, Debug)]
pub enum FrameStatus {
    Localizing,
    Tracking,
}

impl Frame {
    pub fn new(
        intensity_image_left: opencv::core::Mat,
        intensity_image_right: opencv::core::Mat,
    ) -> Self {
        Self {
            keypoints_left: opencv::core::Vector::<KeyPoint>::new(),
            keypoints_right: opencv::core::Vector::<KeyPoint>::new(),

            descriptors_left: Mat::default(),
            descriptors_right: Mat::default(),

            intensity_image_left,
            intensity_image_right,

            number_of_detected_keypoints: 0,

            status: FrameStatus::Localizing,
        }
    }

    pub fn create_framepoint(
        &self,
        feature_left: &IntensityFeature,
        feature_right: &IntensityFeature,
        keypoint_right: &KeyPoint,
        descriptor_right: &Mat,
    ) {
    }
}
