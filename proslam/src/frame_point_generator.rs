use anyhow::Result;
use opencv::{
    core::{KeyPoint, KeyPointTrait, KeyPointTraitConst, MatTraitConst, Point2f, Ptr, Rect, Rect2i},
    features2d::{self, FastFeatureDetector, FastFeatureDetector_DetectorType},
    prelude::{FastFeatureDetectorTraitConst, Feature2DTrait},
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct FramePointGeneratorCfg {
    number_of_detectors_vertical: usize,
    number_of_detectors_horizontal: usize,
    target_number_of_keypoints_tolerance: f32,
    detector_threshold_maximum_change: f32,
    detector_threshold_initial: usize,
    detector_threshold_minimum: usize,
    detector_threshold_maximum: usize,

    bin_size_pixels: usize,
}

impl Default for FramePointGeneratorCfg {
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
        }
    }
}

impl FramePointGeneratorCfg {
    pub fn finalize(self, width: usize, height: usize) -> Result<FramePointGenerator> {
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

        Ok(FramePointGenerator {
            detectors,
            target_number_of_keypoints_per_detector,
            target_number_of_keypoints_tolerance: self.target_number_of_keypoints_tolerance,
            detector_threshold_maximum_change: self.detector_threshold_maximum_change,
            detector_threshold_minimum: self.detector_threshold_minimum as f32,
        })
    }
}

pub struct FramePointGenerator {
    detectors: Vec<Vec<(Ptr<features2d::FastFeatureDetector>, Rect2i, f32)>>,
    target_number_of_keypoints_per_detector: f32,
    target_number_of_keypoints_tolerance: f32,
    detector_threshold_maximum_change: f32,
    detector_threshold_minimum: f32,
}

impl FramePointGenerator {
    pub fn detect_keypoints(
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
                let keypoints_per_detector: opencv::core::Vector<_> = keypoints_per_detector.iter().map(|mut x| {
                    x.set_pt(x.pt() + offset);
                    x
                }).collect();

                key_points.extend(keypoints_per_detector.iter());

            }
        }
        Ok(key_points)
    }
}
