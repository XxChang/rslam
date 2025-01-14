use std::sync::atomic::AtomicUsize;
use rslam_core::PointCoordinates;
use opencv::core::{KeyPoint, KeyPointTraitConst, Mat};

static IDENTIFIER: AtomicUsize = AtomicUsize::new(0);

pub struct StereoFramepoint {
    keypoint_left: opencv::core::KeyPoint,
    keypoint_right: opencv::core::KeyPoint,
    descriptor_left: opencv::core::Mat,
    descriptor_right: opencv::core::Mat,
    disparity_pixels: f32,

    // row and col in the image
    pub row: i32,
    pub col: i32,

    identifier: usize,

    image_coordinates_left: PointCoordinates,
    image_coordinates_right: PointCoordinates,
}

impl StereoFramepoint {
    pub fn new(
        keypoint_left: &KeyPoint,
        keypoint_right: &KeyPoint,
        descriptor_left: &Mat,
        descriptor_right: &Mat,
    ) -> StereoFramepoint {
        let r = StereoFramepoint {
            row: keypoint_left.pt().y as i32,
            col: keypoint_left.pt().x as i32,
            identifier: IDENTIFIER.load(std::sync::atomic::Ordering::SeqCst),

            keypoint_left: keypoint_left.clone(),
            keypoint_right: keypoint_right.clone(),
            descriptor_left: descriptor_left.clone(),
            descriptor_right: descriptor_right.clone(),
            disparity_pixels: keypoint_left.pt().x - keypoint_right.pt().y,

            image_coordinates_left: PointCoordinates::new(keypoint_left.pt().x as f64, keypoint_left.pt().y as f64, 1.0),
            image_coordinates_right: PointCoordinates::new(keypoint_right.pt().x as f64, keypoint_right.pt().y as f64, 1.0),
        };

        IDENTIFIER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        r
    }

    pub fn new_with_intensity_feature(
        feature_left: &IntensityFeature,
        feature_right: &IntensityFeature,
    ) -> StereoFramepoint {
        StereoFramepoint::new(
            &feature_left.keypoint, 
            &feature_right.keypoint, 
            &feature_left.descriptor, 
            &feature_right.descriptor)
    }
}

pub struct IntensityFeature {
    pub keypoint: KeyPoint,
    pub descriptor: Mat,

    pub row: i32,
    pub col: i32,

    // inverted index for vector containing this
    pub index_in_vector: usize,
}

impl IntensityFeature {
    pub fn new(keypoint: &KeyPoint, descriptor: &Mat, index_in_vector: usize) -> IntensityFeature {
        IntensityFeature {
            keypoint: keypoint.clone(),
            descriptor: descriptor.clone(),
            row: keypoint.pt().y as i32,
            col: keypoint.pt().x as i32,
            index_in_vector,
        }
    }
}