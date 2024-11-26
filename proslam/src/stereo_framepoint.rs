use opencv::core::{KeyPoint, KeyPointTraitConst, Mat};

pub struct StereoFramepoint {
    keypoint_left: opencv::core::KeyPoint,
    keypoint_right: opencv::core::KeyPoint,
    descriptor_left: opencv::core::Mat,
    descriptor_right: opencv::core::Mat,
    disparity_pixels: f32,
}

impl StereoFramepoint {
    pub fn new(
        keypoint_left: &KeyPoint,
        keypoint_right: &KeyPoint,
        descriptor_left: &Mat,
        descriptor_right: &Mat,
    ) -> StereoFramepoint {
        StereoFramepoint {
            keypoint_left: keypoint_left.clone(),
            keypoint_right: keypoint_right.clone(),
            descriptor_left: descriptor_left.clone(),
            descriptor_right: descriptor_right.clone(),
            disparity_pixels: keypoint_left.pt().x - keypoint_right.pt().y,
        }
    }
}
