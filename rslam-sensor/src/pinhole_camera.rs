use sophus::{lie::Isometry3F64, sensor::camera_enum::perspective_camera::PinholeCameraF64};

#[derive(Clone, Debug)]
pub struct PinholeCamera {
    pub model: PinholeCameraF64,

    // camera to robot transform (usually constant during operation)
    camera_to_robot: Isometry3F64,
}

impl rslam_core::Camera for PinholeCamera {
    fn cols(&self) -> usize {
        self.model.image_size().width
    }

    fn rows(&self) -> usize {
        self.model.image_size().height
    }

    fn camera_to_robot(&self) -> &sophus::lie::Isometry3F64 {
        &self.camera_to_robot
    }
}

impl PinholeCamera {
    pub fn new(model: PinholeCameraF64) -> Self {
        Self {
            model,

            camera_to_robot: Isometry3F64::identity(),
        }
    }
}
