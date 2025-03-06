pub trait Camera {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;

    fn camera_to_robot(&self) -> &sophus::lie::Isometry3F64;
}
