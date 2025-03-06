use sophus::nalgebra::Vector3;

pub struct FramePoint {
    // 3D point in left camera coordinate frame
    camera_coordinates_left: Vector3<f64>,
    // 3D point in robot coordinate frame
    robot_coordinates: Vector3<f64>,
    // 3D point in world coordinate frame
    world_coordinates: Vector3<f64>,
    depth_meters: f64,
}

impl FramePoint {
    pub fn new(camera_coordinates_left: Vector3<f64>, robot_coordinates: Vector3<f64>, world_coordinates: Vector3<f64>) -> Self {
        Self {
            camera_coordinates_left,
            robot_coordinates,
            world_coordinates,
            depth_meters: -1.0
        }
    }
}