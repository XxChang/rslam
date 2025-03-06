mod dataset;
pub use dataset::*;
mod camera;
pub use camera::*;
pub mod frame;
pub mod framepoint;

use sophus::nalgebra::Vector3;

pub type Real = f64;
pub type PointCoordinates = Vector3<Real>;
pub type ImageCoordinates = Vector3<Real>;
