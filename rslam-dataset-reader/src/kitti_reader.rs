use sophus::{
    core::linalg::VecF64, image::ImageSize, lie::Isometry3F64, sensor::camera_enum::perspective_camera::PinholeCameraF64
};
use std::{io::BufRead, path::PathBuf};

pub struct KittiReader {
    dataset_path: PathBuf,
    cameras: Vec<PinholeCameraF64>,
    cameras_pos: Vec<Isometry3F64>,
}

impl KittiReader {
    pub fn new(dataset_path: &str) -> Self {
        KittiReader {
            dataset_path: PathBuf::from(dataset_path),
            cameras: vec![],
            cameras_pos: vec![],
        }
    }

    pub fn load_camera(&mut self) {
        let calib_file_path = self.dataset_path.join("calib.txt");

        let file = std::fs::File::open(&calib_file_path).unwrap();
        let file = std::io::BufReader::new(file);

        let mut lines = file.lines().filter_map(|l| l.ok());
        while let Some(mut l) = lines.next() {
            let l = l.split_off(4);
            let message_collect: Vec<&str> = l.split_whitespace().collect();
            // println!("{message_collect:?}");
            let fx = message_collect[0].parse::<f64>().unwrap();
            let fy = message_collect[5].parse::<f64>().unwrap();
            let cx = message_collect[2].parse::<f64>().unwrap();
            let cy = message_collect[6].parse::<f64>().unwrap();
            let x = -1f64 * message_collect[3].parse::<f64>().unwrap();
            let y = -1f64 * message_collect[7].parse::<f64>().unwrap();
            let z = -1f64 * message_collect[11].parse::<f64>().unwrap();

            self.cameras.push(PinholeCameraF64::from_params_and_size(
                &VecF64::<4>::new(fx, fy, cx, cy),
                ImageSize::new(1241, 376),
            ));
            println!("{:?}", self.cameras);

            self.cameras_pos.push(Isometry3F64::from_translation(&VecF64::<3>::new(x/fx, y/fy, z)));
        }
    }
}
