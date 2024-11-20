use opencv::{imgcodecs::imread_def, prelude::*};
use rslam_core::Camera;
use rslam_sensor::HasStereoCamera;
use sophus::{
    core::linalg::VecF64, image::ImageSize, lie::Isometry3F64,
    sensor::camera_enum::perspective_camera::PinholeCameraF64,
};
use std::{io::BufRead, path::PathBuf};

pub struct KittiReader {
    dataset_path: PathBuf,
    cameras: Vec<PinholeCameraF64>,
    cameras_pos: Vec<Isometry3F64>,
    timestamp: Vec<f64>,
    current_frame_index: usize,
}

impl KittiReader {
    pub fn new(dataset_path: &str) -> Self {
        KittiReader {
            dataset_path: PathBuf::from(dataset_path),
            cameras: vec![],
            cameras_pos: vec![],
            timestamp: vec![],
            current_frame_index: 0,
        }
    }

    pub fn load_camera(&mut self) {
        let calib_file_path = self.dataset_path.join("calib.txt");

        let file = std::fs::File::open(&calib_file_path).unwrap();
        let file = std::io::BufReader::new(file);

        let lines = file.lines().filter_map(|l| l.ok());
        for (i, mut l) in lines.enumerate() {
            let init_image = self
                .dataset_path
                .join(format!("image_{}", i))
                .join("000000.png");
            if init_image.exists() {
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

                self.cameras_pos
                    .push(Isometry3F64::from_translation(&VecF64::<3>::new(
                        x / fx,
                        y / fy,
                        z,
                    )));
            }
        }
    }

    pub fn load_timestamp(&mut self) {
        let timestamp_file_path = self.dataset_path.join("times.txt");

        let file = std::fs::File::open(&timestamp_file_path).unwrap();
        let file = std::io::BufReader::new(file);

        let lines = file.lines().filter_map(|l| l.ok());
        for l in lines {
            let timestamp = l.parse::<f64>().unwrap();
            self.timestamp.push(timestamp);
        }
    }
}

impl HasStereoCamera for &mut KittiReader {
    type FrameItem = Mat;

    fn get_stereo_frame(self) -> Option<(Self::FrameItem, Self::FrameItem)> {
        let left_image_path = self
            .dataset_path
            .join("image_0")
            .join(format!("{:06}.png", self.current_frame_index));
        let right_image_path = self
            .dataset_path
            .join("image_1")
            .join(format!("{:06}.png", self.current_frame_index));

        if let (Ok(left_image), Ok(right_image)) = (
            imread_def(&left_image_path.display().to_string()),
            imread_def(&right_image_path.display().to_string()),
        ) {
            self.current_frame_index += 1;
            Some((left_image, right_image))
        } else {
            None
        }
    }
}

impl Camera for KittiReader {
    fn rows(&self) -> usize {
        self.cameras[0].image_size().height
    }

    fn cols(&self) -> usize {
        self.cameras[0].image_size().width
    }
}
