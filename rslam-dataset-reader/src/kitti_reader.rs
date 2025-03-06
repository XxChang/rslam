use opencv::{
    imgcodecs::{imread, IMREAD_GRAYSCALE},
    prelude::*,
};
use rslam_core::Camera;
use rslam_sensor::{pinhole_camera::PinholeCamera, HasStereoCamera};
use sophus::{
    core::linalg::VecF64, image::ImageSize, lie::Isometry3F64, nalgebra::Vector3, sensor::camera_enum::perspective_camera::PinholeCameraF64
};
use std::{io::BufRead, path::PathBuf};

pub struct KittiReader {
    dataset_path: PathBuf,
    cameras: Vec<PinholeCamera>,
    cameras_pos: Vec<Isometry3F64>,
    timestamp: Vec<f64>,
    current_frame_index: usize,
    pub baseline_pixel: Vector3<f64>,
}

impl KittiReader {
    pub fn new(dataset_path: &str) -> Self {
        KittiReader {
            dataset_path: PathBuf::from(dataset_path),
            cameras: vec![],
            cameras_pos: vec![],
            timestamp: vec![],
            current_frame_index: 0,
            baseline_pixel: Vector3::zeros(),
        }
    }

    pub fn get_timestamp(&mut self) -> f64 {
        let t = self.timestamp[self.current_frame_index];
        self.current_frame_index += 1;
        t
    }

    pub fn get_cameras(&self) -> &Vec<PinholeCamera> {
        &self.cameras
    }

    pub fn get_cameres_pos(&self) -> &Vec<Isometry3F64> {
        &self.cameras_pos
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
                let x = message_collect[3].parse::<f64>().unwrap();
                let y =  message_collect[7].parse::<f64>().unwrap();
                let z =  message_collect[11].parse::<f64>().unwrap();

                let camera = PinholeCameraF64::from_params_and_size(
                    &VecF64::<4>::new(fx, fy, cx, cy),
                    ImageSize::new(1241, 376),
                );
                log::debug!("loaded camera calibration matrix: {:?}", camera);
                self.cameras.push(PinholeCamera::new(camera));
                if x != 0.0 {
                    let baseline_pixels = Vector3::<f64>::new(x, y, z);
                    log::debug!("with baseline (pixels): {}", baseline_pixels.transpose());
                    self.baseline_pixel = baseline_pixels;
                }
                self.cameras_pos
                    .push(Isometry3F64::from_translation(&VecF64::<3>::new(
                        x,
                        y,
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
        log::debug!(
            "left_image_path: {:?}, right_image_path: {:?}",
            left_image_path,
            right_image_path
        );
        if let (Ok(left_image), Ok(right_image)) = (
            imread(&left_image_path.display().to_string(), IMREAD_GRAYSCALE),
            imread(&right_image_path.display().to_string(), IMREAD_GRAYSCALE),
        ) {
            if left_image.empty() || right_image.empty() {
                return None;
            }
            Some((left_image, right_image))
        } else {
            None
        }
    }
}

