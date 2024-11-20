use proslam::frame_point_generator::FramePointGeneratorCfg;
use rslam_core::Camera;
use rslam_dataset_reader::kitti_reader::KittiReader;
use rslam_sensor::HasStereoCamera;

fn main() {
    env_logger::init();

    let mut reader = KittiReader::new("datasets/01");
    reader.load_camera();
    reader.load_timestamp();

    let number_of_rows_image = reader.rows();
    let number_of_cols_image = reader.cols();

    let frame_point_cfg = FramePointGeneratorCfg::default();
    let mut frame_point_generator = frame_point_cfg
        .finalize(number_of_cols_image, number_of_rows_image)
        .unwrap();

    while let Some((left, right)) = reader.get_stereo_frame() {
        let _left_keypoints = frame_point_generator.detect_keypoints(&left).unwrap();
        let _right_keypoints = frame_point_generator.detect_keypoints(&right).unwrap();
    }
}
