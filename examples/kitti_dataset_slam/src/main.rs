// use opencv::core::{KeyPointTraitConst, MatTraitConst, MatTraitConstManual};
use proslam::stereo_frame_point_generator::{Frame, StereoFramePointGeneratorCfg};
// use rerun::ColorModel;
use rslam_core::Camera;
use rslam_dataset_reader::kitti_reader::KittiReader;
use rslam_sensor::HasStereoCamera;

fn main() {
    env_logger::init();

    // let rec = rerun::RecordingStreamBuilder::new("kitti_dataset_image")
    //     .spawn()
    //     .unwrap();

    let mut reader = KittiReader::new("datasets/01");
    reader.load_camera();
    reader.load_timestamp();

    let number_of_rows_image = reader.get_cameras()[0].rows();
    let number_of_cols_image = reader.get_cameras()[0].cols();

    let cameras = reader.get_cameras();
    let cameras_pos = reader.get_cameres_pos();

    let baseline_x = reader.baseline_pixel;
    let frame_point_cfg = StereoFramePointGeneratorCfg::default();
    let mut frame_point_generator = frame_point_cfg
        .finalize(
            number_of_cols_image,
            number_of_rows_image,
            cameras[0].clone(),
            cameras[1].clone(),
            baseline_x,
        )
        .unwrap();
    log::debug!("camera_left: {:?}, camera pos: {:?}", cameras[0], cameras_pos[0].matrix());
    log::debug!("camera_right: {:?}, camera pos: {:?}", cameras[1], cameras_pos[1].matrix());
    log::debug!("baseline_x: {}", baseline_x);
    log::debug!("focal_length_pixels: {}", cameras[0].model.params()[0]);

    // while let Some((left, right)) = reader.get_stereo_frame() {
    //     let _secs = reader.get_timestamp();
        // rec.set_time_seconds("sample_time", secs);

        // let mut frame = Frame::new(left.clone(), right.clone());
        // frame_point_generator.initialize(&mut frame, true).unwrap();

        // frame_point_generator.compute_frame_point(&mut frame).unwrap();
        // let (left_keypoints, right_keypoints) = frame_point_generator.get_epipolar_matches(&mut frame, 0).unwrap();

        // let left_pts: Vec<_> = left_keypoints
        //     .iter()
        //     .map(|x| rerun::Position2D::new(x.pt().x, x.pt().y))
        //     .collect();
        // let right_pts: Vec<_> = right_keypoints
        //     .iter()
        //     .map(|x| rerun::Position2D::new(x.pt().x, x.pt().y))
        //     .collect();

        // rec.log(
        //     "/camera/image_left",
        //     &rerun::Image::from_elements(
        //         left.data_bytes().unwrap(),
        //         [left.cols() as u32, left.rows() as u32],
        //         ColorModel::L,
        //     ),
        // )
        // .unwrap();

        // rec.log("/camera/image_left/keypoints", &rerun::Points2D::new(left_pts))
        //     .unwrap();

        // rec.log(
        //     "/camera/image_right",
        //     &rerun::Image::from_elements(
        //         right.data_bytes().unwrap(),
        //         [right.cols() as u32, right.rows() as u32],
        //         ColorModel::L,
        //     ),
        // )
        // .unwrap();

        // rec.log("/camera/image_right/keypoints", &rerun::Points2D::new(right_pts))
        //     .unwrap();

        // let associations: Vec<_> = left_keypoints
        //     .iter()
        //     .zip(right_keypoints.iter())
        //     .map(|(l, r)| (rerun::Position2D::new(l.pt().x, l.pt().y), rerun::Position2D::new(r.pt().x, r.pt().y)))
        //     .collect();
        // rerun::
    // }
}
