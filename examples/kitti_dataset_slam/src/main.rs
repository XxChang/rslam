use rslam_dataset_reader::kitti_reader::KittiReader;

fn main() {
    let mut reader = KittiReader::new("datasets/01");
    reader.load_camera();
}
