pub mod pinhole_camera;

pub trait HasStereoCamera {
    type FrameItem;

    fn get_stereo_frame(self) -> Option<(Self::FrameItem, Self::FrameItem)>;
}
