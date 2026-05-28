use crate::backend::SlamBackend;
use std::ptr::NonNull;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameMetadata {
    pub frame_index: u64,
    pub timestamp_seconds: f64,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BorrowedBufferView {
    pub address: usize,
    pub byte_count: usize,
    pub stride_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackendInput {
    pub frame: FrameMetadata,
    pub disparity: BorrowedBufferView,
    pub descriptors: BorrowedBufferView,
    pub keypoint_count: u32,
    pub match_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackendResult {
    pub frame_index: u64,
    pub tracking_status: u32,
    pub pose_right_handed_column_major: [f64; 16],
    pub fused_point_count: u64,
}

#[no_mangle]
pub extern "C" fn vultur_slam_process_frame(input: BackendInput) -> BackendResult {
    let mut backend = SlamBackend::default();
    backend.process_frame(input)
}

#[no_mangle]
pub extern "C" fn vultur_slam_backend_create() -> *mut SlamBackend {
    Box::into_raw(Box::new(SlamBackend::default()))
}

#[no_mangle]
pub unsafe extern "C" fn vultur_slam_backend_destroy(backend: *mut SlamBackend) {
    if let Some(backend) = NonNull::new(backend) {
        drop(Box::from_raw(backend.as_ptr()));
    }
}

#[no_mangle]
pub unsafe extern "C" fn vultur_slam_backend_process_frame(
    backend: *mut SlamBackend,
    input: BackendInput,
) -> BackendResult {
    match NonNull::new(backend) {
        Some(mut backend) => backend.as_mut().process_frame(input),
        None => BackendResult {
            frame_index: input.frame.frame_index,
            tracking_status: 0,
            pose_right_handed_column_major: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            fused_point_count: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TrackingStatus;

    fn input(frame_index: u64, keypoints: u32, matches: u32) -> BackendInput {
        BackendInput {
            frame: FrameMetadata {
                frame_index,
                timestamp_seconds: frame_index as f64 / 30.0,
                width: 640,
                height: 480,
            },
            disparity: BorrowedBufferView {
                address: 4096,
                byte_count: 640 * 4,
                stride_bytes: 640,
            },
            descriptors: BorrowedBufferView {
                address: 8192,
                byte_count: keypoints as usize * 256,
                stride_bytes: 256,
            },
            keypoint_count: keypoints,
            match_count: matches,
        }
    }

    #[test]
    fn stateless_ffi_processes_one_frame() {
        let result = vultur_slam_process_frame(input(7, 64, 32));

        assert_eq!(result.frame_index, 7);
        assert_eq!(
            result.tracking_status,
            TrackingStatus::Initializing.as_u32()
        );
        assert!(result.fused_point_count > 0);
        assert_eq!(result.pose_right_handed_column_major[0], 1.0);
        assert_eq!(result.pose_right_handed_column_major[15], 1.0);
    }

    #[test]
    fn stateful_ffi_accumulates_mapping_and_fusion_state() {
        let backend = vultur_slam_backend_create();

        let first = unsafe { vultur_slam_backend_process_frame(backend, input(0, 80, 40)) };
        let second = unsafe { vultur_slam_backend_process_frame(backend, input(1, 80, 40)) };

        unsafe { vultur_slam_backend_destroy(backend) };

        assert_eq!(first.tracking_status, TrackingStatus::Initializing.as_u32());
        assert_eq!(second.tracking_status, TrackingStatus::Tracking.as_u32());
        assert!(second.fused_point_count > first.fused_point_count);
        assert!(
            second.pose_right_handed_column_major[14] > first.pose_right_handed_column_major[14]
        );
    }

    #[test]
    fn null_backend_pointer_returns_invalid_result() {
        let result =
            unsafe { vultur_slam_backend_process_frame(std::ptr::null_mut(), input(2, 80, 40)) };

        assert_eq!(
            result.tracking_status,
            TrackingStatus::InvalidInput.as_u32()
        );
        assert_eq!(result.fused_point_count, 0);
    }
}
