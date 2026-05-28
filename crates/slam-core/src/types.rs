use crate::ffi::{BackendInput, BorrowedBufferView, FrameMetadata};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum TrackingStatus {
    InvalidInput = 0,
    Initializing = 1,
    Tracking = 2,
    Degraded = 3,
    Lost = 4,
}

impl TrackingStatus {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferKind {
    Empty,
    Borrowed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BufferView {
    pub address: usize,
    pub byte_count: usize,
    pub stride_bytes: usize,
    pub kind: BufferKind,
}

impl From<BorrowedBufferView> for BufferView {
    fn from(value: BorrowedBufferView) -> Self {
        let kind = if value.address == 0 || value.byte_count == 0 {
            BufferKind::Empty
        } else {
            BufferKind::Borrowed
        };

        Self {
            address: value.address,
            byte_count: value.byte_count,
            stride_bytes: value.stride_bytes,
            kind,
        }
    }
}

impl BufferView {
    pub fn is_present(self) -> bool {
        self.kind == BufferKind::Borrowed
    }

    pub fn row_count(self) -> usize {
        if self.stride_bytes == 0 {
            0
        } else {
            self.byte_count / self.stride_bytes
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameObservation {
    pub metadata: FrameMetadata,
    pub disparity: BufferView,
    pub descriptors: BufferView,
    pub keypoint_count: u32,
    pub match_count: u32,
}

impl From<BackendInput> for FrameObservation {
    fn from(value: BackendInput) -> Self {
        Self {
            metadata: value.frame,
            disparity: value.disparity.into(),
            descriptors: value.descriptors.into(),
            keypoint_count: value.keypoint_count,
            match_count: value.match_count,
        }
    }
}

impl FrameObservation {
    pub fn is_valid(self) -> bool {
        self.metadata.width > 0
            && self.metadata.height > 0
            && self.metadata.timestamp_seconds.is_finite()
    }
}
