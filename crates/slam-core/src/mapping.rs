use crate::math::Pose3;
use crate::tracking::TrackingEstimate;
use crate::types::{FrameObservation, TrackingStatus};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyframeSummary {
    pub frame_index: u64,
    pub timestamp_seconds: f64,
    pub keypoint_count: u32,
    pub inlier_count: u32,
    pub pose_world_from_camera: Pose3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MappingConfig {
    pub max_keyframes: usize,
}

impl Default for MappingConfig {
    fn default() -> Self {
        Self { max_keyframes: 32 }
    }
}

#[derive(Clone, Debug)]
pub struct LocalMap {
    config: MappingConfig,
    keyframes: Vec<KeyframeSummary>,
}

impl LocalMap {
    pub fn new(config: MappingConfig) -> Self {
        Self {
            config,
            keyframes: Vec::new(),
        }
    }

    pub fn update(
        &mut self,
        observation: FrameObservation,
        estimate: TrackingEstimate,
    ) -> Option<KeyframeSummary> {
        let should_insert = matches!(estimate.status, TrackingStatus::Initializing)
            || estimate.should_create_keyframe
            || self.keyframes.is_empty();

        if !should_insert {
            return None;
        }

        let keyframe = KeyframeSummary {
            frame_index: observation.metadata.frame_index,
            timestamp_seconds: observation.metadata.timestamp_seconds,
            keypoint_count: observation.keypoint_count,
            inlier_count: estimate.inlier_count,
            pose_world_from_camera: estimate.pose_world_from_camera,
        };
        self.keyframes.push(keyframe);

        if self.keyframes.len() > self.config.max_keyframes {
            let overflow = self.keyframes.len() - self.config.max_keyframes;
            self.keyframes.drain(0..overflow);
        }

        Some(keyframe)
    }

    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    pub fn keyframes(&self) -> &[KeyframeSummary] {
        &self.keyframes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{BorrowedBufferView, FrameMetadata};
    use crate::math::Pose3;
    use crate::types::{BufferView, FrameObservation};

    fn observation(frame_index: u64) -> FrameObservation {
        FrameObservation {
            metadata: FrameMetadata {
                frame_index,
                timestamp_seconds: frame_index as f64 / 30.0,
                width: 640,
                height: 480,
            },
            disparity: BufferView::from(BorrowedBufferView {
                address: 1,
                byte_count: 1280,
                stride_bytes: 640,
            }),
            descriptors: BufferView::from(BorrowedBufferView {
                address: 2,
                byte_count: 2048,
                stride_bytes: 256,
            }),
            keypoint_count: 64,
            match_count: 40,
        }
    }

    fn estimate(status: TrackingStatus, should_create_keyframe: bool) -> TrackingEstimate {
        TrackingEstimate {
            status,
            pose_world_from_camera: Pose3::identity(),
            inlier_count: 40,
            should_create_keyframe,
        }
    }

    #[test]
    fn map_inserts_initializing_keyframe() {
        let mut map = LocalMap::new(MappingConfig::default());

        let keyframe = map.update(
            observation(0),
            estimate(TrackingStatus::Initializing, false),
        );

        assert!(keyframe.is_some());
        assert_eq!(map.keyframe_count(), 1);
    }

    #[test]
    fn map_prunes_old_keyframes_to_configured_window() {
        let mut map = LocalMap::new(MappingConfig { max_keyframes: 2 });

        let _ = map.update(
            observation(0),
            estimate(TrackingStatus::Initializing, false),
        );
        let _ = map.update(observation(1), estimate(TrackingStatus::Tracking, true));
        let _ = map.update(observation(2), estimate(TrackingStatus::Tracking, true));

        assert_eq!(map.keyframe_count(), 2);
        assert_eq!(map.keyframes()[0].frame_index, 1);
    }
}
