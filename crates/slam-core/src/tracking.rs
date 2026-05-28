use crate::math::Pose3;
use crate::types::{FrameObservation, TrackingStatus};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackingConfig {
    pub min_matches_for_tracking: u32,
    pub min_keypoints_for_tracking: u32,
    pub nominal_frame_period_seconds: f64,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            min_matches_for_tracking: 24,
            min_keypoints_for_tracking: 32,
            nominal_frame_period_seconds: 1.0 / 30.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackingEstimate {
    pub status: TrackingStatus,
    pub pose_world_from_camera: Pose3,
    pub inlier_count: u32,
    pub should_create_keyframe: bool,
}

#[derive(Clone, Debug)]
pub struct VisualOdometry {
    config: TrackingConfig,
    last_frame_index: Option<u64>,
    last_pose: Pose3,
}

impl VisualOdometry {
    pub fn new(config: TrackingConfig) -> Self {
        Self {
            config,
            last_frame_index: None,
            last_pose: Pose3::identity(),
        }
    }

    pub fn estimate(&mut self, observation: FrameObservation) -> TrackingEstimate {
        if !observation.is_valid() {
            return TrackingEstimate {
                status: TrackingStatus::InvalidInput,
                pose_world_from_camera: self.last_pose,
                inlier_count: 0,
                should_create_keyframe: false,
            };
        }

        if self.last_frame_index.is_none() {
            self.last_frame_index = Some(observation.metadata.frame_index);
            self.last_pose = Pose3::identity();
            return TrackingEstimate {
                status: TrackingStatus::Initializing,
                pose_world_from_camera: self.last_pose,
                inlier_count: observation.match_count,
                should_create_keyframe: true,
            };
        }

        let status = if observation.match_count >= self.config.min_matches_for_tracking
            && observation.keypoint_count >= self.config.min_keypoints_for_tracking
        {
            TrackingStatus::Tracking
        } else if observation.match_count > 0 {
            TrackingStatus::Degraded
        } else {
            TrackingStatus::Lost
        };

        let previous_index = self
            .last_frame_index
            .unwrap_or(observation.metadata.frame_index);
        let frame_delta = observation
            .metadata
            .frame_index
            .saturating_sub(previous_index)
            .max(1) as f64;
        let match_confidence = if observation.keypoint_count == 0 {
            0.0
        } else {
            (observation.match_count as f64 / observation.keypoint_count as f64).clamp(0.0, 1.0)
        };
        let forward_step = match status {
            TrackingStatus::Tracking => 0.025 * frame_delta * match_confidence.max(0.25),
            TrackingStatus::Degraded => 0.005 * frame_delta,
            _ => 0.0,
        };

        let mut translation = self.last_pose.translation();
        translation[2] += forward_step;
        self.last_pose = self.last_pose.with_translation(translation);
        self.last_frame_index = Some(observation.metadata.frame_index);

        TrackingEstimate {
            status,
            pose_world_from_camera: self.last_pose,
            inlier_count: observation.match_count,
            should_create_keyframe: status == TrackingStatus::Tracking
                && observation.match_count >= self.config.min_matches_for_tracking * 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{BorrowedBufferView, FrameMetadata};
    use crate::types::{BufferView, FrameObservation};

    fn observation(frame_index: u64, keypoint_count: u32, match_count: u32) -> FrameObservation {
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
                byte_count: keypoint_count as usize * 256,
                stride_bytes: 256,
            }),
            keypoint_count,
            match_count,
        }
    }

    #[test]
    fn first_valid_frame_initializes_tracking() {
        let mut tracker = VisualOdometry::new(TrackingConfig::default());

        let estimate = tracker.estimate(observation(0, 80, 40));

        assert_eq!(estimate.status, TrackingStatus::Initializing);
        assert!(estimate.should_create_keyframe);
        assert_eq!(
            estimate.pose_world_from_camera.translation(),
            [0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn second_frame_tracks_and_advances_pose_when_matches_are_sufficient() {
        let mut tracker = VisualOdometry::new(TrackingConfig::default());
        let _ = tracker.estimate(observation(0, 80, 40));

        let estimate = tracker.estimate(observation(1, 80, 40));

        assert_eq!(estimate.status, TrackingStatus::Tracking);
        assert!(estimate.pose_world_from_camera.translation()[2] > 0.0);
    }

    #[test]
    fn missing_matches_marks_tracking_lost() {
        let mut tracker = VisualOdometry::new(TrackingConfig::default());
        let _ = tracker.estimate(observation(0, 80, 40));

        let estimate = tracker.estimate(observation(1, 80, 0));

        assert_eq!(estimate.status, TrackingStatus::Lost);
    }
}
