use crate::mapping::KeyframeSummary;
use crate::tracking::TrackingEstimate;
use crate::types::{FrameObservation, TrackingStatus};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusionConfig {
    pub max_points: u64,
    pub points_per_match: u64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            max_points: 2_000_000,
            points_per_match: 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusionUpdate {
    pub inserted_points: u64,
    pub total_points: u64,
}

#[derive(Clone, Debug)]
pub struct DepthFusion {
    config: FusionConfig,
    total_points: u64,
}

impl DepthFusion {
    pub fn new(config: FusionConfig) -> Self {
        Self {
            config,
            total_points: 0,
        }
    }

    pub fn integrate(
        &mut self,
        observation: FrameObservation,
        estimate: TrackingEstimate,
        keyframe: Option<KeyframeSummary>,
    ) -> FusionUpdate {
        let can_integrate = matches!(
            estimate.status,
            TrackingStatus::Initializing | TrackingStatus::Tracking | TrackingStatus::Degraded
        );

        if !can_integrate {
            return FusionUpdate {
                inserted_points: 0,
                total_points: self.total_points,
            };
        }

        let disparity_rows = observation.disparity.row_count() as u64;
        let disparity_hint = if observation.disparity.is_present() {
            disparity_rows.max(1).min(4096)
        } else {
            0
        };
        let sparse_hint = observation.match_count as u64 * self.config.points_per_match;
        let keyframe_bonus = keyframe
            .map(|_| observation.keypoint_count as u64 / 8)
            .unwrap_or(0);
        let inserted = disparity_hint + sparse_hint + keyframe_bonus;

        self.total_points = self
            .total_points
            .saturating_add(inserted)
            .min(self.config.max_points);

        FusionUpdate {
            inserted_points: inserted,
            total_points: self.total_points,
        }
    }

    pub fn total_points(&self) -> u64 {
        self.total_points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{BorrowedBufferView, FrameMetadata};
    use crate::math::Pose3;
    use crate::types::{BufferView, FrameObservation};

    fn observation(match_count: u32) -> FrameObservation {
        FrameObservation {
            metadata: FrameMetadata {
                frame_index: 0,
                timestamp_seconds: 0.0,
                width: 640,
                height: 480,
            },
            disparity: BufferView::from(BorrowedBufferView {
                address: 1,
                byte_count: 640 * 4,
                stride_bytes: 640,
            }),
            descriptors: BufferView::from(BorrowedBufferView {
                address: 2,
                byte_count: 4096,
                stride_bytes: 256,
            }),
            keypoint_count: 64,
            match_count,
        }
    }

    fn estimate(status: TrackingStatus) -> TrackingEstimate {
        TrackingEstimate {
            status,
            pose_world_from_camera: Pose3::identity(),
            inlier_count: 32,
            should_create_keyframe: false,
        }
    }

    #[test]
    fn fusion_integrates_depth_and_sparse_match_hints() {
        let mut fusion = DepthFusion::new(FusionConfig::default());

        let update = fusion.integrate(observation(32), estimate(TrackingStatus::Tracking), None);

        assert!(update.inserted_points > 32);
        assert_eq!(update.total_points, fusion.total_points());
    }

    #[test]
    fn fusion_does_not_integrate_lost_tracking() {
        let mut fusion = DepthFusion::new(FusionConfig::default());

        let update = fusion.integrate(observation(32), estimate(TrackingStatus::Lost), None);

        assert_eq!(update.inserted_points, 0);
        assert_eq!(update.total_points, 0);
    }
}
