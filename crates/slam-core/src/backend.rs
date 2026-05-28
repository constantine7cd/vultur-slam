use crate::ffi::{BackendInput, BackendResult};
use crate::fusion::{DepthFusion, FusionConfig};
use crate::mapping::{LocalMap, MappingConfig};
use crate::tracking::{TrackingConfig, VisualOdometry};
use crate::types::FrameObservation;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BackendConfig {
    pub tracking: TrackingConfig,
    pub mapping: MappingConfig,
    pub fusion: FusionConfig,
}

#[derive(Clone, Debug)]
pub struct SlamBackend {
    tracker: VisualOdometry,
    local_map: LocalMap,
    fusion: DepthFusion,
}

impl Default for SlamBackend {
    fn default() -> Self {
        Self::new(BackendConfig::default())
    }
}

impl SlamBackend {
    pub fn new(config: BackendConfig) -> Self {
        Self {
            tracker: VisualOdometry::new(config.tracking),
            local_map: LocalMap::new(config.mapping),
            fusion: DepthFusion::new(config.fusion),
        }
    }

    pub fn process_frame(&mut self, input: BackendInput) -> BackendResult {
        let observation = FrameObservation::from(input);
        let estimate = self.tracker.estimate(observation);
        let keyframe = self.local_map.update(observation, estimate);
        let fusion = self.fusion.integrate(observation, estimate, keyframe);

        BackendResult {
            frame_index: observation.metadata.frame_index,
            tracking_status: estimate.status.as_u32(),
            pose_right_handed_column_major: estimate.pose_world_from_camera.column_major(),
            fused_point_count: fusion.total_points,
        }
    }

    pub fn keyframe_count(&self) -> usize {
        self.local_map.keyframe_count()
    }

    pub fn fused_point_count(&self) -> u64 {
        self.fusion.total_points()
    }
}
