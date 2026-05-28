# vultur-slam

Apple Silicon-first stereo SLAM scaffold.

The first implementation provides the application architecture, not the final
SLAM algorithms:

- SwiftPM CLI host for Apple framework integration.
- IOSurface/CVPixelBuffer-oriented frame ownership.
- Placeholder Metal rectification, disparity, feature detection, matching, and
  Rust backend stages.
- Pipeline configuration, event reporting, frame resource retention, and
  explicit FFI boundary types for zero-copy handoff.
- Rust `slam-core` workspace with FFI-safe backend contracts.
- Offline smoke pipeline and config validation commands.

## Build and test

```sh
swift test
swift run vultur-slam validate-config --calibration Fixtures/calibration.example.json
swift run vultur-slam run-config --config Fixtures/pipeline.offline.example.json
```

Rust is not required to build the Swift scaffold, but the backend crate is laid
out for future integration:

```sh
cargo test
```

## CLI

```sh
vultur-slam run-config --config <pipeline.json>
vultur-slam validate-config --calibration <calibration.json>
vultur-slam list-cameras
vultur-slam offline --left <file-or-dir> --right <file-or-dir> --calibration <calibration.json> [--output <dir>] [--max-frames <n>]
vultur-slam online --calibration <calibration.json> [--left-device <id>] [--right-device <id>] [--fps <n>]
```

`online` is intentionally present as an interface placeholder. The first
concrete online source should implement the same `StereoFrameSource` contract
using AVFoundation, with vendor camera SDKs added behind that abstraction later.

## Architecture notes

- `SLAMPipeline` retains Swift-owned frame resources while Rust-compatible
  metadata and borrowed buffer views flow through backend contracts.
- Disparity and feature detection are launched concurrently per rectified frame;
  matching and backend processing run after their dependencies are ready.
- `PipelineEventSink` makes stage timings and lifecycle events observable
  without coupling the pipeline to a logging implementation.
- `MetalRuntime` centralizes Metal device, command queue, and
  `CVMetalTextureCache` creation for future rectification kernels.

## Rust backend

The Rust crate is organized around the SLAM backend responsibilities:

- `tracking`: visual odometry state machine and placeholder pose propagation.
- `mapping`: bounded local keyframe window.
- `fusion`: dense/sparse depth integration accounting.
- `backend`: orchestrates tracking, mapping, and fusion per frame.
- `ffi`: stateless and stateful C ABI entrypoints for Swift.

The stateful ABI keeps backend state across frames:

```c
SlamBackend *vultur_slam_backend_create(void);
BackendResult vultur_slam_backend_process_frame(SlamBackend *, BackendInput);
void vultur_slam_backend_destroy(SlamBackend *);
```
