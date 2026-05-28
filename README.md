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
- Native FastFoundationStereoMPS package integration for Apple Silicon disparity
  inference from rectified stereo pairs.

## Build and test

```sh
swift test
swift run vultur-slam validate-config --calibration Fixtures/calibration.example.json
swift run vultur-slam run-config --config Fixtures/pipeline.offline.example.json
swift run vultur-slam fast-foundation-stereo --output demo_data/disparity.fp32
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
vultur-slam fast-foundation-stereo [--left <image>] [--right <image>] [--resources <Contents/Resources>] [--output <disparity.fp32>] [--valid-iters <n>] [--cost-precision <float32|float16>]
vultur-slam online --calibration <calibration.json> [--left-device <id>] [--right-device <id>] [--fps <n>]
```

`online` is intentionally present as an interface placeholder. The first
concrete online source should implement the same `StereoFrameSource` contract
using AVFoundation, with vendor camera SDKs added behind that abstraction later.

## FastFoundationStereoMPS

`Sources/VulturSLAMCore/ML/FastFoundationStereoMPS` is linked as a local SwiftPM
package. The root `VulturSLAMCore` target excludes the nested package sources so
the model runner remains an independent library while `VulturSLAMCLI` can import
`FastFoundationStereoMPS`.

Model resources are expected under `Contents/Resources` by default:

```text
Contents/Resources/
  feature_projection.mlpackage
  FastFoundationStereoWeights/
    manifest.json
    *.bin
```

The demo command loads `demo_data/left.png` and `demo_data/right.png` unless
overridden, runs FastFoundationStereoMPS, and writes the full-resolution
disparity tensor:

```sh
swift run vultur-slam fast-foundation-stereo --left demo_data/left.png --right demo_data/right.png --output demo_data/disparity.fp32
```

The output is raw FP32 tensor data. A JSON sidecar is written beside it, for
example `demo_data/disparity.fp32.json`, with shape and layout metadata. The
current fixed output shape is `[1, 480, 640, 1]` in `nhwc` layout.

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
