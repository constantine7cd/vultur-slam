# FastFoundationStereoMPS

Native macOS runtime for the fixed-shape FP32 FastFoundationStereo port.

## Current Boundary

The runtime keeps `Contents/Resources/feature_projection.mlpackage` as the Core ML entry point. Its NCHW outputs are copied into shared `MTLBuffer`s, converted to NHWC, and then consumed by native Metal/MPSGraph stages.

## Packaging for Another macOS Project

Use this directory as a SwiftPM package and link the `FastFoundationStereoMPS` library product from the consuming macOS app. The package owns native Swift code and shader resources only. The consuming app owns the model artifacts and should bundle them as app resources.

Recommended app resources:

```text
App/
  Contents/Resources/
    feature_projection.mlmodelc/      # preferred for release builds
    FastFoundationStereoWeights/
      manifest.json
      *.bin
```

`feature_projection.mlpackage` also works. `FeatureProjectionRunner` compiles `.mlpackage` at runtime, while `.mlmodelc` avoids that startup cost.

Usage example:

```swift
import FastFoundationStereoMPS

let runner = try FastFoundationStereoMPSRunner()
let resources = try FastFoundationStereoMPSResources.resolve(in: .main)
let featureRunner = try FeatureProjectionRunner(
    modelURL: resources.featureModelURL,
    arena: runner.arena
)
let weights = try NativeWeights(
    directory: resources.weightsDirectoryURL,
    device: runner.context.device
)

let inputTensors = try StereoInputTensors(
    leftRGBNCHW: rectifiedLeftRGB,
    rightRGBNCHW: rectifiedRightRGB
)
let disparity = try runner.run(
    inputTensors: inputTensors,
    featureRunner: featureRunner,
    weights: weights,
    validIterations: 8
)
```

`rectifiedLeftRGB` and `rectifiedRightRGB` must be contiguous FP32 RGB tensors in NCHW order with shape `[1, 3, 480, 640]`. Values should use the same scale as `ImageTensorIO.loadRGBTensor`, currently RGB channel values in `0...255`. If an app already owns `MLMultiArray` tensors, use `StereoInputTensors(left:right:)`; the initializer validates float32 contiguous NCHW layout before inference.
