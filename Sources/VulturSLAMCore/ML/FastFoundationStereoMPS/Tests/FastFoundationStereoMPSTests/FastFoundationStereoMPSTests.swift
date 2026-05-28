import XCTest
import CoreML
import Metal
@testable import FastFoundationStereoMPS

final class FastFoundationStereoMPSTests: XCTestCase {
    func testTensorShapeByteCount() {
        let shape = TensorShape([1, 120, 160, 12], layout: .nhwc)
        XCTAssertEqual(shape.elementCount, 230_400)
        XCTAssertEqual(shape.byteCountFP32, 921_600)
    }

    func testStereoInputTensorsBuildContiguousRGBNCHWTensors() throws {
        let count = StereoInputTensors.imageElementCount
        let leftValues = Array(repeating: Float(1), count: count)
        let rightValues = Array(repeating: Float(2), count: count)

        let tensors = try StereoInputTensors(leftRGBNCHW: leftValues, rightRGBNCHW: rightValues)

        XCTAssertEqual(tensors.left.shape.map(\.intValue), [1, 3, 480, 640])
        XCTAssertEqual(tensors.right.shape.map(\.intValue), [1, 3, 480, 640])
        XCTAssertEqual(tensors.left.strides.map(\.intValue), [921_600, 307_200, 640, 1])
        XCTAssertEqual(tensors.right.strides.map(\.intValue), [921_600, 307_200, 640, 1])
        XCTAssertEqual(tensors.left.dataPointer.bindMemory(to: Float.self, capacity: count)[0], 1)
        XCTAssertEqual(tensors.right.dataPointer.bindMemory(to: Float.self, capacity: count)[0], 2)
    }

    func testStereoInputTensorsRejectWrongElementCount() {
        XCTAssertThrowsError(try StereoInputTensors(leftRGBNCHW: [0], rightRGBNCHW: [0])) { error in
            XCTAssertEqual(
                String(describing: error),
                "Invalid tensor shape: left expected 921600 float values, got 1"
            )
        }
    }

    func testStereoInputTensorsRejectWrongMLMultiArrayShape() throws {
        let left = try MLMultiArray(shape: [1, 3, 120, 160], dataType: .float32)
        let right = try MLMultiArray(shape: [1, 3, 480, 640], dataType: .float32)

        XCTAssertThrowsError(try StereoInputTensors(left: left, right: right)) { error in
            XCTAssertEqual(
                String(describing: error),
                "Invalid tensor shape: left expected shape [1, 3, 480, 640], got [1, 3, 120, 160]"
            )
        }
    }

    func testPackagedResourcesResolveBundledModelAndWeights() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FastFoundationStereoMPSResources-\(UUID().uuidString)")
        let modelDirectory = directory.appendingPathComponent("feature_projection.mlmodelc", isDirectory: true)
        let weightsDirectory = directory.appendingPathComponent("FastFoundationStereoWeights", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: weightsDirectory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: weightsDirectory.appendingPathComponent("manifest.json"))
        defer {
            try? FileManager.default.removeItem(at: directory)
        }

        let resources = try FastFoundationStereoMPSResources.resolve(in: directory)

        XCTAssertEqual(resources.featureModelURL, modelDirectory)
        XCTAssertEqual(resources.weightsDirectoryURL, weightsDirectory)
    }

    func testPackagedResourcesCanResolveMLPackage() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FastFoundationStereoMPSResources-\(UUID().uuidString)")
        let modelDirectory = directory.appendingPathComponent("feature_projection.mlpackage", isDirectory: true)
        let weightsDirectory = directory.appendingPathComponent("FastFoundationStereoWeights", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: weightsDirectory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: weightsDirectory.appendingPathComponent("manifest.json"))
        defer {
            try? FileManager.default.removeItem(at: directory)
        }

        let resources = try FastFoundationStereoMPSResources.resolve(in: directory)

        XCTAssertEqual(resources.featureModelURL, modelDirectory)
        XCTAssertEqual(resources.weightsDirectoryURL, weightsDirectory)
    }

    func testPackagedResourcesRequireWeightsManifest() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FastFoundationStereoMPSResources-\(UUID().uuidString)")
        let modelDirectory = directory.appendingPathComponent("feature_projection.mlmodelc", isDirectory: true)
        let weightsDirectory = directory.appendingPathComponent("FastFoundationStereoWeights", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: weightsDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: directory)
        }

        XCTAssertThrowsError(try FastFoundationStereoMPSResources.resolve(in: directory)) { error in
            XCTAssertEqual(
                String(describing: error),
                "Required FastFoundationStereoMPS resource is missing: FastFoundationStereoWeights/manifest.json."
            )
        }
    }

    func testMetalContextLoadsKernelsWhenDeviceIsAvailable() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is not available.")
        }
        let context = try MetalContext()
        _ = try MetalKernels(context: context)
    }

    func testRunnerInitializesCompiledGraphs() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is not available.")
        }
        let runner = try FastFoundationStereoMPSRunner()
        XCTAssertNotNil(runner.costRegularizationGraph.graph)
        XCTAssertNotNil(runner.costRegularizationGraph.outputs)
        XCTAssertNotNil(runner.updateStepGraph.graph)
        XCTAssertNotNil(runner.updateStepGraph.outputs)
    }

    func testCostRegularizationGraphDeclaresFixedShapeBoundary() throws {
        let graph = CostRegularizationGraph()
        try graph.compile()
        XCTAssertEqual(graph.inputSpecs.map(\.name), ["combined_volume", "left_04", "left_08", "left_16", "left_32"])
        XCTAssertEqual(graph.outputSpecs.map(\.name), ["regularized_volume", "logits"])
        XCTAssertEqual(graph.outputSpecs[0].shape, TensorShape([1, 48, 120, 160, 28], layout: .ndhwc))
        XCTAssertEqual(graph.outputSpecs[1].shape, TensorShape([1, 120, 160, 48], layout: .nhwc))
        XCTAssertTrue(graph.parameterSpecs.contains { $0.name == "corr_feature_att.layers.0.conv.weight" })
        XCTAssertTrue(graph.parameterSpecs.contains { $0.name == "cost_agg.post8_to_4.upsample.1.sa.0.self_attn.q_proj.weight" })
        XCTAssertTrue(graph.parameterSpecs.contains { $0.name == "classifier.layers.0.weight" })
    }

    func testCostRegularizationGraphSupportsFP16Precision() throws {
        let graph = CostRegularizationGraph(precision: .float16)
        try graph.compile()
        XCTAssertEqual(graph.precision, .float16)
        XCTAssertEqual(graph.inputSpecs.map(\.name), ["combined_volume", "left_04", "left_08", "left_16", "left_32"])
        XCTAssertEqual(graph.outputSpecs.map(\.name), ["regularized_volume", "logits"])
        XCTAssertEqual(graph.outputSpecs[0].shape, TensorShape([1, 48, 120, 160, 28], layout: .ndhwc))
        XCTAssertEqual(graph.outputSpecs[1].shape, TensorShape([1, 120, 160, 48], layout: .nhwc))
        XCTAssertTrue(graph.parameterSpecs.contains { $0.name == "corr_feature_att.layers.0.conv.weight" })
        XCTAssertTrue(graph.parameterSpecs.contains { $0.name == "classifier.layers.0.weight" })
    }

    func testCostRegularizationFP32GraphUsesFP32TensorTypes() throws {
        let graph = CostRegularizationGraph(precision: .float32)
        try graph.compile()

        XCTAssertEqual(graph.inputTensors["combined_volume"]?.dataType, .float32)
        XCTAssertEqual(graph.parameterTensors["classifier.layers.0.weight"]?.dataType, .float32)
        XCTAssertEqual(graph.outputs?.regularizedVolume.dataType, .float32)
        XCTAssertEqual(graph.outputs?.logits.dataType, .float32)
        XCTAssertEqual(graph.outputs?.debugTensors["cost.corr_feature_att"]?.dataType, .float32)
    }

    func testCostRegularizationFP16GraphUsesFP16InternalsWithFP32Boundary() throws {
        let graph = CostRegularizationGraph(precision: .float16)
        try graph.compile()

        XCTAssertEqual(graph.inputTensors["combined_volume"]?.dataType, .float32)
        XCTAssertEqual(graph.inputTensors["left_04"]?.dataType, .float32)
        XCTAssertEqual(graph.parameterTensors["classifier.layers.0.weight"]?.dataType, .float16)
        XCTAssertEqual(graph.parameterTensors["classifier.layers.0.bias"]?.dataType, .float16)
        XCTAssertEqual(graph.outputs?.regularizedVolume.dataType, .float32)
        XCTAssertEqual(graph.outputs?.logits.dataType, .float32)
        XCTAssertEqual(graph.outputs?.debugTensors["cost.corr_feature_att"]?.dataType, .float16)
    }

    func testNativeWeightsLoadsFP16FromFP32Export() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available.")
        }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FastFoundationStereoMPSTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: directory)
        }

        let manifest = """
        {
          "tensors": [
            {
              "name": "tiny.weight",
              "file": "tiny.bin",
              "shape": [2],
              "layout": "nhwc"
            }
          ]
        }
        """
        try manifest.data(using: .utf8)!.write(to: directory.appendingPathComponent("manifest.json"))
        try writeFloats([1.0, -2.0], to: directory.appendingPathComponent("tiny.bin"))

        let weights = try NativeWeights(directory: directory, device: device)
        let spec = NativeGraphTensorSpec(name: "tiny.weight", shape: TensorShape([2], layout: .nhwc))
        _ = try weights.tensorData(for: spec, precision: .float32)
        _ = try weights.tensorData(for: spec, precision: .float16)
    }

    func testUpdateStepGraphDeclaresFixedShapeBoundary() throws {
        let graph = UpdateStepGraph()
        try graph.compile()
        XCTAssertEqual(graph.inputSpecs.map(\.name), ["net_04", "inp_04", "geometry_lookup", "disparity", "attention_04", "stem_2x"])
        XCTAssertEqual(graph.outputSpecs.map(\.name), ["next_net_04", "mask_feat_4", "delta_disparity", "up_weights"])
        XCTAssertEqual(graph.inputSpecs[2].shape, TensorShape([1, 120, 160, 522], layout: .nhwc))
        XCTAssertEqual(graph.inputSpecs[0].shape, TensorShape([1, 120, 160, 60], layout: .nhwc))
        XCTAssertEqual(graph.inputSpecs[1].shape, TensorShape([1, 120, 160, 48], layout: .nhwc))
        XCTAssertEqual(graph.inputSpecs[5].shape, TensorShape([1, 240, 320, 16], layout: .nhwc))
        XCTAssertEqual(graph.outputSpecs[0].shape, TensorShape([1, 120, 160, 60], layout: .nhwc))
        XCTAssertEqual(graph.outputSpecs[1].shape, TensorShape([1, 120, 160, 16], layout: .nhwc))
        XCTAssertEqual(graph.outputSpecs[2].shape, TensorShape([1, 120, 160, 1], layout: .nhwc))
        XCTAssertEqual(graph.outputSpecs[3].shape, TensorShape([1, 480, 640, 9], layout: .nhwc))
        XCTAssertTrue(graph.parameterSpecs.contains { $0.name == "update_block.encoder.convc1.weight" })
        XCTAssertTrue(graph.parameterSpecs.contains { $0.name == "update_block.gru04.small_gru.convz.weight" })
        XCTAssertTrue(graph.parameterSpecs.contains { $0.name == "update_block.disp_head.conv.2.dwconv.weight" })
        XCTAssertTrue(graph.parameterSpecs.contains { $0.name == "spx_gru.0.weight" })
    }

    func testUpdateStepGraphDeclaresExportedParameterShapes() throws {
        let graph = UpdateStepGraph()
        try graph.compile()
        XCTAssertTrue(graph.parameterSpecs.contains {
            $0.name == "update_block.encoder.conv.weight"
                && $0.shape == TensorShape([3, 3, 108, 48], layout: .nhwc)
        })
        XCTAssertTrue(graph.parameterSpecs.contains {
            $0.name == "update_block.gru04.conv0.0.weight"
                && $0.shape == TensorShape([3, 3, 97, 100], layout: .nhwc)
        })
        XCTAssertTrue(graph.parameterSpecs.contains {
            $0.name == "spx_2_gru.conv2.conv.weight"
                && $0.shape == TensorShape([3, 3, 28, 24], layout: .nhwc)
        })
    }

    func testInitialDisparityUniformLogitsMatchesPyTorchSemantics() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is not available.")
        }
        let context = try MetalContext()
        let arena = TensorArena(device: context.device)
        let kernels = try MetalKernels(context: context)
        let logits = try arena.allocate(
            name: "logits",
            shape: TensorShape([1, 120, 160, 48], layout: .nhwc)
        )
        let disparity = try arena.allocate(
            name: "disparity",
            shape: TensorShape([1, 120, 160, 1], layout: .nhwc)
        )
        fill(logits, value: 0)
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        kernels.initialDisparity(logits: logits, output: disparity, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let values = floats(disparity)
        XCTAssertEqual(values[0], 23.5, accuracy: 1.0e-5)
        XCTAssertEqual(values[values.count - 1], 23.5, accuracy: 1.0e-5)
    }

    func testBuildCombinedVolumeSimpleValues() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is not available.")
        }
        let context = try MetalContext()
        let arena = TensorArena(device: context.device)
        let kernels = try MetalKernels(context: context)
        let projectedLeft = try arena.allocate(name: "projectedLeft", shape: TensorShape([1, 120, 160, 12], layout: .nhwc))
        let projectedRight = try arena.allocate(name: "projectedRight", shape: TensorShape([1, 120, 160, 12], layout: .nhwc))
        let left = try arena.allocate(name: "left", shape: TensorShape([1, 120, 160, 224], layout: .nhwc))
        let right = try arena.allocate(name: "right", shape: TensorShape([1, 120, 160, 224], layout: .nhwc))
        let output = try arena.allocate(name: "combined", shape: TensorShape([1, 48, 120, 160, 32], layout: .ndhwc))
        fill(projectedLeft, value: 2)
        fill(projectedRight, value: 3)
        fill(left, value: 1)
        fill(right, value: 1)
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        kernels.buildCombinedVolume(
            projectedLeft04: projectedLeft,
            projectedRight04: projectedRight,
            left04: left,
            right04: right,
            output: output,
            commandBuffer: commandBuffer
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let values = floats(output)
        XCTAssertEqual(values[combinedIndex(d: 0, y: 0, x: 0, c: 0)], 1, accuracy: 1.0e-5)
        XCTAssertEqual(values[combinedIndex(d: 0, y: 0, x: 0, c: 8)], 2, accuracy: 1.0e-5)
        XCTAssertEqual(values[combinedIndex(d: 0, y: 0, x: 0, c: 20)], 3, accuracy: 1.0e-5)
        XCTAssertEqual(values[combinedIndex(d: 1, y: 0, x: 0, c: 0)], 0, accuracy: 1.0e-5)
        XCTAssertEqual(values[combinedIndex(d: 1, y: 0, x: 0, c: 20)], 0, accuracy: 1.0e-5)
    }

    func testContextUpsampleCenterWeights() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is not available.")
        }
        let context = try MetalContext()
        let arena = TensorArena(device: context.device)
        let kernels = try MetalKernels(context: context)
        let disparity = try arena.allocate(name: "disp_low", shape: TensorShape([1, 120, 160, 1], layout: .nhwc))
        let weights = try arena.allocate(name: "weights", shape: TensorShape([1, 480, 640, 9], layout: .nhwc))
        let output = try arena.allocate(name: "disp_up", shape: TensorShape([1, 480, 640, 1], layout: .nhwc))
        fill(disparity, value: 7)
        fill(weights, value: 0)
        let weightValues = weights.buffer.contents().bindMemory(to: Float.self, capacity: weights.shape.elementCount)
        for pixel in 0..<(480 * 640) {
            weightValues[pixel * 9 + 4] = 1
        }
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        kernels.contextUpsample(disparityLow: disparity, upWeights: weights, output: output, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let values = floats(output)
        XCTAssertEqual(values[0], 7, accuracy: 1.0e-5)
        XCTAssertEqual(values[values.count - 1], 7, accuracy: 1.0e-5)
    }

    func testGeometryLookupSamplesVolumeAndCorrelationPyramids() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is not available.")
        }
        let context = try MetalContext()
        let arena = TensorArena(device: context.device)
        let kernels = try MetalKernels(context: context)
        let left = try arena.allocate(name: "left", shape: TensorShape([1, 120, 160, 224], layout: .nhwc))
        let right = try arena.allocate(name: "right", shape: TensorShape([1, 120, 160, 224], layout: .nhwc))
        let volume = try arena.allocate(name: "volume", shape: TensorShape([1, 48, 120, 160, 28], layout: .ndhwc))
        let disparity = try arena.allocate(name: "disp", shape: TensorShape([1, 120, 160, 1], layout: .nhwc))
        let output = try arena.allocate(name: "geometry", shape: TensorShape([1, 120, 160, 522], layout: .nhwc))
        fill(left, value: 1)
        fill(right, value: 1)
        fill(disparity, value: 4.5)
        let volumeValues = volume.buffer.contents().bindMemory(to: Float.self, capacity: volume.shape.elementCount)
        for d in 0..<48 {
            for index in 0..<(120 * 160 * 28) {
                volumeValues[d * 120 * 160 * 28 + index] = Float(d)
            }
        }
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        kernels.geometryLookup(
            left04: left,
            right04: right,
            regularizedVolume: volume,
            disparity: disparity,
            output: output,
            commandBuffer: commandBuffer
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let values = floats(output)
        XCTAssertEqual(values[geometryIndex(y: 10, x: 10, c: 4)], 4.5, accuracy: 1.0e-5)
        XCTAssertEqual(values[geometryIndex(y: 10, x: 10, c: 265)], 5.0, accuracy: 1.0e-5)
        XCTAssertEqual(values[geometryIndex(y: 10, x: 10, c: 256)], 1.0, accuracy: 1.0e-5)
        XCTAssertEqual(values[geometryIndex(y: 0, x: 0, c: 256)], 0.0, accuracy: 1.0e-5)
    }
}

private func fill(_ tensor: MetalTensor, value: Float) {
    let values = tensor.buffer.contents().bindMemory(to: Float.self, capacity: tensor.shape.elementCount)
    for index in 0..<tensor.shape.elementCount {
        values[index] = value
    }
}

private func floats(_ tensor: MetalTensor) -> [Float] {
    let values = tensor.buffer.contents().bindMemory(to: Float.self, capacity: tensor.shape.elementCount)
    return Array(UnsafeBufferPointer(start: values, count: tensor.shape.elementCount))
}

private func writeFloats(_ values: [Float], to url: URL) throws {
    let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
    try data.write(to: url)
}

private func combinedIndex(d: Int, y: Int, x: Int, c: Int) -> Int {
    (((d * 120) + y) * 160 + x) * 32 + c
}

private func geometryIndex(y: Int, x: Int, c: Int) -> Int {
    ((y * 160) + x) * 522 + c
}
