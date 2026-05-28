import Foundation
import CoreML
import Metal

public struct PreparedFeatureTensors {
    public let left04: MetalTensor
    public let left08: MetalTensor
    public let left16: MetalTensor
    public let left32: MetalTensor
    public let right04: MetalTensor
    public let projectedLeft04: MetalTensor
    public let projectedRight04: MetalTensor
    public let stem2x: MetalTensor
    public let netList: MetalTensor
    public let inpList: MetalTensor
    public let attention: MetalTensor
}

public final class FastFoundationStereoMPSRunner {
    public let context: MetalContext
    public let arena: TensorArena
    public let kernels: MetalKernels
    public let costRegularizationGraph: CostRegularizationGraph
    public let updateStepGraph: UpdateStepGraph

    public init(costRegularizationPrecision: NativeGraphPrecision = .float32) throws {
        self.context = try MetalContext()
        self.arena = TensorArena(device: context.device)
        self.kernels = try MetalKernels(context: context)
        let costRegularizationGraph = CostRegularizationGraph(precision: costRegularizationPrecision)
        try costRegularizationGraph.compile()
        self.costRegularizationGraph = costRegularizationGraph
        let updateStepGraph = UpdateStepGraph()
        try updateStepGraph.compile()
        self.updateStepGraph = updateStepGraph
    }

    public func prepareFeatureOutputs(_ outputs: FeatureProjectionOutputs) throws -> PreparedFeatureTensors {
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw FastFoundationStereoMPSError.commandQueueUnavailable
        }

        let left04 = try arena.allocate(name: "left_04_nhwc", shape: TensorShape([1, 120, 160, 224], layout: .nhwc))
        let left08 = try arena.allocate(name: "left_08_nhwc", shape: TensorShape([1, 60, 80, 192], layout: .nhwc))
        let left16 = try arena.allocate(name: "left_16_nhwc", shape: TensorShape([1, 30, 40, 320], layout: .nhwc))
        let left32 = try arena.allocate(name: "left_32_nhwc", shape: TensorShape([1, 15, 20, 304], layout: .nhwc))
        let right04 = try arena.allocate(name: "right_04_nhwc", shape: TensorShape([1, 120, 160, 224], layout: .nhwc))
        let projectedLeft04 = try arena.allocate(name: "proj_left_04_nhwc", shape: TensorShape([1, 120, 160, 12], layout: .nhwc))
        let projectedRight04 = try arena.allocate(name: "proj_right_04_nhwc", shape: TensorShape([1, 120, 160, 12], layout: .nhwc))
        let stem2x = try arena.allocate(name: "stem_2x_nhwc", shape: TensorShape([1, 240, 320, 16], layout: .nhwc))
        let netList = try arena.allocate(name: "net_list_nhwc", shape: TensorShape([1, 120, 160, 60], layout: .nhwc))
        let inpList = try arena.allocate(name: "inp_list_nhwc", shape: TensorShape([1, 120, 160, 48], layout: .nhwc))
        let attention = try arena.allocate(name: "att_nhwc", shape: TensorShape([1, 120, 160, 1], layout: .nhwc))

        kernels.convertNCHWToNHWC(source: outputs.left04, destination: left04, commandBuffer: commandBuffer)
        kernels.convertNCHWToNHWC(source: outputs.left08, destination: left08, commandBuffer: commandBuffer)
        kernels.convertNCHWToNHWC(source: outputs.left16, destination: left16, commandBuffer: commandBuffer)
        kernels.convertNCHWToNHWC(source: outputs.left32, destination: left32, commandBuffer: commandBuffer)
        kernels.convertNCHWToNHWC(source: outputs.right04, destination: right04, commandBuffer: commandBuffer)
        kernels.convertNCHWToNHWC(source: outputs.projectedLeft04, destination: projectedLeft04, commandBuffer: commandBuffer)
        kernels.convertNCHWToNHWC(source: outputs.projectedRight04, destination: projectedRight04, commandBuffer: commandBuffer)
        kernels.convertNCHWToNHWC(source: outputs.stem2x, destination: stem2x, commandBuffer: commandBuffer)
        kernels.convertNCHWToNHWC(source: outputs.netList, destination: netList, commandBuffer: commandBuffer)
        kernels.convertNCHWToNHWC(source: outputs.inpList, destination: inpList, commandBuffer: commandBuffer)
        kernels.convertNCHWToNHWC(source: outputs.attention, destination: attention, commandBuffer: commandBuffer)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return PreparedFeatureTensors(
            left04: left04,
            left08: left08,
            left16: left16,
            left32: left32,
            right04: right04,
            projectedLeft04: projectedLeft04,
            projectedRight04: projectedRight04,
            stem2x: stem2x,
            netList: netList,
            inpList: inpList,
            attention: attention
        )
    }

    public func buildCombinedVolume(from tensors: PreparedFeatureTensors) throws -> MetalTensor {
        let output = try arena.allocate(
            name: "combined_volume_ndhwc",
            shape: TensorShape([1, 48, 120, 160, 32], layout: .ndhwc)
        )
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw FastFoundationStereoMPSError.commandQueueUnavailable
        }
        kernels.buildCombinedVolume(
            projectedLeft04: tensors.projectedLeft04,
            projectedRight04: tensors.projectedRight04,
            left04: tensors.left04,
            right04: tensors.right04,
            output: output,
            commandBuffer: commandBuffer
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return output
    }

    public func costRegularization(
        tensors: PreparedFeatureTensors,
        combinedVolume: MetalTensor,
        weights: NativeWeights,
        debugDumper: DebugTensorDumper? = nil
    ) throws -> (regularizedVolume: MetalTensor, logits: MetalTensor) {
        let regularizedVolume = try arena.allocate(
            name: "regularized_volume_ndhwc",
            shape: TensorShape([1, 48, 120, 160, 28], layout: .ndhwc)
        )
        let logits = try arena.allocate(
            name: "logits_nhwc",
            shape: TensorShape([1, 120, 160, 48], layout: .nhwc)
        )
        let runtime = MPSGraphRuntime(commandQueue: context.commandQueue, weights: weights)
        try runtime.runCostRegularization(
            graph: costRegularizationGraph,
            combinedVolume: combinedVolume,
            left04: tensors.left04,
            left08: tensors.left08,
            left16: tensors.left16,
            left32: tensors.left32,
            regularizedVolume: regularizedVolume,
            logits: logits,
            debugDumper: debugDumper
        )
        return (regularizedVolume, logits)
    }

    public func initialDisparity(logits: MetalTensor, name: String = "disparity_initial_nhwc") throws -> MetalTensor {
        let output = try arena.allocate(
            name: name,
            shape: TensorShape([1, 120, 160, 1], layout: .nhwc)
        )
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw FastFoundationStereoMPSError.commandQueueUnavailable
        }
        kernels.initialDisparity(logits: logits, output: output, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return output
    }

    public func geometryLookup(
        left04: MetalTensor,
        right04: MetalTensor,
        regularizedVolume: MetalTensor,
        disparity: MetalTensor
    ) throws -> MetalTensor {
        let outputChannels = FastFoundationStereoMPSConstants.correlationLevels
            * (FastFoundationStereoMPSConstants.regularizedVolumeChannels + 1)
            * (FastFoundationStereoMPSConstants.correlationRadius * 2 + 1)
        let output = try arena.allocate(
            name: "geometry_lookup_nhwc",
            shape: TensorShape([1, 120, 160, outputChannels], layout: .nhwc)
        )
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw FastFoundationStereoMPSError.commandQueueUnavailable
        }
        kernels.geometryLookup(
            left04: left04,
            right04: right04,
            regularizedVolume: regularizedVolume,
            disparity: disparity,
            output: output,
            commandBuffer: commandBuffer
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return output
    }

    public func refinementStep(
        weights: NativeWeights,
        net: MetalTensor,
        input: MetalTensor,
        geometry: MetalTensor,
        disparity: MetalTensor,
        attention: MetalTensor,
        stem2x: MetalTensor,
        nextNet: MetalTensor,
        nextDisparity: MetalTensor,
        deltaDisparity: MetalTensor,
        maskFeature: MetalTensor,
        upsampleWeights: MetalTensor
    ) throws {
        let runtime = MPSGraphRuntime(commandQueue: context.commandQueue, weights: weights)
        try runtime.runUpdateStep(
            graph: updateStepGraph,
            net: net,
            input: input,
            geometry: geometry,
            disparity: disparity,
            attention: attention,
            stem2x: stem2x,
            nextNet: nextNet,
            maskFeature: maskFeature,
            deltaDisparity: deltaDisparity,
            upsampleWeights: upsampleWeights
        )
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw FastFoundationStereoMPSError.commandQueueUnavailable
        }
        kernels.add(left: disparity, right: deltaDisparity, output: nextDisparity, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    public func contextUpsample(disparity: MetalTensor, upsampleWeights: MetalTensor) throws -> MetalTensor {
        let scaledDisparity = try arena.allocate(
            name: "disparity_quarter_scaled_nhwc",
            shape: TensorShape([1, 120, 160, 1], layout: .nhwc)
        )
        let output = try arena.allocate(
            name: "disparity_full_nhwc",
            shape: TensorShape([1, 480, 640, 1], layout: .nhwc)
        )
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw FastFoundationStereoMPSError.commandQueueUnavailable
        }
        kernels.scale(source: disparity, output: scaledDisparity, factor: 4.0, commandBuffer: commandBuffer)
        kernels.contextUpsample(disparityLow: scaledDisparity, upWeights: upsampleWeights, output: output, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return output
    }

    public func run(
        inputTensors: StereoInputTensors,
        featureRunner: FeatureProjectionRunner,
        weights: NativeWeights,
        validIterations: Int,
        debugDumper: DebugTensorDumper? = nil
    ) throws -> MetalTensor {
        try run(
            leftImage: inputTensors.left,
            rightImage: inputTensors.right,
            featureRunner: featureRunner,
            weights: weights,
            validIterations: validIterations,
            debugDumper: debugDumper
        )
    }

    public func run(
        leftImage: MLMultiArray,
        rightImage: MLMultiArray,
        featureRunner: FeatureProjectionRunner,
        weights: NativeWeights,
        validIterations: Int,
        debugDumper: DebugTensorDumper? = nil
    ) throws -> MetalTensor {
        try StereoInputTensors.validateImageTensor(leftImage, name: "leftImage")
        try StereoInputTensors.validateImageTensor(rightImage, name: "rightImage")
        var inferenceDuration = 0.0
        func measure<T>(_ name: String, _ body: () throws -> T) throws -> T {
            let start = CFAbsoluteTimeGetCurrent()
            defer {
                let duration = CFAbsoluteTimeGetCurrent() - start
                inferenceDuration += duration
                print(String(format: "FastFoundationStereoMPS \(name): %.2f ms", duration * 1000.0))
            }
            return try body()
        }
        func dump(_ tensor: MetalTensor, as name: String) throws {
            guard let debugDumper else {
                return
            }
            try debugDumper.dump(tensor, as: name)
        }
        func dumpFloatArray(pointer: UnsafeRawPointer, shape: TensorShape, as name: String) throws {
            guard let debugDumper else {
                return
            }
            try debugDumper.dumpFloatArray(pointer: pointer, shape: shape, as: name)
        }

        let imageShape = StereoInputTensors.imageShape
        try dumpFloatArray(pointer: leftImage.dataPointer, shape: imageShape, as: "input.left_image")
        try dumpFloatArray(pointer: rightImage.dataPointer, shape: imageShape, as: "input.right_image")
        let featureOutputs = try measure("feature projection") {
            try featureRunner.predict(leftImage: leftImage, rightImage: rightImage)
        }
        let tensors = try measure("prepare features") {
            try prepareFeatureOutputs(featureOutputs)
        }
        try dump(tensors.left04, as: "feature.left04")
        try dump(tensors.left08, as: "feature.left08")
        try dump(tensors.left16, as: "feature.left16")
        try dump(tensors.left32, as: "feature.left32")
        try dump(tensors.right04, as: "feature.right04")
        try dump(tensors.projectedLeft04, as: "feature.projectedLeft04")
        try dump(tensors.projectedRight04, as: "feature.projectedRight04")
        try dump(tensors.stem2x, as: "feature.stem2x")
        try dump(tensors.netList, as: "feature.netList")
        try dump(tensors.inpList, as: "feature.inpList")
        try dump(tensors.attention, as: "feature.attention")
        let combinedVolume = try measure("combined volume") {
            try buildCombinedVolume(from: tensors)
        }
        try dump(combinedVolume, as: "combined_volume")
        let costOutputs = try measure("cost regularization") {
            try costRegularization(tensors: tensors, combinedVolume: combinedVolume, weights: weights)
        }
        try dump(costOutputs.regularizedVolume, as: "regularized_volume")
        try dump(costOutputs.logits, as: "logits")
        let initial = try measure("initial disparity") {
            try initialDisparity(logits: costOutputs.logits)
        }
        try dump(initial, as: "initial_disparity")

        var currentNet = tensors.netList
        var currentDisparity = initial
        let refinementBuffers = try measure("allocate refinement buffers") {
            (
                try arena.allocate(name: "next_net_04_a", shape: TensorShape([1, 120, 160, 60], layout: .nhwc)),
                try arena.allocate(name: "next_net_04_b", shape: TensorShape([1, 120, 160, 60], layout: .nhwc)),
                try arena.allocate(name: "next_disparity_a", shape: TensorShape([1, 120, 160, 1], layout: .nhwc)),
                try arena.allocate(name: "next_disparity_b", shape: TensorShape([1, 120, 160, 1], layout: .nhwc)),
                try arena.allocate(name: "delta_disparity_nhwc", shape: TensorShape([1, 120, 160, 1], layout: .nhwc)),
                try arena.allocate(name: "mask_feat_4_nhwc", shape: TensorShape([1, 120, 160, 16], layout: .nhwc)),
                try arena.allocate(name: "up_weights_nhwc", shape: TensorShape([1, 480, 640, 9], layout: .nhwc))
            )
        }
        let nextNetA = refinementBuffers.0
        let nextNetB = refinementBuffers.1
        let nextDisparityA = refinementBuffers.2
        let nextDisparityB = refinementBuffers.3
        let deltaDisparity = refinementBuffers.4
        let maskFeature = refinementBuffers.5
        let upsampleWeights = refinementBuffers.6

        for iteration in 0..<validIterations {
            let geometry = try measure("iteration \(iteration) geometry lookup") {
                try geometryLookup(
                    left04: tensors.left04,
                    right04: tensors.right04,
                    regularizedVolume: costOutputs.regularizedVolume,
                    disparity: currentDisparity
                )
            }
            try dump(geometry, as: "iteration_\(iteration).geometry_lookup")
            let nextNet = iteration.isMultiple(of: 2) ? nextNetA : nextNetB
            let nextDisparity = iteration.isMultiple(of: 2) ? nextDisparityA : nextDisparityB
            try measure("iteration \(iteration) refinement") {
                try refinementStep(
                    weights: weights,
                    net: currentNet,
                    input: tensors.inpList,
                    geometry: geometry,
                    disparity: currentDisparity,
                    attention: tensors.attention,
                    stem2x: tensors.stem2x,
                    nextNet: nextNet,
                    nextDisparity: nextDisparity,
                    deltaDisparity: deltaDisparity,
                    maskFeature: maskFeature,
                    upsampleWeights: upsampleWeights
                )
            }
            try dump(deltaDisparity, as: "iteration_\(iteration).delta_disparity")
            try dump(nextNet, as: "iteration_\(iteration).next_net")
            try dump(maskFeature, as: "iteration_\(iteration).mask_feat_4")
            try dump(upsampleWeights, as: "iteration_\(iteration).up_weights")
            try dump(nextDisparity, as: "iteration_\(iteration).quarter_disparity")
            currentNet = nextNet
            currentDisparity = nextDisparity
        }

        let output = try measure("context upsample") {
            try contextUpsample(disparity: currentDisparity, upsampleWeights: upsampleWeights)
        }
        try dump(output, as: "final_disparity")
        print(String(format: "FastFoundationStereoMPS inference time: %.2f ms", inferenceDuration * 1000.0))
        return output
    }
}
