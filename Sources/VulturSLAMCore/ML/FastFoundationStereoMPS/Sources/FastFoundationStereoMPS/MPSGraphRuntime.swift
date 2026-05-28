import Foundation
import MetalPerformanceShadersGraph

public final class MPSGraphRuntime {
    private let commandQueue: MTLCommandQueue
    private let weights: NativeWeights

    public init(commandQueue: MTLCommandQueue, weights: NativeWeights) {
        self.commandQueue = commandQueue
        self.weights = weights
    }

    public func runCostRegularization(
        graph nativeGraph: CostRegularizationGraph,
        combinedVolume: MetalTensor,
        left04: MetalTensor,
        left08: MetalTensor,
        left16: MetalTensor,
        left32: MetalTensor,
        regularizedVolume: MetalTensor,
        logits: MetalTensor,
        debugDumper: DebugTensorDumper? = nil
    ) throws {
        guard let graph = nativeGraph.graph, let outputs = nativeGraph.outputs else {
            throw FastFoundationStereoMPSError.graphNotCompiled(nativeGraph.name)
        }
        let feeds = try feeds(
            inputTensors: nativeGraph.inputTensors,
            inputValues: [
                "combined_volume": combinedVolume,
                "left_04": left04,
                "left_08": left08,
                "left_16": left16,
                "left_32": left32,
            ],
            parameterTensors: nativeGraph.parameterTensors,
            parameterSpecs: nativeGraph.parameterSpecs,
            parameterPrecision: nativeGraph.precision
        )
        var debugOutputs: [(spec: NativeGraphTensorSpec, tensor: MetalTensor)] = []
        var resultTensors: [MPSGraphTensor: MetalTensor] = [
            outputs.regularizedVolume: regularizedVolume,
            outputs.logits: logits,
        ]
        var results: [MPSGraphTensor: MPSGraphTensorData] = [
            outputs.regularizedVolume: tensorData(regularizedVolume),
            outputs.logits: tensorData(logits),
        ]
        if debugDumper != nil {
            for spec in nativeGraph.debugOutputSpecs {
                guard let graphTensor = outputs.debugTensors[spec.name] else {
                    throw FastFoundationStereoMPSError.graphInputMissing(spec.name)
                }
                let tensor: MetalTensor
                if let existingTensor = resultTensors[graphTensor] {
                    tensor = existingTensor
                } else {
                    tensor = try allocateTensor(name: spec.name, shape: spec.shape)
                    results[graphTensor] = tensorData(tensor)
                    resultTensors[graphTensor] = tensor
                }
                debugOutputs.append((spec, tensor))
            }
        }
        graph.run(with: commandQueue, feeds: feeds, targetOperations: nil, resultsDictionary: results)
        for item in debugOutputs {
            try debugDumper?.dump(item.tensor, as: item.spec.name)
        }
    }

    public func runUpdateStep(
        graph nativeGraph: UpdateStepGraph,
        net: MetalTensor,
        input: MetalTensor,
        geometry: MetalTensor,
        disparity: MetalTensor,
        attention: MetalTensor,
        stem2x: MetalTensor,
        nextNet: MetalTensor,
        maskFeature: MetalTensor,
        deltaDisparity: MetalTensor,
        upsampleWeights: MetalTensor
    ) throws {
        guard let graph = nativeGraph.graph, let outputs = nativeGraph.outputs else {
            throw FastFoundationStereoMPSError.graphNotCompiled(nativeGraph.name)
        }
        let feeds = try feeds(
            inputTensors: nativeGraph.inputTensors,
            inputValues: [
                "net_04": net,
                "inp_04": input,
                "geometry_lookup": geometry,
                "disparity": disparity,
                "attention_04": attention,
                "stem_2x": stem2x,
            ],
            parameterTensors: nativeGraph.parameterTensors,
            parameterSpecs: nativeGraph.parameterSpecs,
            parameterPrecision: .float32
        )
        let results: [MPSGraphTensor: MPSGraphTensorData] = [
            outputs.nextNet: tensorData(nextNet),
            outputs.maskFeature: tensorData(maskFeature),
            outputs.deltaDisparity: tensorData(deltaDisparity),
            outputs.upsampleWeights: tensorData(upsampleWeights),
        ]
        graph.run(with: commandQueue, feeds: feeds, targetOperations: nil, resultsDictionary: results)
    }

    private func feeds(
        inputTensors: [String: MPSGraphTensor],
        inputValues: [String: MetalTensor],
        parameterTensors: [String: MPSGraphTensor],
        parameterSpecs: [NativeGraphTensorSpec],
        parameterPrecision: NativeGraphPrecision
    ) throws -> [MPSGraphTensor: MPSGraphTensorData] {
        var values: [MPSGraphTensor: MPSGraphTensorData] = [:]
        for (name, tensor) in inputValues {
            guard let graphTensor = inputTensors[name] else {
                throw FastFoundationStereoMPSError.graphInputMissing(name)
            }
            values[graphTensor] = tensorData(tensor)
        }
        for spec in parameterSpecs {
            guard let graphTensor = parameterTensors[spec.name] else {
                throw FastFoundationStereoMPSError.graphInputMissing(spec.name)
            }
            values[graphTensor] = try weights.tensorData(for: spec, precision: parameterPrecision)
        }
        return values
    }

    private func tensorData(_ tensor: MetalTensor) -> MPSGraphTensorData {
        MPSGraphTensorData(tensor.buffer, shape: mpsShape(tensor.shape), dataType: .float32)
    }

    private func allocateTensor(name: String, shape: TensorShape) throws -> MetalTensor {
        guard let buffer = commandQueue.device.makeBuffer(length: shape.byteCountFP32, options: [.storageModeShared]) else {
            throw FastFoundationStereoMPSError.allocationFailed(name)
        }
        return MetalTensor(name: name, shape: shape, buffer: buffer)
    }
}
