import CoreML
import Foundation
import Metal

public struct FeatureProjectionOutputs {
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

public final class FeatureProjectionRunner {
    private let model: MLModel
    private let arena: TensorArena

    public init(modelURL: URL, arena: TensorArena) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        if modelURL.pathExtension == "mlpackage" {
            let compiledURL = try MLModel.compileModel(at: modelURL)
            self.model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        } else {
            self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
        }
        self.arena = arena
    }

    public func predict(inputTensors: StereoInputTensors) throws -> FeatureProjectionOutputs {
        try predict(leftImage: inputTensors.left, rightImage: inputTensors.right)
    }

    public func predict(leftImage: MLMultiArray, rightImage: MLMultiArray) throws -> FeatureProjectionOutputs {
        try StereoInputTensors.validateImageTensor(leftImage, name: "leftImage")
        try StereoInputTensors.validateImageTensor(rightImage, name: "rightImage")
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "left_image": MLFeatureValue(multiArray: leftImage),
            "right_image": MLFeatureValue(multiArray: rightImage),
        ])
        let outputs = try model.prediction(from: provider)
        return try copyOutputs(outputs)
    }

    private func copyOutputs(_ outputs: MLFeatureProvider) throws -> FeatureProjectionOutputs {
        let left04 = try copyOutput(outputs, name: "left_04", shape: TensorShape([1, 224, 120, 160], layout: .nchw))
        let left08 = try copyOutput(outputs, name: "left_08", shape: TensorShape([1, 192, 60, 80], layout: .nchw))
        let left16 = try copyOutput(outputs, name: "left_16", shape: TensorShape([1, 320, 30, 40], layout: .nchw))
        let left32 = try copyOutput(outputs, name: "left_32", shape: TensorShape([1, 304, 15, 20], layout: .nchw))
        let right04 = try copyOutput(outputs, name: "right_04", shape: TensorShape([1, 224, 120, 160], layout: .nchw))
        let projectedLeft04 = try copyOutput(outputs, name: "proj_left_04", shape: TensorShape([1, 12, 120, 160], layout: .nchw))
        let projectedRight04 = try copyOutput(outputs, name: "proj_right_04", shape: TensorShape([1, 12, 120, 160], layout: .nchw))
        let stem2x = try copyOutput(outputs, name: "stem_2x", shape: TensorShape([1, 16, 240, 320], layout: .nchw))
        let netList = try copyOutput(outputs, name: "net_list", shape: TensorShape([1, 60, 120, 160], layout: .nchw))
        let inpList = try copyOutput(outputs, name: "inp_list", shape: TensorShape([1, 48, 120, 160], layout: .nchw))
        let attention = try copyOutput(outputs, name: "att", shape: TensorShape([1, 1, 120, 160], layout: .nchw))
        return FeatureProjectionOutputs(
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

    private func copyOutput(_ outputs: MLFeatureProvider, name: String, shape: TensorShape) throws -> MetalTensor {
        guard let multiArray = outputs.featureValue(for: name)?.multiArrayValue else {
            throw FastFoundationStereoMPSError.coreMLOutputMissing(name)
        }
        guard multiArray.count == shape.elementCount else {
            throw FastFoundationStereoMPSError.invalidShape("\(name) expected \(shape.elementCount), got \(multiArray.count)")
        }
        let tensor = try arena.allocate(name: name, shape: shape)
        let byteCount = shape.byteCountFP32
        memcpy(tensor.buffer.contents(), multiArray.dataPointer, byteCount)
        return tensor
    }
}
