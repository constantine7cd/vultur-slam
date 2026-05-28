import Foundation
import MetalPerformanceShadersGraph

public protocol NativeGraph {
    var name: String { get }
    func compile() throws
}

public struct NativeGraphTensorSpec: Equatable, Sendable {
    public let name: String
    public let shape: TensorShape

    public init(name: String, shape: TensorShape) {
        self.name = name
        self.shape = shape
    }
}

public enum NativeGraphPrecision: String, Sendable {
    case float32 = "fp32"
    case float16 = "fp16"

    var dataType: MPSDataType {
        switch self {
        case .float32:
            return .float32
        case .float16:
            return .float16
        }
    }
}

public struct CostRegularizationGraphOutputs {
    public let regularizedVolume: MPSGraphTensor
    public let logits: MPSGraphTensor
    public let debugTensors: [String: MPSGraphTensor]
}

public struct UpdateStepGraphOutputs {
    public let nextNet: MPSGraphTensor
    public let maskFeature: MPSGraphTensor
    public let deltaDisparity: MPSGraphTensor
    public let upsampleWeights: MPSGraphTensor
}

public final class CostRegularizationGraph: NativeGraph {
    public let name = "CostRegularizationGraph"
    public let precision: NativeGraphPrecision
    public private(set) var parameterSpecs: [NativeGraphTensorSpec] = []
    public private(set) var inputSpecs: [NativeGraphTensorSpec] = []
    public private(set) var outputSpecs: [NativeGraphTensorSpec] = []
    public private(set) var debugOutputSpecs: [NativeGraphTensorSpec] = []
    public private(set) var graph: MPSGraph?
    public private(set) var outputs: CostRegularizationGraphOutputs?
    public private(set) var inputTensors: [String: MPSGraphTensor] = [:]
    public private(set) var parameterTensors: [String: MPSGraphTensor] = [:]

    public init(precision: NativeGraphPrecision = .float32) {
        self.precision = precision
    }

    public func compile() throws {
        let builder = CostRegularizationGraphBuilder(precision: precision)
        let built = builder.build()
        graph = built.graph
        outputs = built.outputs
        inputSpecs = built.inputSpecs
        outputSpecs = built.outputSpecs
        debugOutputSpecs = built.debugOutputSpecs
        parameterSpecs = built.parameterSpecs
        inputTensors = built.inputTensors
        parameterTensors = built.parameterTensors
    }
}

public final class UpdateStepGraph: NativeGraph {
    public let name = "UpdateStepGraph"
    public private(set) var parameterSpecs: [NativeGraphTensorSpec] = []
    public private(set) var inputSpecs: [NativeGraphTensorSpec] = []
    public private(set) var outputSpecs: [NativeGraphTensorSpec] = []
    public private(set) var graph: MPSGraph?
    public private(set) var outputs: UpdateStepGraphOutputs?
    public private(set) var inputTensors: [String: MPSGraphTensor] = [:]
    public private(set) var parameterTensors: [String: MPSGraphTensor] = [:]

    public init() {}

    public func compile() throws {
        let builder = UpdateStepGraphBuilder()
        let built = builder.build()
        graph = built.graph
        outputs = built.outputs
        inputSpecs = built.inputSpecs
        outputSpecs = built.outputSpecs
        parameterSpecs = built.parameterSpecs
        inputTensors = built.inputTensors
        parameterTensors = built.parameterTensors
    }
}

private struct BuiltUpdateStepGraph {
    let graph: MPSGraph
    let outputs: UpdateStepGraphOutputs
    let inputSpecs: [NativeGraphTensorSpec]
    let outputSpecs: [NativeGraphTensorSpec]
    let parameterSpecs: [NativeGraphTensorSpec]
    let inputTensors: [String: MPSGraphTensor]
    let parameterTensors: [String: MPSGraphTensor]
}

private final class UpdateStepGraphBuilder {
    private let graph = MPSGraph()
    private var parameterSpecs: [NativeGraphTensorSpec] = []
    private var inputTensors: [String: MPSGraphTensor] = [:]
    private var parameterTensors: [String: MPSGraphTensor] = [:]
    private let hiddenChannels = 60
    private let inputChannels = 48
    private let stemChannels = 16
    private let motionCorrelationChannels = 56
    private let motionCorrelationHiddenChannels = 96
    private let motionDisparityChannels = 16
    private let motionDisparityHiddenChannels = 12
    private let motionOutputChannels = 48
    private let gruInputChannels = 97
    private let gruConvChannels = 100
    private let gruHXChannels = 168
    private let disparityHeadChannels = 36
    private let maskFeatureChannels = 16
    private let spxChannels = 12
    private let spxMergedChannels = 28
    private let spxOutputChannels = 24
    private let geometryChannels = FastFoundationStereoMPSConstants.correlationLevels
        * (FastFoundationStereoMPSConstants.regularizedVolumeChannels + 1)
        * (FastFoundationStereoMPSConstants.correlationRadius * 2 + 1)
    private let height = FastFoundationStereoMPSConstants.quarterHeight
    private let width = FastFoundationStereoMPSConstants.quarterWidth
    private let halfHeight = FastFoundationStereoMPSConstants.halfHeight
    private let halfWidth = FastFoundationStereoMPSConstants.halfWidth
    private let imageHeight = FastFoundationStereoMPSConstants.imageHeight
    private let imageWidth = FastFoundationStereoMPSConstants.imageWidth

    func build() -> BuiltUpdateStepGraph {
        let net = placeholder("net_04", [1, height, width, hiddenChannels], layout: .nhwc)
        let input = placeholder("inp_04", [1, height, width, inputChannels], layout: .nhwc)
        let geometry = placeholder("geometry_lookup", [1, height, width, geometryChannels], layout: .nhwc)
        let disparity = placeholder("disparity", [1, height, width, 1], layout: .nhwc)
        let attention = placeholder("attention_04", [1, height, width, 1], layout: .nhwc)
        let stem2x = placeholder("stem_2x", [1, halfHeight, halfWidth, stemChannels], layout: .nhwc)

        let motion = motionEncoder(disparity: disparity, geometry: geometry)
        let gruInput = graph.concatTensors([input, motion], dimension: 3, name: "update_block.motion_input")
        let nextNet = selectiveGRU(attention: attention, hidden: net, input: gruInput)
        let deltaDisparity = disparityHead(nextNet)
        let maskFeature = graph.multiplication(
            graph.constant(0.25, dataType: .float32),
            mask(nextNet),
            name: "update_block.mask.scale"
        )
        let upsampleWeights = spxWeights(maskFeature: maskFeature, stem2x: stem2x)

        let inputs = [
            NativeGraphTensorSpec(name: "net_04", shape: TensorShape([1, height, width, hiddenChannels], layout: .nhwc)),
            NativeGraphTensorSpec(name: "inp_04", shape: TensorShape([1, height, width, inputChannels], layout: .nhwc)),
            NativeGraphTensorSpec(name: "geometry_lookup", shape: TensorShape([1, height, width, geometryChannels], layout: .nhwc)),
            NativeGraphTensorSpec(name: "disparity", shape: TensorShape([1, height, width, 1], layout: .nhwc)),
            NativeGraphTensorSpec(name: "attention_04", shape: TensorShape([1, height, width, 1], layout: .nhwc)),
            NativeGraphTensorSpec(name: "stem_2x", shape: TensorShape([1, halfHeight, halfWidth, stemChannels], layout: .nhwc)),
        ]
        let outputs = [
            NativeGraphTensorSpec(name: "next_net_04", shape: TensorShape([1, height, width, hiddenChannels], layout: .nhwc)),
            NativeGraphTensorSpec(name: "mask_feat_4", shape: TensorShape([1, height, width, maskFeatureChannels], layout: .nhwc)),
            NativeGraphTensorSpec(name: "delta_disparity", shape: TensorShape([1, height, width, 1], layout: .nhwc)),
            NativeGraphTensorSpec(name: "up_weights", shape: TensorShape([1, imageHeight, imageWidth, 9], layout: .nhwc)),
        ]
        return BuiltUpdateStepGraph(
            graph: graph,
            outputs: UpdateStepGraphOutputs(nextNet: nextNet, maskFeature: maskFeature, deltaDisparity: deltaDisparity, upsampleWeights: upsampleWeights),
            inputSpecs: inputs,
            outputSpecs: outputs,
            parameterSpecs: parameterSpecs,
            inputTensors: inputTensors,
            parameterTensors: parameterTensors
        )
    }

    private func motionEncoder(disparity: MPSGraphTensor, geometry: MPSGraphTensor) -> MPSGraphTensor {
        var correlation = conv2D(geometry, weightName: "update_block.encoder.convc1.weight", biasName: "update_block.encoder.convc1.bias", inChannels: geometryChannels, outChannels: motionCorrelationChannels, kernel: (1, 1), padding: (0, 0), activation: .relu)
        correlation = conv2D(correlation, weightName: "update_block.encoder.convc2.weight", biasName: "update_block.encoder.convc2.bias", inChannels: motionCorrelationChannels, outChannels: motionCorrelationHiddenChannels, kernel: (3, 3), padding: (1, 1), activation: .relu)
        var disp = conv2D(disparity, weightName: "update_block.encoder.convd1.weight", biasName: "update_block.encoder.convd1.bias", inChannels: 1, outChannels: motionDisparityChannels, kernel: (7, 7), padding: (3, 3), activation: .relu)
        disp = conv2D(disp, weightName: "update_block.encoder.convd2.weight", biasName: "update_block.encoder.convd2.bias", inChannels: motionDisparityChannels, outChannels: motionDisparityHiddenChannels, kernel: (3, 3), padding: (1, 1), activation: .relu)
        let combined = graph.concatTensors([correlation, disp], dimension: 3, name: "update_block.encoder.concat")
        let encoded = conv2D(combined, weightName: "update_block.encoder.conv.weight", biasName: "update_block.encoder.conv.bias", inChannels: motionCorrelationHiddenChannels + motionDisparityHiddenChannels, outChannels: motionOutputChannels, kernel: (3, 3), padding: (1, 1), activation: .relu)
        return graph.concatTensors([encoded, disparity], dimension: 3, name: "update_block.encoder.motion_with_disparity")
    }

    private func selectiveGRU(attention: MPSGraphTensor, hidden: MPSGraphTensor, input: MPSGraphTensor) -> MPSGraphTensor {
        let x = conv2D(input, weightName: "update_block.gru04.conv0.0.weight", biasName: "update_block.gru04.conv0.0.bias", inChannels: gruInputChannels, outChannels: gruConvChannels, kernel: (3, 3), padding: (1, 1), activation: .relu)
        let hxInput = graph.concatTensors([x, hidden], dimension: 3, name: "update_block.gru04.hx_input")
        let hx = conv2D(hxInput, weightName: "update_block.gru04.conv1.0.weight", biasName: "update_block.gru04.conv1.0.bias", inChannels: gruConvChannels + hiddenChannels, outChannels: gruHXChannels, kernel: (3, 3), padding: (1, 1), activation: .relu)
        let small = raftGRU(prefix: "update_block.gru04.small_gru", hidden: hidden, input: x, hx: hx, kernel: (1, 1), padding: (0, 0))
        let large = raftGRU(prefix: "update_block.gru04.large_gru", hidden: hidden, input: x, hx: hx, kernel: (3, 3), padding: (1, 1))
        let inverseAttention = graph.subtraction(graph.constant(1.0, dataType: .float32), attention, name: "update_block.gru04.inverse_attention")
        let selectedSmall = graph.multiplication(small, attention, name: "update_block.gru04.select_small")
        let selectedLarge = graph.multiplication(large, inverseAttention, name: "update_block.gru04.select_large")
        return graph.addition(selectedSmall, selectedLarge, name: "update_block.gru04.next_hidden")
    }

    private func raftGRU(prefix: String, hidden: MPSGraphTensor, input: MPSGraphTensor, hx: MPSGraphTensor, kernel: (Int, Int), padding: (Int, Int)) -> MPSGraphTensor {
        let z = graph.sigmoid(with: conv2D(hx, weightName: "\(prefix).convz.weight", biasName: "\(prefix).convz.bias", inChannels: gruHXChannels, outChannels: hiddenChannels, kernel: kernel, padding: padding, activation: nil), name: "\(prefix).z")
        let r = graph.sigmoid(with: conv2D(hx, weightName: "\(prefix).convr.weight", biasName: "\(prefix).convr.bias", inChannels: gruHXChannels, outChannels: hiddenChannels, kernel: kernel, padding: padding, activation: nil), name: "\(prefix).r")
        let resetHidden = graph.multiplication(r, hidden, name: "\(prefix).reset_hidden")
        let qInput = graph.concatTensors([resetHidden, input], dimension: 3, name: "\(prefix).q_input")
        let q = graph.tanh(with: conv2D(qInput, weightName: "\(prefix).convq.weight", biasName: "\(prefix).convq.bias", inChannels: hiddenChannels + gruConvChannels, outChannels: hiddenChannels, kernel: kernel, padding: padding, activation: nil), name: "\(prefix).q")
        let keep = graph.multiplication(graph.subtraction(graph.constant(1.0, dataType: .float32), z, name: "\(prefix).one_minus_z"), hidden, name: "\(prefix).keep_hidden")
        let update = graph.multiplication(z, q, name: "\(prefix).update_hidden")
        return graph.addition(keep, update, name: "\(prefix).hidden")
    }

    private func disparityHead(_ input: MPSGraphTensor) -> MPSGraphTensor {
        var x = conv2D(input, weightName: "update_block.disp_head.conv.0.weight", biasName: "update_block.disp_head.conv.0.bias", inChannels: hiddenChannels, outChannels: disparityHeadChannels, kernel: (3, 3), padding: (1, 1), activation: .relu)
        x = edgeNext(x, prefix: "update_block.disp_head.conv.2", channels: disparityHeadChannels, expandedChannels: 212)
        x = edgeNext(x, prefix: "update_block.disp_head.conv.3", channels: disparityHeadChannels, expandedChannels: 244)
        return conv2D(x, weightName: "update_block.disp_head.conv.4.weight", biasName: "update_block.disp_head.conv.4.bias", inChannels: disparityHeadChannels, outChannels: 1, kernel: (3, 3), padding: (1, 1), activation: nil)
    }

    private func edgeNext(_ input: MPSGraphTensor, prefix: String, channels: Int, expandedChannels: Int) -> MPSGraphTensor {
        var x = conv2D(input, weightName: "\(prefix).dwconv.weight", biasName: "\(prefix).dwconv.bias", inChannels: channels, outChannels: channels, kernel: (7, 7), padding: (3, 3), groups: channels, activation: nil)
        x = linear(x, weightName: "\(prefix).pwconv1.weight", biasName: "\(prefix).pwconv1.bias", inChannels: channels, outChannels: expandedChannels)
        x = gelu(x, name: "\(prefix).gelu")
        x = linear(x, weightName: "\(prefix).pwconv2.weight", biasName: "\(prefix).pwconv2.bias", inChannels: expandedChannels, outChannels: channels)
        x = graph.multiplication(x, parameter("\(prefix).gamma", [1, 1, 1, channels], layout: .nhwc), name: "\(prefix).gamma_scale")
        return graph.addition(input, x, name: "\(prefix).residual")
    }

    private func mask(_ input: MPSGraphTensor) -> MPSGraphTensor {
        var x = conv2D(input, weightName: "update_block.mask.0.weight", biasName: "update_block.mask.0.bias", inChannels: hiddenChannels, outChannels: 32, kernel: (3, 3), padding: (1, 1), activation: .relu)
        x = conv2D(x, weightName: "update_block.mask.2.weight", biasName: "update_block.mask.2.bias", inChannels: 32, outChannels: maskFeatureChannels, kernel: (3, 3), padding: (1, 1), activation: .relu)
        return x
    }

    private func spxWeights(maskFeature: MPSGraphTensor, stem2x: MPSGraphTensor) -> MPSGraphTensor {
        var x = deconv2D(maskFeature, weightName: "spx_2_gru.conv1.conv.weight", biasName: nil, inChannels: maskFeatureChannels, outChannels: spxChannels, outputShape: [1, halfHeight, halfWidth, spxChannels], activation: .leakyReLU)
        x = graph.concatTensors([x, stem2x], dimension: 3, name: "spx_2_gru.concat")
        x = conv2D(x, weightName: "spx_2_gru.conv2.conv.weight", biasName: nil, inChannels: spxMergedChannels, outChannels: spxOutputChannels, kernel: (3, 3), padding: (1, 1), activation: .leakyReLU)
        x = deconv2D(x, weightName: "spx_gru.0.weight", biasName: "spx_gru.0.bias", inChannels: spxOutputChannels, outChannels: 9, outputShape: [1, imageHeight, imageWidth, 9], activation: nil)
        return graph.softMax(with: x, axis: 3, name: "spx_gru.softmax")
    }

    private enum Activation {
        case relu
        case leakyReLU
    }

    private func conv2D(
        _ input: MPSGraphTensor,
        weightName: String,
        biasName: String?,
        inChannels: Int,
        outChannels: Int,
        kernel: (Int, Int),
        padding: (Int, Int),
        groups: Int = 1,
        activation: Activation?
    ) -> MPSGraphTensor {
        let weights = parameter(weightName, [kernel.0, kernel.1, inChannels / groups, outChannels], layout: .nhwc)
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: groups,
            paddingLeft: padding.1,
            paddingRight: padding.1,
            paddingTop: padding.0,
            paddingBottom: padding.0,
            paddingStyle: .explicit,
            dataLayout: .NHWC,
            weightsLayout: .HWIO
        )!
        var x = graph.convolution2D(input, weights: weights, descriptor: descriptor, name: weightName.replacingOccurrences(of: ".weight", with: ""))
        if let biasName {
            x = graph.addition(x, parameter(biasName, [1, 1, 1, outChannels], layout: .nhwc), name: biasName)
        }
        return activate(x, activation: activation, name: weightName)
    }

    private func deconv2D(
        _ input: MPSGraphTensor,
        weightName: String,
        biasName: String?,
        inChannels: Int,
        outChannels: Int,
        outputShape: [Int],
        activation: Activation?
    ) -> MPSGraphTensor {
        let weights = parameter(weightName, [4, 4, outChannels, inChannels], layout: .nhwc)
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: 2,
            strideInY: 2,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingLeft: 1,
            paddingRight: 1,
            paddingTop: 1,
            paddingBottom: 1,
            paddingStyle: .explicit,
            dataLayout: .NHWC,
            weightsLayout: .HWIO
        )!
        var x = graph.convolution2DDataGradient(input, weights: weights, outputShape: outputShape.map { NSNumber(value: $0) }, forwardConvolutionDescriptor: descriptor, name: weightName.replacingOccurrences(of: ".weight", with: ""))
        if let biasName {
            x = graph.addition(x, parameter(biasName, [1, 1, 1, outChannels], layout: .nhwc), name: biasName)
        }
        return activate(x, activation: activation, name: weightName)
    }

    private func linear(_ input: MPSGraphTensor, weightName: String, biasName: String, inChannels: Int, outChannels: Int) -> MPSGraphTensor {
        let x = graph.matrixMultiplication(primary: input, secondary: parameter(weightName, [inChannels, outChannels], layout: .nhwc), name: weightName.replacingOccurrences(of: ".weight", with: ".matmul"))
        return graph.addition(x, parameter(biasName, [1, 1, 1, outChannels], layout: .nhwc), name: biasName)
    }

    private func gelu(_ input: MPSGraphTensor, name: String) -> MPSGraphTensor {
        let half = graph.constant(0.5, dataType: .float32)
        let one = graph.constant(1.0, dataType: .float32)
        let sqrtTwo = graph.constant(sqrt(2.0), dataType: .float32)
        let scaled = graph.division(input, sqrtTwo, name: "\(name).scale")
        let erf = graph.erf(with: scaled, name: "\(name).erf")
        let cdf = graph.multiplication(half, graph.addition(one, erf, name: "\(name).one_plus_erf"), name: "\(name).cdf")
        return graph.multiplication(input, cdf, name: name)
    }

    private func activate(_ input: MPSGraphTensor, activation: Activation?, name: String) -> MPSGraphTensor {
        switch activation {
        case .relu:
            return graph.reLU(with: input, name: "\(name).relu")
        case .leakyReLU:
            return graph.leakyReLU(with: input, alpha: 0.01, name: "\(name).relu")
        case nil:
            return input
        }
    }

    private func placeholder(_ name: String, _ shape: [Int], layout: TensorLayout) -> MPSGraphTensor {
        let tensor = graph.placeholder(shape: shape.map { NSNumber(value: $0) }, dataType: .float32, name: name)
        inputTensors[name] = tensor
        return tensor
    }

    private func parameter(_ name: String, _ shape: [Int], layout: TensorLayout) -> MPSGraphTensor {
        parameterSpecs.append(NativeGraphTensorSpec(name: name, shape: TensorShape(shape, layout: layout)))
        let tensor = graph.placeholder(shape: shape.map { NSNumber(value: $0) }, dataType: .float32, name: name)
        parameterTensors[name] = tensor
        return tensor
    }

}

private struct BuiltCostRegularizationGraph {
    let graph: MPSGraph
    let outputs: CostRegularizationGraphOutputs
    let inputSpecs: [NativeGraphTensorSpec]
    let outputSpecs: [NativeGraphTensorSpec]
    let debugOutputSpecs: [NativeGraphTensorSpec]
    let parameterSpecs: [NativeGraphTensorSpec]
    let inputTensors: [String: MPSGraphTensor]
    let parameterTensors: [String: MPSGraphTensor]
}

private final class CostRegularizationGraphBuilder {
    private let graph = MPSGraph()
    private let precision: NativeGraphPrecision
    private var parameterSpecs: [NativeGraphTensorSpec] = []
    private var inputTensors: [String: MPSGraphTensor] = [:]
    private var parameterTensors: [String: MPSGraphTensor] = [:]
    private var debugTensors: [String: MPSGraphTensor] = [:]
    private var debugOutputSpecs: [NativeGraphTensorSpec] = []
    private let volumeChannels = FastFoundationStereoMPSConstants.regularizedVolumeChannels
    private let disparity = FastFoundationStereoMPSConstants.disparityQuarter
    private let height = FastFoundationStereoMPSConstants.quarterHeight
    private let width = FastFoundationStereoMPSConstants.quarterWidth

    init(precision: NativeGraphPrecision) {
        self.precision = precision
    }

    func build() -> BuiltCostRegularizationGraph {
        let combinedVolume = placeholder("combined_volume", [1, disparity, height, width, 32], layout: .ndhwc)
        let left04 = placeholder("left_04", [1, height, width, 224], layout: .nhwc)
        let left08 = placeholder("left_08", [1, height / 2, width / 2, 192], layout: .nhwc)
        let left16 = placeholder("left_16", [1, height / 4, width / 4, 320], layout: .nhwc)
        let left32 = placeholder("left_32", [1, height / 8, width / 8, 304], layout: .nhwc)

        var volume = prunedCorrFeatureAttention(computeInput(combinedVolume, name: "combined_volume"), left04: computeInput(left04, name: "left_04"))
        tap(volume, name: "cost.corr_feature_att", shape: [1, disparity, height, width, volumeChannels], layout: .ndhwc)
        volume = prunedHourglass(
            volume,
            left08: computeInput(left08, name: "left_08"),
            left16: computeInput(left16, name: "left_16"),
            left32: computeInput(left32, name: "left_32")
        )
        let logitsNDHWC = rawConv3D(volume, weightName: "classifier.layers.0.weight", biasName: "classifier.layers.0.bias", inChannels: volumeChannels, outChannels: 1, kernel: (3, 3, 3), padding: (1, 1, 1), activation: nil, groups: 1)
        let logitsNDHW = graph.squeeze(logitsNDHWC, axis: 4, name: "classifier_logits_squeeze")
        let logitsNHWD = graph.transpose(logitsNDHW, permutation: [0, 2, 3, 1], name: "classifier_logits_nhwd")
        let outputVolume = outputTensor(volume, name: "regularized_volume")
        let outputLogits = outputTensor(logitsNHWD, name: "logits")

        let inputs = [
            NativeGraphTensorSpec(name: "combined_volume", shape: TensorShape([1, disparity, height, width, 32], layout: .ndhwc)),
            NativeGraphTensorSpec(name: "left_04", shape: TensorShape([1, height, width, 224], layout: .nhwc)),
            NativeGraphTensorSpec(name: "left_08", shape: TensorShape([1, height / 2, width / 2, 192], layout: .nhwc)),
            NativeGraphTensorSpec(name: "left_16", shape: TensorShape([1, height / 4, width / 4, 320], layout: .nhwc)),
            NativeGraphTensorSpec(name: "left_32", shape: TensorShape([1, height / 8, width / 8, 304], layout: .nhwc)),
        ]
        let outputs = [
            NativeGraphTensorSpec(name: "regularized_volume", shape: TensorShape([1, disparity, height, width, volumeChannels], layout: .ndhwc)),
            NativeGraphTensorSpec(name: "logits", shape: TensorShape([1, height, width, disparity], layout: .nhwc)),
        ]
        return BuiltCostRegularizationGraph(
            graph: graph,
            outputs: CostRegularizationGraphOutputs(regularizedVolume: outputVolume, logits: outputLogits, debugTensors: debugTensors),
            inputSpecs: inputs,
            outputSpecs: outputs,
            debugOutputSpecs: debugOutputSpecs,
            parameterSpecs: parameterSpecs,
            inputTensors: inputTensors,
            parameterTensors: parameterTensors
        )
    }

    private func prunedCorrFeatureAttention(_ input: MPSGraphTensor, left04: MPSGraphTensor) -> MPSGraphTensor {
        var x = basicConv3D(input, prefix: "corr_feature_att.layers.0", inChannels: 32, outChannels: 28, kernel: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1), activation: .leakyReLU)
        x = basicConv3D(x, prefix: "corr_feature_att.layers.1", inChannels: 28, outChannels: 28, kernel: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1), activation: .leakyReLU)
        return featureAttention(prefix: "corr_feature_att.layers.2", volume: x, feature: left04, featureChannels: 224, volumeChannels: 28, featureHeight: height, featureWidth: width)
    }

    private func prunedHourglass(_ input: MPSGraphTensor, left08: MPSGraphTensor, left16: MPSGraphTensor, left32: MPSGraphTensor) -> MPSGraphTensor {
        let conv1 = prunedFeatureAtt8(input, left08: left08)
        tap(conv1, name: "cost.feature_att_8", shape: [1, disparity / 2, height / 2, width / 2, 56], layout: .ndhwc)
        let conv2 = prunedFeatureAtt16(conv1)
        tap(conv2, name: "cost.feature_att_16", shape: [1, disparity / 4, height / 4, width / 4, 112], layout: .ndhwc)
        var conv3 = basicConv3D(conv2, prefix: "cost_agg.conv3.0", inChannels: 112, outChannels: 168, kernel: (3, 3, 3), stride: (2, 2, 2), padding: (1, 1, 1), activation: .leakyReLU)
        conv3 = reduced3DSequential(conv3, prefix: "cost_agg.conv3.1", inChannels: 168, hiddenChannels: 168, outChannels: 168, spatialKernel: 3, disparityKernel: 17)
        conv3 = featureAttention(prefix: "cost_agg.feature_att_32", volume: conv3, feature: left32, featureChannels: 304, volumeChannels: 168, featureHeight: height / 8, featureWidth: width / 8)
        tap(conv3, name: "cost.feature_att_32", shape: [1, disparity / 8, height / 8, width / 8, 168], layout: .ndhwc)

        var merged2 = deconv3DGeneral(conv3, weightName: "cost_agg.post32_to_16.upsample.0.conv.weight", biasName: nil, bnPrefix: "cost_agg.post32_to_16.upsample.0.bn", inChannels: 168, outChannels: 112, kernel: (3, 3, 3), outputShape: [1, disparity / 4, height / 4, width / 4, 112], activation: .leakyReLU)
        merged2 = graph.addition(merged2, conv2, name: "cost_agg.post32_to_16.sum")
        merged2 = featureAttention(prefix: "cost_agg.post32_to_16.out.0", volume: merged2, feature: left16, featureChannels: 320, volumeChannels: 112, featureHeight: height / 4, featureWidth: width / 4)
        merged2 = basicConv3D(merged2, prefix: "cost_agg.post32_to_16.out.1", inChannels: 112, outChannels: 112, kernel: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1), activation: .leakyReLU)
        merged2 = reduced3DSequential(merged2, prefix: "cost_agg.post32_to_16.out.2", inChannels: 112, hiddenChannels: 112, outChannels: 112, spatialKernel: 3, disparityKernel: 9)
        merged2 = basicConv3D(merged2, prefix: "cost_agg.post32_to_16.out.3", inChannels: 112, outChannels: 112, kernel: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1), activation: .leakyReLU)
        tap(merged2, name: "cost.post32_to_16", shape: [1, disparity / 4, height / 4, width / 4, 112], layout: .ndhwc)

        var merged1 = deconv3DGeneral(merged2, weightName: "cost_agg.post16_to_8.upsample.0.conv.weight", biasName: nil, bnPrefix: "cost_agg.post16_to_8.upsample.0.bn", inChannels: 112, outChannels: 56, kernel: (4, 4, 4), outputShape: [1, disparity / 2, height / 2, width / 2, 56], activation: .leakyReLU)
        merged1 = graph.addition(merged1, conv1, name: "cost_agg.post16_to_8.sum")
        merged1 = basicConv3D(merged1, prefix: "cost_agg.post16_to_8.out.0", inChannels: 56, outChannels: 56, kernel: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1), activation: .leakyReLU)
        merged1 = featureAttention(prefix: "cost_agg.post16_to_8.out.1", volume: merged1, feature: left08, featureChannels: 192, volumeChannels: 56, featureHeight: height / 2, featureWidth: width / 2)
        merged1 = reduced3DSequential(merged1, prefix: "cost_agg.post16_to_8.out.2", inChannels: 56, hiddenChannels: 56, outChannels: 56, spatialKernel: 3, disparityKernel: 9)
        tap(merged1, name: "cost.post16_to_8", shape: [1, disparity / 2, height / 2, width / 2, 56], layout: .ndhwc)

        var conv = deconv3DGeneral(merged1, weightName: "cost_agg.conv1_up.conv.weight", biasName: nil, bnPrefix: "cost_agg.conv1_up.bn", inChannels: 56, outChannels: 28, kernel: (4, 4, 4), outputShape: [1, disparity, height, width, 28], activation: .leakyReLU)
        tap(conv, name: "cost.conv1_up", shape: [1, disparity, height, width, 28], layout: .ndhwc)
        var patch = rawConv3D(conv, weightName: "cost_agg.post8_to_4.upsample.0.conv.weight", biasName: nil, inChannels: 28, outChannels: 28, kernel: (4, 4, 4), stride: (4, 4, 4), padding: (0, 0, 0), activation: nil, groups: 1)
        tap(patch, name: "cost.post8_to_4.patch_raw_conv", shape: [1, disparity / 4, height / 4, width / 4, 28], layout: .ndhwc)
        patch = batchNorm3D(patch, prefix: "cost_agg.post8_to_4.upsample.0.bn", channels: 28)
        patch = graph.leakyReLU(with: patch, alpha: 0.01, name: "cost_agg.post8_to_4.upsample.0.relu")
        tap(patch, name: "cost.post8_to_4.patch_conv", shape: [1, disparity / 4, height / 4, width / 4, 28], layout: .ndhwc)
        patch = disparityAttention(patch, prefix: "cost_agg.post8_to_4.upsample.1", channels: 28, heads: 4, layers: 1, featureHeight: height / 4, featureWidth: width / 4, sequenceDisparity: disparity / 4, feedForwardChannels: 112, useSinusoidalPosition: false)
        tap(patch, name: "cost.post8_to_4.patch_attention", shape: [1, disparity / 4, height / 4, width / 4, 28], layout: .ndhwc)
        patch = trilinearUpsample4x(patch, name: "cost_agg.post8_to_4.upsample.2")
        tap(patch, name: "cost.post8_to_4.patch", shape: [1, disparity, height, width, 28], layout: .ndhwc)
        conv = graph.addition(input, patch, name: "cost_agg.post8_to_4.sum")
        tap(conv, name: "cost.post8_to_4.sum", shape: [1, disparity, height, width, 28], layout: .ndhwc)
        conv = basicConv3D(conv, prefix: "cost_agg.post8_to_4.out.0", inChannels: 28, outChannels: 28, kernel: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1), activation: .leakyReLU)
        tap(conv, name: "cost.post8_to_4.out0", shape: [1, disparity, height, width, 28], layout: .ndhwc)
        conv = resnetBasicBlock3D(conv, prefix: "cost_agg.post8_to_4.out.1", channels: 28)
        tap(conv, name: "cost.post8_to_4", shape: [1, disparity, height, width, 28], layout: .ndhwc)
        return conv
    }

    private func prunedFeatureAtt8(_ input: MPSGraphTensor, left08: MPSGraphTensor) -> MPSGraphTensor {
        var x = basicConv3D(input, prefix: "cost_agg.feature_att_8.layers.0", inChannels: 28, outChannels: 28, kernel: (3, 3, 3), stride: (2, 2, 2), padding: (1, 1, 1), activation: .leakyReLU)
        x = reduced3DSequential(x, prefix: "cost_agg.feature_att_8.layers.1", inChannels: 28, hiddenChannels: 56, outChannels: 56, spatialKernel: 3, disparityKernel: 17)
        return featureAttention(prefix: "cost_agg.feature_att_8.layers.2", volume: x, feature: left08, featureChannels: 192, volumeChannels: 56, featureHeight: height / 2, featureWidth: width / 2)
    }

    private func prunedFeatureAtt16(_ input: MPSGraphTensor) -> MPSGraphTensor {
        var x = basicConv3D(input, prefix: "cost_agg.feature_att_16.layers.0", inChannels: 56, outChannels: 112, kernel: (3, 3, 3), stride: (2, 2, 2), padding: (1, 1, 1), activation: .leakyReLU)
        x = reduced3DSequential(x, prefix: "cost_agg.feature_att_16.layers.1", inChannels: 112, hiddenChannels: 56, outChannels: 56, spatialKernel: 3, disparityKernel: 13)
        return reduced3DSequential(x, prefix: "cost_agg.feature_att_16.layers.2", inChannels: 56, hiddenChannels: 112, outChannels: 112, spatialKernel: 3, disparityKernel: 3)
    }

    private func corrStem(_ input: MPSGraphTensor) -> MPSGraphTensor {
        var x = conv3D(input, prefix: "corr_stem.0", inChannels: 32, outChannels: volumeChannels, kernel: (1, 1, 1), padding: (0, 0, 0), activation: nil, useBias: true, useBatchNorm: false)
        x = conv3D(x, prefix: "corr_stem.1", inChannels: volumeChannels, outChannels: volumeChannels, kernel: (3, 3, 3), padding: (1, 1, 1), activation: .leakyReLU, useBias: false, useBatchNorm: true)
        x = residual3D(x, prefix: "corr_stem.2", channels: volumeChannels)
        x = residual3D(x, prefix: "corr_stem.3", channels: volumeChannels)
        return x
    }

    private func hourglass(_ input: MPSGraphTensor, left08: MPSGraphTensor, left16: MPSGraphTensor, left32: MPSGraphTensor) -> MPSGraphTensor {
        var conv1 = conv3D(input, prefix: "cost_agg.conv1.0", inChannels: volumeChannels, outChannels: volumeChannels * 2, kernel: (3, 3, 3), stride: (2, 2, 2), padding: (1, 1, 1), activation: .leakyReLU, useBias: false, useBatchNorm: true)
        conv1 = reduced3D(conv1, prefix: "cost_agg.conv1.1", inChannels: volumeChannels * 2, outChannels: volumeChannels * 2)
        conv1 = featureAttention(prefix: "cost_agg.feature_att_8", volume: conv1, feature: left08, featureChannels: 192, volumeChannels: volumeChannels * 2, featureHeight: height / 2, featureWidth: width / 2)

        var conv2 = conv3D(conv1, prefix: "cost_agg.conv2.0", inChannels: volumeChannels * 2, outChannels: volumeChannels * 4, kernel: (3, 3, 3), stride: (2, 2, 2), padding: (1, 1, 1), activation: .leakyReLU, useBias: false, useBatchNorm: true)
        conv2 = reduced3D(conv2, prefix: "cost_agg.conv2.1", inChannels: volumeChannels * 4, outChannels: volumeChannels * 4)
        conv2 = featureAttention(prefix: "cost_agg.feature_att_16", volume: conv2, feature: left16, featureChannels: 320, volumeChannels: volumeChannels * 4, featureHeight: height / 4, featureWidth: width / 4)

        var conv3 = conv3D(conv2, prefix: "cost_agg.conv3.0", inChannels: volumeChannels * 4, outChannels: volumeChannels * 6, kernel: (3, 3, 3), stride: (2, 2, 2), padding: (1, 1, 1), activation: .leakyReLU, useBias: false, useBatchNorm: true)
        conv3 = reduced3D(conv3, prefix: "cost_agg.conv3.1", inChannels: volumeChannels * 6, outChannels: volumeChannels * 6)
        conv3 = featureAttention(prefix: "cost_agg.feature_att_32", volume: conv3, feature: left32, featureChannels: 304, volumeChannels: volumeChannels * 6, featureHeight: height / 8, featureWidth: width / 8)

        let conv3Up = deconv3D(conv3, prefix: "cost_agg.conv3_up", inChannels: volumeChannels * 6, outChannels: volumeChannels * 4, outputShape: [1, disparity / 4, height / 4, width / 4, volumeChannels * 4])
        var merged2 = graph.concatTensors([conv3Up, conv2], dimension: 4, name: "cost_agg.merge_32_to_16")
        merged2 = conv3D(merged2, prefix: "cost_agg.agg_0.0", inChannels: volumeChannels * 8, outChannels: volumeChannels * 4, kernel: (1, 1, 1), padding: (0, 0, 0), activation: .leakyReLU, useBias: false, useBatchNorm: true)
        merged2 = reduced3D(merged2, prefix: "cost_agg.agg_0.1", inChannels: volumeChannels * 4, outChannels: volumeChannels * 4)
        merged2 = reduced3D(merged2, prefix: "cost_agg.agg_0.2", inChannels: volumeChannels * 4, outChannels: volumeChannels * 4)
        merged2 = featureAttention(prefix: "cost_agg.feature_att_up_16", volume: merged2, feature: left16, featureChannels: 320, volumeChannels: volumeChannels * 4, featureHeight: height / 4, featureWidth: width / 4)

        let conv2Up = deconv3D(merged2, prefix: "cost_agg.conv2_up", inChannels: volumeChannels * 4, outChannels: volumeChannels * 2, outputShape: [1, disparity / 2, height / 2, width / 2, volumeChannels * 2])
        var merged1 = graph.concatTensors([conv2Up, conv1], dimension: 4, name: "cost_agg.merge_16_to_8")
        merged1 = conv3D(merged1, prefix: "cost_agg.agg_1.0", inChannels: volumeChannels * 4, outChannels: volumeChannels * 2, kernel: (1, 1, 1), padding: (0, 0, 0), activation: .leakyReLU, useBias: false, useBatchNorm: true)
        merged1 = reduced3D(merged1, prefix: "cost_agg.agg_1.1", inChannels: volumeChannels * 2, outChannels: volumeChannels * 2)
        merged1 = reduced3D(merged1, prefix: "cost_agg.agg_1.2", inChannels: volumeChannels * 2, outChannels: volumeChannels * 2)
        merged1 = featureAttention(prefix: "cost_agg.feature_att_up_8", volume: merged1, feature: left08, featureChannels: 192, volumeChannels: volumeChannels * 2, featureHeight: height / 2, featureWidth: width / 2)

        var conv = deconv3D(merged1, prefix: "cost_agg.conv1_up", inChannels: volumeChannels * 2, outChannels: volumeChannels, outputShape: [1, disparity, height, width, volumeChannels])
        let patch = disparityAttentionPatch(input)
        conv = graph.addition(conv, patch, name: "cost_agg.patch_residual")
        conv = reduced3D(conv, prefix: "cost_agg.conv_out.0", inChannels: volumeChannels, outChannels: volumeChannels)
        conv = reduced3D(conv, prefix: "cost_agg.conv_out.1", inChannels: volumeChannels, outChannels: volumeChannels)
        return conv
    }

    private func classifier(_ input: MPSGraphTensor) -> MPSGraphTensor {
        var x = conv3D(input, prefix: "classifier.0", inChannels: volumeChannels, outChannels: volumeChannels / 2, kernel: (3, 3, 3), padding: (1, 1, 1), activation: .leakyReLU, useBias: false, useBatchNorm: true)
        x = residual3D(x, prefix: "classifier.1", channels: volumeChannels / 2)
        return conv3D(x, prefix: "classifier.2", inChannels: volumeChannels / 2, outChannels: 1, kernel: (7, 7, 7), padding: (3, 3, 3), activation: nil, useBias: true, useBatchNorm: false)
    }

    private func disparityAttentionPatch(_ input: MPSGraphTensor) -> MPSGraphTensor {
        var x = conv3D(input, prefix: "cost_agg.conv_patch.0", inChannels: volumeChannels, outChannels: volumeChannels, kernel: (4, 4, 4), stride: (4, 4, 4), padding: (0, 0, 0), activation: nil, groups: volumeChannels, useBias: false, useBatchNorm: false)
        x = batchNorm3D(x, prefix: "cost_agg.conv_patch.1", channels: volumeChannels)
        x = disparityAttention(x, prefix: "cost_agg.atts.4", channels: volumeChannels, heads: 4, layers: 4)
        return deconv3D(x, prefix: "cost_agg.attention_upsample", inChannels: volumeChannels, outChannels: volumeChannels, outputShape: [1, disparity, height, width, volumeChannels], parameterPrefix: "cost_agg.conv_patch_upsample")
    }

    private func disparityAttention(
        _ input: MPSGraphTensor,
        prefix: String,
        channels: Int,
        heads: Int,
        layers: Int,
        featureHeight: Int? = nil,
        featureWidth: Int? = nil,
        sequenceDisparity: Int? = nil,
        feedForwardChannels: Int? = nil,
        useSinusoidalPosition: Bool = false
    ) -> MPSGraphTensor {
        let attentionHeight = featureHeight ?? height
        let attentionWidth = featureWidth ?? width
        let reducedDisparity = sequenceDisparity ?? disparity / 4
        let sequenceCount = attentionHeight * attentionWidth
        let sequenceInput = graph.transpose(input, permutation: [0, 2, 3, 1, 4], name: "\(prefix).bhwdc_in")
        var x = graph.reshape(sequenceInput, shape: [sequenceCount as NSNumber, reducedDisparity as NSNumber, channels as NSNumber], name: "\(prefix).reshape_in")
        let position: MPSGraphTensor
        if useSinusoidalPosition {
            position = sinusoidalPosition(disparity: reducedDisparity, channels: channels)
        } else {
            position = parameter("\(prefix).pos_embed0.pe", [1, reducedDisparity, channels], layout: .nhwc)
        }
        x = graph.addition(x, position, name: "\(prefix).pos_embed0.add")
        if prefix == "cost_agg.post8_to_4.upsample.1" {
            tap(x, name: "cost.post8_to_4.attention.pos_add", shape: [sequenceCount, reducedDisparity, channels], layout: .nhwc)
        }
        for layer in 0..<layers {
            x = transformerLayer(x, prefix: "\(prefix).sa.\(layer)", channels: channels, heads: heads, sequenceCount: sequenceCount, sequenceDisparity: reducedDisparity, feedForwardChannels: feedForwardChannels ?? channels)
        }
        x = graph.reshape(x, shape: [1, attentionHeight, attentionWidth, reducedDisparity, channels] as [NSNumber], name: "\(prefix).reshape_out")
        return graph.transpose(x, permutation: [0, 3, 1, 2, 4], name: "\(prefix).ndhwc_out")
    }

    private func transformerLayer(_ input: MPSGraphTensor, prefix: String, channels: Int, heads: Int, sequenceCount: Int, sequenceDisparity: Int, feedForwardChannels: Int) -> MPSGraphTensor {
        let headDim = channels / heads
        let q = attentionProjection(input, prefix: "\(prefix).self_attn.q_proj", channels: channels, heads: heads, headDim: headDim, sequenceCount: sequenceCount, sequenceDisparity: sequenceDisparity)
        let k = attentionProjection(input, prefix: "\(prefix).self_attn.k_proj", channels: channels, heads: heads, headDim: headDim, sequenceCount: sequenceCount, sequenceDisparity: sequenceDisparity)
        let v = attentionProjection(input, prefix: "\(prefix).self_attn.v_proj", channels: channels, heads: heads, headDim: headDim, sequenceCount: sequenceCount, sequenceDisparity: sequenceDisparity)
        var attn = scaledDotProductAttention(query: q, key: k, value: v, headDim: headDim, name: "\(prefix).self_attn.sdpa")
        attn = graph.reshape(attn, shape: [sequenceCount as NSNumber, sequenceDisparity as NSNumber, channels as NSNumber], name: "\(prefix).self_attn.merge_heads")
        attn = linear(attn, prefix: "\(prefix).self_attn.out_proj", inChannels: channels, outChannels: channels)
        if prefix == "cost_agg.post8_to_4.upsample.1.sa.0" {
            tap(attn, name: "cost.post8_to_4.attention.self_attn", shape: [sequenceCount, sequenceDisparity, channels], layout: .nhwc)
        }
        var x = graph.addition(input, attn, name: "\(prefix).residual_attn")
        x = layerNorm(x, prefix: "\(prefix).norm1", channels: channels)
        if prefix == "cost_agg.post8_to_4.upsample.1.sa.0" {
            tap(x, name: "cost.post8_to_4.attention.norm1", shape: [sequenceCount, sequenceDisparity, channels], layout: .nhwc)
        }
        var feedForward = linear(x, prefix: "\(prefix).linear1", inChannels: channels, outChannels: feedForwardChannels)
        feedForward = gelu(feedForward, name: "\(prefix).gelu")
        feedForward = linear(feedForward, prefix: "\(prefix).linear2", inChannels: feedForwardChannels, outChannels: channels)
        if prefix == "cost_agg.post8_to_4.upsample.1.sa.0" {
            tap(feedForward, name: "cost.post8_to_4.attention.feed_forward", shape: [sequenceCount, sequenceDisparity, channels], layout: .nhwc)
        }
        x = graph.addition(x, feedForward, name: "\(prefix).residual_ffn")
        let output = layerNorm(x, prefix: "\(prefix).norm2", channels: channels)
        if prefix == "cost_agg.post8_to_4.upsample.1.sa.0" {
            tap(output, name: "cost.post8_to_4.attention.norm2", shape: [sequenceCount, sequenceDisparity, channels], layout: .nhwc)
        }
        return output
    }

    private func sinusoidalPosition(disparity: Int, channels: Int) -> MPSGraphTensor {
        var values = [Float](repeating: 0, count: disparity * channels)
        for position in 0..<disparity {
            for channel in stride(from: 0, to: channels, by: 2) {
                let div = exp(Float(channel) * -Float(log(10000.0)) / Float(channels))
                values[position * channels + channel] = sin(Float(position) * div)
                if channel + 1 < channels {
                    values[position * channels + channel + 1] = cos(Float(position) * div)
                }
            }
        }
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) } as NSData
        return graph.constant(data as Data, shape: [1, disparity, channels] as [NSNumber], dataType: .float32)
    }

    private func trilinearUpsample4x(_ input: MPSGraphTensor, name: String) -> MPSGraphTensor {
        let spatialInput = graph.reshape(input, shape: [disparity / 4 as NSNumber, height / 4 as NSNumber, width / 4 as NSNumber, volumeChannels as NSNumber], name: "\(name).spatial_reshape")
        let spatial = graph.resize(spatialInput, size: [height as NSNumber, width as NSNumber], mode: .bilinear, centerResult: true, alignCorners: false, layout: .NHWC, name: "\(name).spatial")
        let withBatch = graph.reshape(spatial, shape: [1, disparity / 4, height, width, volumeChannels] as [NSNumber], name: "\(name).with_batch")
        let depthFirst = graph.transpose(withBatch, permutation: [0, 2, 3, 1, 4], name: "\(name).depth_first")
        let depthInput = graph.reshape(depthFirst, shape: [height * width as NSNumber, disparity / 4 as NSNumber, 1 as NSNumber, volumeChannels as NSNumber], name: "\(name).depth_reshape")
        let depth = graph.resize(depthInput, size: [disparity as NSNumber, 1 as NSNumber], mode: .bilinear, centerResult: true, alignCorners: false, layout: .NHWC, name: "\(name).depth")
        let restored = graph.reshape(depth, shape: [1, height, width, disparity, volumeChannels] as [NSNumber], name: "\(name).restore")
        return graph.transpose(restored, permutation: [0, 3, 1, 2, 4], name: "\(name).ndhwc")
    }

    private func attentionProjection(_ input: MPSGraphTensor, prefix: String, channels: Int, heads: Int, headDim: Int, sequenceCount: Int, sequenceDisparity: Int) -> MPSGraphTensor {
        let x = linear(input, prefix: prefix, inChannels: channels, outChannels: channels)
        return graph.reshape(x, shape: [sequenceCount as NSNumber, sequenceDisparity as NSNumber, heads as NSNumber, headDim as NSNumber], name: "\(prefix).split_heads")
    }

    private func scaledDotProductAttention(query: MPSGraphTensor, key: MPSGraphTensor, value: MPSGraphTensor, headDim: Int, name: String) -> MPSGraphTensor {
        let keyTransposed = graph.transpose(key, permutation: [0, 1, 3, 2], name: "\(name).key_transpose")
        var scores = graph.matrixMultiplication(primary: query, secondary: keyTransposed, name: "\(name).scores")
        scores = graph.multiplication(scores, graph.constant(1.0 / sqrt(Double(headDim)), dataType: precision.dataType), name: "\(name).scale")
        let probabilities = graph.softMax(with: scores, axis: 3, name: "\(name).softmax")
        return graph.matrixMultiplication(primary: probabilities, secondary: value, name: "\(name).value")
    }

    private func featureAttention(prefix: String, volume: MPSGraphTensor, feature: MPSGraphTensor, featureChannels: Int, volumeChannels: Int, featureHeight: Int, featureWidth: Int) -> MPSGraphTensor {
        var attention = conv2D(feature, prefix: "\(prefix).feat_att.0", inChannels: featureChannels, outChannels: featureChannels / 2, kernel: (1, 1), padding: (0, 0), activation: .leakyReLU, useBias: false, useBatchNorm: true)
        attention = rawConv2D(attention, weightName: "\(prefix).feat_att.1.weight", biasName: "\(prefix).feat_att.1.bias", inChannels: featureChannels / 2, outChannels: volumeChannels, kernel: (1, 1), padding: (0, 0), activation: nil)
        attention = graph.sigmoid(with: attention, name: "\(prefix).sigmoid")
        attention = graph.reshape(attention, shape: [1, 1, featureHeight, featureWidth, volumeChannels] as [NSNumber], name: "\(prefix).unsqueeze_disparity")
        return graph.multiplication(volume, attention, name: "\(prefix).gate")
    }

    private func reduced3D(_ input: MPSGraphTensor, prefix: String, inChannels: Int, outChannels: Int) -> MPSGraphTensor {
        var x = conv3D(input, prefix: "\(prefix).conv1", inChannels: inChannels, outChannels: outChannels, kernel: (1, 3, 3), padding: (0, 1, 1), activation: .relu, useBias: true, useBatchNorm: true)
        x = conv3D(x, prefix: "\(prefix).conv2", inChannels: outChannels, outChannels: outChannels, kernel: (17, 1, 1), padding: (8, 0, 0), activation: .relu, useBias: true, useBatchNorm: true)
        return x
    }

    private func reduced3DSequential(_ input: MPSGraphTensor, prefix: String, inChannels: Int, hiddenChannels: Int, outChannels: Int, spatialKernel: Int, disparityKernel: Int) -> MPSGraphTensor {
        var x = rawConv3D(input, weightName: "\(prefix).conv1.0.weight", biasName: "\(prefix).conv1.0.bias", inChannels: inChannels, outChannels: hiddenChannels, kernel: (1, spatialKernel, spatialKernel), padding: (0, spatialKernel / 2, spatialKernel / 2), activation: nil, groups: 1)
        x = batchNorm3D(x, prefix: "\(prefix).conv1.1", channels: hiddenChannels)
        x = graph.reLU(with: x, name: "\(prefix).conv1.2")
        x = rawConv3D(x, weightName: "\(prefix).conv2.0.weight", biasName: "\(prefix).conv2.0.bias", inChannels: hiddenChannels, outChannels: outChannels, kernel: (disparityKernel, 1, 1), padding: (disparityKernel / 2, 0, 0), activation: nil, groups: 1)
        x = batchNorm3D(x, prefix: "\(prefix).conv2.1", channels: outChannels)
        return graph.reLU(with: x, name: "\(prefix).conv2.2")
    }

    private func basicConv3D(
        _ input: MPSGraphTensor,
        prefix: String,
        inChannels: Int,
        outChannels: Int,
        kernel: (Int, Int, Int),
        stride: (Int, Int, Int),
        padding: (Int, Int, Int),
        activation: Activation?
    ) -> MPSGraphTensor {
        var x = rawConv3D(input, weightName: "\(prefix).conv.weight", biasName: nil, inChannels: inChannels, outChannels: outChannels, kernel: kernel, stride: stride, padding: padding, activation: nil, groups: 1)
        x = batchNorm3D(x, prefix: "\(prefix).bn", channels: outChannels)
        return activate(x, activation: activation, name: prefix)
    }

    private func rawConv3D(
        _ input: MPSGraphTensor,
        weightName: String,
        biasName: String?,
        inChannels: Int,
        outChannels: Int,
        kernel: (Int, Int, Int),
        stride: (Int, Int, Int) = (1, 1, 1),
        padding: (Int, Int, Int),
        activation: Activation?,
        groups: Int
    ) -> MPSGraphTensor {
        let weights = parameter(weightName, [kernel.0, kernel.1, kernel.2, inChannels / groups, outChannels], layout: .ndhwc)
        let descriptor = MPSGraphConvolution3DOpDescriptor(
            strideInX: stride.2,
            strideInY: stride.1,
            strideInZ: stride.0,
            dilationRateInX: 1,
            dilationRateInY: 1,
            dilationRateInZ: 1,
            groups: groups,
            paddingLeft: padding.2,
            paddingRight: padding.2,
            paddingTop: padding.1,
            paddingBottom: padding.1,
            paddingFront: padding.0,
            paddingBack: padding.0,
            paddingStyle: .explicit,
            dataLayout: .NDHWC,
            weightsLayout: .DHWIO
        )!
        var x = graph.convolution3D(input, weights: weights, descriptor: descriptor, name: weightName.replacingOccurrences(of: ".weight", with: ""))
        if let biasName {
            x = graph.addition(x, parameter(biasName, [1, 1, 1, 1, outChannels], layout: .ndhwc), name: biasName)
        }
        return activate(x, activation: activation, name: weightName)
    }

    private func deconv3DGeneral(
        _ input: MPSGraphTensor,
        weightName: String,
        biasName: String?,
        bnPrefix: String,
        inChannels: Int,
        outChannels: Int,
        kernel: (Int, Int, Int),
        outputShape: [Int],
        activation: Activation?
    ) -> MPSGraphTensor {
        let weights = parameter(weightName, [kernel.0, kernel.1, kernel.2, outChannels, inChannels], layout: .ndhwc)
        let descriptor = MPSGraphConvolution3DOpDescriptor(
            strideInX: 2,
            strideInY: 2,
            strideInZ: 2,
            dilationRateInX: 1,
            dilationRateInY: 1,
            dilationRateInZ: 1,
            groups: 1,
            paddingLeft: 1,
            paddingRight: 1,
            paddingTop: 1,
            paddingBottom: 1,
            paddingFront: 1,
            paddingBack: 1,
            paddingStyle: .explicit,
            dataLayout: .NDHWC,
            weightsLayout: .DHWIO
        )!
        var x = graph.convolution3DDataGradient(input, weights: weights, outputShape: outputShape.map { NSNumber(value: $0) }, forwardConvolutionDescriptor: descriptor, name: weightName.replacingOccurrences(of: ".weight", with: ""))
        if let biasName {
            x = graph.addition(x, parameter(biasName, [1, 1, 1, 1, outChannels], layout: .ndhwc), name: biasName)
        }
        x = batchNorm3D(x, prefix: bnPrefix, channels: outChannels)
        return activate(x, activation: activation, name: weightName)
    }

    private func resnetBasicBlock3D(_ input: MPSGraphTensor, prefix: String, channels: Int) -> MPSGraphTensor {
        var x = rawConv3D(input, weightName: "\(prefix).conv1.weight", biasName: nil, inChannels: channels, outChannels: channels, kernel: (3, 3, 3), padding: (1, 1, 1), activation: nil, groups: 1)
        if prefix == "cost_agg.post8_to_4.out.1" {
            tap(x, name: "cost.post8_to_4.resblock.conv1_raw", shape: [1, disparity, height, width, channels], layout: .ndhwc)
        }
        x = batchNorm3D(x, prefix: "\(prefix).bn1", channels: channels)
        x = graph.reLU(with: x, name: "\(prefix).relu1")
        if prefix == "cost_agg.post8_to_4.out.1" {
            tap(x, name: "cost.post8_to_4.resblock.relu1", shape: [1, disparity, height, width, channels], layout: .ndhwc)
        }
        x = rawConv3D(x, weightName: "\(prefix).conv2.weight", biasName: nil, inChannels: channels, outChannels: channels, kernel: (3, 3, 3), padding: (1, 1, 1), activation: nil, groups: 1)
        if prefix == "cost_agg.post8_to_4.out.1" {
            tap(x, name: "cost.post8_to_4.resblock.conv2_raw", shape: [1, disparity, height, width, channels], layout: .ndhwc)
        }
        x = batchNorm3D(x, prefix: "\(prefix).bn2", channels: channels)
        x = graph.addition(x, input, name: "\(prefix).add")
        if prefix == "cost_agg.post8_to_4.out.1" {
            tap(x, name: "cost.post8_to_4.resblock.add", shape: [1, disparity, height, width, channels], layout: .ndhwc)
        }
        return graph.reLU(with: x, name: "\(prefix).relu2")
    }

    private func residual3D(_ input: MPSGraphTensor, prefix: String, channels: Int) -> MPSGraphTensor {
        var x = conv3D(input, prefix: "\(prefix).conv1", inChannels: channels, outChannels: channels, kernel: (3, 3, 3), padding: (1, 1, 1), activation: nil, useBias: false, useBatchNorm: true)
        x = graph.reLU(with: x, name: "\(prefix).relu1")
        x = conv3D(x, prefix: "\(prefix).conv2", inChannels: channels, outChannels: channels, kernel: (3, 3, 3), padding: (1, 1, 1), activation: nil, useBias: false, useBatchNorm: true)
        x = graph.addition(x, input, name: "\(prefix).add")
        return graph.reLU(with: x, name: "\(prefix).relu2")
    }

    private enum Activation {
        case relu
        case leakyReLU
    }

    private func conv3D(
        _ input: MPSGraphTensor,
        prefix: String,
        inChannels: Int,
        outChannels: Int,
        kernel: (Int, Int, Int),
        stride: (Int, Int, Int) = (1, 1, 1),
        padding: (Int, Int, Int),
        activation: Activation?,
        groups: Int = 1,
        useBias: Bool,
        useBatchNorm: Bool
    ) -> MPSGraphTensor {
        let weights = parameter("\(prefix).conv.weight", [kernel.0, kernel.1, kernel.2, inChannels / groups, outChannels], layout: .ndhwc)
        let descriptor = MPSGraphConvolution3DOpDescriptor(
            strideInX: stride.2,
            strideInY: stride.1,
            strideInZ: stride.0,
            dilationRateInX: 1,
            dilationRateInY: 1,
            dilationRateInZ: 1,
            groups: groups,
            paddingLeft: padding.2,
            paddingRight: padding.2,
            paddingTop: padding.1,
            paddingBottom: padding.1,
            paddingFront: padding.0,
            paddingBack: padding.0,
            paddingStyle: .explicit,
            dataLayout: .NDHWC,
            weightsLayout: .DHWIO
        )!
        var x = graph.convolution3D(input, weights: weights, descriptor: descriptor, name: "\(prefix).conv")
        if useBias {
            x = graph.addition(x, parameter("\(prefix).conv.bias", [1, 1, 1, 1, outChannels], layout: .ndhwc), name: "\(prefix).bias")
        }
        if useBatchNorm {
            x = batchNorm3D(x, prefix: "\(prefix).bn", channels: outChannels)
        }
        return activate(x, activation: activation, name: prefix)
    }

    private func deconv3D(
        _ input: MPSGraphTensor,
        prefix: String,
        inChannels: Int,
        outChannels: Int,
        outputShape: [Int],
        parameterPrefix: String? = nil
    ) -> MPSGraphTensor {
        let parameterName = parameterPrefix ?? prefix
        let weights = parameter("\(parameterName).conv.weight", [4, 4, 4, outChannels, inChannels], layout: .ndhwc)
        let descriptor = MPSGraphConvolution3DOpDescriptor(
            strideInX: 2,
            strideInY: 2,
            strideInZ: 2,
            dilationRateInX: 1,
            dilationRateInY: 1,
            dilationRateInZ: 1,
            groups: 1,
            paddingLeft: 1,
            paddingRight: 1,
            paddingTop: 1,
            paddingBottom: 1,
            paddingFront: 1,
            paddingBack: 1,
            paddingStyle: .explicit,
            dataLayout: .NDHWC,
            weightsLayout: .DHWIO
        )!
        var x = graph.convolution3DDataGradient(input, weights: weights, outputShape: outputShape.map { NSNumber(value: $0) }, forwardConvolutionDescriptor: descriptor, name: "\(prefix).conv_transpose")
        x = batchNorm3D(x, prefix: "\(prefix).bn", channels: outChannels)
        return graph.leakyReLU(with: x, alpha: 0.01, name: "\(prefix).relu")
    }

    private func conv2D(
        _ input: MPSGraphTensor,
        prefix: String,
        inChannels: Int,
        outChannels: Int,
        kernel: (Int, Int),
        padding: (Int, Int),
        activation: Activation?,
        useBias: Bool,
        useBatchNorm: Bool
    ) -> MPSGraphTensor {
        let weights = parameter("\(prefix).conv.weight", [kernel.0, kernel.1, inChannels, outChannels], layout: .nhwc)
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingLeft: padding.1,
            paddingRight: padding.1,
            paddingTop: padding.0,
            paddingBottom: padding.0,
            paddingStyle: .explicit,
            dataLayout: .NHWC,
            weightsLayout: .HWIO
        )!
        var x = graph.convolution2D(input, weights: weights, descriptor: descriptor, name: "\(prefix).conv")
        if useBias {
            x = graph.addition(x, parameter("\(prefix).conv.bias", [1, 1, 1, outChannels], layout: .nhwc), name: "\(prefix).bias")
        }
        if useBatchNorm {
            x = batchNorm2D(x, prefix: "\(prefix).bn", channels: outChannels)
        }
        return activate(x, activation: activation, name: prefix)
    }

    private func rawConv2D(
        _ input: MPSGraphTensor,
        weightName: String,
        biasName: String?,
        inChannels: Int,
        outChannels: Int,
        kernel: (Int, Int),
        padding: (Int, Int),
        activation: Activation?
    ) -> MPSGraphTensor {
        let weights = parameter(weightName, [kernel.0, kernel.1, inChannels, outChannels], layout: .nhwc)
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingLeft: padding.1,
            paddingRight: padding.1,
            paddingTop: padding.0,
            paddingBottom: padding.0,
            paddingStyle: .explicit,
            dataLayout: .NHWC,
            weightsLayout: .HWIO
        )!
        var x = graph.convolution2D(input, weights: weights, descriptor: descriptor, name: weightName.replacingOccurrences(of: ".weight", with: ""))
        if let biasName {
            x = graph.addition(x, parameter(biasName, [1, 1, 1, outChannels], layout: .nhwc), name: biasName)
        }
        return activate(x, activation: activation, name: weightName)
    }

    private func batchNorm3D(_ input: MPSGraphTensor, prefix: String, channels: Int) -> MPSGraphTensor {
        graph.normalize(
            input,
            mean: parameter("\(prefix).running_mean", [1, 1, 1, 1, channels], layout: .ndhwc),
            variance: parameter("\(prefix).running_var", [1, 1, 1, 1, channels], layout: .ndhwc),
            gamma: parameter("\(prefix).weight", [1, 1, 1, 1, channels], layout: .ndhwc),
            beta: parameter("\(prefix).bias", [1, 1, 1, 1, channels], layout: .ndhwc),
            epsilon: 1.0e-5,
            name: prefix
        )
    }

    private func batchNorm2D(_ input: MPSGraphTensor, prefix: String, channels: Int) -> MPSGraphTensor {
        graph.normalize(
            input,
            mean: parameter("\(prefix).running_mean", [1, 1, 1, channels], layout: .nhwc),
            variance: parameter("\(prefix).running_var", [1, 1, 1, channels], layout: .nhwc),
            gamma: parameter("\(prefix).weight", [1, 1, 1, channels], layout: .nhwc),
            beta: parameter("\(prefix).bias", [1, 1, 1, channels], layout: .nhwc),
            epsilon: 1.0e-5,
            name: prefix
        )
    }

    private func layerNorm(_ input: MPSGraphTensor, prefix: String, channels: Int) -> MPSGraphTensor {
        let mean = graph.mean(of: input, axes: [2], name: "\(prefix).mean")
        let variance = graph.variance(of: input, mean: mean, axes: [2], name: "\(prefix).variance")
        return graph.normalize(
            input,
            mean: mean,
            variance: variance,
            gamma: parameter("\(prefix).weight", [1, 1, channels], layout: .nhwc),
            beta: parameter("\(prefix).bias", [1, 1, channels], layout: .nhwc),
            epsilon: 1.0e-5,
            name: prefix
        )
    }

    private func linear(_ input: MPSGraphTensor, prefix: String, inChannels: Int, outChannels: Int) -> MPSGraphTensor {
        let x = graph.matrixMultiplication(primary: input, secondary: parameter("\(prefix).weight", [inChannels, outChannels], layout: .nhwc), name: "\(prefix).matmul")
        return graph.addition(x, parameter("\(prefix).bias", [1, 1, outChannels], layout: .nhwc), name: "\(prefix).bias")
    }

    private func gelu(_ input: MPSGraphTensor, name: String) -> MPSGraphTensor {
        let half = graph.constant(0.5, dataType: precision.dataType)
        let one = graph.constant(1.0, dataType: precision.dataType)
        let sqrtTwo = graph.constant(sqrt(2.0), dataType: precision.dataType)
        let scaled = graph.division(input, sqrtTwo, name: "\(name).scale")
        let erf = graph.erf(with: scaled, name: "\(name).erf")
        let cdf = graph.multiplication(half, graph.addition(one, erf, name: "\(name).one_plus_erf"), name: "\(name).cdf")
        return graph.multiplication(input, cdf, name: name)
    }

    private func activate(_ input: MPSGraphTensor, activation: Activation?, name: String) -> MPSGraphTensor {
        switch activation {
        case .relu:
            return graph.reLU(with: input, name: "\(name).relu")
        case .leakyReLU:
            return graph.leakyReLU(with: input, alpha: 0.01, name: "\(name).relu")
        case nil:
            return input
        }
    }

    private func placeholder(_ name: String, _ shape: [Int], layout: TensorLayout) -> MPSGraphTensor {
        let tensor = graph.placeholder(shape: shape.map { NSNumber(value: $0) }, dataType: .float32, name: name)
        inputTensors[name] = tensor
        return tensor
    }

    private func parameter(_ name: String, _ shape: [Int], layout: TensorLayout) -> MPSGraphTensor {
        parameterSpecs.append(NativeGraphTensorSpec(name: name, shape: TensorShape(shape, layout: layout)))
        let tensor = graph.placeholder(shape: shape.map { NSNumber(value: $0) }, dataType: precision.dataType, name: name)
        parameterTensors[name] = tensor
        return tensor
    }

    private func computeInput(_ tensor: MPSGraphTensor, name: String) -> MPSGraphTensor {
        guard precision == .float16 else {
            return tensor
        }
        return graph.cast(tensor, to: .float16, name: "\(name).fp16")
    }

    private func outputTensor(_ tensor: MPSGraphTensor, name: String) -> MPSGraphTensor {
        guard precision == .float16 else {
            return tensor
        }
        return graph.cast(tensor, to: .float32, name: "\(name).fp32")
    }

    private func tap(_ tensor: MPSGraphTensor, name: String, shape: [Int], layout: TensorLayout) {
        debugTensors[name] = tensor
        debugOutputSpecs.append(NativeGraphTensorSpec(name: name, shape: TensorShape(shape, layout: layout)))
    }
}
