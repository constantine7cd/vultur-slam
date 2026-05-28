import FastFoundationStereoMPS
import Foundation
import Metal
import MetalPerformanceShadersGraph

private struct Options {
    let iterations: Int
    let warmup: Int
    let convolutionCount: Int
    let includeUInt8Direct: Bool

    init(arguments: [String]) {
        var iterations = 30
        var warmup = 5
        var convolutionCount = 2
        var includeUInt8Direct = false
        var iterator = arguments.dropFirst().makeIterator()
        while let argument = iterator.next() {
            if argument == "--iterations", let value = iterator.next(), let parsed = Int(value) {
                iterations = parsed
            } else if argument == "--warmup", let value = iterator.next(), let parsed = Int(value) {
                warmup = parsed
            } else if argument == "--conv-count", let value = iterator.next(), let parsed = Int(value) {
                convolutionCount = parsed
            } else if argument == "--include-u8-direct" {
                includeUInt8Direct = true
            }
        }
        self.iterations = max(iterations, 1)
        self.warmup = max(warmup, 0)
        self.convolutionCount = min(max(convolutionCount, 1), 2)
        self.includeUInt8Direct = includeUInt8Direct
    }
}

private struct TensorDescriptor {
    let shape: [Int]
    let dataType: MPSDataType

    var elementCount: Int {
        shape.reduce(1, *)
    }

    var byteCount: Int {
        switch dataType {
        case .float32:
            return elementCount * MemoryLayout<Float>.stride
        case .float16, .bFloat16:
            return elementCount * MemoryLayout<Float16>.stride
        case .uInt8, .int8:
            return elementCount * MemoryLayout<UInt8>.stride
        default:
            preconditionFailure("Unsupported benchmark data type \(dataType)")
        }
    }
}

private struct BenchmarkBuffers {
    let input: MPSGraphTensorData
    let weight0: MPSGraphTensorData
    let bias0: MPSGraphTensorData
    let weight1: MPSGraphTensorData
    let bias1: MPSGraphTensorData
    let output: MPSGraphTensorData
}

private struct QuantizedBenchmarkBuffers {
    let input: MPSGraphTensorData
    let weight0: MPSGraphTensorData
    let bias0: MPSGraphTensorData
    let weight1: MPSGraphTensorData
    let bias1: MPSGraphTensorData
    let output: MPSGraphTensorData
}

private struct DirectIntegerBenchmarkBuffers {
    let input: MPSGraphTensorData
    let weight0: MPSGraphTensorData
    let weight1: MPSGraphTensorData
    let output: MPSGraphTensorData
}

private final class Conv3DBenchmarkGraph {
    let graph = MPSGraph()
    let input: MPSGraphTensor
    let weight0: MPSGraphTensor
    let bias0: MPSGraphTensor
    let weight1: MPSGraphTensor
    let bias1: MPSGraphTensor
    let output: MPSGraphTensor

    init(dataType: MPSDataType, convolutionCount: Int) {
        let disparity = FastFoundationStereoMPSConstants.disparityQuarter
        let height = FastFoundationStereoMPSConstants.quarterHeight
        let width = FastFoundationStereoMPSConstants.quarterWidth
        input = graph.placeholder(
            shape: [1, disparity, height, width, 32] as [NSNumber],
            dataType: dataType,
            name: "combined_volume"
        )
        weight0 = graph.placeholder(shape: [3, 3, 3, 32, 28] as [NSNumber], dataType: dataType, name: "corr_feature_att.layers.0.conv.weight")
        bias0 = graph.placeholder(shape: [1, 1, 1, 1, 28] as [NSNumber], dataType: dataType, name: "corr_feature_att.layers.0.conv.bias")
        weight1 = graph.placeholder(shape: [3, 3, 3, 28, 28] as [NSNumber], dataType: dataType, name: "corr_feature_att.layers.1.conv.weight")
        bias1 = graph.placeholder(shape: [1, 1, 1, 1, 28] as [NSNumber], dataType: dataType, name: "corr_feature_att.layers.1.conv.bias")

        var x = Conv3DBenchmarkGraph.convolution3D(graph: graph, input: input, weights: weight0, name: "corr_feature_att.layers.0.conv")
        x = graph.addition(x, bias0, name: "corr_feature_att.layers.0.bias")
        x = graph.leakyReLU(with: x, alpha: 0.01, name: "corr_feature_att.layers.0.relu")
        if convolutionCount == 2 {
            x = Conv3DBenchmarkGraph.convolution3D(graph: graph, input: x, weights: weight1, name: "corr_feature_att.layers.1.conv")
            x = graph.addition(x, bias1, name: "corr_feature_att.layers.1.bias")
            x = graph.leakyReLU(with: x, alpha: 0.01, name: "corr_feature_att.layers.1.relu")
        }
        output = x
    }

    fileprivate static func convolution3D(graph: MPSGraph, input: MPSGraphTensor, weights: MPSGraphTensor, name: String) -> MPSGraphTensor {
        let descriptor = MPSGraphConvolution3DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            strideInZ: 1,
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
        return graph.convolution3D(input, weights: weights, descriptor: descriptor, name: name)
    }

    func run(commandQueue: MTLCommandQueue, buffers: BenchmarkBuffers) {
        graph.run(
            with: commandQueue,
            feeds: [
                input: buffers.input,
                weight0: buffers.weight0,
                bias0: buffers.bias0,
                weight1: buffers.weight1,
                bias1: buffers.bias1,
            ],
            targetOperations: nil,
            resultsDictionary: [output: buffers.output]
        )
    }
}

private final class DirectIntegerConv3DBenchmarkGraph {
    let graph = MPSGraph()
    let input: MPSGraphTensor
    let weight0: MPSGraphTensor
    let weight1: MPSGraphTensor
    let output: MPSGraphTensor

    init(dataType: MPSDataType, convolutionCount: Int) {
        let disparity = FastFoundationStereoMPSConstants.disparityQuarter
        let height = FastFoundationStereoMPSConstants.quarterHeight
        let width = FastFoundationStereoMPSConstants.quarterWidth
        input = graph.placeholder(
            shape: [1, disparity, height, width, 32] as [NSNumber],
            dataType: dataType,
            name: "combined_volume.integer.direct"
        )
        weight0 = graph.placeholder(shape: [3, 3, 3, 32, 28] as [NSNumber], dataType: dataType, name: "conv0.weight.integer.direct")
        weight1 = graph.placeholder(shape: [3, 3, 3, 28, 28] as [NSNumber], dataType: dataType, name: "conv1.weight.integer.direct")

        var x = Conv3DBenchmarkGraph.convolution3D(graph: graph, input: input, weights: weight0, name: "conv0.integer.direct")
        if convolutionCount == 2 {
            x = Conv3DBenchmarkGraph.convolution3D(graph: graph, input: x, weights: weight1, name: "conv1.integer.direct")
        }
        output = x
    }

    func run(commandQueue: MTLCommandQueue, buffers: DirectIntegerBenchmarkBuffers) {
        graph.run(
            with: commandQueue,
            feeds: [
                input: buffers.input,
                weight0: buffers.weight0,
                weight1: buffers.weight1,
            ],
            targetOperations: nil,
            resultsDictionary: [output: buffers.output]
        )
    }
}

private final class QuantizedDequantizedConv3DBenchmarkGraph {
    let graph = MPSGraph()
    let input: MPSGraphTensor
    let weight0: MPSGraphTensor
    let bias0: MPSGraphTensor
    let weight1: MPSGraphTensor
    let bias1: MPSGraphTensor
    let output: MPSGraphTensor

    init(computeType: MPSDataType, convolutionCount: Int) {
        let disparity = FastFoundationStereoMPSConstants.disparityQuarter
        let height = FastFoundationStereoMPSConstants.quarterHeight
        let width = FastFoundationStereoMPSConstants.quarterWidth
        input = graph.placeholder(
            shape: [1, disparity, height, width, 32] as [NSNumber],
            dataType: .uInt8,
            name: "combined_volume.u8"
        )
        weight0 = graph.placeholder(shape: [3, 3, 3, 32, 28] as [NSNumber], dataType: .uInt8, name: "corr_feature_att.layers.0.conv.weight.u8")
        bias0 = graph.placeholder(shape: [1, 1, 1, 1, 28] as [NSNumber], dataType: computeType, name: "corr_feature_att.layers.0.conv.bias")
        weight1 = graph.placeholder(shape: [3, 3, 3, 28, 28] as [NSNumber], dataType: .uInt8, name: "corr_feature_att.layers.1.conv.weight.u8")
        bias1 = graph.placeholder(shape: [1, 1, 1, 1, 28] as [NSNumber], dataType: computeType, name: "corr_feature_att.layers.1.conv.bias")

        var x = graph.dequantize(input, scale: 1.0 / 255.0, zeroPoint: 128.0, dataType: computeType, name: "combined_volume.dequant")
        let w0 = graph.dequantize(weight0, scale: 1.0 / 255.0, zeroPoint: 128.0, dataType: computeType, name: "corr_feature_att.layers.0.conv.weight.dequant")
        x = Conv3DBenchmarkGraph.convolution3D(graph: graph, input: x, weights: w0, name: "corr_feature_att.layers.0.conv.qdq")
        x = graph.addition(x, bias0, name: "corr_feature_att.layers.0.bias.qdq")
        x = graph.leakyReLU(with: x, alpha: 0.01, name: "corr_feature_att.layers.0.relu.qdq")
        if convolutionCount == 2 {
            let w1 = graph.dequantize(weight1, scale: 1.0 / 255.0, zeroPoint: 128.0, dataType: computeType, name: "corr_feature_att.layers.1.conv.weight.dequant")
            x = Conv3DBenchmarkGraph.convolution3D(graph: graph, input: x, weights: w1, name: "corr_feature_att.layers.1.conv.qdq")
            x = graph.addition(x, bias1, name: "corr_feature_att.layers.1.bias.qdq")
            x = graph.leakyReLU(with: x, alpha: 0.01, name: "corr_feature_att.layers.1.relu.qdq")
        }
        output = graph.quantize(x, scale: 1.0 / 255.0, zeroPoint: 128.0, dataType: .uInt8, name: "output.quant")
    }

    func run(commandQueue: MTLCommandQueue, buffers: QuantizedBenchmarkBuffers) {
        graph.run(
            with: commandQueue,
            feeds: [
                input: buffers.input,
                weight0: buffers.weight0,
                bias0: buffers.bias0,
                weight1: buffers.weight1,
                bias1: buffers.bias1,
            ],
            targetOperations: nil,
            resultsDictionary: [output: buffers.output]
        )
    }
}

private func makeBuffer(device: MTLDevice, descriptor: TensorDescriptor, seed: UInt32) throws -> MPSGraphTensorData {
    guard let buffer = device.makeBuffer(length: descriptor.byteCount, options: [.storageModeShared]) else {
        throw FastFoundationStereoMPSError.allocationFailed("conv3d-benchmark")
    }
    fill(buffer: buffer, descriptor: descriptor, seed: seed)
    return MPSGraphTensorData(buffer, shape: descriptor.shape.map { NSNumber(value: $0) }, dataType: descriptor.dataType)
}

private func fill(buffer: MTLBuffer, descriptor: TensorDescriptor, seed: UInt32) {
    var state = seed
    func nextValue() -> Float {
        state = 1_664_525 &* state &+ 1_013_904_223
        let unit = Float(state & 0x00ff_ffff) / Float(0x00ff_ffff)
        return (unit - 0.5) * 0.125
    }

    switch descriptor.dataType {
    case .float32:
        let values = buffer.contents().bindMemory(to: Float.self, capacity: descriptor.elementCount)
        for index in 0..<descriptor.elementCount {
            values[index] = nextValue()
        }
    case .float16:
        let values = buffer.contents().bindMemory(to: Float16.self, capacity: descriptor.elementCount)
        for index in 0..<descriptor.elementCount {
            values[index] = Float16(nextValue())
        }
    case .bFloat16:
        let values = buffer.contents().bindMemory(to: UInt16.self, capacity: descriptor.elementCount)
        for index in 0..<descriptor.elementCount {
            values[index] = bfloat16Bits(nextValue())
        }
    case .uInt8:
        let values = buffer.contents().bindMemory(to: UInt8.self, capacity: descriptor.elementCount)
        for index in 0..<descriptor.elementCount {
            values[index] = UInt8(truncatingIfNeeded: UInt32((nextValue() + 0.0625) * 2040.0))
        }
    case .int8:
        let values = buffer.contents().bindMemory(to: Int8.self, capacity: descriptor.elementCount)
        for index in 0..<descriptor.elementCount {
            values[index] = Int8(truncatingIfNeeded: Int((nextValue() + 0.0625) * 1020.0) - 64)
        }
    default:
        preconditionFailure("Unsupported benchmark data type \(descriptor.dataType)")
    }
}

private func bfloat16Bits(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let roundingBias = ((bits >> 16) & 1) + 0x7fff
    return UInt16((bits &+ roundingBias) >> 16)
}

private func makeBuffers(device: MTLDevice, dataType: MPSDataType) throws -> BenchmarkBuffers {
    let disparity = FastFoundationStereoMPSConstants.disparityQuarter
    let height = FastFoundationStereoMPSConstants.quarterHeight
    let width = FastFoundationStereoMPSConstants.quarterWidth
    return BenchmarkBuffers(
        input: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [1, disparity, height, width, 32], dataType: dataType), seed: 1),
        weight0: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [3, 3, 3, 32, 28], dataType: dataType), seed: 2),
        bias0: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [1, 1, 1, 1, 28], dataType: dataType), seed: 3),
        weight1: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [3, 3, 3, 28, 28], dataType: dataType), seed: 4),
        bias1: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [1, 1, 1, 1, 28], dataType: dataType), seed: 5),
        output: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [1, disparity, height, width, 28], dataType: dataType), seed: 6)
    )
}

private func makeQuantizedBuffers(device: MTLDevice, computeType: MPSDataType) throws -> QuantizedBenchmarkBuffers {
    let disparity = FastFoundationStereoMPSConstants.disparityQuarter
    let height = FastFoundationStereoMPSConstants.quarterHeight
    let width = FastFoundationStereoMPSConstants.quarterWidth
    return QuantizedBenchmarkBuffers(
        input: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [1, disparity, height, width, 32], dataType: .uInt8), seed: 1),
        weight0: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [3, 3, 3, 32, 28], dataType: .uInt8), seed: 2),
        bias0: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [1, 1, 1, 1, 28], dataType: computeType), seed: 3),
        weight1: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [3, 3, 3, 28, 28], dataType: .uInt8), seed: 4),
        bias1: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [1, 1, 1, 1, 28], dataType: computeType), seed: 5),
        output: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [1, disparity, height, width, 28], dataType: .uInt8), seed: 6)
    )
}

private func makeDirectIntegerBuffers(device: MTLDevice, dataType: MPSDataType) throws -> DirectIntegerBenchmarkBuffers {
    let disparity = FastFoundationStereoMPSConstants.disparityQuarter
    let height = FastFoundationStereoMPSConstants.quarterHeight
    let width = FastFoundationStereoMPSConstants.quarterWidth
    return DirectIntegerBenchmarkBuffers(
        input: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [1, disparity, height, width, 32], dataType: dataType), seed: 1),
        weight0: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [3, 3, 3, 32, 28], dataType: dataType), seed: 2),
        weight1: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [3, 3, 3, 28, 28], dataType: dataType), seed: 4),
        output: try makeBuffer(device: device, descriptor: TensorDescriptor(shape: [1, disparity, height, width, 28], dataType: dataType), seed: 6)
    )
}

private func measure(
    label: String,
    graph: Conv3DBenchmarkGraph,
    commandQueue: MTLCommandQueue,
    buffers: BenchmarkBuffers,
    warmup: Int,
    iterations: Int
) {
    for _ in 0..<warmup {
        graph.run(commandQueue: commandQueue, buffers: buffers)
    }

    var samples: [Double] = []
    samples.reserveCapacity(iterations)
    for _ in 0..<iterations {
        let start = CFAbsoluteTimeGetCurrent()
        graph.run(commandQueue: commandQueue, buffers: buffers)
        samples.append((CFAbsoluteTimeGetCurrent() - start) * 1000.0)
    }

    let sorted = samples.sorted()
    let average = samples.reduce(0.0, +) / Double(samples.count)
    let median = sorted[sorted.count / 2]
    let minValue = sorted.first ?? 0.0
    let maxValue = sorted.last ?? 0.0
    print(String(format: "%@: avg %.3f ms, median %.3f ms, min %.3f ms, max %.3f ms", label, average, median, minValue, maxValue))
}

private func measureDirectInteger(
    label: String,
    graph: DirectIntegerConv3DBenchmarkGraph,
    commandQueue: MTLCommandQueue,
    buffers: DirectIntegerBenchmarkBuffers,
    warmup: Int,
    iterations: Int
) {
    for _ in 0..<warmup {
        graph.run(commandQueue: commandQueue, buffers: buffers)
    }

    var samples: [Double] = []
    samples.reserveCapacity(iterations)
    for _ in 0..<iterations {
        let start = CFAbsoluteTimeGetCurrent()
        graph.run(commandQueue: commandQueue, buffers: buffers)
        samples.append((CFAbsoluteTimeGetCurrent() - start) * 1000.0)
    }

    let sorted = samples.sorted()
    let average = samples.reduce(0.0, +) / Double(samples.count)
    let median = sorted[sorted.count / 2]
    let minValue = sorted.first ?? 0.0
    let maxValue = sorted.last ?? 0.0
    print(String(format: "%@: avg %.3f ms, median %.3f ms, min %.3f ms, max %.3f ms", label, average, median, minValue, maxValue))
}

private func measureQuantized(
    label: String,
    graph: QuantizedDequantizedConv3DBenchmarkGraph,
    commandQueue: MTLCommandQueue,
    buffers: QuantizedBenchmarkBuffers,
    warmup: Int,
    iterations: Int
) {
    for _ in 0..<warmup {
        graph.run(commandQueue: commandQueue, buffers: buffers)
    }

    var samples: [Double] = []
    samples.reserveCapacity(iterations)
    for _ in 0..<iterations {
        let start = CFAbsoluteTimeGetCurrent()
        graph.run(commandQueue: commandQueue, buffers: buffers)
        samples.append((CFAbsoluteTimeGetCurrent() - start) * 1000.0)
    }

    let sorted = samples.sorted()
    let average = samples.reduce(0.0, +) / Double(samples.count)
    let median = sorted[sorted.count / 2]
    let minValue = sorted.first ?? 0.0
    let maxValue = sorted.last ?? 0.0
    print(String(format: "%@: avg %.3f ms, median %.3f ms, min %.3f ms, max %.3f ms", label, average, median, minValue, maxValue))
}

private let options = Options(arguments: CommandLine.arguments)
guard let device = MTLCreateSystemDefaultDevice() else {
    throw FastFoundationStereoMPSError.noMetalDevice
}
guard let commandQueue = device.makeCommandQueue() else {
    throw FastFoundationStereoMPSError.commandQueueUnavailable
}

print("FastFoundationStereoMPS Conv3D benchmark on \(device.name)")
print("Shape: input [1, 48, 120, 160, 32], output [1, 48, 120, 160, 28], conv-count \(options.convolutionCount)")
print("Warmup: \(options.warmup), iterations: \(options.iterations)")

private let fp32Graph = Conv3DBenchmarkGraph(dataType: .float32, convolutionCount: options.convolutionCount)
private let fp16Graph = Conv3DBenchmarkGraph(dataType: .float16, convolutionCount: options.convolutionCount)
private let bf16Graph = Conv3DBenchmarkGraph(dataType: .bFloat16, convolutionCount: options.convolutionCount)
private let qdqFp16Graph = QuantizedDequantizedConv3DBenchmarkGraph(computeType: .float16, convolutionCount: options.convolutionCount)
private let fp32Buffers = try makeBuffers(device: device, dataType: .float32)
private let fp16Buffers = try makeBuffers(device: device, dataType: .float16)
private let bf16Buffers = try makeBuffers(device: device, dataType: .bFloat16)
private let qdqFp16Buffers = try makeQuantizedBuffers(device: device, computeType: .float16)

measure(label: "fp32 full-boundary", graph: fp32Graph, commandQueue: commandQueue, buffers: fp32Buffers, warmup: options.warmup, iterations: options.iterations)
measure(label: "fp16 full-boundary", graph: fp16Graph, commandQueue: commandQueue, buffers: fp16Buffers, warmup: options.warmup, iterations: options.iterations)
measure(label: "bf16 full-boundary", graph: bf16Graph, commandQueue: commandQueue, buffers: bf16Buffers, warmup: options.warmup, iterations: options.iterations)
measureQuantized(label: "u8 qdq -> fp16 conv -> u8", graph: qdqFp16Graph, commandQueue: commandQueue, buffers: qdqFp16Buffers, warmup: options.warmup, iterations: options.iterations)

if options.includeUInt8Direct {
    let u8DirectGraph = DirectIntegerConv3DBenchmarkGraph(dataType: .uInt8, convolutionCount: options.convolutionCount)
    let i8DirectGraph = DirectIntegerConv3DBenchmarkGraph(dataType: .int8, convolutionCount: options.convolutionCount)
    let u8DirectBuffers = try makeDirectIntegerBuffers(device: device, dataType: .uInt8)
    let i8DirectBuffers = try makeDirectIntegerBuffers(device: device, dataType: .int8)
    measureDirectInteger(label: "u8 direct conv only", graph: u8DirectGraph, commandQueue: commandQueue, buffers: u8DirectBuffers, warmup: options.warmup, iterations: options.iterations)
    measureDirectInteger(label: "i8 direct conv only", graph: i8DirectGraph, commandQueue: commandQueue, buffers: i8DirectBuffers, warmup: options.warmup, iterations: options.iterations)
}
