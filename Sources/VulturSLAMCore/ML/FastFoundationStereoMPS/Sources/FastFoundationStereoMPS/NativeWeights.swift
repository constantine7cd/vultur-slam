import Foundation
import Metal
import MetalPerformanceShadersGraph

public final class NativeWeights {
    private struct Manifest: Decodable {
        let tensors: [ManifestTensor]
    }

    private struct ManifestTensor: Decodable {
        let name: String
        let source: String?
        let file: String
        let shape: [Int]
        let layout: String
    }

    private let directory: URL
    private let device: MTLDevice
    private let tensorsByName: [String: ManifestTensor]
    private var buffers: [String: MTLBuffer] = [:]
    private var fp16Buffers: [String: MTLBuffer] = [:]

    public init(directory: URL, device: MTLDevice) throws {
        self.directory = directory
        self.device = device
        let manifestURL = directory.appendingPathComponent("manifest.json")
        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        self.tensorsByName = Dictionary(uniqueKeysWithValues: manifest.tensors.map { ($0.name, $0) })
    }

    public func tensorData(for spec: NativeGraphTensorSpec, precision: NativeGraphPrecision = .float32) throws -> MPSGraphTensorData {
        let tensor = try manifestTensor(for: spec)
        let buffer: MTLBuffer
        switch precision {
        case .float32:
            buffer = try metalBuffer(for: tensor)
        case .float16:
            buffer = try fp16MetalBuffer(for: tensor)
        }
        return MPSGraphTensorData(buffer, shape: mpsShape(spec.shape), dataType: precision.dataType)
    }

    private func manifestTensor(for spec: NativeGraphTensorSpec) throws -> ManifestTensor {
        let candidates = candidateNames(for: spec)
        for name in candidates {
            guard let tensor = tensorsByName[name] else {
                continue
            }
            if tensor.shape.reduce(1, *) == spec.shape.elementCount {
                return tensor
            }
        }
        throw FastFoundationStereoMPSError.weightMissing(spec.name)
    }

    private func candidateNames(for spec: NativeGraphTensorSpec) -> [String] {
        if spec.name.hasSuffix("pos_embed0.pe") {
            return ["\(spec.name).d12", spec.name]
        }
        guard spec.name.hasSuffix(".weight") else {
            return [spec.name]
        }
        switch spec.shape.dimensions.count {
        case 2:
            return ["\(spec.name).io", spec.name]
        case 4:
            return ["\(spec.name).hwio", "\(spec.name).hwoi_transpose", spec.name]
        case 5:
            return ["\(spec.name).dhwio", "\(spec.name).dhwoi_transpose", spec.name]
        default:
            return [spec.name]
        }
    }

    private func metalBuffer(for tensor: ManifestTensor) throws -> MTLBuffer {
        if let buffer = buffers[tensor.name] {
            return buffer
        }
        let url = directory.appendingPathComponent(tensor.file)
        let data = try Data(contentsOf: url)
        guard data.count == tensor.shape.reduce(1, *) * MemoryLayout<Float>.stride else {
            throw FastFoundationStereoMPSError.invalidShape("\(tensor.name) has \(data.count) bytes")
        }
        guard let buffer = device.makeBuffer(bytes: Array(data), length: data.count, options: [.storageModeShared]) else {
            throw FastFoundationStereoMPSError.allocationFailed(tensor.name)
        }
        buffers[tensor.name] = buffer
        return buffer
    }

    private func fp16MetalBuffer(for tensor: ManifestTensor) throws -> MTLBuffer {
        if let buffer = fp16Buffers[tensor.name] {
            return buffer
        }
        let url = directory.appendingPathComponent(tensor.file)
        let data = try Data(contentsOf: url)
        let elementCount = tensor.shape.reduce(1, *)
        guard data.count == elementCount * MemoryLayout<Float>.stride else {
            throw FastFoundationStereoMPSError.invalidShape("\(tensor.name) has \(data.count) bytes")
        }
        let fp16Values: [Float16] = data.withUnsafeBytes { rawBuffer in
            let values = rawBuffer.bindMemory(to: Float.self)
            return values.map { Float16($0) }
        }
        guard let buffer = fp16Values.withUnsafeBufferPointer({ pointer in
            device.makeBuffer(
                bytes: pointer.baseAddress!,
                length: fp16Values.count * MemoryLayout<Float16>.stride,
                options: [.storageModeShared]
            )
        }) else {
            throw FastFoundationStereoMPSError.allocationFailed(tensor.name)
        }
        fp16Buffers[tensor.name] = buffer
        return buffer
    }
}

func mpsShape(_ shape: TensorShape) -> [NSNumber] {
    shape.dimensions.map { NSNumber(value: $0) }
}
