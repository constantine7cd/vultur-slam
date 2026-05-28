import Foundation
import Metal

public final class TensorArena {
    public let device: MTLDevice
    private var tensors: [String: MetalTensor] = [:]

    public init(device: MTLDevice) {
        self.device = device
    }

    @discardableResult
    public func allocate(name: String, shape: TensorShape) throws -> MetalTensor {
        if let tensor = tensors[name] {
            precondition(tensor.shape == shape, "Tensor \(name) was requested with a different shape.")
            return tensor
        }
        guard let buffer = device.makeBuffer(length: shape.byteCountFP32, options: [.storageModeShared]) else {
            throw FastFoundationStereoMPSError.allocationFailed(name)
        }
        let tensor = MetalTensor(name: name, shape: shape, buffer: buffer)
        tensors[name] = tensor
        return tensor
    }

    public func tensor(named name: String) -> MetalTensor {
        guard let tensor = tensors[name] else {
            preconditionFailure("Missing tensor \(name).")
        }
        return tensor
    }

    public func removeAll() {
        tensors.removeAll(keepingCapacity: true)
    }
}

public enum FastFoundationStereoMPSError: Error, CustomStringConvertible {
    case noMetalDevice
    case commandQueueUnavailable
    case allocationFailed(String)
    case libraryLoadFailed(String)
    case functionMissing(String)
    case coreMLModelLoadFailed(URL)
    case coreMLOutputMissing(String)
    case invalidShape(String)
    case weightMissing(String)
    case graphNotCompiled(String)
    case graphInputMissing(String)
    case imageLoadFailed(URL)
    case resourceMissing(String)

    public var description: String {
        switch self {
        case .noMetalDevice:
            return "No Metal device is available."
        case .commandQueueUnavailable:
            return "Metal command queue could not be created."
        case .allocationFailed(let name):
            return "Could not allocate Metal buffer for \(name)."
        case .libraryLoadFailed(let message):
            return "Could not load Metal library: \(message)"
        case .functionMissing(let name):
            return "Metal function \(name) was not found."
        case .coreMLModelLoadFailed(let url):
            return "Could not load Core ML model at \(url.path)."
        case .coreMLOutputMissing(let name):
            return "Core ML output \(name) is missing."
        case .invalidShape(let message):
            return "Invalid tensor shape: \(message)"
        case .weightMissing(let name):
            return "Native weight \(name) is missing from the exported manifest."
        case .graphNotCompiled(let name):
            return "MPSGraph \(name) is not compiled."
        case .graphInputMissing(let name):
            return "MPSGraph input \(name) is not declared."
        case .imageLoadFailed(let url):
            return "Could not load image at \(url.path)."
        case .resourceMissing(let name):
            return "Required FastFoundationStereoMPS resource is missing: \(name)."
        }
    }
}
