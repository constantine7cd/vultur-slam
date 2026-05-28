import Foundation
import Metal

public final class MetalContext {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary

    public init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        guard let device else {
            throw FastFoundationStereoMPSError.noMetalDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw FastFoundationStereoMPSError.commandQueueUnavailable
        }
        self.device = device
        self.commandQueue = queue

        let shaderURL = Bundle.module.url(
            forResource: "FastFoundationStereoKernels",
            withExtension: "metal"
        )
        guard let shaderURL else {
            throw FastFoundationStereoMPSError.libraryLoadFailed("shader resource is missing")
        }
        let source = try String(contentsOf: shaderURL, encoding: .utf8)
        do {
            self.library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw FastFoundationStereoMPSError.libraryLoadFailed(error.localizedDescription)
        }
    }

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw FastFoundationStereoMPSError.functionMissing(name)
        }
        return try device.makeComputePipelineState(function: function)
    }
}

