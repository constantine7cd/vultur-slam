#if canImport(Metal)
import CoreVideo
import Foundation
import Metal

public final class MetalRuntime: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let textureCache: CVMetalTextureCache

    public init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        guard let device else {
            throw MetalRuntimeError.deviceUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalRuntimeError.commandQueueUnavailable
        }

        var cache: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(
            kCFAllocatorDefault,
            nil,
            device,
            nil,
            &cache
        )
        guard status == kCVReturnSuccess, let cache else {
            throw MetalRuntimeError.textureCacheUnavailable(status)
        }

        self.device = device
        self.commandQueue = commandQueue
        self.textureCache = cache
    }

    public func makeTexture(from pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat = .bgra8Unorm) throws -> MTLTexture {
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            pixelFormat,
            CVPixelBufferGetWidth(pixelBuffer),
            CVPixelBufferGetHeight(pixelBuffer),
            0,
            &cvTexture
        )

        guard status == kCVReturnSuccess, let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else {
            throw MetalRuntimeError.textureCreationFailed(status)
        }

        return texture
    }
}

public enum MetalRuntimeError: Error, Equatable, CustomStringConvertible {
    case deviceUnavailable
    case commandQueueUnavailable
    case textureCacheUnavailable(CVReturn)
    case textureCreationFailed(CVReturn)

    public var description: String {
        switch self {
        case .deviceUnavailable:
            return "Metal device unavailable"
        case .commandQueueUnavailable:
            return "Metal command queue unavailable"
        case let .textureCacheUnavailable(status):
            return "CVMetalTextureCacheCreate failed with status \(status)"
        case let .textureCreationFailed(status):
            return "CVMetalTextureCacheCreateTextureFromImage failed with status \(status)"
        }
    }
}
#endif
