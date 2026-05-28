import CoreVideo
import Foundation

public struct FrameMetadata: Equatable, Sendable {
    public var frameIndex: Int
    public var timestampSeconds: Double
    public var width: Int
    public var height: Int

    public init(frameIndex: Int, timestampSeconds: Double, width: Int, height: Int) {
        self.frameIndex = frameIndex
        self.timestampSeconds = timestampSeconds
        self.width = width
        self.height = height
    }
}

public final class StereoFrame: @unchecked Sendable {
    public let metadata: FrameMetadata
    public let leftPixelBuffer: CVPixelBuffer?
    public let rightPixelBuffer: CVPixelBuffer?
    public let leftSourceURL: URL?
    public let rightSourceURL: URL?

    public init(
        metadata: FrameMetadata,
        leftPixelBuffer: CVPixelBuffer? = nil,
        rightPixelBuffer: CVPixelBuffer? = nil,
        leftSourceURL: URL? = nil,
        rightSourceURL: URL? = nil
    ) {
        self.metadata = metadata
        self.leftPixelBuffer = leftPixelBuffer
        self.rightPixelBuffer = rightPixelBuffer
        self.leftSourceURL = leftSourceURL
        self.rightSourceURL = rightSourceURL
    }
}

public struct BorrowedBufferView: Equatable, Sendable {
    public var address: UInt
    public var byteCount: Int
    public var strideBytes: Int

    public init(address: UInt, byteCount: Int, strideBytes: Int) {
        self.address = address
        self.byteCount = byteCount
        self.strideBytes = strideBytes
    }
}

public enum PixelBufferFactory {
    public static func makeIOSurfaceBackedBGRA(width: Int, height: Int) throws -> CVPixelBuffer {
        let attributes: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw PixelBufferError.creationFailed(status)
        }

        return pixelBuffer
    }
}

public enum PixelBufferError: Error, Equatable, CustomStringConvertible {
    case creationFailed(CVReturn)

    public var description: String {
        switch self {
        case let .creationFailed(status):
            return "CVPixelBufferCreate failed with status \(status)"
        }
    }
}
