import Foundation
import Metal

public enum TensorLayout: String, Codable, Sendable {
    case nchw
    case nhwc
    case ndhwc
}

public struct TensorShape: Equatable, Codable, Sendable {
    public let dimensions: [Int]
    public let layout: TensorLayout

    public init(_ dimensions: [Int], layout: TensorLayout) {
        self.dimensions = dimensions
        self.layout = layout
    }

    public var elementCount: Int {
        dimensions.reduce(1, *)
    }

    public var byteCountFP32: Int {
        elementCount * MemoryLayout<Float>.stride
    }
}

public final class MetalTensor {
    public let name: String
    public let shape: TensorShape
    public let buffer: MTLBuffer

    public init(name: String, shape: TensorShape, buffer: MTLBuffer) {
        self.name = name
        self.shape = shape
        self.buffer = buffer
    }
}

public enum FastFoundationStereoMPSConstants {
    public static let batch = 1
    public static let imageHeight = 480
    public static let imageWidth = 640
    public static let quarterHeight = 120
    public static let quarterWidth = 160
    public static let halfHeight = 240
    public static let halfWidth = 320
    public static let maxDisparity = 192
    public static let disparityQuarter = 48
    public static let correlationGroups = 8
    public static let correlationRadius = 4
    public static let correlationLevels = 2
    public static let combinedVolumeChannels = 32
    public static let regularizedVolumeChannels = 28
}

