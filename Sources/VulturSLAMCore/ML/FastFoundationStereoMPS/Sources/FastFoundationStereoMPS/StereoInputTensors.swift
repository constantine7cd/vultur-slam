import CoreML
import Foundation

public struct StereoInputTensors {
    public let left: MLMultiArray
    public let right: MLMultiArray

    public init(left: MLMultiArray, right: MLMultiArray) throws {
        try Self.validateImageTensor(left, name: "left")
        try Self.validateImageTensor(right, name: "right")
        self.left = left
        self.right = right
    }

    public init(leftRGBNCHW: [Float], rightRGBNCHW: [Float]) throws {
        self.left = try Self.makeImageTensor(values: leftRGBNCHW, name: "left")
        self.right = try Self.makeImageTensor(values: rightRGBNCHW, name: "right")
    }

    public static var imageShape: TensorShape {
        TensorShape(
            [
                FastFoundationStereoMPSConstants.batch,
                3,
                FastFoundationStereoMPSConstants.imageHeight,
                FastFoundationStereoMPSConstants.imageWidth,
            ],
            layout: .nchw
        )
    }

    public static var imageElementCount: Int {
        imageShape.elementCount
    }

    public static func makeImageTensor(values: [Float], name: String = "image") throws -> MLMultiArray {
        guard values.count == imageElementCount else {
            throw FastFoundationStereoMPSError.invalidShape(
                "\(name) expected \(imageElementCount) float values, got \(values.count)"
            )
        }
        let tensor = try MLMultiArray(
            shape: [
                NSNumber(value: FastFoundationStereoMPSConstants.batch),
                3,
                NSNumber(value: FastFoundationStereoMPSConstants.imageHeight),
                NSNumber(value: FastFoundationStereoMPSConstants.imageWidth),
            ],
            dataType: .float32
        )
        let destination = tensor.dataPointer.bindMemory(to: Float.self, capacity: imageElementCount)
        values.withUnsafeBufferPointer { source in
            destination.update(from: source.baseAddress!, count: imageElementCount)
        }
        return tensor
    }

    public static func validateImageTensor(_ tensor: MLMultiArray, name: String = "image") throws {
        guard tensor.dataType == .float32 else {
            throw FastFoundationStereoMPSError.invalidShape("\(name) must be float32")
        }
        let expectedShape = imageShape.dimensions
        let actualShape = tensor.shape.map(\.intValue)
        guard actualShape == expectedShape else {
            throw FastFoundationStereoMPSError.invalidShape("\(name) expected shape \(expectedShape), got \(actualShape)")
        }
        let expectedStrides = [
            3 * FastFoundationStereoMPSConstants.imageHeight * FastFoundationStereoMPSConstants.imageWidth,
            FastFoundationStereoMPSConstants.imageHeight * FastFoundationStereoMPSConstants.imageWidth,
            FastFoundationStereoMPSConstants.imageWidth,
            1,
        ]
        let actualStrides = tensor.strides.map(\.intValue)
        guard actualStrides == expectedStrides else {
            throw FastFoundationStereoMPSError.invalidShape(
                "\(name) expected contiguous NCHW strides \(expectedStrides), got \(actualStrides)"
            )
        }
    }
}
