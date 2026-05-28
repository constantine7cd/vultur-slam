import CoreGraphics
import CoreML
import Foundation
import ImageIO

public enum ImageTensorIO {
    public static func loadRGBTensors(leftURL: URL, rightURL: URL) throws -> StereoInputTensors {
        try StereoInputTensors(
            left: loadRGBTensor(url: leftURL),
            right: loadRGBTensor(url: rightURL)
        )
    }

    public static func loadRGBTensor(url: URL, height: Int = 480, width: Int = 640) throws -> MLMultiArray {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw FastFoundationStereoMPSError.imageLoadFailed(url)
        }
        let sourceWidth = image.width
        let sourceHeight = image.height
        var rgba = [UInt8](repeating: 0, count: sourceWidth * sourceHeight * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &rgba,
            width: sourceWidth,
            height: sourceHeight,
            bitsPerComponent: 8,
            bytesPerRow: sourceWidth * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw FastFoundationStereoMPSError.imageLoadFailed(url)
        }
        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: sourceWidth, height: sourceHeight))

        let tensor = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        let values = tensor.dataPointer.bindMemory(to: Float.self, capacity: 3 * height * width)
        let xScale = Double(sourceWidth) / Double(width)
        let yScale = Double(sourceHeight) / Double(height)
        for y in 0..<height {
            let sourceY = (Double(y) + 0.5) * yScale - 0.5
            let y0 = max(0, min(sourceHeight - 1, Int(floor(sourceY))))
            let y1 = max(0, min(sourceHeight - 1, y0 + 1))
            let yWeight = Float(sourceY - floor(sourceY))
            for x in 0..<width {
                let sourceX = (Double(x) + 0.5) * xScale - 0.5
                let x0 = max(0, min(sourceWidth - 1, Int(floor(sourceX))))
                let x1 = max(0, min(sourceWidth - 1, x0 + 1))
                let xWeight = Float(sourceX - floor(sourceX))
                let pixel00 = (y0 * sourceWidth + x0) * 4
                let pixel01 = (y0 * sourceWidth + x1) * 4
                let pixel10 = (y1 * sourceWidth + x0) * 4
                let pixel11 = (y1 * sourceWidth + x1) * 4
                let tensorPixel = y * width + x
                for channel in 0..<3 {
                    let top = Float(rgba[pixel00 + channel]) * (1 - xWeight) + Float(rgba[pixel01 + channel]) * xWeight
                    let bottom = Float(rgba[pixel10 + channel]) * (1 - xWeight) + Float(rgba[pixel11 + channel]) * xWeight
                    let resized = top * (1 - yWeight) + bottom * yWeight
                    values[channel * height * width + tensorPixel] = resized.rounded()
                }
            }
        }
        return tensor
    }

    public static func writeFP32Tensor(_ tensor: MetalTensor, url: URL) throws {
        let data = Data(bytes: tensor.buffer.contents(), count: tensor.shape.byteCountFP32)
        try data.write(to: url)
    }
}
