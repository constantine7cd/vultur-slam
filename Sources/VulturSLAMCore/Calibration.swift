import Foundation

public struct CameraIntrinsics: Codable, Equatable, Sendable {
    public var fx: Double
    public var fy: Double
    public var cx: Double
    public var cy: Double

    public init(fx: Double, fy: Double, cx: Double, cy: Double) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    }
}

public struct CameraModel: Codable, Equatable, Sendable {
    public var intrinsics: CameraIntrinsics
    public var distortion: [Double]
    public var width: Int
    public var height: Int

    public init(intrinsics: CameraIntrinsics, distortion: [Double], width: Int, height: Int) {
        self.intrinsics = intrinsics
        self.distortion = distortion
        self.width = width
        self.height = height
    }
}

public struct StereoCalibration: Codable, Equatable, Sendable {
    public var left: CameraModel
    public var right: CameraModel
    public var rotationRightToLeft: [Double]
    public var translationRightToLeft: [Double]
    public var baselineMeters: Double
    public var rectificationScale: Double

    public init(
        left: CameraModel,
        right: CameraModel,
        rotationRightToLeft: [Double],
        translationRightToLeft: [Double],
        baselineMeters: Double,
        rectificationScale: Double
    ) {
        self.left = left
        self.right = right
        self.rotationRightToLeft = rotationRightToLeft
        self.translationRightToLeft = translationRightToLeft
        self.baselineMeters = baselineMeters
        self.rectificationScale = rectificationScale
    }

    public static func load(from url: URL) throws -> StereoCalibration {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(StereoCalibration.self, from: data)
    }

    public func validate() throws {
        try validateCamera(left, name: "left")
        try validateCamera(right, name: "right")

        guard left.width == right.width, left.height == right.height else {
            throw CalibrationError.mismatchedImageSize
        }
        guard rotationRightToLeft.count == 9 else {
            throw CalibrationError.invalidRotationElementCount(rotationRightToLeft.count)
        }
        guard translationRightToLeft.count == 3 else {
            throw CalibrationError.invalidTranslationElementCount(translationRightToLeft.count)
        }
        guard baselineMeters > 0 else {
            throw CalibrationError.invalidBaseline(baselineMeters)
        }
        guard rectificationScale > 0 else {
            throw CalibrationError.invalidRectificationScale(rectificationScale)
        }
    }

    private func validateCamera(_ camera: CameraModel, name: String) throws {
        guard camera.width > 0, camera.height > 0 else {
            throw CalibrationError.invalidImageSize(camera: name, width: camera.width, height: camera.height)
        }
        guard camera.intrinsics.fx > 0, camera.intrinsics.fy > 0 else {
            throw CalibrationError.invalidFocalLength(camera: name)
        }
    }
}

public enum CalibrationError: Error, Equatable, CustomStringConvertible {
    case invalidImageSize(camera: String, width: Int, height: Int)
    case invalidFocalLength(camera: String)
    case mismatchedImageSize
    case invalidRotationElementCount(Int)
    case invalidTranslationElementCount(Int)
    case invalidBaseline(Double)
    case invalidRectificationScale(Double)

    public var description: String {
        switch self {
        case let .invalidImageSize(camera, width, height):
            return "\(camera) camera has invalid image size \(width)x\(height)"
        case let .invalidFocalLength(camera):
            return "\(camera) camera has invalid focal length"
        case .mismatchedImageSize:
            return "left and right image sizes must match"
        case let .invalidRotationElementCount(count):
            return "rotation_right_to_left must contain 9 values, got \(count)"
        case let .invalidTranslationElementCount(count):
            return "translation_right_to_left must contain 3 values, got \(count)"
        case let .invalidBaseline(value):
            return "baseline_meters must be positive, got \(value)"
        case let .invalidRectificationScale(value):
            return "rectification_scale must be positive, got \(value)"
        }
    }
}
