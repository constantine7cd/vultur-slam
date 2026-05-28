import Foundation

public protocol StereoFrameSource: Sendable {
    func frames() throws -> [StereoFrame]
}

public struct OfflineStereoSource: StereoFrameSource {
    public var left: URL
    public var right: URL
    public var calibration: StereoCalibration

    public init(left: URL, right: URL, calibration: StereoCalibration) {
        self.left = left
        self.right = right
        self.calibration = calibration
    }

    public func frames() throws -> [StereoFrame] {
        let leftFrames = try frameURLs(from: left)
        let rightFrames = try frameURLs(from: right)
        guard leftFrames.count == rightFrames.count else {
            throw StereoSourceError.mismatchedFrameCount(left: leftFrames.count, right: rightFrames.count)
        }

        return zip(leftFrames, rightFrames).enumerated().map { index, pair in
            StereoFrame(
                metadata: FrameMetadata(
                    frameIndex: index,
                    timestampSeconds: Double(index) / 30.0,
                    width: calibration.left.width,
                    height: calibration.left.height
                ),
                leftSourceURL: pair.0,
                rightSourceURL: pair.1
            )
        }
    }

    private func frameURLs(from url: URL) throws -> [URL] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw StereoSourceError.missingInput(url.path)
        }

        if !isDirectory.boolValue {
            return [url]
        }

        let urls = try FileManager.default.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: nil
        )
        .filter { !$0.lastPathComponent.hasPrefix(".") }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !urls.isEmpty else {
            throw StereoSourceError.emptyDirectory(url.path)
        }
        return urls
    }
}

public struct SyntheticStereoSource: StereoFrameSource {
    public var frameCount: Int
    public var width: Int
    public var height: Int

    public init(frameCount: Int, width: Int, height: Int) {
        self.frameCount = frameCount
        self.width = width
        self.height = height
    }

    public func frames() throws -> [StereoFrame] {
        guard frameCount >= 0 else {
            throw StereoSourceError.invalidFrameCount(frameCount)
        }

        return try (0..<frameCount).map { index in
            StereoFrame(
                metadata: FrameMetadata(
                    frameIndex: index,
                    timestampSeconds: Double(index) / 30.0,
                    width: width,
                    height: height
                ),
                leftPixelBuffer: try PixelBufferFactory.makeIOSurfaceBackedBGRA(width: width, height: height),
                rightPixelBuffer: try PixelBufferFactory.makeIOSurfaceBackedBGRA(width: width, height: height)
            )
        }
    }
}

public enum StereoSourceError: Error, Equatable, CustomStringConvertible {
    case missingInput(String)
    case emptyDirectory(String)
    case mismatchedFrameCount(left: Int, right: Int)
    case invalidFrameCount(Int)

    public var description: String {
        switch self {
        case let .missingInput(path):
            return "missing stereo input at \(path)"
        case let .emptyDirectory(path):
            return "stereo input directory is empty: \(path)"
        case let .mismatchedFrameCount(left, right):
            return "left/right frame count mismatch: \(left) != \(right)"
        case let .invalidFrameCount(count):
            return "invalid frame count: \(count)"
        }
    }
}
