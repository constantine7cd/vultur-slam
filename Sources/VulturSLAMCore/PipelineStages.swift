import Foundation

public struct RectifiedStereoFrame: Sendable {
    public var frame: StereoFrame

    public init(frame: StereoFrame) {
        self.frame = frame
    }
}

public struct DisparityMap: Equatable, Sendable {
    public var width: Int
    public var height: Int
    public var storage: BorrowedBufferView?

    public init(width: Int, height: Int, storage: BorrowedBufferView? = nil) {
        self.width = width
        self.height = height
        self.storage = storage
    }
}

public struct FeatureSet: Equatable, Sendable {
    public var keypointCount: Int
    public var descriptorStorage: BorrowedBufferView?

    public init(keypointCount: Int, descriptorStorage: BorrowedBufferView? = nil) {
        self.keypointCount = keypointCount
        self.descriptorStorage = descriptorStorage
    }
}

public struct FeatureMatches: Equatable, Sendable {
    public var matchCount: Int

    public init(matchCount: Int) {
        self.matchCount = matchCount
    }
}

public struct BackendInput: Sendable {
    public var frame: RectifiedStereoFrame
    public var disparity: DisparityMap
    public var features: FeatureSet
    public var matches: FeatureMatches

    public init(
        frame: RectifiedStereoFrame,
        disparity: DisparityMap,
        features: FeatureSet,
        matches: FeatureMatches
    ) {
        self.frame = frame
        self.disparity = disparity
        self.features = features
        self.matches = matches
    }
}

public struct BackendResult: Codable, Equatable, Sendable {
    public var frameIndex: Int
    public var trackingStatus: String
    public var poseRightHandedColumnMajor: [Double]
    public var fusedPointCount: Int

    public init(
        frameIndex: Int,
        trackingStatus: String,
        poseRightHandedColumnMajor: [Double],
        fusedPointCount: Int
    ) {
        self.frameIndex = frameIndex
        self.trackingStatus = trackingStatus
        self.poseRightHandedColumnMajor = poseRightHandedColumnMajor
        self.fusedPointCount = fusedPointCount
    }
}

public protocol RectificationStage: Sendable {
    func rectify(_ frame: StereoFrame, calibration: StereoCalibration) async throws -> RectifiedStereoFrame
}

public protocol DisparityStage: Sendable {
    func estimateDisparity(_ frame: RectifiedStereoFrame) async throws -> DisparityMap
}

public protocol FeatureDetectionStage: Sendable {
    func detectFeatures(_ frame: RectifiedStereoFrame) async throws -> FeatureSet
}

public protocol MatchingStage: Sendable {
    func matchFeatures(_ features: FeatureSet, frame: RectifiedStereoFrame) async throws -> FeatureMatches
}

public protocol SLAMBackend: Sendable {
    func process(_ input: BackendInput) async throws -> BackendResult
}

public struct MetalRectificationPlaceholder: RectificationStage {
    public init() {}

    public func rectify(_ frame: StereoFrame, calibration: StereoCalibration) async throws -> RectifiedStereoFrame {
        _ = calibration
        return RectifiedStereoFrame(frame: frame)
    }
}

public struct DisparityPlaceholder: DisparityStage {
    public init() {}

    public func estimateDisparity(_ frame: RectifiedStereoFrame) async throws -> DisparityMap {
        DisparityMap(width: frame.frame.metadata.width, height: frame.frame.metadata.height)
    }
}

public struct FeatureDetectionPlaceholder: FeatureDetectionStage {
    public init() {}

    public func detectFeatures(_ frame: RectifiedStereoFrame) async throws -> FeatureSet {
        FeatureSet(keypointCount: max(1, frame.frame.metadata.width * frame.frame.metadata.height / 4096))
    }
}

public struct MatchingPlaceholder: MatchingStage {
    public init() {}

    public func matchFeatures(_ features: FeatureSet, frame: RectifiedStereoFrame) async throws -> FeatureMatches {
        _ = frame
        return FeatureMatches(matchCount: features.keypointCount / 2)
    }
}

public struct RustBackendPlaceholder: SLAMBackend {
    public init() {}

    public func process(_ input: BackendInput) async throws -> BackendResult {
        BackendResult(
            frameIndex: input.frame.frame.metadata.frameIndex,
            trackingStatus: "placeholder",
            poseRightHandedColumnMajor: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
            ],
            fusedPointCount: input.matches.matchCount
        )
    }
}
