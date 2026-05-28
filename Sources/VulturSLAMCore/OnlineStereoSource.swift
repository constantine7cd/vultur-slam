import Foundation

#if canImport(AVFoundation)
import AVFoundation
#endif

public struct OnlineStereoSource: StereoFrameSource {
    public var configuration: OnlineSourceConfiguration
    public var calibration: StereoCalibration

    public init(configuration: OnlineSourceConfiguration, calibration: StereoCalibration) {
        self.configuration = configuration
        self.calibration = calibration
    }

    public func frames() throws -> [StereoFrame] {
        try configuration.validate()
        throw OnlineStereoSourceError.captureNotImplemented
    }

    public static func availableDeviceSummaries() -> [String] {
        #if canImport(AVFoundation)
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external, .builtInWideAngleCamera],
            mediaType: .video,
            position: .unspecified
        )
        return discovery.devices.map { "\($0.localizedName) [\($0.uniqueID)]" }
        #else
        return []
        #endif
    }
}

public enum OnlineStereoSourceError: Error, Equatable, CustomStringConvertible {
    case captureNotImplemented

    public var description: String {
        switch self {
        case .captureNotImplemented:
            return "online capture adapter is configured but frame capture is not implemented yet"
        }
    }
}
