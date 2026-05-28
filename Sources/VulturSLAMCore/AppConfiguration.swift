import Foundation

public enum RuntimeMode: String, Codable, Equatable, Sendable {
    case offline
    case online
}

public struct PipelineConfiguration: Codable, Equatable, Sendable {
    public var mode: RuntimeMode
    public var calibrationPath: String
    public var maxFrames: Int?
    public var outputDirectory: String?
    public var offline: OfflineSourceConfiguration?
    public var online: OnlineSourceConfiguration?

    public init(
        mode: RuntimeMode,
        calibrationPath: String,
        maxFrames: Int? = nil,
        outputDirectory: String? = nil,
        offline: OfflineSourceConfiguration? = nil,
        online: OnlineSourceConfiguration? = nil
    ) {
        self.mode = mode
        self.calibrationPath = calibrationPath
        self.maxFrames = maxFrames
        self.outputDirectory = outputDirectory
        self.offline = offline
        self.online = online
    }

    public static func load(from url: URL) throws -> PipelineConfiguration {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(PipelineConfiguration.self, from: data)
    }

    public func validate() throws {
        if let maxFrames, maxFrames < 0 {
            throw PipelineConfigurationError.invalidMaxFrames(maxFrames)
        }

        switch mode {
        case .offline:
            guard offline != nil else {
                throw PipelineConfigurationError.missingOfflineConfiguration
            }
        case .online:
            guard online != nil else {
                throw PipelineConfigurationError.missingOnlineConfiguration
            }
        }
    }
}

public struct OfflineSourceConfiguration: Codable, Equatable, Sendable {
    public var leftPath: String
    public var rightPath: String

    public init(leftPath: String, rightPath: String) {
        self.leftPath = leftPath
        self.rightPath = rightPath
    }
}

public struct OnlineSourceConfiguration: Codable, Equatable, Sendable {
    public var leftDeviceUniqueID: String?
    public var rightDeviceUniqueID: String?
    public var requestedFPS: Double

    public init(leftDeviceUniqueID: String? = nil, rightDeviceUniqueID: String? = nil, requestedFPS: Double = 30) {
        self.leftDeviceUniqueID = leftDeviceUniqueID
        self.rightDeviceUniqueID = rightDeviceUniqueID
        self.requestedFPS = requestedFPS
    }

    public func validate() throws {
        guard requestedFPS > 0 else {
            throw PipelineConfigurationError.invalidFPS(requestedFPS)
        }
    }
}

public struct PipelineRunOptions: Equatable, Sendable {
    public var maxFrames: Int?

    public init(maxFrames: Int? = nil) {
        self.maxFrames = maxFrames
    }
}

public enum PipelineConfigurationError: Error, Equatable, CustomStringConvertible {
    case missingOfflineConfiguration
    case missingOnlineConfiguration
    case invalidMaxFrames(Int)
    case invalidFPS(Double)

    public var description: String {
        switch self {
        case .missingOfflineConfiguration:
            return "offline mode requires offline source configuration"
        case .missingOnlineConfiguration:
            return "online mode requires online source configuration"
        case let .invalidMaxFrames(value):
            return "max_frames must be non-negative, got \(value)"
        case let .invalidFPS(value):
            return "requested_fps must be positive, got \(value)"
        }
    }
}
