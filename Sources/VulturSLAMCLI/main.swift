import Foundation
import FastFoundationStereoMPS
import VulturSLAMCore

@main
struct VulturSLAMCommand {
    static func main() async {
        do {
            try await run(arguments: Array(CommandLine.arguments.dropFirst()))
        } catch let error as UsageError {
            FileHandle.standardError.write(Data((error.description + "\n").utf8))
            printUsage()
            Foundation.exit(2)
        } catch {
            FileHandle.standardError.write(Data(("error: \(error)\n").utf8))
            Foundation.exit(1)
        }
    }

    private static func run(arguments: [String]) async throws {
        guard let command = arguments.first else {
            throw UsageError.missingCommand
        }

        let options = try CLIOptions(arguments: Array(arguments.dropFirst()))
        switch command {
        case "run-config":
            let configURL = try options.requiredURL("config")
            let config = try PipelineConfiguration.load(from: configURL)
            try config.validate()
            try await run(configuration: config)
        case "validate-config":
            let calibration = try StereoCalibration.load(from: try options.requiredURL("calibration"))
            try calibration.validate()
            print("calibration valid")
        case "list-cameras":
            let devices = OnlineStereoSource.availableDeviceSummaries()
            if devices.isEmpty {
                print("no video capture devices discovered")
            } else {
                devices.forEach { print($0) }
            }
        case "offline":
            let calibration = try StereoCalibration.load(from: try options.requiredURL("calibration"))
            try calibration.validate()

            let source = OfflineStereoSource(
                left: try options.requiredURL("left"),
                right: try options.requiredURL("right"),
                calibration: calibration
            )
            let summary = try await SLAMPipeline().run(
                source: source,
                calibration: calibration,
                options: PipelineRunOptions(maxFrames: try options.int("max-frames")),
                mode: .offline
            )
            let outputDirectory = options.url("output") ?? URL(fileURLWithPath: ".")
            try write(summary: summary, outputDirectory: outputDirectory)
        case "fast-foundation-stereo":
            try runFastFoundationStereo(options: options)
        case "online":
            let calibration = try StereoCalibration.load(from: try options.requiredURL("calibration"))
            try calibration.validate()
            let configuration = OnlineSourceConfiguration(
                leftDeviceUniqueID: options.string("left-device"),
                rightDeviceUniqueID: options.string("right-device"),
                requestedFPS: try options.double("fps") ?? 30
            )
            try configuration.validate()
            let source = OnlineStereoSource(configuration: configuration, calibration: calibration)
            _ = source
            throw UsageError.notImplemented("online source configured; capture implementation is the next adapter layer")
        default:
            throw UsageError.unknownCommand(command)
        }
    }

    private static func runFastFoundationStereo(options: CLIOptions) throws {
        let resourcesDirectory = options.url("resources") ?? URL(fileURLWithPath: "Contents/Resources")
        let resolvedResources = try FastFoundationStereoMPSResources.resolve(in: resourcesDirectory)
        let featureModelURL = options.url("feature-package") ?? resolvedResources.featureModelURL
        let weightsDirectoryURL = options.url("weights-dir") ?? resolvedResources.weightsDirectoryURL
        let leftURL = options.url("left") ?? URL(fileURLWithPath: "demo_data/left.png")
        let rightURL = options.url("right") ?? URL(fileURLWithPath: "demo_data/right.png")
        let outputURL = options.url("output") ?? URL(fileURLWithPath: "demo_data/disparity.fp32")
        let validIterations = try options.int("valid-iters") ?? 8
        let costPrecision = try options.nativeGraphPrecision("cost-precision") ?? .float32

        let runner = try FastFoundationStereoMPSRunner(costRegularizationPrecision: costPrecision)
        print("FastFoundationStereoMPS ready on \(runner.context.device.name)")
        print("feature model: \(featureModelURL.path)")
        print("weights: \(weightsDirectoryURL.path)")
        print("left: \(leftURL.path)")
        print("right: \(rightURL.path)")

        let featureRunner = try FeatureProjectionRunner(modelURL: featureModelURL, arena: runner.arena)
        let weights = try NativeWeights(directory: weightsDirectoryURL, device: runner.context.device)
        let inputTensors = try ImageTensorIO.loadRGBTensors(leftURL: leftURL, rightURL: rightURL)
        let disparity = try runner.run(
            inputTensors: inputTensors,
            featureRunner: featureRunner,
            weights: weights,
            validIterations: validIterations
        )

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try ImageTensorIO.writeFP32Tensor(disparity, url: outputURL)
        try writeDisparityMetadata(for: disparity, outputURL: outputURL)
        let metadataURL = outputURL.appendingPathExtension("json")
        print("disparity: \(outputURL.path)")
        print("metadata: \(metadataURL.path)")
    }

    private static func run(configuration: PipelineConfiguration) async throws {
        let calibration = try StereoCalibration.load(from: URL(fileURLWithPath: configuration.calibrationPath))
        try calibration.validate()

        switch configuration.mode {
        case .offline:
            guard let offline = configuration.offline else {
                throw PipelineConfigurationError.missingOfflineConfiguration
            }
            let source = OfflineStereoSource(
                left: URL(fileURLWithPath: offline.leftPath),
                right: URL(fileURLWithPath: offline.rightPath),
                calibration: calibration
            )
            let summary = try await SLAMPipeline().run(
                source: source,
                calibration: calibration,
                options: PipelineRunOptions(maxFrames: configuration.maxFrames),
                mode: .offline
            )
            try write(
                summary: summary,
                outputDirectory: URL(fileURLWithPath: configuration.outputDirectory ?? ".")
            )
        case .online:
            guard let online = configuration.online else {
                throw PipelineConfigurationError.missingOnlineConfiguration
            }
            try online.validate()
            _ = OnlineStereoSource(configuration: online, calibration: calibration)
            throw UsageError.notImplemented("online source configured; capture implementation is the next adapter layer")
        }
    }

    private static func write(summary: PipelineRunSummary, outputDirectory: URL) throws {
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let outputURL = outputDirectory.appendingPathComponent("metrics.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(summary).write(to: outputURL)
        print("processed \(summary.processedFrameCount) frame(s)")
        print("metrics: \(outputURL.path)")
    }

    private static func writeDisparityMetadata(for tensor: MetalTensor, outputURL: URL) throws {
        let metadata = DisparityMetadata(
            format: "fp32",
            shape: tensor.shape.dimensions,
            layout: tensor.shape.layout.rawValue
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metadata).write(to: outputURL.appendingPathExtension("json"))
    }

    private static func printUsage() {
        let usage = """
        usage:
          vultur-slam run-config --config <pipeline.json>
          vultur-slam validate-config --calibration <calibration.json>
          vultur-slam list-cameras
          vultur-slam offline --left <file-or-dir> --right <file-or-dir> --calibration <calibration.json> [--output <dir>] [--max-frames <n>]
          vultur-slam fast-foundation-stereo [--left <image>] [--right <image>] [--resources <Contents/Resources>] [--output <disparity.fp32>] [--valid-iters <n>] [--cost-precision <float32|float16>]
          vultur-slam online --calibration <calibration.json> [--left-device <id>] [--right-device <id>] [--fps <n>]
        """
        FileHandle.standardError.write(Data((usage + "\n").utf8))
    }
}

private struct DisparityMetadata: Encodable {
    let format: String
    let shape: [Int]
    let layout: String
}

private struct CLIOptions {
    private var values: [String: String] = [:]

    init(arguments: [String]) throws {
        var index = arguments.startIndex
        while index < arguments.endIndex {
            let token = arguments[index]
            guard token.hasPrefix("--") else {
                throw UsageError.unexpectedArgument(token)
            }

            let key = String(token.dropFirst(2))
            let valueIndex = arguments.index(after: index)
            guard valueIndex < arguments.endIndex else {
                throw UsageError.missingValue(key)
            }
            values[key] = arguments[valueIndex]
            index = arguments.index(after: valueIndex)
        }
    }

    func url(_ key: String) -> URL? {
        values[key].map { URL(fileURLWithPath: $0) }
    }

    func string(_ key: String) -> String? {
        values[key]
    }

    func int(_ key: String) throws -> Int? {
        guard let value = values[key] else {
            return nil
        }
        guard let intValue = Int(value) else {
            throw UsageError.invalidValue(option: key, value: value)
        }
        return intValue
    }

    func double(_ key: String) throws -> Double? {
        guard let value = values[key] else {
            return nil
        }
        guard let doubleValue = Double(value) else {
            throw UsageError.invalidValue(option: key, value: value)
        }
        return doubleValue
    }

    func nativeGraphPrecision(_ key: String) throws -> NativeGraphPrecision? {
        guard let value = values[key] else {
            return nil
        }
        guard let precision = NativeGraphPrecision(rawValue: value) else {
            throw UsageError.invalidValue(option: key, value: value)
        }
        return precision
    }

    func requiredURL(_ key: String) throws -> URL {
        guard let url = url(key) else {
            throw UsageError.missingOption(key)
        }
        return url
    }
}

private enum UsageError: Error, CustomStringConvertible {
    case missingCommand
    case unknownCommand(String)
    case unexpectedArgument(String)
    case missingOption(String)
    case missingValue(String)
    case invalidValue(option: String, value: String)
    case notImplemented(String)

    var description: String {
        switch self {
        case .missingCommand:
            return "missing command"
        case let .unknownCommand(command):
            return "unknown command: \(command)"
        case let .unexpectedArgument(argument):
            return "unexpected argument: \(argument)"
        case let .missingOption(option):
            return "missing required option --\(option)"
        case let .missingValue(option):
            return "missing value for --\(option)"
        case let .invalidValue(option, value):
            return "invalid value for --\(option): \(value)"
        case let .notImplemented(message):
            return message
        }
    }
}
