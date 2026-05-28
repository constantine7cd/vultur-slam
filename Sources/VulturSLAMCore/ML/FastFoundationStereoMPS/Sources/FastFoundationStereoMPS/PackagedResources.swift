import Foundation

public struct FastFoundationStereoMPSResources: Equatable, Sendable {
    public let featureModelURL: URL
    public let weightsDirectoryURL: URL

    public init(featureModelURL: URL, weightsDirectoryURL: URL) {
        self.featureModelURL = featureModelURL
        self.weightsDirectoryURL = weightsDirectoryURL
    }

    public static func resolve(
        in bundle: Bundle = .main,
        featureModelName: String = "feature_projection",
        preferredFeatureModelExtension: String? = nil,
        weightsDirectoryName: String = "FastFoundationStereoWeights"
    ) throws -> FastFoundationStereoMPSResources {
        let resourceURL = bundle.resourceURL ?? bundle.bundleURL
        return try resolve(
            in: resourceURL,
            featureModelName: featureModelName,
            preferredFeatureModelExtension: preferredFeatureModelExtension,
            weightsDirectoryName: weightsDirectoryName
        )
    }

    public static func resolve(
        in resourceDirectoryURL: URL,
        featureModelName: String = "feature_projection",
        preferredFeatureModelExtension: String? = nil,
        weightsDirectoryName: String = "FastFoundationStereoWeights"
    ) throws -> FastFoundationStereoMPSResources {
        let featureModelURL = try resolveFeatureModelURL(
            in: resourceDirectoryURL,
            name: featureModelName,
            preferredExtension: preferredFeatureModelExtension
        )
        let weightsDirectoryURL = resourceDirectoryURL.appendingPathComponent(weightsDirectoryName, isDirectory: true)
        try requireDirectory(weightsDirectoryURL, resourceName: weightsDirectoryName)
        try requireFile(
            weightsDirectoryURL.appendingPathComponent("manifest.json"),
            resourceName: "\(weightsDirectoryName)/manifest.json"
        )
        return FastFoundationStereoMPSResources(
            featureModelURL: featureModelURL,
            weightsDirectoryURL: weightsDirectoryURL
        )
    }

    private static func resolveFeatureModelURL(
        in resourceDirectoryURL: URL,
        name: String,
        preferredExtension: String?
    ) throws -> URL {
        let extensions = preferredExtension.map { [$0] } ?? ["mlmodelc", "mlpackage"]
        for fileExtension in extensions {
            let url = resourceDirectoryURL.appendingPathComponent("\(name).\(fileExtension)")
            if isExistingDirectory(url) {
                return url
            }
        }
        let expectedNames = extensions.map { "\(name).\($0)" }.joined(separator: " or ")
        throw FastFoundationStereoMPSError.resourceMissing(expectedNames)
    }

    private static func requireDirectory(_ url: URL, resourceName: String) throws {
        guard isExistingDirectory(url) else {
            throw FastFoundationStereoMPSError.resourceMissing(resourceName)
        }
    }

    private static func requireFile(_ url: URL, resourceName: String) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            throw FastFoundationStereoMPSError.resourceMissing(resourceName)
        }
    }

    private static func isExistingDirectory(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
    }
}
