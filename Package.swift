// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "vultur-slam",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "VulturSLAMCore",
            targets: ["VulturSLAMCore"]
        ),
        .executable(
            name: "vultur-slam",
            targets: ["VulturSLAMCLI"]
        )
    ],
    targets: [
        .target(
            name: "VulturSLAMCore"
        ),
        .executableTarget(
            name: "VulturSLAMCLI",
            dependencies: ["VulturSLAMCore"]
        ),
        .testTarget(
            name: "VulturSLAMCoreTests",
            dependencies: ["VulturSLAMCore"]
        )
    ]
)
