// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "vultur-slam",
    platforms: [
        .macOS(.v15)
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
    dependencies: [
        .package(path: "Sources/VulturSLAMCore/ML/FastFoundationStereoMPS")
    ],
    targets: [
        .target(
            name: "VulturSLAMCore",
            exclude: ["ML/FastFoundationStereoMPS"]
        ),
        .executableTarget(
            name: "VulturSLAMCLI",
            dependencies: [
                "VulturSLAMCore",
                .product(name: "FastFoundationStereoMPS", package: "FastFoundationStereoMPS"),
            ]
        ),
        .testTarget(
            name: "VulturSLAMCoreTests",
            dependencies: ["VulturSLAMCore"]
        )
    ]
)
