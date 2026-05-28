// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "FastFoundationStereoMPS",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "FastFoundationStereoMPS", targets: ["FastFoundationStereoMPS"]),
        .executable(name: "ffs-mps-runner", targets: ["FFSMPSRunner"]),
        .executable(name: "ffs-mps-conv3d-benchmark", targets: ["FFSMPSConv3DBenchmark"]),
    ],
    targets: [
        .target(
            name: "FastFoundationStereoMPS",
            resources: [.process("Shaders")]
        ),
        .executableTarget(
            name: "FFSMPSRunner",
            dependencies: ["FastFoundationStereoMPS"]
        ),
        .executableTarget(
            name: "FFSMPSConv3DBenchmark",
            dependencies: ["FastFoundationStereoMPS"]
        ),
        .testTarget(
            name: "FastFoundationStereoMPSTests",
            dependencies: ["FastFoundationStereoMPS"]
        ),
    ]
)
