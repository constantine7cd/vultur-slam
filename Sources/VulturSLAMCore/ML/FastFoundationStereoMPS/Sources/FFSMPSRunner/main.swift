import CoreML
import FastFoundationStereoMPS
import Foundation

struct CommandLineOptions {
    let modelURL: URL?
    let weightsURL: URL?
    let leftURL: URL?
    let rightURL: URL?
    let outputURL: URL?
    let validIterations: Int
    let debugURL: URL?
    let costPrecision: NativeGraphPrecision

    init(arguments: [String]) {
        var modelURL: URL?
        var weightsURL: URL?
        var leftURL: URL?
        var rightURL: URL?
        var outputURL: URL?
        var validIterations = 8
        var debugURL: URL?
        var costPrecision: NativeGraphPrecision = .float32
        var iterator = arguments.dropFirst().makeIterator()
        while let argument = iterator.next() {
            if argument == "--feature-package", let value = iterator.next() {
                modelURL = URL(fileURLWithPath: value)
            } else if argument == "--weights-dir", let value = iterator.next() {
                weightsURL = URL(fileURLWithPath: value)
            } else if argument == "--left", let value = iterator.next() {
                leftURL = URL(fileURLWithPath: value)
            } else if argument == "--right", let value = iterator.next() {
                rightURL = URL(fileURLWithPath: value)
            } else if argument == "--out", let value = iterator.next() {
                outputURL = URL(fileURLWithPath: value)
            } else if argument == "--valid-iters", let value = iterator.next(), let parsed = Int(value) {
                validIterations = parsed
            } else if argument == "--debug-dir", let value = iterator.next() {
                debugURL = URL(fileURLWithPath: value)
            } else if argument == "--cost-precision", let value = iterator.next(), let parsed = NativeGraphPrecision(rawValue: value) {
                costPrecision = parsed
            }
        }
        self.modelURL = modelURL
        self.weightsURL = weightsURL
        self.leftURL = leftURL
        self.rightURL = rightURL
        self.outputURL = outputURL
        self.validIterations = validIterations
        self.debugURL = debugURL
        self.costPrecision = costPrecision
    }
}

let options = CommandLineOptions(arguments: CommandLine.arguments)
let runner = try FastFoundationStereoMPSRunner(costRegularizationPrecision: options.costPrecision)
print("FastFoundationStereoMPS ready on \(runner.context.device.name)")
print("FastFoundationStereoMPS cost regularization precision: \(options.costPrecision.rawValue)")

if let modelURL = options.modelURL {
    let featureRunner = try FeatureProjectionRunner(modelURL: modelURL, arena: runner.arena)
    print("Loaded feature projection package: \(modelURL.path)")

    if let weightsURL = options.weightsURL,
        let leftURL = options.leftURL,
        let rightURL = options.rightURL,
        let outputURL = options.outputURL {
        let weights = try NativeWeights(directory: weightsURL, device: runner.context.device)
        let left = try ImageTensorIO.loadRGBTensor(url: leftURL)
        let right = try ImageTensorIO.loadRGBTensor(url: rightURL)
        let dumper = try options.debugURL.map { try DebugTensorDumper(directory: $0) }
        let inferenceRuns = 5
        print("FastFoundationStereoMPS run 1/\(inferenceRuns)")
        var disparity = try runner.run(
            leftImage: left,
            rightImage: right,
            featureRunner: featureRunner,
            weights: weights,
            validIterations: options.validIterations,
            debugDumper: dumper
        )
        if inferenceRuns > 1 {
            for run in 2...inferenceRuns {
                print("FastFoundationStereoMPS run \(run)/\(inferenceRuns)")
                disparity = try runner.run(
                    leftImage: left,
                    rightImage: right,
                    featureRunner: featureRunner,
                    weights: weights,
                    validIterations: options.validIterations,
                    debugDumper: dumper
                )
            }
        }
        let outputDirectory = outputURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        try ImageTensorIO.writeFP32Tensor(disparity, url: outputURL)
        try dumper?.finish()
        print("Wrote disparity tensor: \(outputURL.path)")
    }
}
