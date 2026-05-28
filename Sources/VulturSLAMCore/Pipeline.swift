import Foundation

public struct StageTiming: Codable, Equatable, Sendable {
    public var name: String
    public var milliseconds: Double

    public init(name: String, milliseconds: Double) {
        self.name = name
        self.milliseconds = milliseconds
    }
}

public struct FramePipelineMetrics: Codable, Equatable, Sendable {
    public var frameIndex: Int
    public var resourceID: FrameResourceID
    public var timings: [StageTiming]
    public var backend: BackendResult

    public init(frameIndex: Int, resourceID: FrameResourceID, timings: [StageTiming], backend: BackendResult) {
        self.frameIndex = frameIndex
        self.resourceID = resourceID
        self.timings = timings
        self.backend = backend
    }
}

public struct PipelineRunSummary: Codable, Equatable, Sendable {
    public var processedFrameCount: Int
    public var droppedFrameCount: Int
    public var frames: [FramePipelineMetrics]

    public init(processedFrameCount: Int, droppedFrameCount: Int, frames: [FramePipelineMetrics]) {
        self.processedFrameCount = processedFrameCount
        self.droppedFrameCount = droppedFrameCount
        self.frames = frames
    }
}

public struct SLAMPipeline: Sendable {
    public var rectification: any RectificationStage
    public var disparity: any DisparityStage
    public var featureDetection: any FeatureDetectionStage
    public var matching: any MatchingStage
    public var backend: any SLAMBackend
    public var resourceStore: FrameResourceStore
    public var eventSink: any PipelineEventSink

    public init(
        rectification: any RectificationStage = MetalRectificationPlaceholder(),
        disparity: any DisparityStage = DisparityPlaceholder(),
        featureDetection: any FeatureDetectionStage = FeatureDetectionPlaceholder(),
        matching: any MatchingStage = MatchingPlaceholder(),
        backend: any SLAMBackend = RustBackendBridge(),
        resourceStore: FrameResourceStore = FrameResourceStore(),
        eventSink: any PipelineEventSink = NoOpPipelineEventSink()
    ) {
        self.rectification = rectification
        self.disparity = disparity
        self.featureDetection = featureDetection
        self.matching = matching
        self.backend = backend
        self.resourceStore = resourceStore
        self.eventSink = eventSink
    }

    public func run(
        source: any StereoFrameSource,
        calibration: StereoCalibration,
        options: PipelineRunOptions = PipelineRunOptions(),
        mode: RuntimeMode? = nil
    ) async throws -> PipelineRunSummary {
        var frames = try source.frames()
        if let maxFrames = options.maxFrames {
            frames = Array(frames.prefix(maxFrames))
        }

        var frameMetrics: [FramePipelineMetrics] = []
        frameMetrics.reserveCapacity(frames.count)

        await eventSink.record(.runStarted(mode: mode, frameLimit: options.maxFrames))

        for frame in frames {
            var timings: [StageTiming] = []
            let resourceID = await resourceStore.retain(frame: frame)

            await eventSink.record(
                .frameStarted(index: frame.metadata.frameIndex, timestampSeconds: frame.metadata.timestampSeconds)
            )

            let rectified = try await timed("rectification", timings: &timings) {
                try await rectification.rectify(frame, calibration: calibration)
            }
            await eventSink.record(timings.last!.stageEvent(frameIndex: frame.metadata.frameIndex))

            async let disparityTask: TimedValue<DisparityMap> = timedValue("disparity") {
                try await disparity.estimateDisparity(rectified)
            }
            async let featuresTask: TimedValue<FeatureSet> = timedValue("feature_detection") {
                try await featureDetection.detectFeatures(rectified)
            }

            let disparityResult = try await disparityTask
            let featuresResult = try await featuresTask
            timings.append(disparityResult.timing)
            timings.append(featuresResult.timing)
            await eventSink.record(disparityResult.timing.stageEvent(frameIndex: frame.metadata.frameIndex))
            await eventSink.record(featuresResult.timing.stageEvent(frameIndex: frame.metadata.frameIndex))

            let matches = try await timed("matching", timings: &timings) {
                try await matching.matchFeatures(featuresResult.value, frame: rectified)
            }
            await eventSink.record(timings.last!.stageEvent(frameIndex: frame.metadata.frameIndex))

            let backendResult = try await timed("backend", timings: &timings) {
                try await backend.process(
                    BackendInput(
                        frame: rectified,
                        disparity: disparityResult.value,
                        features: featuresResult.value,
                        matches: matches
                    )
                )
            }
            await eventSink.record(timings.last!.stageEvent(frameIndex: frame.metadata.frameIndex))

            frameMetrics.append(
                FramePipelineMetrics(
                    frameIndex: frame.metadata.frameIndex,
                    resourceID: resourceID,
                    timings: timings,
                    backend: backendResult
                )
            )
            await eventSink.record(
                .frameCompleted(index: frame.metadata.frameIndex, trackingStatus: backendResult.trackingStatus)
            )
            await resourceStore.release(resourceID)
        }

        let summary = PipelineRunSummary(
            processedFrameCount: frameMetrics.count,
            droppedFrameCount: 0,
            frames: frameMetrics
        )
        await eventSink.record(
            .runCompleted(processedFrameCount: summary.processedFrameCount, droppedFrameCount: summary.droppedFrameCount)
        )
        return summary
    }
}

private struct TimedValue<Value>: Sendable where Value: Sendable {
    var value: Value
    var timing: StageTiming
}

private func timed<Value>(
    _ name: String,
    timings: inout [StageTiming],
    operation: () async throws -> Value
) async throws -> Value {
    let start = ContinuousClock.now
    let value = try await operation()
    let duration = start.duration(to: .now)
    timings.append(StageTiming(name: name, milliseconds: duration.milliseconds))
    return value
}

private func timedValue<Value: Sendable>(
    _ name: String,
    operation: () async throws -> Value
) async throws -> TimedValue<Value> {
    let start = ContinuousClock.now
    let value = try await operation()
    let duration = start.duration(to: .now)
    return TimedValue(
        value: value,
        timing: StageTiming(name: name, milliseconds: duration.milliseconds)
    )
}

private extension Duration {
    var milliseconds: Double {
        Double(components.seconds) * 1000 + Double(components.attoseconds) / 1e15
    }
}

private extension StageTiming {
    func stageEvent(frameIndex: Int) -> PipelineEvent {
        .stageCompleted(frameIndex: frameIndex, stage: name, milliseconds: milliseconds)
    }
}
