import Foundation

public enum PipelineEvent: Equatable, Sendable {
    case runStarted(mode: RuntimeMode?, frameLimit: Int?)
    case frameStarted(index: Int, timestampSeconds: Double)
    case stageCompleted(frameIndex: Int, stage: String, milliseconds: Double)
    case frameCompleted(index: Int, trackingStatus: String)
    case frameDropped(index: Int, reason: String)
    case runCompleted(processedFrameCount: Int, droppedFrameCount: Int)
}

public protocol PipelineEventSink: Sendable {
    func record(_ event: PipelineEvent) async
}

public actor InMemoryPipelineEventSink: PipelineEventSink {
    private var storage: [PipelineEvent] = []

    public init() {}

    public func record(_ event: PipelineEvent) {
        storage.append(event)
    }

    public func events() -> [PipelineEvent] {
        storage
    }
}

public struct NoOpPipelineEventSink: PipelineEventSink {
    public init() {}

    public func record(_ event: PipelineEvent) async {
        _ = event
    }
}
