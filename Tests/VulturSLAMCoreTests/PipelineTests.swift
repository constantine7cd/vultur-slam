import XCTest
@testable import VulturSLAMCore

final class PipelineTests: XCTestCase {
    func testSyntheticPipelineInvokesAllStages() async throws {
        let source = SyntheticStereoSource(frameCount: 2, width: 64, height: 48)
        let summary = try await SLAMPipeline().run(source: source, calibration: Fixtures.calibration)

        XCTAssertEqual(summary.processedFrameCount, 2)
        XCTAssertEqual(summary.droppedFrameCount, 0)
        XCTAssertEqual(summary.frames.map(\.frameIndex), [0, 1])
        XCTAssertEqual(
            summary.frames.first?.timings.map(\.name),
            ["rectification", "disparity", "feature_detection", "matching", "backend"]
        )
        XCTAssertEqual(summary.frames.first?.backend.trackingStatus, "rust-ffi-placeholder")
    }

    func testPipelineRespectsFrameLimitAndRecordsEvents() async throws {
        let eventSink = InMemoryPipelineEventSink()
        let resourceStore = FrameResourceStore()
        let pipeline = SLAMPipeline(resourceStore: resourceStore, eventSink: eventSink)
        let source = SyntheticStereoSource(frameCount: 3, width: 64, height: 48)

        let summary = try await pipeline.run(
            source: source,
            calibration: Fixtures.calibration,
            options: PipelineRunOptions(maxFrames: 1),
            mode: .offline
        )

        XCTAssertEqual(summary.processedFrameCount, 1)
        let retainedCount = await resourceStore.retainedCount()
        XCTAssertEqual(retainedCount, 0)

        let events = await eventSink.events()
        XCTAssertEqual(events.first, .runStarted(mode: .offline, frameLimit: 1))
        XCTAssertTrue(events.contains(.frameCompleted(index: 0, trackingStatus: "rust-ffi-placeholder")))
        XCTAssertEqual(events.last, .runCompleted(processedFrameCount: 1, droppedFrameCount: 0))
    }
}
