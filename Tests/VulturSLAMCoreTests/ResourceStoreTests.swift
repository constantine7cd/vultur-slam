import XCTest
@testable import VulturSLAMCore

final class ResourceStoreTests: XCTestCase {
    func testResourceStoreRetainsAndReleasesFramesByID() async throws {
        let store = FrameResourceStore()
        let frame = StereoFrame(
            metadata: FrameMetadata(frameIndex: 3, timestampSeconds: 0.1, width: 16, height: 12)
        )

        let id = await store.retain(frame: frame)

        let retainedCount = await store.retainedCount()
        let retainedFrameIndex = await store.resource(for: id)?.frame.metadata.frameIndex
        XCTAssertEqual(retainedCount, 1)
        XCTAssertEqual(retainedFrameIndex, 3)

        await store.release(id)

        let releasedCount = await store.retainedCount()
        let releasedResource = await store.resource(for: id)
        XCTAssertEqual(releasedCount, 0)
        XCTAssertNil(releasedResource)
    }
}
