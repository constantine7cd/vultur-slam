import XCTest
@testable import VulturSLAMCore

final class RustBackendBridgeTests: XCTestCase {
    func testFFIInputMapsBackendInputWithoutOwningImageData() {
        let frame = StereoFrame(
            metadata: FrameMetadata(frameIndex: 4, timestampSeconds: 0.2, width: 64, height: 48)
        )
        let input = BackendInput(
            frame: RectifiedStereoFrame(frame: frame),
            disparity: DisparityMap(
                width: 64,
                height: 48,
                storage: BorrowedBufferView(address: 10, byteCount: 20, strideBytes: 5)
            ),
            features: FeatureSet(
                keypointCount: 12,
                descriptorStorage: BorrowedBufferView(address: 30, byteCount: 40, strideBytes: 10)
            ),
            matches: FeatureMatches(matchCount: 6)
        )

        let ffi = RustFFIBackendInput(input)

        XCTAssertEqual(ffi.frame.frameIndex, 4)
        XCTAssertEqual(ffi.disparity.address, 10)
        XCTAssertEqual(ffi.descriptors.byteCount, 40)
        XCTAssertEqual(ffi.keypointCount, 12)
        XCTAssertEqual(ffi.matchCount, 6)
    }
}
