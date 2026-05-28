import Foundation
import XCTest
@testable import VulturSLAMCore

final class StereoSourceTests: XCTestCase {
    func testOfflineSourcePairsDirectoriesInSortedOrder() throws {
        let temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let left = temporaryDirectory.appendingPathComponent("left", isDirectory: true)
        let right = temporaryDirectory.appendingPathComponent("right", isDirectory: true)
        try FileManager.default.createDirectory(at: left, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: right, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        try Data().write(to: left.appendingPathComponent("0002.png"))
        try Data().write(to: left.appendingPathComponent("0001.png"))
        try Data().write(to: right.appendingPathComponent("0002.png"))
        try Data().write(to: right.appendingPathComponent("0001.png"))

        let frames = try OfflineStereoSource(left: left, right: right, calibration: Fixtures.calibration).frames()

        XCTAssertEqual(frames.count, 2)
        XCTAssertEqual(frames[0].leftSourceURL?.lastPathComponent, "0001.png")
        XCTAssertEqual(frames[0].rightSourceURL?.lastPathComponent, "0001.png")
        XCTAssertEqual(frames[1].leftSourceURL?.lastPathComponent, "0002.png")
    }

    func testSyntheticSourceCreatesPixelBuffers() throws {
        let frames = try SyntheticStereoSource(frameCount: 1, width: 32, height: 24).frames()

        XCTAssertEqual(frames.count, 1)
        XCTAssertNotNil(frames[0].leftPixelBuffer)
        XCTAssertNotNil(frames[0].rightPixelBuffer)
    }
}
