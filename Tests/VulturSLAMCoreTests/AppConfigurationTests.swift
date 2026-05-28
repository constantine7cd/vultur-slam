import Foundation
import XCTest
@testable import VulturSLAMCore

final class AppConfigurationTests: XCTestCase {
    func testOfflinePipelineConfigurationRequiresOfflineSource() {
        let configuration = PipelineConfiguration(
            mode: .offline,
            calibrationPath: "calibration.json"
        )

        XCTAssertThrowsError(try configuration.validate()) { error in
            XCTAssertEqual(error as? PipelineConfigurationError, .missingOfflineConfiguration)
        }
    }

    func testOnlineConfigurationValidatesFPS() {
        let configuration = OnlineSourceConfiguration(requestedFPS: 0)

        XCTAssertThrowsError(try configuration.validate()) { error in
            XCTAssertEqual(error as? PipelineConfigurationError, .invalidFPS(0))
        }
    }

    func testPipelineConfigurationLoadsSnakeCaseJSON() throws {
        let json = """
        {
          "mode": "offline",
          "calibration_path": "Fixtures/calibration.example.json",
          "max_frames": 1,
          "offline": {
            "left_path": "left",
            "right_path": "right"
          }
        }
        """
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        try Data(json.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let configuration = try PipelineConfiguration.load(from: url)

        XCTAssertEqual(configuration.mode, .offline)
        XCTAssertEqual(configuration.maxFrames, 1)
        XCTAssertEqual(configuration.offline?.leftPath, "left")
        try configuration.validate()
    }
}
