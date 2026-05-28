import XCTest
@testable import VulturSLAMCore

final class CalibrationTests: XCTestCase {
    func testValidCalibrationPassesValidation() throws {
        try Fixtures.calibration.validate()
    }

    func testInvalidBaselineFailsValidation() {
        var calibration = Fixtures.calibration
        calibration.baselineMeters = 0

        XCTAssertThrowsError(try calibration.validate()) { error in
            XCTAssertEqual(error as? CalibrationError, .invalidBaseline(0))
        }
    }
}
