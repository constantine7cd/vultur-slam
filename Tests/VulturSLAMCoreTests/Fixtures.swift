@testable import VulturSLAMCore

enum Fixtures {
    static let calibration = StereoCalibration(
        left: CameraModel(
            intrinsics: CameraIntrinsics(fx: 400, fy: 400, cx: 32, cy: 24),
            distortion: [0, 0, 0, 0, 0],
            width: 64,
            height: 48
        ),
        right: CameraModel(
            intrinsics: CameraIntrinsics(fx: 400, fy: 400, cx: 32, cy: 24),
            distortion: [0, 0, 0, 0, 0],
            width: 64,
            height: 48
        ),
        rotationRightToLeft: [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ],
        translationRightToLeft: [-0.1, 0, 0],
        baselineMeters: 0.1,
        rectificationScale: 1
    )
}
