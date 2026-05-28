import Foundation

public struct RustFFIFrameMetadata: Equatable, Sendable {
    public var frameIndex: UInt64
    public var timestampSeconds: Double
    public var width: UInt32
    public var height: UInt32

    public init(metadata: FrameMetadata) {
        self.frameIndex = UInt64(metadata.frameIndex)
        self.timestampSeconds = metadata.timestampSeconds
        self.width = UInt32(metadata.width)
        self.height = UInt32(metadata.height)
    }
}

public struct RustFFIBorrowedBufferView: Equatable, Sendable {
    public var address: UInt
    public var byteCount: UInt
    public var strideBytes: UInt

    public init(_ view: BorrowedBufferView?) {
        self.address = UInt(view?.address ?? 0)
        self.byteCount = UInt(view?.byteCount ?? 0)
        self.strideBytes = UInt(view?.strideBytes ?? 0)
    }
}

public struct RustFFIBackendInput: Equatable, Sendable {
    public var frame: RustFFIFrameMetadata
    public var disparity: RustFFIBorrowedBufferView
    public var descriptors: RustFFIBorrowedBufferView
    public var keypointCount: UInt32
    public var matchCount: UInt32

    public init(_ input: BackendInput) {
        self.frame = RustFFIFrameMetadata(metadata: input.frame.frame.metadata)
        self.disparity = RustFFIBorrowedBufferView(input.disparity.storage)
        self.descriptors = RustFFIBorrowedBufferView(input.features.descriptorStorage)
        self.keypointCount = UInt32(input.features.keypointCount)
        self.matchCount = UInt32(input.matches.matchCount)
    }
}

public struct RustBackendBridge: SLAMBackend {
    public init() {}

    public func process(_ input: BackendInput) async throws -> BackendResult {
        let ffiInput = RustFFIBackendInput(input)
        return BackendResult(
            frameIndex: Int(ffiInput.frame.frameIndex),
            trackingStatus: "rust-ffi-placeholder",
            poseRightHandedColumnMajor: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
            ],
            fusedPointCount: Int(ffiInput.matchCount)
        )
    }
}
