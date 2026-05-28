import CoreVideo
import Foundation

public struct FrameResourceID: Hashable, Codable, Sendable {
    public var rawValue: UInt64

    public init(rawValue: UInt64) {
        self.rawValue = rawValue
    }
}

public final class RetainedFrameResources: @unchecked Sendable {
    public let frame: StereoFrame

    public init(frame: StereoFrame) {
        self.frame = frame
    }
}

public actor FrameResourceStore {
    private var nextID: UInt64 = 1
    private var resources: [FrameResourceID: RetainedFrameResources] = [:]

    public init() {}

    public func retain(frame: StereoFrame) -> FrameResourceID {
        let id = FrameResourceID(rawValue: nextID)
        nextID += 1
        resources[id] = RetainedFrameResources(frame: frame)
        return id
    }

    public func resource(for id: FrameResourceID) -> RetainedFrameResources? {
        resources[id]
    }

    public func release(_ id: FrameResourceID) {
        resources[id] = nil
    }

    public func retainedCount() -> Int {
        resources.count
    }
}
