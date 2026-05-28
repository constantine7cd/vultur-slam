import Foundation
import Metal

public final class MetalKernels {
    private let context: MetalContext
    private let nchwToNhwcPipeline: MTLComputePipelineState
    private let buildCombinedVolumePipeline: MTLComputePipelineState
    private let initialDisparityPipeline: MTLComputePipelineState
    private let geometryLookupPipeline: MTLComputePipelineState
    private let contextUpsamplePipeline: MTLComputePipelineState
    private let addPipeline: MTLComputePipelineState
    private let scalePipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.nchwToNhwcPipeline = try context.pipeline("nchw_to_nhwc_fp32")
        self.buildCombinedVolumePipeline = try context.pipeline("build_combined_volume_fp32")
        self.initialDisparityPipeline = try context.pipeline("initial_disparity_fp32")
        self.geometryLookupPipeline = try context.pipeline("geometry_lookup_fp32")
        self.contextUpsamplePipeline = try context.pipeline("context_upsample_fp32")
        self.addPipeline = try context.pipeline("add_fp32")
        self.scalePipeline = try context.pipeline("scale_fp32")
    }

    public func convertNCHWToNHWC(
        source: MetalTensor,
        destination: MetalTensor,
        commandBuffer: MTLCommandBuffer
    ) {
        precondition(source.shape.layout == .nchw)
        precondition(destination.shape.layout == .nhwc)
        var constants = LayoutConstants(
            n: UInt32(source.shape.dimensions[0]),
            c: UInt32(source.shape.dimensions[1]),
            h: UInt32(source.shape.dimensions[2]),
            w: UInt32(source.shape.dimensions[3])
        )
        encode(
            pipeline: nchwToNhwcPipeline,
            buffers: [source.buffer, destination.buffer],
            constants: &constants,
            constantLength: MemoryLayout<LayoutConstants>.stride,
            threads: destination.shape.elementCount,
            commandBuffer: commandBuffer
        )
    }

    public func buildCombinedVolume(
        projectedLeft04: MetalTensor,
        projectedRight04: MetalTensor,
        left04: MetalTensor,
        right04: MetalTensor,
        output: MetalTensor,
        commandBuffer: MTLCommandBuffer
    ) {
        precondition(projectedLeft04.shape.layout == .nhwc)
        precondition(projectedRight04.shape.layout == .nhwc)
        precondition(left04.shape.layout == .nhwc)
        precondition(right04.shape.layout == .nhwc)
        precondition(output.shape.layout == .ndhwc)
        var constants = CombinedVolumeConstants(
            h: UInt32(FastFoundationStereoMPSConstants.quarterHeight),
            w: UInt32(FastFoundationStereoMPSConstants.quarterWidth),
            d: UInt32(FastFoundationStereoMPSConstants.disparityQuarter),
            featureChannels: UInt32(left04.shape.dimensions[3]),
            projectionChannels: UInt32(projectedLeft04.shape.dimensions[3]),
            groups: UInt32(FastFoundationStereoMPSConstants.correlationGroups),
            outputChannels: UInt32(FastFoundationStereoMPSConstants.combinedVolumeChannels)
        )
        encode(
            pipeline: buildCombinedVolumePipeline,
            buffers: [
                projectedLeft04.buffer,
                projectedRight04.buffer,
                left04.buffer,
                right04.buffer,
                output.buffer,
            ],
            constants: &constants,
            constantLength: MemoryLayout<CombinedVolumeConstants>.stride,
            threads: output.shape.elementCount,
            commandBuffer: commandBuffer
        )
    }

    public func initialDisparity(
        logits: MetalTensor,
        output: MetalTensor,
        commandBuffer: MTLCommandBuffer
    ) {
        precondition(logits.shape.layout == .nhwc)
        precondition(output.shape.layout == .nhwc)
        var constants = InitialDisparityConstants(
            h: UInt32(FastFoundationStereoMPSConstants.quarterHeight),
            w: UInt32(FastFoundationStereoMPSConstants.quarterWidth),
            d: UInt32(FastFoundationStereoMPSConstants.disparityQuarter)
        )
        encode(
            pipeline: initialDisparityPipeline,
            buffers: [logits.buffer, output.buffer],
            constants: &constants,
            constantLength: MemoryLayout<InitialDisparityConstants>.stride,
            threads: FastFoundationStereoMPSConstants.quarterHeight * FastFoundationStereoMPSConstants.quarterWidth,
            commandBuffer: commandBuffer
        )
    }

    public func geometryLookup(
        left04: MetalTensor,
        right04: MetalTensor,
        regularizedVolume: MetalTensor,
        disparity: MetalTensor,
        output: MetalTensor,
        commandBuffer: MTLCommandBuffer
    ) {
        precondition(left04.shape.layout == .nhwc)
        precondition(right04.shape.layout == .nhwc)
        precondition(regularizedVolume.shape.layout == .ndhwc)
        precondition(disparity.shape.layout == .nhwc)
        precondition(output.shape.layout == .nhwc)
        var constants = GeometryLookupConstants(
            h: UInt32(FastFoundationStereoMPSConstants.quarterHeight),
            w: UInt32(FastFoundationStereoMPSConstants.quarterWidth),
            d: UInt32(FastFoundationStereoMPSConstants.disparityQuarter),
            featureChannels: UInt32(left04.shape.dimensions[3]),
            volumeChannels: UInt32(FastFoundationStereoMPSConstants.regularizedVolumeChannels),
            radius: UInt32(FastFoundationStereoMPSConstants.correlationRadius),
            levels: UInt32(FastFoundationStereoMPSConstants.correlationLevels),
            outputChannels: UInt32(output.shape.dimensions[3])
        )
        encode(
            pipeline: geometryLookupPipeline,
            buffers: [
                left04.buffer,
                right04.buffer,
                regularizedVolume.buffer,
                disparity.buffer,
                output.buffer,
            ],
            constants: &constants,
            constantLength: MemoryLayout<GeometryLookupConstants>.stride,
            threads: output.shape.elementCount,
            commandBuffer: commandBuffer
        )
    }

    public func contextUpsample(
        disparityLow: MetalTensor,
        upWeights: MetalTensor,
        output: MetalTensor,
        commandBuffer: MTLCommandBuffer
    ) {
        precondition(disparityLow.shape.layout == .nhwc)
        precondition(upWeights.shape.layout == .nhwc)
        precondition(output.shape.layout == .nhwc)
        var constants = ContextUpsampleConstants(
            lowH: UInt32(FastFoundationStereoMPSConstants.quarterHeight),
            lowW: UInt32(FastFoundationStereoMPSConstants.quarterWidth),
            highH: UInt32(FastFoundationStereoMPSConstants.imageHeight),
            highW: UInt32(FastFoundationStereoMPSConstants.imageWidth)
        )
        encode(
            pipeline: contextUpsamplePipeline,
            buffers: [disparityLow.buffer, upWeights.buffer, output.buffer],
            constants: &constants,
            constantLength: MemoryLayout<ContextUpsampleConstants>.stride,
            threads: FastFoundationStereoMPSConstants.imageHeight * FastFoundationStereoMPSConstants.imageWidth,
            commandBuffer: commandBuffer
        )
    }

    public func add(
        left: MetalTensor,
        right: MetalTensor,
        output: MetalTensor,
        commandBuffer: MTLCommandBuffer
    ) {
        precondition(left.shape == right.shape)
        precondition(left.shape == output.shape)
        var constants = ElementwiseConstants(count: UInt32(output.shape.elementCount), scale: 1.0)
        encode(
            pipeline: addPipeline,
            buffers: [left.buffer, right.buffer, output.buffer],
            constants: &constants,
            constantLength: MemoryLayout<ElementwiseConstants>.stride,
            threads: output.shape.elementCount,
            commandBuffer: commandBuffer
        )
    }

    public func scale(
        source: MetalTensor,
        output: MetalTensor,
        factor: Float,
        commandBuffer: MTLCommandBuffer
    ) {
        precondition(source.shape == output.shape)
        var constants = ElementwiseConstants(count: UInt32(output.shape.elementCount), scale: factor)
        encode(
            pipeline: scalePipeline,
            buffers: [source.buffer, output.buffer],
            constants: &constants,
            constantLength: MemoryLayout<ElementwiseConstants>.stride,
            threads: output.shape.elementCount,
            commandBuffer: commandBuffer
        )
    }

    private func encode<T>(
        pipeline: MTLComputePipelineState,
        buffers: [MTLBuffer],
        constants: inout T,
        constantLength: Int,
        threads: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            preconditionFailure("Could not create compute encoder.")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        withUnsafeBytes(of: &constants) { rawBuffer in
            encoder.setBytes(rawBuffer.baseAddress!, length: constantLength, index: buffers.count)
        }
        let width = pipeline.threadExecutionWidth
        let grid = MTLSize(width: threads, height: 1, depth: 1)
        let group = MTLSize(width: width, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: group)
        encoder.endEncoding()
    }
}

private struct LayoutConstants {
    let n: UInt32
    let c: UInt32
    let h: UInt32
    let w: UInt32
}

private struct CombinedVolumeConstants {
    let h: UInt32
    let w: UInt32
    let d: UInt32
    let featureChannels: UInt32
    let projectionChannels: UInt32
    let groups: UInt32
    let outputChannels: UInt32
}

private struct InitialDisparityConstants {
    let h: UInt32
    let w: UInt32
    let d: UInt32
}

private struct GeometryLookupConstants {
    let h: UInt32
    let w: UInt32
    let d: UInt32
    let featureChannels: UInt32
    let volumeChannels: UInt32
    let radius: UInt32
    let levels: UInt32
    let outputChannels: UInt32
}

private struct ContextUpsampleConstants {
    let lowH: UInt32
    let lowW: UInt32
    let highH: UInt32
    let highW: UInt32
}

private struct ElementwiseConstants {
    let count: UInt32
    let scale: Float
}
