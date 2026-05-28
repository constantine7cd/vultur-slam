import Foundation

public final class DebugTensorDumper {
    private struct TensorRecord: Encodable {
        let name: String
        let file: String
        let shape: [Int]
        let layout: String
        let dtype: String
    }

    private struct Manifest: Encodable {
        let format: String
        let tensors: [TensorRecord]
    }

    private let directory: URL
    private var records: [TensorRecord] = []

    public init(directory: URL) throws {
        self.directory = directory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    public func dump(_ tensor: MetalTensor, as name: String? = nil) throws {
        let tensorName = name ?? tensor.name
        let fileName = "\(safeFileName(tensorName)).bin"
        let fileURL = directory.appendingPathComponent(fileName)
        try ImageTensorIO.writeFP32Tensor(tensor, url: fileURL)
        records.append(
            TensorRecord(
                name: tensorName,
                file: fileName,
                shape: tensor.shape.dimensions,
                layout: tensor.shape.layout.rawValue,
                dtype: "float32"
            )
        )
    }

    public func dumpFloatArray(pointer: UnsafeRawPointer, shape: TensorShape, as name: String) throws {
        let fileName = "\(safeFileName(name)).bin"
        let fileURL = directory.appendingPathComponent(fileName)
        let data = Data(bytes: pointer, count: shape.byteCountFP32)
        try data.write(to: fileURL)
        records.append(
            TensorRecord(
                name: name,
                file: fileName,
                shape: shape.dimensions,
                layout: shape.layout.rawValue,
                dtype: "float32"
            )
        )
    }

    public func finish() throws {
        let manifest = Manifest(format: "fast_foundation_stereo_debug_tensors_v1", tensors: records)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: directory.appendingPathComponent("manifest.json"))
    }

    private func safeFileName(_ name: String) -> String {
        name.map { character in
            if character.isLetter || character.isNumber || character == "_" || character == "-" {
                return character
            }
            return "_"
        }.reduce(into: "") { partial, character in
            partial.append(character)
        }
    }
}
