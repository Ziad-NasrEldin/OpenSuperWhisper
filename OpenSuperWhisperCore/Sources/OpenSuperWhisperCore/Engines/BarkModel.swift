import Foundation
import CoreML
import FluidAudio

/// Underdog Bark 3.0 Pro (ConwayResearch), an English Parakeet TDT v3 derivative.
/// It is not downloaded by the app: the Core ML folder from the Hugging Face repo
/// (`hf download ConwayResearch/Underdog-Bark-3.0-Pro --include 'coreml/*'`) is
/// expected at `~/Models/bark-3.0-pro/coreml`. Loading follows the model card's
/// `runtime/coreml/Sources/main.swift`.
public enum BarkModel {
    public static let version = "bark3"

    public static var directory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Models/bark-3.0-pro/coreml", isDirectory: true)
    }

    /// The model was evaluated with 0.25 s of leading silence.
    public static let leadingSilenceSamples = 4000
    /// The encoder also ships an 8 s `short8` function; `full15` handles every length
    /// (FluidAudio chunks longer audio), so only one copy of the encoder is loaded.
    private static let encoderFunction = "full15"

    private static let componentNames = ["Preprocessor", "Encoder", "Decoder", "JointDecisionv3"]
    private static let vocabularyName = "parakeet_v3_vocab.json"

    public static var isInstalled: Bool {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: directory.appendingPathComponent(vocabularyName).path) else {
            return false
        }
        return componentNames.allSatisfy { name in
            fileManager.fileExists(atPath: directory.appendingPathComponent(name + ".mlmodelc").path)
                || fileManager.fileExists(atPath: directory.appendingPathComponent(name + ".mlpackage").path)
        }
    }

    /// Loads the models with the encoder on the Neural Engine and the rest on the CPU,
    /// as the model card recommends.
    public static func loadModels() throws -> AsrModels {
        let raw = try JSONDecoder().decode(
            [String: String].self,
            from: Data(contentsOf: directory.appendingPathComponent(vocabularyName)))
        let vocabulary = Dictionary(uniqueKeysWithValues: raw.compactMap { key, value in
            Int(key).map { ($0, value) }
        })
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        return AsrModels(
            encoder: try load("Encoder", computeUnits: .cpuAndNeuralEngine, function: encoderFunction),
            preprocessor: try load("Preprocessor", computeUnits: .cpuOnly),
            decoder: try load("Decoder", computeUnits: .cpuOnly),
            joint: try load("JointDecisionv3", computeUnits: .cpuOnly),
            configuration: configuration,
            vocabulary: vocabulary,
            version: .ultra)
    }

    /// Prefers a compiled `.mlmodelc`; otherwise compiles the `.mlpackage` once and keeps
    /// the result next to it so later launches load quickly.
    private static func load(_ name: String, computeUnits: MLComputeUnits, function: String? = nil) throws -> MLModel {
        let fileManager = FileManager.default
        let compiledURL = directory.appendingPathComponent(name + ".mlmodelc")
        if !fileManager.fileExists(atPath: compiledURL.path) {
            let temporaryURL = try MLModel.compileModel(at: directory.appendingPathComponent(name + ".mlpackage"))
            do {
                try fileManager.moveItem(at: temporaryURL, to: compiledURL)
            } catch {
                return try MLModel(contentsOf: temporaryURL, configuration: configuration(computeUnits, function))
            }
        }
        return try MLModel(contentsOf: compiledURL, configuration: configuration(computeUnits, function))
    }

    private static func configuration(_ computeUnits: MLComputeUnits, _ function: String?) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        // The encoder is a multifunction model (macOS 15+); without a function name
        // Core ML loads its default function.
        if #available(macOS 15.0, *) {
            config.functionName = function
        }
        return config
    }
}
