import Foundation
import MLX
import MLXLLM
import MLXRandom

/// On-device LLM inference using Apple's MLX framework.
/// Talks directly to the memory palace for context assembly.
@MainActor
final class InferenceService: ObservableObject {
    let palace: LumenPalaceService

    @Published var isLoading = false
    @Published var currentModelName: String?

    private var modelContainer: ModelContainer?

    init(palace: LumenPalaceService) {
        self.palace = palace
    }

    // MARK: - Model Loading

    func loadModel(modelID: String = "mlx-community/Qwen3-1.7B-mlx") async throws {
        isLoading = true
        defer { isLoading = false }

        // MLXLLM stream loads from Hugging Face Hub
        let modelConfiguration = ModelConfiguration(
            id: modelID,
            overrideTokenizer: nil
        )

        modelContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: modelConfiguration
        ) { progress in
            print("Downloading model: \(Int(progress.fractionCompleted * 100))%")
        }

        currentModelName = modelID
    }

    // MARK: - Generation

    func generate(prompt: String, useMemory: Bool = true) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    guard let container = modelContainer else {
                        throw InferenceError.modelNotLoaded
                    }

                    var fullPrompt = prompt

                    // Assemble context from palace
                    if useMemory {
                        let context = try await palace.assembleContext(query: prompt, maxTokens: 1024)
                        if !context.isEmpty {
                            fullPrompt = "Context from memory palace:\n\(context)\n---\nUser: \(prompt)\nAssistant:"
                        }
                    }

                    let result = try await container.perform { model, tokenizer in
                        // Tokenize
                        let inputTokens = tokenizer.encode(text: fullPrompt)
                        let inputArray = MLXArray(inputTokens)

                        // Greedy generation (simplified — replace with sampling for production)
                        var generatedTokens = inputTokens
                        let maxNewTokens = 512

                        for _ in 0..<maxNewTokens {
                            let input = MLXArray(generatedTokens)
                            let logits = model(input, cache: nil)
                            let nextToken = logits[0, -1].argmax().item(Int.self)
                            generatedTokens.append(nextToken)

                            let piece = tokenizer.decode(tokens: [nextToken])
                            continuation.yield(piece)

                            if nextToken == tokenizer.eosTokenId {
                                break
                            }
                        }

                        return tokenizer.decode(tokens: generatedTokens)
                    }

                    // Store this turn in the palace
                    try await palace.store(
                        content: "User: \(prompt)\nAssistant: \(result)",
                        room: "conversations",
                        sourceType: "agent_turn"
                    )

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Streaming Chat (Agent-friendly)

    func chatStream(userMessage: String, systemPrompt: String? = nil) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    var prompt = ""
                    if let system = systemPrompt {
                        prompt += "<|im_start|>system\n\(system)<|im_end|>\n"
                    }
                    prompt += "<|im_start|>user\n\(userMessage)<|im_end|>\n<|im_start|>assistant\n"

                    let stream = generate(prompt: prompt, useMemory: true)
                    for try await token in stream {
                        continuation.yield(token)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Helpers

    func unloadModel() {
        modelContainer = nil
        currentModelName = nil
    }

    var isModelLoaded: Bool {
        modelContainer != nil
    }
}

enum InferenceError: LocalizedError {
    case modelNotLoaded
    case generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "No model loaded. Download a model in Settings first."
        case .generationFailed(let msg):
            return "Generation failed: \(msg)"
        }
    }
}
