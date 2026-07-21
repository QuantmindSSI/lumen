import Foundation

struct ChatMessage: Identifiable, Equatable {
    let id: UUID
    var role: ChatRole
    var content: String
    var timestamp: Date
    var isStreaming: Bool

    init(id: UUID = UUID(), role: ChatRole, content: String, timestamp: Date = Date(), isStreaming: Bool = false) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.isStreaming = isStreaming
    }
}

enum ChatRole: String, Equatable {
    case user = "user"
    case assistant = "assistant"
    case system = "system"
}

struct ModelDownload: Identifiable {
    let id: String
    let name: String
    let sizeMB: Int
    let quantization: String
    let parameters: String
    let url: URL
}

extension ModelDownload {
    static let recommended: [ModelDownload] = [
        ModelDownload(
            id: "qwen3-1.7b-q4",
            name: "Qwen3 1.7B",
            sizeMB: 1100,
            quantization: "Q4_K_M",
            parameters: "1.7B",
            url: URL(string: "https://huggingface.co/mlx-community/Qwen3-1.7B-mlx")!
        ),
        ModelDownload(
            id: "qwen3-4b-q4",
            name: "Qwen3 4B",
            sizeMB: 2500,
            quantization: "Q4_K_M",
            parameters: "4B",
            url: URL(string: "https://huggingface.co/mlx-community/Qwen3-4B-mlx")!
        ),
        ModelDownload(
            id: "phi-4-mini-q4",
            name: "Phi-4 Mini",
            sizeMB: 2300,
            quantization: "Q4_K_M",
            parameters: "3.8B",
            url: URL(string: "https://huggingface.co/mlx-community/Phi-4-mini-instruct-mlx")!
        ),
    ]
}
