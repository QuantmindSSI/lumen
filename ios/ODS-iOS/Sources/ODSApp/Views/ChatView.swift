import SwiftUI

struct ChatView: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var viewModel = ChatViewModel()
    @State private var messageText = ""
    @State private var scrollProxy: ScrollViewProxy? = nil

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("ODS Chat")
                        .font(.headline)
                    if let model = appState.inference.currentModelName {
                        Text(model.components(separatedBy: "/").last ?? model)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    } else {
                        Text("No model loaded")
                            .font(.caption2)
                            .foregroundStyle(.red)
                    }
                }
                Spacer()
                if appState.inference.isLoading {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial)

            Divider()

            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                    }
                    .padding()
                }
                .onAppear { scrollProxy = proxy }
                .onChange(of: viewModel.messages.count) { _, _ in
                    scrollToBottom()
                }
            }

            // Input
            VStack(spacing: 8) {
                if appState.voice.transcribedText.isEmpty == false && !appState.voice.isRecording {
                    HStack {
                        Text("Voice: \(appState.voice.transcribedText)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                        Spacer()
                        Button("Use") {
                            messageText = appState.voice.transcribedText
                            appState.voice.transcribedText = ""
                        }
                        .font(.caption)
                    }
                    .padding(.horizontal)
                }

                HStack(spacing: 8) {
                    // Voice button
                    Button {
                        if appState.voice.isRecording {
                            appState.voice.stopRecording()
                            if !appState.voice.transcribedText.isEmpty {
                                messageText = appState.voice.transcribedText
                                appState.voice.transcribedText = ""
                            }
                        } else {
                            Task {
                                try? await appState.voice.startRecording()
                            }
                        }
                    } label: {
                        Image(systemName: appState.voice.isRecording ? "waveform" : "mic")
                            .font(.title2)
                            .foregroundStyle(appState.voice.isRecording ? .red : .accentColor)
                            .frame(width: 40, height: 40)
                            .background(appState.voice.isRecording ? Color.red.opacity(0.1) : Color.clear)
                            .clipShape(Circle())
                    }
                    .animation(.easeInOut, value: appState.voice.isRecording)

                    // Text field
                    TextField("Message...", text: $messageText, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...5)

                    // Send button
                    Button {
                        Task {
                            await sendMessage()
                        }
                    } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.title2)
                            .foregroundStyle(messageText.isEmpty ? .secondary : .accentColor)
                    }
                    .disabled(messageText.isEmpty)
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
            }
            .background(.ultraThinMaterial)
        }
        .task {
            // Auto-load default model if available
            if !appState.inference.isModelLoaded {
                // Model must be downloaded first — user does this in Settings
            }
        }
    }

    private func sendMessage() async {
        let userText = messageText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !userText.isEmpty else { return }
        messageText = ""

        let userMessage = ChatMessage(role: .user, content: userText)
        viewModel.addMessage(userMessage)

        var assistantMessage = ChatMessage(role: .assistant, content: "", isStreaming: true)
        viewModel.addMessage(assistantMessage)

        do {
            let stream = appState.inference.chatStream(userMessage: userText)
            for try await token in stream {
                if let index = viewModel.messages.firstIndex(where: { $0.id == assistantMessage.id }) {
                    viewModel.messages[index].content += token
                }
            }
            if let index = viewModel.messages.firstIndex(where: { $0.id == assistantMessage.id }) {
                viewModel.messages[index].isStreaming = false
            }
        } catch {
            if let index = viewModel.messages.firstIndex(where: { $0.id == assistantMessage.id }) {
                viewModel.messages[index].content = "Error: \(error.localizedDescription)"
                viewModel.messages[index].isStreaming = false
            }
        }

        scrollToBottom()
    }

    private func scrollToBottom() {
        if let last = viewModel.messages.last {
            scrollProxy?.scrollTo(last.id, anchor: .bottom)
        }
    }
}

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 40) }

            VStack(alignment: .leading, spacing: 4) {
                Text(message.content)
                    .padding(12)
                    .background(message.role == .user ? Color.accentColor : Color(.systemGray5))
                    .foregroundStyle(message.role == .user ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                if message.isStreaming {
                    HStack(spacing: 4) {
                        Circle()
                            .frame(width: 4, height: 4)
                            .opacity(0.5)
                        Circle()
                            .frame(width: 4, height: 4)
                            .opacity(0.3)
                        Circle()
                            .frame(width: 4, height: 4)
                            .opacity(0.1)
                    }
                    .foregroundStyle(.secondary)
                    .padding(.leading, 4)
                }
            }

            if message.role == .assistant { Spacer(minLength: 40) }
        }
    }
}

#Preview {
    ChatView()
        .environmentObject(AppState())
}
