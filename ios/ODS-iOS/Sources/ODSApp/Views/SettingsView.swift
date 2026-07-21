import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @State private var temperature: Double = 0.7
    @State private var maxContext: Double = 2048
    @State private var selectedModel: ModelDownload?
    @State private var showingModelPicker = false
    @State private var isDownloading = false
    @State private var downloadProgress: Double = 0

    var body: some View {
        NavigationStack {
            List {
                Section("Model") {
                    HStack {
                        Text("Current Model")
                        Spacer()
                        Text(appState.inference.currentModelName?.components(separatedBy: "/").last ?? "None")
                            .foregroundStyle(.secondary)
                    }

                    Button("Download Model...") {
                        showingModelPicker = true
                    }

                    if isDownloading {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Downloading...")
                                .font(.caption)
                            ProgressView(value: downloadProgress)
                        }
                    }

                    Button("Unload Model") {
                        appState.inference.unloadModel()
                    }
                    .disabled(!appState.inference.isModelLoaded)
                    .foregroundStyle(.red)
                }

                Section("Generation") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Temperature")
                            Spacer()
                            Text(String(format: "%.2f", temperature))
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $temperature, in: 0...2, step: 0.1)
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Max Context")
                            Spacer()
                            Text("\(Int(maxContext))")
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $maxContext, in: 512...8192, step: 512)
                    }
                }

                Section("Memory Palace") {
                    Button("Reset Palace (Dangerous)") {
                        // Show confirmation alert
                    }
                    .foregroundStyle(.red)

                    Button("Export Palace") {
                        // Share sheet with .db file
                    }
                }

                Section("About") {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("0.1.0-alpha")
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("License")
                        Spacer()
                        Text("Apache-2.0")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
            .sheet(isPresented: $showingModelPicker) {
                ModelPickerSheet(
                    selectedModel: $selectedModel,
                    isDownloading: $isDownloading,
                    downloadProgress: $downloadProgress,
                    inference: appState.inference
                )
            }
        }
    }
}

struct ModelPickerSheet: View {
    @Binding var selectedModel: ModelDownload?
    @Binding var isDownloading: Bool
    @Binding var downloadProgress: Double
    let inference: InferenceService
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Recommended for iPhone") {
                    ForEach(ModelDownload.recommended) { model in
                        ModelRow(model: model, isLoaded: inference.currentModelName?.contains(model.id) ?? false)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                Task {
                                    await downloadAndLoad(model: model)
                                }
                            }
                    }
                }
            }
            .navigationTitle("Download Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }

    private func downloadAndLoad(model: ModelDownload) async {
        isDownloading = true
        downloadProgress = 0
        defer { isDownloading = false }

        do {
            // MLXLLM auto-downloads from Hugging Face on loadContainer
            try await inference.loadModel(modelID: model.url.absoluteString)
            dismiss()
        } catch {
            print("Model download failed: \(error)")
        }
    }
}

struct ModelRow: View {
    let model: ModelDownload
    let isLoaded: Bool

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.name)
                    .font(.headline)
                Text("\(model.parameters) • \(model.quantization) • \(model.sizeMB) MB")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if isLoaded {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            }
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    SettingsView()
        .environmentObject(AppState())
}
