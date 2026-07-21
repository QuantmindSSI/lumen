import SwiftUI

struct DashboardView: View {
    @EnvironmentObject var appState: AppState
    @State private var status: PalaceStatus?
    @State private var deviceProfile = "iPhone"
    @State private var memoryPressure = "Normal"

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Model Status Card
                    ModelStatusCard(inference: appState.inference)

                    // Palace Health Card
                    PalaceHealthCard(status: status)

                    // Device Stats Card
                    DeviceStatsCard(
                        deviceProfile: deviceProfile,
                        memoryPressure: memoryPressure
                    )

                    // Voice Status
                    VoiceStatusCard(voice: appState.voice)
                }
                .padding()
            }
            .navigationTitle("Dashboard")
            .task {
                await refreshStatus()
            }
            .refreshable {
                await refreshStatus()
            }
        }
    }

    private func refreshStatus() async {
        do {
            status = try await appState.palace.status()
        } catch {
            status = nil
        }

        // Determine device profile
        let info = ProcessInfo.processInfo
        #if targetEnvironment(simulator)
        deviceProfile = "Simulator"
        #else
        let physical = info.physicalMemory
        if physical < 6_000_000_000 {
            deviceProfile = "iPhone (6 GB)"
        } else if physical < 9_000_000_000 {
            deviceProfile = "iPhone Pro (8 GB)"
        } else {
            deviceProfile = "iPad Pro / Ultra (16 GB+)"
        }
        #endif

        // Memory pressure
        let used = info.physicalMemory - UInt64(availableMemoryMB()) * 1_000_000
        let ratio = Double(used) / Double(info.physicalMemory)
        if ratio > 0.9 {
            memoryPressure = "Critical"
        } else if ratio > 0.8 {
            memoryPressure = "Warning"
        } else {
            memoryPressure = "Normal"
        }
    }

    private func availableMemoryMB() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return 0 }
        return UInt64(info.resident_size) / 1024 / 1024
    }
}

struct ModelStatusCard: View {
    @ObservedObject var inference: InferenceService

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Inference Engine", systemImage: "cpu")
                .font(.headline)

            HStack {
                Circle()
                    .fill(inference.isModelLoaded ? Color.green : Color.red)
                    .frame(width: 10, height: 10)
                Text(inference.isModelLoaded ? "Model Loaded" : "No Model")
                    .font(.subheadline)
                Spacer()
            }

            if let name = inference.currentModelName {
                Text(name)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            if inference.isLoading {
                ProgressView("Loading...")
                    .scaleEffect(0.8)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

struct PalaceHealthCard: View {
    let status: PalaceStatus?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Memory Palace", systemImage: "building.columns")
                .font(.headline)

            HStack(spacing: 24) {
                DashboardMetric(label: "Rooms", value: status?.rooms ?? 0)
                DashboardMetric(label: "Loci", value: status?.loci ?? 0)
                DashboardMetric(label: "Chunks", value: status?.chunks ?? 0)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

struct DeviceStatsCard: View {
    let deviceProfile: String
    let memoryPressure: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Device", systemImage: "iphone")
                .font(.headline)

            HStack {
                Text("Profile")
                Spacer()
                Text(deviceProfile)
                    .foregroundStyle(.secondary)
            }
            .font(.subheadline)

            HStack {
                Text("Memory")
                Spacer()
                Text(memoryPressure)
                    .foregroundStyle(
                        memoryPressure == "Normal" ? .green :
                        memoryPressure == "Warning" ? .orange : .red
                    )
                    .fontWeight(.semibold)
            }
            .font(.subheadline)
        }
        .padding()
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

struct VoiceStatusCard: View {
    @ObservedObject var voice: VoiceService

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Voice", systemImage: "waveform")
                .font(.headline)

            HStack {
                Text("Speech-to-Text")
                Spacer()
                Image(systemName: voice.isRecording ? "waveform.circle.fill" : "mic.circle")
                    .foregroundStyle(voice.isRecording ? .red : .green)
            }
            .font(.subheadline)

            HStack {
                Text("Text-to-Speech")
                Spacer()
                Image(systemName: voice.isSpeaking ? "speaker.wave.3.fill" : "speaker.slash")
                    .foregroundStyle(voice.isSpeaking ? .green : .secondary)
            }
            .font(.subheadline)
        }
        .padding()
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

struct DashboardMetric: View {
    let label: String
    let value: Int

    var body: some View {
        VStack(spacing: 4) {
            Text("\(value)")
                .font(.title2)
                .fontWeight(.bold)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

#Preview {
    DashboardView()
        .environmentObject(AppState())
}
