import SwiftUI

@main
struct ODSApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
    }
}

@MainActor
final class AppState: ObservableObject {
    let palace: LumenPalaceService
    let inference: InferenceService
    let voice: VoiceService

    init() {
        self.palace = LumenPalaceService()
        self.inference = InferenceService(palace: palace)
        self.voice = VoiceService()
    }
}

struct ContentView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        TabView {
            ChatView()
                .tabItem {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right")
                }

            MemoryPalaceView()
                .tabItem {
                    Label("Palace", systemImage: "building.columns")
                }

            DashboardView()
                .tabItem {
                    Label("Status", systemImage: "gauge.with.dots.needle.67percent")
                }

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
        }
    }
}
