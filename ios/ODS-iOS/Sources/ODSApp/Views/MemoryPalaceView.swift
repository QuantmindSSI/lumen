import SwiftUI

struct MemoryPalaceView: View {
    @EnvironmentObject var appState: AppState
    @State private var rooms: [Room] = []
    @State private var searchQuery = ""
    @State private var searchResults: [RetrievedMemory] = []
    @State private var isSearching = false
    @State private var selectedRoom: Room?
    @State private var showingAddSheet = false

    var body: some View {
        NavigationStack {
            List {
                Section("Search Palace") {
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundStyle(.secondary)
                        TextField("Search your memories...", text: $searchQuery)
                            .submitLabel(.search)
                            .onSubmit {
                                Task { await performSearch() }
                            }
                        if isSearching {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                    }

                    if !searchResults.isEmpty {
                        ForEach(searchResults, id: \.chunk.id) { result in
                            MemoryRow(memory: result)
                        }
                    }
                }

                Section("Rooms") {
                    ForEach(rooms) { room in
                        NavigationLink(value: room) {
                            RoomRow(room: room, palace: appState.palace)
                        }
                    }
                }

                Section("Status") {
                    PalaceStatusRow(palace: appState.palace)
                }
            }
            .navigationTitle("Memory Palace")
            .navigationDestination(for: Room.self) { room in
                RoomDetailView(room: room, palace: appState.palace)
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("+ Memory") {
                        showingAddSheet = true
                    }
                }
            }
            .sheet(isPresented: $showingAddSheet) {
                AddMemorySheet(palace: appState.palace, rooms: rooms)
            }
            .task {
                await loadRooms()
            }
            .refreshable {
                await loadRooms()
            }
        }
    }

    private func loadRooms() async {
        do {
            rooms = try await appState.palace.db.read { db in
                try Room.fetchAll(db)
            }
        } catch {
            print("Failed to load rooms: \(error)")
        }
    }

    private func performSearch() async {
        guard !searchQuery.isEmpty else {
            searchResults = []
            return
        }
        isSearching = true
        defer { isSearching = false }
        do {
            searchResults = try await appState.palace.search(query: searchQuery)
        } catch {
            print("Search failed: \(error)")
        }
    }
}

struct RoomRow: View {
    let room: Room
    let palace: LumenPalaceService
    @State private var chunkCount: Int = 0

    var body: some View {
        HStack {
            Image(systemName: "building.columns.fill")
                .foregroundStyle(.accent)
            VStack(alignment: .leading, spacing: 2) {
                Text(room.name.capitalized)
                    .font(.headline)
                if let desc = room.description {
                    Text(desc)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            Text("\(chunkCount)")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 8)
                .padding(.vertical, 2)
                .background(Color(.systemGray6))
                .clipShape(Capsule())
        }
        .task {
            do {
                chunkCount = try await palace.db.read { db in
                    try MemoryChunk.filter(MemoryChunk.Columns.roomId == room.id!).fetchCount(db)
                }
            } catch {
                chunkCount = 0
            }
        }
    }
}

struct MemoryRow: View {
    let memory: RetrievedMemory

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(memory.chunk.content)
                    .font(.subheadline)
                    .lineLimit(3)
                Spacer()
                MatchBadge(type: memory.matchType)
            }
            HStack {
                Text(memory.chunk.sourceType)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.2f", memory.score))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

struct MatchBadge: View {
    let type: MatchType

    var body: some View {
        Text(type.rawValue.uppercased())
            .font(.caption2)
            .fontWeight(.semibold)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                type == .dense ? Color.blue.opacity(0.15) :
                type == .bm25 ? Color.green.opacity(0.15) :
                Color.purple.opacity(0.15)
            )
            .foregroundStyle(
                type == .dense ? .blue :
                type == .bm25 ? .green :
                .purple
            )
            .clipShape(Capsule())
    }
}

struct PalaceStatusRow: View {
    let palace: LumenPalaceService
    @State private var status: PalaceStatus?

    var body: some View {
        HStack(spacing: 16) {
            StatusItem(label: "Rooms", value: status?.rooms ?? 0)
            StatusItem(label: "Loci", value: status?.loci ?? 0)
            StatusItem(label: "Chunks", value: status?.chunks ?? 0)
        }
        .task {
            do {
                status = try await palace.status()
            } catch {
                status = nil
            }
        }
    }
}

struct StatusItem: View {
    let label: String
    let value: Int

    var body: some View {
        VStack(spacing: 2) {
            Text("\(value)")
                .font(.title3)
                .fontWeight(.bold)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

#Preview {
    MemoryPalaceView()
        .environmentObject(AppState())
}
