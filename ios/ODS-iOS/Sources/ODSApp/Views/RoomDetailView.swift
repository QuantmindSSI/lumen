import SwiftUI

struct RoomDetailView: View {
    let room: Room
    let palace: LumenPalaceService
    @State private var chunks: [MemoryChunk] = []
    @State private var showingAddMemory = false

    var body: some View {
        List {
            ForEach(chunks, id: \.id) { chunk in
                VStack(alignment: .leading, spacing: 6) {
                    Text(chunk.content)
                        .font(.body)
                    HStack {
                        Text(chunk.sourceType)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        if let accessed = chunk.accessedAt {
                            Text(accessed, style: .relative)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.vertical, 4)
            }
        }
        .navigationTitle(room.name.capitalized)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("+") {
                    showingAddMemory = true
                }
            }
        }
        .sheet(isPresented: $showingAddMemory) {
            AddMemorySheet(palace: palace, rooms: [room], preselectedRoom: room)
        }
        .task {
            await loadChunks()
        }
        .refreshable {
            await loadChunks()
        }
    }

    private func loadChunks() async {
        do {
            chunks = try await palace.db.read { db in
                try MemoryChunk
                    .filter(MemoryChunk.Columns.roomId == room.id!)
                    .order(MemoryChunk.Columns.createdAt.desc)
                    .fetchAll(db)
            }
        } catch {
            print("Failed to load chunks: \(error)")
        }
    }
}

struct AddMemorySheet: View {
    let palace: LumenPalaceService
    let rooms: [Room]
    var preselectedRoom: Room? = nil

    @Environment(\.dismiss) private var dismiss
    @State private var content = ""
    @State private var selectedRoomName: String
    @State private var sourceType = "user_input"
    @State private var isSaving = false

    init(palace: LumenPalaceService, rooms: [Room], preselectedRoom: Room? = nil) {
        self.palace = palace
        self.rooms = rooms
        self.preselectedRoom = preselectedRoom
        _selectedRoomName = State(initialValue: preselectedRoom?.name ?? (rooms.first?.name ?? "conversations"))
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Memory Content") {
                    TextEditor(text: $content)
                        .frame(minHeight: 120)
                }

                Section("Room") {
                    Picker("Room", selection: $selectedRoomName) {
                        ForEach(rooms, id: \.name) { room in
                            Text(room.name.capitalized).tag(room.name)
                        }
                    }
                }

                Section("Source") {
                    Picker("Source Type", selection: $sourceType) {
                        Text("User Input").tag("user_input")
                        Text("Agent Reasoning").tag("agent_reasoning")
                        Text("Consolidation").tag("consolidation")
                        Text("Snippet").tag("snippet")
                    }
                }
            }
            .navigationTitle("Add Memory")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        Task { await saveMemory() }
                    }
                    .disabled(content.isEmpty || isSaving)
                }
            }
        }
    }

    private func saveMemory() async {
        isSaving = true
        defer { isSaving = false }
        do {
            try await palace.store(
                content: content,
                room: selectedRoomName,
                sourceType: sourceType
            )
            dismiss()
        } catch {
            print("Failed to save memory: \(error)")
        }
    }
}
