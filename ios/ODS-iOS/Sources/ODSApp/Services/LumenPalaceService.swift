import Foundation
import GRDB
import Accelerate

/// Lumen Memory Palace service for iOS.
/// Direct Swift port of Lumen's mnemonic force: store, retrieve, forget.
@MainActor
final class LumenPalaceService: ObservableObject {
    private let db = LumenDatabase.shared.dbQueue

    // MARK: - Lifecycle

    init() {
        // Ensure default rooms exist
        try? db.write { db in
            let defaults = ["preferences", "conversations", "projects", "people", "decisions"]
            for name in defaults {
                if try !Room.exists(db, key: ["name": name]) {
                    var room = Room(name: name, description: nil, createdAt: Date())
                    try room.insert(db)
                }
            }
        }
    }

    // MARK: - Store

    func store(content: String, room: String, locus: String? = nil, sourceType: String = "user_input") async throws {
        try db.write { db in
            // Resolve or create room
            var theRoom = try Room.fetchOne(db, key: ["name": room])
            if theRoom == nil {
                theRoom = Room(name: room, description: nil, createdAt: Date())
                try theRoom!.insert(db)
            }
            let roomId = theRoom!.id!

            // Resolve or create locus
            var locusId: Int64? = nil
            if let locusName = locus {
                var theLocus = try Locus
                    .filter(Locus.Columns.roomId == roomId && Locus.Columns.name == locusName)
                    .fetchOne(db)
                if theLocus == nil {
                    theLocus = Locus(roomId: roomId, name: locusName, description: nil, createdAt: Date())
                    try theLocus!.insert(db)
                }
                locusId = theLocus!.id
            }

            // Insert chunk
            var chunk = MemoryChunk(
                id: nil,
                roomId: roomId,
                locusId: locusId,
                content: content,
                sourceType: sourceType,
                createdAt: Date(),
                accessedAt: nil,
                accessCount: 0,
                opticalLevel: 0,
                vmScore: 0.5,
                vector: nil // On-device embedding computed lazily
            )
            try chunk.insert(db)
        }
    }

    // MARK: - Search (Hybrid: BM25 + Dense cosine)

    func search(query: String, room: String? = nil, topK: Int = 5) async throws -> [RetrievedMemory] {
        let results = try db.read { db -> [RetrievedMemory] in
            var bm25Results: [RetrievedMemory] = []
            var denseResults: [RetrievedMemory] = []

            // 1. BM25 via FTS5
            let pattern = FTS5Pattern(matchingAllTokensIn: query) ?? FTS5Pattern(matchingPhrase: query)
            let roomFilter: SQLExpression = room != nil
                ? Room.Columns.name == room!
                : true.sqlExpression

            let bm25Rows = try ChunkFTSRecord
                .joining(required: ChunkFTSRecord.chunk)
                .joining(required: ChunkFTSRecord.chunk.filter(roomFilter).forKey("room_join"))
                .filter(ChunkFTSRecord.Columns.content.match(pattern))
                .order(ChunkFTSRecord.rowID.desc)
                .limit(topK * 2)
                .fetchAll(db)

            for row in bm25Rows {
                if let chunk = try? MemoryChunk.fetchOne(db, id: row.chunkId) {
                    let rank = try? DatabaseRegion.computeFTSRank(
                        db: db,
                        table: "chunk_fts",
                        column: "content",
                        rowID: row.chunkId
                    ) ?? 0.0
                    bm25Results.append(RetrievedMemory(chunk: chunk, score: 1.0 / (1.0 + rank), matchType: .bm25))
                }
            }

            // 2. Dense retrieval (simplified: keyword-in-content fallback until embedding model loaded)
            let denseChunks = try MemoryChunk
                .filter(MemoryChunk.Columns.content.like("%\(query)%"))
                .limit(topK * 2)
                .fetchAll(db)

            for chunk in denseChunks {
                denseResults.append(RetrievedMemory(chunk: chunk, score: 0.5, matchType: .dense))
            }

            // 3. RRF fusion
            return self.fuseRRF(bm25: bm25Results, dense: denseResults, topK: topK)
        }

        // Update access stats for retrieved chunks
        try db.write { db in
            for r in results {
                try db.execute(
                    sql: "UPDATE chunk SET access_count = access_count + 1, accessed_at = ? WHERE id = ?",
                    arguments: [Date(), r.chunk.id!]
                )
            }
        }

        return results
    }

    // MARK: - Assemble Context Window

    func assembleContext(query: String, maxTokens: Int = 2048) async throws -> String {
        let memories = try await search(query: query, topK: 10)
        var window = ""
        var tokenCount = 0
        let approxTokensPerChar = 0.25

        for memory in memories {
            let chunkTokens = Int(Double(memory.chunk.content.count) * approxTokensPerChar)
            if tokenCount + chunkTokens > maxTokens { break }
            window += "[\(memory.matchType.rawValue.uppercased())] \(memory.chunk.content)\n\n"
            tokenCount += chunkTokens
        }
        return window
    }

    // MARK: - Status

    func status() async throws -> PalaceStatus {
        try db.read { db in
            let roomCount = try Room.fetchCount(db)
            let chunkCount = try MemoryChunk.fetchCount(db)
            let locusCount = try Locus.fetchCount(db)
            return PalaceStatus(rooms: roomCount, loci: locusCount, chunks: chunkCount)
        }
    }

    // MARK: - Internal

    private func fuseRRF(bm25: [RetrievedMemory], dense: [RetrievedMemory], topK: Int) -> [RetrievedMemory] {
        let k: Double = 60.0
        var scores: [Int64: (memory: MemoryChunk, score: Double)] = [:]

        for (rank, r) in bm25.enumerated() {
            let id = r.chunk.id!
            let current = scores[id]?.score ?? 0.0
            scores[id] = (r.chunk, current + 1.0 / (k + Double(rank + 1)))
        }

        for (rank, r) in dense.enumerated() {
            let id = r.chunk.id!
            let current = scores[id]?.score ?? 0.0
            scores[id] = (r.chunk, current + 1.0 / (k + Double(rank + 1)))
        }

        return scores.values
            .map { RetrievedMemory(chunk: $0.memory, score: $0.score, matchType: .hybrid) }
            .sorted { $0.score > $1.score }
            .prefix(topK)
            .map { $0 }
    }

    // MARK: - Cosine Similarity (Accelerate)

    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float? {
        guard a.count == b.count, !a.isEmpty else { return nil }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        vDSP_svesq(a, 1, &normA, vDSP_Length(a.count))
        vDSP_svesq(b, 1, &normB, vDSP_Length(b.count))
        guard normA > 0, normB > 0 else { return 0 }
        return dot / (sqrt(normA) * sqrt(normB))
    }
}

// MARK: - Supporting Types

struct PalaceStatus {
    let rooms: Int
    let loci: Int
    let chunks: Int
}

// FTS5 bridge table
struct ChunkFTSRecord: Codable, FetchableRecord {
    var chunkId: Int64
}

extension ChunkFTSRecord {
    static let chunk = belongsTo(MemoryChunk.self, using: ForeignKey(["rowid"], to: ["id"]))

    enum Columns {
        static let content = Column("content")
        static let sourceType = Column("source_type")
        static let rowID = Column("rowid")
    }
}
