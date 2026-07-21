import Foundation
import GRDB

// MARK: - Lumen Palace Data Models (Swift port of Lumen schema)

struct Room: Identifiable, Codable, Hashable {
    var id: Int64?
    var name: String
    var description: String?
    var createdAt: Date

    enum Columns {
        static let id = Column("id")
        static let name = Column("name")
        static let description = Column("description")
        static let createdAt = Column("created_at")
    }
}

struct Locus: Identifiable, Codable, Hashable {
    var id: Int64?
    var roomId: Int64
    var name: String
    var description: String?
    var createdAt: Date

    enum Columns {
        static let id = Column("id")
        static let roomId = Column("room_id")
        static let name = Column("name")
        static let description = Column("description")
        static let createdAt = Column("created_at")
    }
}

struct MemoryChunk: Identifiable, Codable, Hashable {
    var id: Int64?
    var roomId: Int64
    var locusId: Int64?
    var content: String
    var sourceType: String
    var createdAt: Date
    var accessedAt: Date?
    var accessCount: Int
    var opticalLevel: Int // 0=FP32, 1=FP16, 2=INT8, 3=BINARY, 4=RELEASED
    var vmScore: Double
    var vector: [Float]? // Dense embedding (384-dim)

    enum Columns {
        static let id = Column("id")
        static let roomId = Column("room_id")
        static let locusId = Column("locus_id")
        static let content = Column("content")
        static let sourceType = Column("source_type")
        static let createdAt = Column("created_at")
        static let accessedAt = Column("accessed_at")
        static let accessCount = Column("access_count")
        static let opticalLevel = Column("optical_level")
        static let vmScore = Column("vm_score")
        static let vector = Column("vector")
    }
}

struct RetrievedMemory {
    let chunk: MemoryChunk
    let score: Double
    let matchType: MatchType
}

enum MatchType: String {
    case bm25 = "bm25"
    case dense = "dense"
    case hybrid = "hybrid"
}

// MARK: - GRDB Database Setup

final class LumenDatabase {
    static let shared = LumenDatabase()
    let dbQueue: DatabaseQueue

    private init() {
        let fileManager = FileManager.default
        let docs = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dbURL = docs.appendingPathComponent("lumen_palace.sqlite")

        dbQueue = try! DatabaseQueue(path: dbURL.path)
        try! migrator.migrate(dbQueue)
    }

    var migrator: DatabaseMigrator {
        var migrator = DatabaseMigrator()

        migrator.registerMigration("createSchema") { db in
            try db.create(table: "room") { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("name", .text).notNull().unique()
                t.column("description", .text)
                t.column("created_at", .datetime).notNull().defaults(sql: "CURRENT_TIMESTAMP")
            }

            try db.create(table: "locus") { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("room_id", .integer).notNull()
                    .references("room", onDelete: .cascade)
                t.column("name", .text).notNull()
                t.column("description", .text)
                t.column("created_at", .datetime).notNull().defaults(sql: "CURRENT_TIMESTAMP")
            }

            try db.create(table: "chunk") { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("room_id", .integer).notNull()
                    .references("room", onDelete: .cascade)
                t.column("locus_id", .integer)
                    .references("locus", onDelete: .setNull)
                t.column("content", .text).notNull()
                t.column("source_type", .text).notNull().defaults(to: "user_input")
                t.column("created_at", .datetime).notNull().defaults(sql: "CURRENT_TIMESTAMP")
                t.column("accessed_at", .datetime)
                t.column("access_count", .integer).notNull().defaults(to: 0)
                t.column("optical_level", .integer).notNull().defaults(to: 0)
                t.column("vm_score", .double).notNull().defaults(to: 0.5)
                t.column("vector", .blob)
            }

            // FTS5 for BM25 lexical search
            try db.create(virtualTable: "chunk_fts", using: FTS5()) { t in
                t.column("content")
                t.column("source_type")
                t.synchronize(withTable: "chunk")
            }
        }

        return migrator
    }
}
