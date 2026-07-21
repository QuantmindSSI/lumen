import XCTest
@testable import ODSApp

final class ODSAppTests: XCTestCase {

    func testLumenPalaceInit() async throws {
        let palace = LumenPalaceService()
        let status = try await palace.status()
        XCTAssertGreaterThanOrEqual(status.rooms, 1, "Default rooms should be seeded")
    }

    func testStoreAndSearch() async throws {
        let palace = LumenPalaceService()
        try await palace.store(
            content: "User prefers dark mode",
            room: "preferences",
            sourceType: "user_input"
        )
        let results = try await palace.search(query: "dark mode", topK: 5)
        XCTAssertGreaterThanOrEqual(results.count, 1, "Should retrieve stored memory")
    }

    func testCosineSimilarity() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [1, 0, 0]
        XCTAssertEqual(LumenPalaceService.cosineSimilarity(a, b), 1.0, accuracy: 0.001)

        let c: [Float] = [0, 1, 0]
        XCTAssertEqual(LumenPalaceService.cosineSimilarity(a, c), 0.0, accuracy: 0.001)
    }
}
