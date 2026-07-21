// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ODS-iOS",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "ODSApp",
            targets: ["ODSApp"]
        ),
    ],
    dependencies: [
        // SQLite wrapper for Lumen Memory Palace
        .package(url: "https://github.com/groue/GRDB.swift.git", from: "6.29.0"),
        // Apple's MLX for on-device LLM inference (iOS 17+)
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.18.0"),
        // HTTP client for remote API fallback
        .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.9.0"),
    ],
    targets: [
        .target(
            name: "ODSApp",
            dependencies: [
                .product(name: "GRDB", package: "GRDB.swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                "Alamofire",
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "ODSAppTests",
            dependencies: ["ODSApp"]
        ),
    ]
)
