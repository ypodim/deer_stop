// swift-tools-version: 5.9
// This Package.swift declares the Swift Package dependencies for the DeerStop iOS app.
// Open ios/DeerStop.xcodeproj in Xcode and add this package via:
//   File → Add Package Dependencies → enter the WebRTC package URL below.

import PackageDescription

let package = Package(
    name: "DeerStop",
    platforms: [.iOS(.v17)],
    dependencies: [
        // Google's precompiled WebRTC framework for iOS
        // https://github.com/stasel/WebRTC
        .package(url: "https://github.com/stasel/WebRTC.git", from: "125.0.0"),
    ],
    targets: [
        .target(
            name: "DeerStop",
            dependencies: [
                .product(name: "WebRTC", package: "WebRTC"),
            ],
            path: "DeerStop"
        ),
    ]
)
