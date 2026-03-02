import Foundation

struct Clip: Identifiable, Codable {
    let id: String
    let path: String
    var reviewed: Bool

    /// Filename portion of `path`, used to build the streaming URL.
    var filename: String {
        URL(fileURLWithPath: path).lastPathComponent
    }
}
