import Foundation

struct Clip: Identifiable, Codable {
    let id: String
    let path: String
    let className: String?
    let confidence: Double?
    let timestamp: String?
    var reviewed: Bool

    enum CodingKeys: String, CodingKey {
        case id, path, confidence, timestamp, reviewed
        case className = "class_name"
    }

    /// Filename portion of `path`, used to build the streaming URL.
    var filename: String {
        URL(fileURLWithPath: path).lastPathComponent
    }

    /// Preview MP4 filename (e.g. "2026-02-26T16-22-03_person_preview.mp4").
    var previewFilename: String {
        let base = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        return "\(base)_preview.mp4"
    }

    /// Human-readable label like "person 93%".
    var label: String {
        let cls = className ?? "unknown"
        if let conf = confidence {
            return "\(cls) \(Int(conf * 100))%"
        }
        return cls
    }
}
