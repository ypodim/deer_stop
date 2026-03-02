import Foundation

/// REST client for the DeerStop detection server API (proxied via Node/nginx).
@MainActor
final class APIService: ObservableObject {
    static let shared = APIService()

    private init() {}

    private var baseURL: URL? {
        guard let raw = KeychainService.load(forKey: "serverURL"),
              let url = URL(string: raw) else { return nil }
        return url
    }

    private var authToken: String? {
        KeychainService.load(forKey: "authToken")
    }

    private func request(path: String, method: String = "GET") -> URLRequest? {
        guard let base = baseURL,
              let url = URL(string: path, relativeTo: base) else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = method
        if let token = authToken {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return req
    }

    // MARK: - Clips

    func fetchClips() async throws -> [Clip] {
        guard let req = request(path: "/api/clips") else {
            throw URLError(.badURL)
        }
        let (data, response) = try await URLSession.shared.data(for: req)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            let body = String(decoding: data, as: UTF8.self)
            throw NSError(domain: "APIService", code: http.statusCode,
                          userInfo: [NSLocalizedDescriptionKey: "Server returned \(http.statusCode): \(body)"])
        }
        do {
            return try JSONDecoder().decode([Clip].self, from: data)
        } catch {
            let preview = String(decoding: data.prefix(200), as: UTF8.self)
            throw NSError(domain: "APIService", code: 0,
                          userInfo: [NSLocalizedDescriptionKey: "JSON decode failed. Response preview: \(preview)"])
        }
    }

    func markReviewed(clipID: String) async throws {
        guard var req = request(path: "/api/clips/\(clipID)/review", method: "POST") else {
            throw URLError(.badURL)
        }
        req.httpBody = Data()
        _ = try await URLSession.shared.data(for: req)
    }

    func clipVideoURL(filename: String) -> URL? {
        guard let base = baseURL else { return nil }
        return URL(string: "/api/clips/files/\(filename)", relativeTo: base)
    }

    func clipThumbnailURL(filename: String) -> URL? {
        guard let base = baseURL else { return nil }
        return URL(string: "/api/clips/files/\(filename)", relativeTo: base)
    }

    func clipPreviewURL(filename: String) -> URL? {
        guard let base = baseURL else { return nil }
        return URL(string: "/api/clips/files/\(filename)", relativeTo: base)
    }

    /// Build a URLRequest with auth headers for image/video loading.
    func authorizedRequest(for url: URL) -> URLRequest {
        var req = URLRequest(url: url)
        if let token = authToken {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return req
    }

    // MARK: - Stats

    func fetchStats() async throws -> [String: Double] {
        guard let req = request(path: "/api/stats") else {
            throw URLError(.badURL)
        }
        let (data, _) = try await URLSession.shared.data(for: req)
        return (try? JSONDecoder().decode([String: Double].self, from: data)) ?? [:]
    }
}
