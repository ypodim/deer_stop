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
        let (data, _) = try await URLSession.shared.data(for: req)
        return try JSONDecoder().decode([Clip].self, from: data)
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

    // MARK: - Stats

    func fetchStats() async throws -> [String: Double] {
        guard let req = request(path: "/api/stats") else {
            throw URLError(.badURL)
        }
        let (data, _) = try await URLSession.shared.data(for: req)
        return (try? JSONDecoder().decode([String: Double].self, from: data)) ?? [:]
    }
}
