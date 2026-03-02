import Foundation
import UserNotifications

/// Connects to the detection server's /events SSE endpoint and fires local
/// notifications whenever a clip is saved.
final class SSEService: NSObject, URLSessionDataDelegate {
    static let shared = SSEService()

    private var task: URLSessionDataTask?
    private lazy var session = URLSession(
        configuration: .default,
        delegate: self,
        delegateQueue: nil
    )

    private override init() { super.init() }

    // MARK: - Public

    func start() {
        guard let raw    = KeychainService.load(forKey: "serverURL"),
              let base   = URL(string: raw),
              let url    = URL(string: "/api/events", relativeTo: base) else { return }
        var req = URLRequest(url: url, timeoutInterval: .infinity)
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        if let token = KeychainService.load(forKey: "authToken") {
            req.setValue("Basic \(token)", forHTTPHeaderField: "Authorization")
        }
        task = session.dataTask(with: req)
        task?.resume()
    }

    func stop() {
        task?.cancel()
        task = nil
    }

    // MARK: - URLSessionDataDelegate

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        guard let text = String(data: data, encoding: .utf8) else { return }
        for line in text.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("data:") else { continue }
            let payload = String(trimmed.dropFirst(5)).trimmingCharacters(in: .whitespaces)
            handleEvent(payload)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        // Reconnect after a short delay on any error
        DispatchQueue.global().asyncAfter(deadline: .now() + 5) { [weak self] in
            self?.start()
        }
    }

    // MARK: - Private

    private func handleEvent(_ jsonString: String) {
        guard let data  = jsonString.data(using: .utf8),
              let clip  = try? JSONDecoder().decode(Clip.self, from: data) else { return }
        fireNotification(for: clip)
    }

    private func fireNotification(for clip: Clip) {
        let content = UNMutableNotificationContent()
        content.title = "Detection recorded"
        content.body  = clip.filename
        content.sound = .default

        let req = UNNotificationRequest(
            identifier: clip.id,
            content: content,
            trigger: nil  // deliver immediately
        )
        UNUserNotificationCenter.current().add(req)
    }
}
