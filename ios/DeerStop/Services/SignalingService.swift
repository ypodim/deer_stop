import Foundation

/// Message types exchanged with the Node signaling server.
enum SignalingMessage: Codable {
    case turnCredentials(username: String, credential: String, turnURL: String)
    case viewerConnected(viewerID: String)  // unused on iOS (viewer side)
    case sdpOffer(viewerID: String, sdp: String)
    case sdpAnswer(viewerID: String, sdp: String)
    case iceCandidate(viewerID: String, candidate: IceCandidate)
    case streamerDisconnected

    struct IceCandidate: Codable {
        let candidate: String
        let sdpMid: String?
        let sdpMLineIndex: Int?
    }
}

/// Manages the persistent WebSocket connection to the Node signaling server.
/// Decodes incoming messages and forwards them to a delegate.
@MainActor
protocol SignalingDelegate: AnyObject {
    func signalingDidReceiveTURNCredentials(username: String, credential: String, turnURL: String)
    func signalingDidReceiveOffer(sdp: String)
    func signalingDidReceiveICECandidate(_ candidate: SignalingMessage.IceCandidate)
    func signalingDidDisconnect()
}

@MainActor
final class SignalingService: NSObject, URLSessionWebSocketDelegate {
    weak var delegate: SignalingDelegate?

    private var webSocketTask: URLSessionWebSocketTask?
    private lazy var session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    private var viewerID: String?  // our ID assigned by the server

    // MARK: - Public

    func connect(to signalingURL: URL, authToken: String) {
        var req = URLRequest(url: signalingURL)
        req.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        webSocketTask = session.webSocketTask(with: req)
        webSocketTask?.resume()
        receiveNext()
    }

    func disconnect() {
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
    }

    func sendAnswer(sdp: String) {
        guard let vid = viewerID else { return }
        send(["type": "sdp_answer", "viewer_id": vid, "sdp": sdp])
    }

    func sendICECandidate(_ candidate: SignalingMessage.IceCandidate) {
        guard let vid = viewerID else { return }
        var dict: [String: Any] = [
            "type": "ice_candidate",
            "viewer_id": vid,
            "candidate": ["candidate": candidate.candidate,
                          "sdpMid": candidate.sdpMid as Any,
                          "sdpMLineIndex": candidate.sdpMLineIndex as Any],
        ]
        send(dict)
    }

    // MARK: - Private

    private func receiveNext() {
        webSocketTask?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let msg):
                if case .string(let text) = msg {
                    Task { @MainActor in self.handleText(text) }
                }
                self.receiveNext()
            case .failure:
                Task { @MainActor in self.delegate?.signalingDidDisconnect() }
            }
        }
    }

    private func handleText(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return }

        switch type {
        case "turn_credentials":
            let username   = json["username"]   as? String ?? ""
            let credential = json["credential"] as? String ?? ""
            let turnURL    = json["turn_url"]   as? String ?? ""
            // Register as viewer after receiving credentials
            send(["type": "register", "role": "viewer"])
            delegate?.signalingDidReceiveTURNCredentials(
                username: username, credential: credential, turnURL: turnURL)

        case "sdp_offer":
            viewerID = json["viewer_id"] as? String
            let sdp  = json["sdp"] as? String ?? ""
            delegate?.signalingDidReceiveOffer(sdp: sdp)

        case "ice_candidate":
            if let raw       = json["candidate"] as? [String: Any],
               let candidate = raw["candidate"] as? String {
                let ic = SignalingMessage.IceCandidate(
                    candidate:     candidate,
                    sdpMid:        raw["sdpMid"] as? String,
                    sdpMLineIndex: raw["sdpMLineIndex"] as? Int
                )
                delegate?.signalingDidReceiveICECandidate(ic)
            }

        case "streamer_disconnected":
            delegate?.signalingDidDisconnect()

        default:
            break
        }
    }

    private func send(_ dict: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let text = String(data: data, encoding: .utf8) else { return }
        webSocketTask?.send(.string(text)) { _ in }
    }

    // MARK: - URLSessionWebSocketDelegate

    nonisolated func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                     didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        Task { @MainActor in self.delegate?.signalingDidDisconnect() }
    }
}
