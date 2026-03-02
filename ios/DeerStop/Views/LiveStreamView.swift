import SwiftUI
import WebRTC

/// Wraps RTCMTLVideoView (Metal-accelerated) for use in SwiftUI.
struct RTCVideoView: UIViewRepresentable {
    let track: RTCVideoTrack?

    func makeUIView(context: Context) -> RTCMTLVideoView {
        let view = RTCMTLVideoView()
        view.videoContentMode = .scaleAspectFit
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: RTCMTLVideoView, context: Context) {
        // Remove from old track first
        context.coordinator.currentTrack?.remove(uiView)
        track?.add(uiView)
        context.coordinator.currentTrack = track
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    class Coordinator {
        var currentTrack: RTCVideoTrack?
    }
}

/// Main live stream screen. Connects to the signaling server, negotiates
/// WebRTC, and renders the incoming video from the detection server.
@MainActor
struct LiveStreamView: View {
    @StateObject private var signalingService = SignalingService()
    @StateObject private var webRTCService: WebRTCService
    @State private var stats: [String: Double] = [:]
    @State private var statsTimer: Timer?
    @State private var connectionStatus = "Connecting…"

    init() {
        let sig = SignalingService()
        _signalingService = StateObject(wrappedValue: sig)
        _webRTCService    = StateObject(wrappedValue: WebRTCService(signalingService: sig))
    }

    var body: some View {
        ZStack(alignment: .topLeading) {
            Color.black.ignoresSafeArea()

            RTCVideoView(track: webRTCService.remoteVideoTrack)
                .ignoresSafeArea()

            // Stats overlay
            VStack(alignment: .leading, spacing: 4) {
                Text(connectionStatus)
                    .font(.caption.monospaced())
                    .foregroundStyle(.white)
                if let fps = stats["stream_fps"] {
                    Text(String(format: "%.1f fps", fps))
                        .font(.caption.monospaced())
                        .foregroundStyle(.white)
                }
            }
            .padding(8)
            .background(.black.opacity(0.45))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .padding()
        }
        .navigationTitle("Live")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { connectSignaling() }
        .onDisappear {
            signalingService.disconnect()
            webRTCService.close()
            statsTimer?.invalidate()
        }
        .onChange(of: webRTCService.isConnected) { _, connected in
            connectionStatus = connected ? "Connected" : "Connecting…"
        }
    }

    // MARK: - Private

    private func connectSignaling() {
        guard let rawURL   = KeychainService.load(forKey: "serverURL"),
              let token    = KeychainService.load(forKey: "authToken"),
              let base     = URL(string: rawURL),
              let sigURL   = URL(string: "/signaling", relativeTo: base) else {
            connectionStatus = "⚠ Configure server in Settings"
            return
        }
        signalingService.delegate = makeDelegate()
        signalingService.connect(to: sigURL, authToken: token)
        startStatsPolling()
    }

    private func makeDelegate() -> some SignalingDelegate {
        StreamDelegate(webRTCService: webRTCService, signalingService: signalingService)
    }

    private func startStatsPolling() {
        statsTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            Task { @MainActor in
                if let fetched = try? await APIService.shared.fetchStats() {
                    self.stats = fetched
                }
            }
        }
    }
}

// MARK: - Signaling delegate shim (avoids circular reference)

@MainActor
private class StreamDelegate: SignalingDelegate {
    private let webRTCService:    WebRTCService
    private let signalingService: SignalingService

    init(webRTCService: WebRTCService, signalingService: SignalingService) {
        self.webRTCService    = webRTCService
        self.signalingService = signalingService
    }

    func signalingDidReceiveTURNCredentials(username: String, credential: String, turnURL: String) {
        webRTCService.configure(turnURL: turnURL, username: username, credential: credential)
    }

    func signalingDidReceiveOffer(sdp: String) {
        Task { await webRTCService.handleOffer(sdp: sdp) }
    }

    func signalingDidReceiveICECandidate(_ candidate: SignalingMessage.IceCandidate) {
        webRTCService.addICECandidate(candidate)
    }

    func signalingDidDisconnect() {
        webRTCService.close()
    }
}
