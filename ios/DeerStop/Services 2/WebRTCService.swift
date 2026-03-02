import Foundation
import WebRTC

/// Manages one RTCPeerConnection session as a viewer.
/// Receives an SDP offer from the signaling server, creates an answer,
/// and surfaces the incoming video track for display.
@MainActor
final class WebRTCService: NSObject, ObservableObject {
    @Published var remoteVideoTrack: RTCVideoTrack?
    @Published var isConnected = false

    private var peerConnection: RTCPeerConnection?
    private let factory: RTCPeerConnectionFactory
    private weak var signalingService: SignalingService?

    // MARK: - Init

    init(signalingService: SignalingService) {
        RTCInitializeSSL()
        let encoderFactory = RTCDefaultVideoEncoderFactory()
        let decoderFactory = RTCDefaultVideoDecoderFactory()
        factory = RTCPeerConnectionFactory(
            encoderFactory: encoderFactory,
            decoderFactory: decoderFactory
        )
        self.signalingService = signalingService
        super.init()
    }

    // MARK: - Public

    /// Call after TURN credentials are available.
    func configure(turnURL: String, username: String, credential: String) {
        let iceServer = RTCIceServer(
            urlStrings: [turnURL],
            username: username,
            credential: credential
        )
        let config = RTCConfiguration()
        config.iceServers = [iceServer]
        config.sdpSemantics = .unifiedPlan
        config.continualGatheringPolicy = .gatherContinually

        let constraints = RTCMediaConstraints(
            mandatoryConstraints: nil,
            optionalConstraints: ["DtlsSrtpKeyAgreement": "true"]
        )
        peerConnection = factory.peerConnection(
            with: config, constraints: constraints, delegate: self
        )
    }

    /// Handle an SDP offer received from the detection server via the signaling relay.
    func handleOffer(sdp: String) async {
        guard let pc = peerConnection else { return }
        let remoteDesc = RTCSessionDescription(type: .offer, sdp: sdp)
        await withCheckedContinuation { cont in
            pc.setRemoteDescription(remoteDesc) { _ in cont.resume() }
        }
        let constraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: nil)
        let answer = await withCheckedContinuation { (cont: CheckedContinuation<RTCSessionDescription, Never>) in
            pc.answer(for: constraints) { sdp, _ in
                if let sdp { cont.resume(returning: sdp) }
            }
        }
        await withCheckedContinuation { cont in
            pc.setLocalDescription(answer) { _ in cont.resume() }
        }
        signalingService?.sendAnswer(sdp: answer.sdp)
    }

    /// Add a remote ICE candidate from the detection server.
    func addICECandidate(_ ic: SignalingMessage.IceCandidate) {
        guard let pc = peerConnection else { return }
        let candidate = RTCIceCandidate(
            sdp:           ic.candidate,
            sdpMLineIndex: Int32(ic.sdpMLineIndex ?? 0),
            sdpMid:        ic.sdpMid
        )
        pc.add(candidate)
    }

    func close() {
        peerConnection?.close()
        peerConnection = nil
        remoteVideoTrack = nil
        isConnected = false
    }
}

// MARK: - RTCPeerConnectionDelegate

extension WebRTCService: RTCPeerConnectionDelegate {
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection,
                         didChange stateChanged: RTCSignalingState) {}

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didAdd stream: RTCMediaStream) {
        guard let track = stream.videoTracks.first else { return }
        Task { @MainActor in
            self.remoteVideoTrack = track
        }
    }

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didRemove stream: RTCMediaStream) {
        Task { @MainActor in self.remoteVideoTrack = nil }
    }

    nonisolated func peerConnectionShouldNegotiate(_ peerConnection: RTCPeerConnection) {}

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection,
                         didChange newState: RTCIceConnectionState) {
        Task { @MainActor in
            self.isConnected = (newState == .connected || newState == .completed)
        }
    }

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection,
                         didChange newState: RTCIceGatheringState) {}

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection,
                         didGenerate candidate: RTCIceCandidate) {
        let ic = SignalingMessage.IceCandidate(
            candidate:     candidate.sdp,
            sdpMid:        candidate.sdpMid,
            sdpMLineIndex: Int(candidate.sdpMLineIndex)
        )
        Task { @MainActor in self.signalingService?.sendICECandidate(ic) }
    }

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection,
                         didRemove candidates: [RTCIceCandidate]) {}

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection,
                         didOpen dataChannel: RTCDataChannel) {}

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection,
                         didAdd rtpReceiver: RTCRtpReceiver,
                         streams mediaStreams: [RTCMediaStream]) {
        guard let track = rtpReceiver.track as? RTCVideoTrack else { return }
        Task { @MainActor in self.remoteVideoTrack = track }
    }
}
