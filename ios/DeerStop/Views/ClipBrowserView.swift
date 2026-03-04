import AVFoundation
import AVKit
import SwiftUI

struct ClipBrowserView: View {
    @State private var clips: [Clip] = []
    @State private var isLoading = false
    @State private var errorMessage: String?

    private let columns = [GridItem(.adaptive(minimum: 160), spacing: 12)]

    var body: some View {
        NavigationStack {
            Group {
                if isLoading {
                    ProgressView()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if clips.isEmpty {
                    ContentUnavailableView(
                        "No clips yet",
                        systemImage: "film.stack",
                        description: Text("Clips appear here when the detection server records a wildlife event.")
                    )
                } else {
                    ScrollView {
                        LazyVGrid(columns: columns, spacing: 12) {
                            ForEach(clips) { clip in
                                NavigationLink(destination: ClipPlayerView(clip: clip)) {
                                    ClipThumbnailCell(clip: clip) {
                                        markReviewed(clip)
                                    }
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Clips")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { Task { await loadClips() } } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .alert("Error", isPresented: Binding(
                get: { errorMessage != nil },
                set: { if !$0 { errorMessage = nil } }
            )) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(errorMessage ?? "")
            }
        }
        .task { await loadClips() }
    }

    // MARK: - Private

    private func loadClips() async {
        isLoading = true
        defer { isLoading = false }
        do {
            clips = try await APIService.shared.fetchClips()
                .sorted { $0.id > $1.id }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func markReviewed(_ clip: Clip) {
        Task {
            try? await APIService.shared.markReviewed(clipID: clip.id)
            if let idx = clips.firstIndex(where: { $0.id == clip.id }) {
                clips[idx].reviewed = true
            }
        }
    }
}

// MARK: - Looping preview player (UIKit-backed, muted, no controls)

private struct LoopingPreviewPlayer: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> UIView {
        let view = PlayerUIView(player: player)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}

    private class PlayerUIView: UIView {
        private let playerLayer = AVPlayerLayer()

        init(player: AVPlayer) {
            super.init(frame: .zero)
            playerLayer.player = player
            playerLayer.videoGravity = .resizeAspectFill
            layer.addSublayer(playerLayer)

            // Loop playback
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(playerDidFinish),
                name: .AVPlayerItemDidPlayToEndTime,
                object: player.currentItem
            )
        }

        required init?(coder: NSCoder) { fatalError() }

        override func layoutSubviews() {
            super.layoutSubviews()
            playerLayer.frame = bounds
        }

        @objc private func playerDidFinish() {
            playerLayer.player?.seek(to: .zero)
            playerLayer.player?.play()
        }
    }
}

// MARK: - Thumbnail cell

private struct ClipThumbnailCell: View {
    let clip: Clip
    let onMarkReviewed: () -> Void

    @State private var previewPlayer: AVPlayer?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ZStack(alignment: .topTrailing) {
                // Preview video or placeholder
                Group {
                    if let player = previewPlayer {
                        LoopingPreviewPlayer(player: player)
                            .aspectRatio(16 / 9, contentMode: .fit)
                    } else {
                        Rectangle()
                            .fill(Color(.secondarySystemBackground))
                            .aspectRatio(16 / 9, contentMode: .fit)
                            .overlay(ProgressView())
                    }
                }

                // Unreviewed indicator
                if !clip.reviewed {
                    Circle()
                        .fill(.orange)
                        .frame(width: 10, height: 10)
                        .padding(6)
                }

                // Class label overlay
                VStack {
                    Spacer()
                    HStack {
                        Text(clip.label)
                            .font(.caption2.bold())
                            .foregroundStyle(.white)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.black.opacity(0.6))
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                        Spacer()
                    }
                    .padding(4)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))

            Text(clip.filename)
                .font(.caption)
                .lineLimit(1)
                .foregroundStyle(.primary)

            Button("Mark reviewed", action: onMarkReviewed)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .disabled(clip.reviewed)
        }
        .task { await loadPreview() }
        .onDisappear {
            previewPlayer?.pause()
            previewPlayer = nil
        }
    }

    private func loadPreview() async {
        guard let url = APIService.shared.clipPreviewURL(filename: clip.previewFilename) else { return }

        // Check if preview exists (HEAD request)
        var headReq = APIService.shared.authorizedRequest(for: url)
        headReq.httpMethod = "HEAD"
        guard let (_, resp) = try? await URLSession.shared.data(for: headReq),
              (resp as? HTTPURLResponse)?.statusCode == 200 else { return }

        // Load preview video with auth
        let token = KeychainService.load(forKey: "authToken") ?? ""
        let asset = AVURLAsset(url: url, options: [
            "AVURLAssetHTTPHeaderFieldsKey": ["Authorization": "Bearer \(token)"]
        ])
        let item = AVPlayerItem(asset: asset)
        let player = AVPlayer(playerItem: item)
        previewPlayer = player
        player.play()
    }
}
