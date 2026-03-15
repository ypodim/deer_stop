import AVFoundation
import AVKit
import SwiftUI

struct ClipBrowserView: View {
    @State private var clips: [Clip] = []
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var isEditing = false
    @State private var selected: Set<String> = []
    @State private var showDeleteConfirm = false

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
                                if isEditing {
                                    ClipThumbnailCell(clip: clip, isSelected: selected.contains(clip.id)) {
                                        markReviewed(clip)
                                    }
                                    .onTapGesture { toggleSelection(clip) }
                                } else {
                                    NavigationLink(destination: ClipPlayerView(clip: clip)) {
                                        ClipThumbnailCell(clip: clip) {
                                            markReviewed(clip)
                                        }
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Clips")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    if !clips.isEmpty {
                        Button(isEditing ? "Done" : "Select") {
                            isEditing.toggle()
                            if !isEditing { selected.removeAll() }
                        }
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    if isEditing && !selected.isEmpty {
                        Button(role: .destructive) {
                            showDeleteConfirm = true
                        } label: {
                            Label("Delete \(selected.count)", systemImage: "trash")
                        }
                        .tint(.red)
                    } else {
                        Button { Task { await loadClips() } } label: {
                            Image(systemName: "arrow.clockwise")
                        }
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
            .alert("Delete \(selected.count) clip\(selected.count == 1 ? "" : "s")?", isPresented: $showDeleteConfirm) {
                Button("Delete", role: .destructive) { deleteSelected() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This will permanently delete the selected clips and their previews.")
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

    private func toggleSelection(_ clip: Clip) {
        if selected.contains(clip.id) {
            selected.remove(clip.id)
        } else {
            selected.insert(clip.id)
        }
    }

    private func deleteSelected() {
        let ids = selected
        Task {
            for id in ids {
                do {
                    try await APIService.shared.deleteClip(clipID: id)
                    clips.removeAll { $0.id == id }
                } catch {
                    errorMessage = error.localizedDescription
                    break
                }
            }
            selected.removeAll()
            isEditing = false
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
    var isSelected: Bool = false
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

                // Selection / review indicator (top-right)
                Image(systemName: isSelected ? "checkmark.circle.fill" : (clip.reviewed ? "checkmark.circle.fill" : "circle"))
                    .font(.title3)
                    .foregroundStyle(isSelected ? .blue : (clip.reviewed ? .green : .white))
                    .shadow(color: .black.opacity(0.5), radius: 2)
                    .frame(width: 44, height: 44)

                // Class label overlay (bottom-left)
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
