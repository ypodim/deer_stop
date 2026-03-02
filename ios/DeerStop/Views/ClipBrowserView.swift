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
                .sorted { !$0.reviewed && $1.reviewed }
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

// MARK: - Thumbnail cell

private struct ClipThumbnailCell: View {
    let clip: Clip
    let onMarkReviewed: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ZStack(alignment: .topTrailing) {
                Rectangle()
                    .fill(Color(.secondarySystemBackground))
                    .aspectRatio(16 / 9, contentMode: .fit)
                    .overlay(Image(systemName: "film").foregroundStyle(.secondary))

                if !clip.reviewed {
                    Circle()
                        .fill(.orange)
                        .frame(width: 10, height: 10)
                        .padding(6)
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
    }
}
