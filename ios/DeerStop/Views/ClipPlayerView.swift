import AVKit
import SwiftUI

struct ClipPlayerView: View {
    let clip: Clip
    @State private var player: AVPlayer?

    var body: some View {
        Group {
            if let player {
                VideoPlayer(player: player)
                    .ignoresSafeArea()
            } else {
                ProgressView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.black)
            }
        }
        .navigationTitle(clip.filename)
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            guard let url = APIService.shared.clipVideoURL(filename: clip.filename) else { return }
            // Pass the Bearer auth header directly to AVURLAsset so nginx lets the request through
            var options: [String: Any] = [:]
            if let token = KeychainService.load(forKey: "authToken") {
                options[AVURLAssetHTTPHeaderFieldsKey] = ["Authorization": "Bearer \(token)"]
            }
            let asset = AVURLAsset(url: url, options: options)
            player = AVPlayer(playerItem: AVPlayerItem(asset: asset))
            player?.play()
        }
        .onDisappear { player?.pause() }
    }
}
