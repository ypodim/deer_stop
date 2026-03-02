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
            var req = URLRequest(url: url)
            if let token = KeychainService.load(forKey: "authToken") {
                req.setValue("Basic \(token)", forHTTPHeaderField: "Authorization")
            }
            let asset = AVURLAsset(url: url)  // AVPlayer handles auth via URLSession config
            player = AVPlayer(playerItem: AVPlayerItem(asset: asset))
            player?.play()
        }
        .onDisappear { player?.pause() }
    }
}
