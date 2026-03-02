import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            NavigationStack {
                LiveStreamView()
            }
            .tabItem {
                Label("Live", systemImage: "video.fill")
            }

            ClipBrowserView()
                .tabItem {
                    Label("Clips", systemImage: "film.stack")
                }

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
        }
    }
}
