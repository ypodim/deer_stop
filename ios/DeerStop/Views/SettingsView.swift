import SwiftUI

/// Lets the user configure server URL and auth token.
/// Values are stored in the iOS Keychain — never in UserDefaults or plain files.
struct SettingsView: View {
    @State private var serverURL  = KeychainService.load(forKey: "serverURL")  ?? ""
    @State private var authToken  = KeychainService.load(forKey: "authToken")  ?? ""
    @State private var saved      = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Node Server") {
                    TextField("https://node.polychronis.gr", text: $serverURL)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                }
                Section {
                    SecureField("Auth token", text: $authToken)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                } header: {
                    Text("Auth Token")
                } footer: {
                    Text("Set in node/.env → AUTH_TOKEN on the Node server.")
                }

                Button("Save") {
                    KeychainService.save(serverURL, forKey: "serverURL")
                    KeychainService.save(authToken,  forKey: "authToken")
                    saved = true
                }
                .disabled(serverURL.isEmpty || authToken.isEmpty)
            }
            .navigationTitle("Settings")
            .alert("Saved", isPresented: $saved) {
                Button("OK", role: .cancel) {}
            }
        }
    }
}
