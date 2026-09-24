// KreaImage.app: a native macOS shell around the Krea 2 Turbo engine's web UI.
//
// The app owns the engine server (ui/server.py, run with the project's .venv Python): it starts
// it on 127.0.0.1 (or reuses one that is already running), shows a native loading screen while the
// engine pages its weights into memory, then shows the UI in a WKWebView. Quitting the app
// or closing its window shuts the server down so its memory is freed; the server is also started
// with --parent-pid so it exits by itself if the app is force-quit or crashes.
//
// Configuration (UserDefaults, domain = bundle id):
//   projectPath  - the project directory (default: baked in at build time)
//   outputsPath  - optional override of where images are saved (default: <project>/outputs)
import AppKit
import Darwin
import SwiftUI
import WebKit

// MARK: - Configuration

enum AppConfig {
  static let bundleID = Bundle.main.bundleIdentifier ?? "local.kreaimage.studio"
  static let preferredPort = 7861  // stable origin keeps the UI's saved settings (localStorage)

  static var projectPath: String {
    get {
      if let p = UserDefaults.standard.string(forKey: "projectPath"), !p.isEmpty { return p }
      return (Bundle.main.object(forInfoDictionaryKey: "KreaProjectPath") as? String) ?? ""
    }
    set { UserDefaults.standard.set(newValue, forKey: "projectPath") }
  }
  static var projectURL: URL { URL(fileURLWithPath: projectPath, isDirectory: true) }
  static var pythonURL: URL { projectURL.appendingPathComponent(".venv/bin/python") }
  static var serverScript: URL { projectURL.appendingPathComponent("ui/server.py") }
  static var logsDir: URL { projectURL.appendingPathComponent("logs", isDirectory: true) }
  static var logURL: URL { logsDir.appendingPathComponent("server.log") }
  static var stateURL: URL { logsDir.appendingPathComponent("server.state.json") }
  static var outputsOverride: String? {
    let o = UserDefaults.standard.string(forKey: "outputsPath")
    return (o?.isEmpty ?? true) ? nil : o
  }
  static var outputsURL: URL {
    outputsOverride.map { URL(fileURLWithPath: $0, isDirectory: true) }
      ?? projectURL.appendingPathComponent("outputs", isDirectory: true)
  }

  /// nil if the project folder looks usable, else a human-readable problem.
  static func problem() -> String? {
    let fm = FileManager.default
    if projectPath.isEmpty { return "No project folder is configured." }
    if !fm.fileExists(atPath: serverScript.path) {
      return "Couldn't find ui/server.py in \(projectPath). Choose the krea_metal project folder."
    }
    if !fm.isExecutableFile(atPath: pythonURL.path) {
      return "Couldn't find the Python environment (.venv/bin/python) in \(projectPath). Create it first (see README)."
    }
    return nil
  }
}

// MARK: - Engine server lifecycle

@MainActor
final class ServerController: ObservableObject {
  static let shared = ServerController()

  enum Phase: Equatable {
    case idle
    case starting
    case loading
    case ready(URL)
    case failed(String)
    case stopping
  }

  @Published private(set) var phase: Phase = .idle
  @Published private(set) var elapsed: TimeInterval = 0
  @Published private(set) var detail = ""

  private var process: Process?
  private var pid: pid_t = 0
  private var port = 0
  private var adopted = false  // reusing a server this app instance did not launch
  private var stopping = false
  private var pollTask: Task<Void, Never>?
  private var startedAt = Date()

  /// Only ever talks to 127.0.0.1.
  private let session: URLSession = {
    let cfg = URLSessionConfiguration.ephemeral
    cfg.timeoutIntervalForRequest = 2
    cfg.connectionProxyDictionary = [:]
    return URLSession(configuration: cfg)
  }()

  var hasServer: Bool { pid > 0 }
  private var baseURL: URL { URL(string: "http://127.0.0.1:\(port)/")! }

  func start() {
    pollTask?.cancel()
    if let problem = AppConfig.problem() {
      phase = .failed(problem)
      return
    }
    phase = .starting
    detail = "Starting the engine server…"
    startedAt = Date()
    elapsed = 0
    Task {
      if let (runningPid, runningPort) = await findRunningServer() {
        pid = runningPid
        port = runningPort
        adopted = true
        detail = "Connected to the running engine server (pid \(runningPid))."
        appendLog("app: reusing running server pid \(runningPid) on port \(runningPort)")
      } else {
        do {
          try launch()
        } catch {
          phase = .failed("Couldn't start the engine server: \(error.localizedDescription)")
          return
        }
      }
      poll()
    }
  }

  func restart() {
    Task {
      await stop()
      start()
    }
  }

  /// Ask the server to exit, escalating to SIGTERM / SIGKILL; returns once it is gone.
  func stop() async {
    pollTask?.cancel()
    guard pid > 0 else { return }
    stopping = true
    phase = .stopping
    let target = pid
    var req = URLRequest(url: baseURL.appendingPathComponent("api/shutdown"))
    req.httpMethod = "POST"
    _ = try? await session.data(for: req)
    if !(await waitForExit(target, seconds: 6)) {
      appendLog("app: server did not exit after shutdown request; sending SIGTERM")
      kill(target, SIGTERM)
      if !(await waitForExit(target, seconds: 3)) {
        appendLog("app: sending SIGKILL")
        kill(target, SIGKILL)
        _ = await waitForExit(target, seconds: 2)
      }
    }
    appendLog("app: server pid \(target) stopped")
    process = nil
    pid = 0
    adopted = false
    stopping = false
    phase = .idle
  }

  /// Quit path. AppKit runs a modal run loop while termination is pending, in which MainActor
  /// tasks are not serviced, so the shutdown runs synchronously on a background thread.
  struct StopTarget { let pid: pid_t; let port: Int; let child: Process? }

  func beginStop() -> StopTarget? {
    pollTask?.cancel()
    guard pid > 0 else { return nil }
    stopping = true
    phase = .stopping
    return StopTarget(pid: pid, port: port, child: process)
  }

  nonisolated static func stopBlocking(_ t: StopTarget) {
    func alive() -> Bool {
      if let c = t.child { return c.isRunning }
      return kill(t.pid, 0) == 0 || errno == EPERM
    }
    func wait(_ seconds: Double) -> Bool {
      let deadline = Date().addingTimeInterval(seconds)
      while Date() < deadline {
        if !alive() { return true }
        usleep(100_000)
      }
      return !alive()
    }
    var req = URLRequest(url: URL(string: "http://127.0.0.1:\(t.port)/api/shutdown")!, timeoutInterval: 2)
    req.httpMethod = "POST"
    let done = DispatchSemaphore(value: 0)
    let cfg = URLSessionConfiguration.ephemeral
    cfg.connectionProxyDictionary = [:]
    URLSession(configuration: cfg).dataTask(with: req) { _, _, _ in done.signal() }.resume()
    _ = done.wait(timeout: .now() + 2.5)
    if !wait(6) {
      log("server did not exit after shutdown request; sending SIGTERM")
      kill(t.pid, SIGTERM)
      if !wait(3) {
        log("sending SIGKILL")
        kill(t.pid, SIGKILL)
        _ = wait(2)
      }
    }
    log("server pid \(t.pid) stopped")
  }

  // MARK: internals

  private func isAlive(_ p: pid_t) -> Bool {
    if let proc = process, proc.processIdentifier == p { return proc.isRunning }  // our child (reaped by Process)
    return kill(p, 0) == 0 || errno == EPERM
  }

  private func waitForExit(_ p: pid_t, seconds: Double) async -> Bool {
    let deadline = Date().addingTimeInterval(seconds)
    while Date() < deadline {
      if !isAlive(p) { return true }
      try? await Task.sleep(nanoseconds: 100_000_000)
    }
    return !isAlive(p)
  }

  private struct Status: Decodable {
    let server: String?
    let pid: Int?
    let engine: String
    let error: String?
  }

  private func status(port: Int) async -> Status? {
    guard let url = URL(string: "http://127.0.0.1:\(port)/api/status"),
          let (data, resp) = try? await session.data(from: url),
          (resp as? HTTPURLResponse)?.statusCode == 200
    else { return nil }
    return try? JSONDecoder().decode(Status.self, from: data)
  }

  /// A server recorded in logs/server.state.json (or answering on the preferred port) that is ours.
  private func findRunningServer() async -> (pid_t, Int)? {
    var candidates: [Int] = []
    if let data = try? Data(contentsOf: AppConfig.stateURL),
       let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
       let p = obj["port"] as? Int, let spid = obj["pid"] as? Int, kill(pid_t(spid), 0) == 0 {
      candidates.append(p)
    }
    candidates.append(AppConfig.preferredPort)
    for p in candidates {
      if let s = await status(port: p), s.server == "krea-image-studio", let spid = s.pid {
        return (pid_t(spid), p)
      }
    }
    return nil
  }

  private func portIsFree(_ p: Int) -> Bool {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { return false }
    defer { close(fd) }
    // Like the server's own socket: a port left in TIME_WAIT by a previous run counts as free.
    var one: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))
    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = in_port_t(UInt16(p).bigEndian)
    addr.sin_addr.s_addr = inet_addr("127.0.0.1")
    return withUnsafePointer(to: &addr) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0 }
    }
  }

  private func freePort() -> Int {
    if portIsFree(AppConfig.preferredPort) { return AppConfig.preferredPort }
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    defer { close(fd) }
    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = 0
    addr.sin_addr.s_addr = inet_addr("127.0.0.1")
    var len = socklen_t(MemoryLayout<sockaddr_in>.size)
    _ = withUnsafePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, len) } }
    _ = withUnsafeMutablePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) } }
    return Int(UInt16(bigEndian: addr.sin_port))
  }

  nonisolated static func log(_ line: String) {
    try? FileManager.default.createDirectory(at: AppConfig.logsDir, withIntermediateDirectories: true)
    let stamp = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate, .withFullTime, .withSpaceBetweenDateAndTime])
    let data = "[\(stamp)] app: \(line)\n".data(using: .utf8)!
    if let h = try? FileHandle(forWritingTo: AppConfig.logURL) {
      h.seekToEndOfFile()
      h.write(data)
      try? h.close()
    }
  }

  private func appendLog(_ line: String) {
    try? FileManager.default.createDirectory(at: AppConfig.logsDir, withIntermediateDirectories: true)
    let stamp = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate, .withFullTime, .withSpaceBetweenDateAndTime])
    let data = "[\(stamp)] \(line)\n".data(using: .utf8)!
    if let h = try? FileHandle(forWritingTo: AppConfig.logURL) {
      h.seekToEndOfFile()
      h.write(data)
      try? h.close()
    } else {
      try? data.write(to: AppConfig.logURL)
    }
  }

  private func launch() throws {
    port = freePort()
    try FileManager.default.createDirectory(at: AppConfig.logsDir, withIntermediateDirectories: true)
    if !FileManager.default.fileExists(atPath: AppConfig.logURL.path) {
      FileManager.default.createFile(atPath: AppConfig.logURL.path, contents: nil)
    }
    appendLog("app: starting engine server on 127.0.0.1:\(port)")
    let log = try FileHandle(forWritingTo: AppConfig.logURL)
    log.seekToEndOfFile()

    let p = Process()
    p.executableURL = AppConfig.pythonURL
    var args = ["-u", AppConfig.serverScript.path, "--host", "127.0.0.1", "--port", String(port),
                "--log", AppConfig.logURL.path, "--quiet", "--parent-pid", String(getpid())]
    if let o = AppConfig.outputsOverride { args += ["--outputs", o] }
    p.arguments = args
    p.currentDirectoryURL = AppConfig.projectURL
    var env = ProcessInfo.processInfo.environment
    env["PYTHONUNBUFFERED"] = "1"
    p.environment = env
    p.standardInput = FileHandle.nullDevice
    p.standardOutput = log  // catches anything printed before the server's own log tee starts
    p.standardError = log
    p.terminationHandler = { proc in
      let code = proc.terminationStatus
      let childPid = proc.processIdentifier
      Task { @MainActor in ServerController.shared.childExited(childPid, status: code) }
    }
    try p.run()
    process = p
    pid = p.processIdentifier
    adopted = false
  }

  fileprivate func childExited(_ childPid: pid_t, status code: Int32) {
    guard !stopping, pid == childPid else { return }
    pid = 0
    process = nil
    pollTask?.cancel()
    appendLog("app: engine server exited unexpectedly (status \(code))")
    phase = .failed("The engine server stopped unexpectedly (exit status \(code)).")
  }

  private func lastLogLine() -> String {
    guard let h = try? FileHandle(forReadingFrom: AppConfig.logURL) else { return "" }
    defer { try? h.close() }
    let size = (try? h.seekToEnd()) ?? 0
    try? h.seek(toOffset: size > 4096 ? size - 4096 : 0)
    let text = String(decoding: h.readDataToEndOfFile(), as: UTF8.self)
    let line = text.split(separator: "\n").last.map(String.init) ?? ""
    // drop the "[date time] " stamp
    if line.hasPrefix("["), let close = line.firstIndex(of: "]") {
      return String(line[line.index(after: close)...]).trimmingCharacters(in: .whitespaces)
    }
    return line
  }

  private func poll() {
    pollTask?.cancel()
    pollTask = Task { [weak self] in
      var sawServer = false
      while !Task.isCancelled {
        guard let self else { return }
        self.elapsed = Date().timeIntervalSince(self.startedAt)
        if self.adopted && !self.isAlive(self.pid) {
          self.pid = 0
          self.phase = .failed("The engine server stopped unexpectedly.")
          return
        }
        if let s = await self.status(port: self.port) {
          sawServer = true
          switch s.engine {
          case "ready":
            if case .ready = self.phase {} else { self.phase = .ready(self.baseURL) }
          case "error":
            self.phase = .failed(s.error.map { "The engine failed to load: \($0)" } ?? "The engine failed to load.")
            return
          default:
            self.phase = .loading
            self.detail = self.lastLogLine()
          }
        } else if !sawServer {
          self.detail = self.lastLogLine()
          if self.elapsed > 90 {
            self.phase = .failed("The engine server did not respond within 90 seconds.")
            return
          }
        }
        let ready: Bool = { if case .ready = self.phase { return true } else { return false } }()
        try? await Task.sleep(nanoseconds: ready ? 3_000_000_000 : 700_000_000)
      }
    }
  }
}

// MARK: - Views

struct RootView: View {
  @ObservedObject var server = ServerController.shared

  var body: some View {
    Group {
      switch server.phase {
      case .ready(let url):
        WebView(url: url)
      case .failed(let message):
        ErrorView(message: message)
      case .stopping:
        StatusView(title: "Stopping the engine…", detail: "Freeing model memory.", elapsed: nil)
      case .idle, .starting, .loading:
        StatusView(title: server.phase == .loading ? "Loading the model…" : "Starting the engine…",
                   detail: server.detail, elapsed: server.elapsed)
      }
    }
    .frame(minWidth: 900, minHeight: 640)
  }
}

struct StatusView: View {
  let title: String
  let detail: String
  let elapsed: TimeInterval?

  var body: some View {
    VStack(spacing: 18) {
      Image(nsImage: NSApp.applicationIconImage)
        .resizable()
        .frame(width: 112, height: 112)
      Text(title).font(.title2.weight(.semibold))
      ProgressView().controlSize(.large)
      VStack(spacing: 6) {
        Text("Krea 2 Turbo pages its weights into memory and prepares the Neural Engine programs. This takes up to a minute.")
          .multilineTextAlignment(.center)
          .foregroundStyle(.secondary)
        if let elapsed {
          Text(String(format: "%.0f s", elapsed)).monospacedDigit().foregroundStyle(.secondary)
        }
        if !detail.isEmpty {
          Text(detail).font(.caption.monospaced()).foregroundStyle(.tertiary).lineLimit(2)
            .multilineTextAlignment(.center)
        }
      }
      .frame(maxWidth: 520)
    }
    .padding(40)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}

struct ErrorView: View {
  let message: String

  var body: some View {
    VStack(spacing: 16) {
      Image(systemName: "exclamationmark.triangle.fill")
        .font(.system(size: 44))
        .foregroundStyle(.orange)
      Text("The engine couldn't start").font(.title2.weight(.semibold))
      Text(message).multilineTextAlignment(.center).foregroundStyle(.secondary).frame(maxWidth: 560)
      HStack(spacing: 10) {
        Button("Show Log") { AppActions.showLog() }
        Button("Choose Project Folder…") { AppActions.chooseProjectFolder() }
        Button("Restart Engine") { ServerController.shared.restart() }.keyboardShortcut(.defaultAction)
      }
      Text("Project: \(AppConfig.projectPath)").font(.caption).foregroundStyle(.tertiary)
    }
    .padding(40)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}

enum AppActions {
  static func showLog() {
    if FileManager.default.fileExists(atPath: AppConfig.logURL.path) {
      NSWorkspace.shared.open(AppConfig.logURL)
    } else {
      NSWorkspace.shared.open(AppConfig.logsDir)
    }
  }

  static func openOutputs() {
    try? FileManager.default.createDirectory(at: AppConfig.outputsURL, withIntermediateDirectories: true)
    NSWorkspace.shared.open(AppConfig.outputsURL)
  }

  @MainActor static func chooseProjectFolder() {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.message = "Choose the krea_metal project folder (it contains ui/server.py and .venv)."
    if panel.runModal() == .OK, let url = panel.url {
      AppConfig.projectPath = url.path
      ServerController.shared.restart()
    }
  }
}

/// The web UI, restricted to the local server; handles the UI's "Download PNG" links.
struct WebView: NSViewRepresentable {
  let url: URL

  func makeCoordinator() -> Coordinator { Coordinator() }

  func makeNSView(context: Context) -> WKWebView {
    let cfg = WKWebViewConfiguration()
    cfg.websiteDataStore = .default()  // persists the UI's saved settings
    let view = WKWebView(frame: .zero, configuration: cfg)
    view.navigationDelegate = context.coordinator
    view.setValue(false, forKey: "drawsBackground")
    view.load(URLRequest(url: url))
    return view
  }

  func updateNSView(_ view: WKWebView, context: Context) {
    if view.url?.port != url.port { view.load(URLRequest(url: url)) }
  }

  final class Coordinator: NSObject, WKNavigationDelegate, WKDownloadDelegate {
    private func isLocal(_ u: URL?) -> Bool {
      guard let u else { return false }
      if u.scheme == "about" || u.scheme == "blob" || u.scheme == "data" { return true }
      return (u.scheme == "http") && (u.host == "127.0.0.1" || u.host == "localhost")
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction) async -> WKNavigationActionPolicy {
      guard isLocal(action.request.url) else { return .cancel }  // no network access beyond localhost
      return action.shouldPerformDownload ? .download : .allow
    }

    func webView(_ webView: WKWebView, decidePolicyFor response: WKNavigationResponse) async -> WKNavigationResponsePolicy {
      response.canShowMIMEType ? .allow : .download
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
      download.delegate = self
    }

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) {
      download.delegate = self
    }

    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse, suggestedFilename: String) async -> URL? {
      let dir = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
      var dest = dir.appendingPathComponent(suggestedFilename)
      let base = dest.deletingPathExtension().lastPathComponent, ext = dest.pathExtension
      var n = 1
      while FileManager.default.fileExists(atPath: dest.path) {
        dest = dir.appendingPathComponent("\(base) \(n).\(ext)")
        n += 1
      }
      lastDestination = dest
      return dest
    }

    private var lastDestination: URL?

    func downloadDidFinish(_ download: WKDownload) {
      if let d = lastDestination { NSWorkspace.shared.activateFileViewerSelecting([d]) }
    }
  }
}

// MARK: - App

final class AppDelegate: NSObject, NSApplicationDelegate {
  private var lockFD: Int32 = -1
  private var signalSources: [DispatchSourceSignal] = []

  /// Single instance: an exclusive lock held for the app's lifetime.
  private func acquireInstanceLock() -> Bool {
    let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("KreaImage", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let fd = open(dir.appendingPathComponent("app.lock").path, O_CREAT | O_RDWR, 0o644)
    guard fd >= 0 else { return true }
    if flock(fd, LOCK_EX | LOCK_NB) != 0 {
      close(fd)
      return false
    }
    lockFD = fd
    return true
  }

  func applicationWillFinishLaunching(_ notification: Notification) {
    if !acquireInstanceLock() {
      NSRunningApplication.runningApplications(withBundleIdentifier: AppConfig.bundleID)
        .first { $0 != NSRunningApplication.current }?
        .activate(options: [])
      exit(0)
    }
  }

  func applicationDidFinishLaunching(_ notification: Notification) {
    // SIGTERM / SIGINT (logout, `kill`, Ctrl-C) take the normal quit path, which stops the server.
    for sig in [SIGTERM, SIGINT] {
      signal(sig, SIG_IGN)
      let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
      src.setEventHandler {
        ServerController.log("received signal \(sig), quitting")
        NSApp.terminate(nil)
      }
      src.resume()
      signalSources.append(src)
    }
    Task { @MainActor in ServerController.shared.start() }
  }

  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

  func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
    ServerController.log("quit requested")
    guard let target = MainActor.assumeIsolated({ ServerController.shared.beginStop() }) else { return .terminateNow }
    DispatchQueue.global(qos: .userInitiated).async {
      ServerController.stopBlocking(target)
      RunLoop.main.perform(inModes: [.common, .modalPanel, .default]) {
        NSApp.reply(toApplicationShouldTerminate: true)
      }
      CFRunLoopWakeUp(CFRunLoopGetMain())
    }
    return .terminateLater
  }
}

@main
struct KreaImageApp: App {
  @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

  var body: some Scene {
    Window("KreaImage", id: "main") {
      RootView()
    }
    .defaultSize(width: 1280, height: 880)
    .commands {
      CommandGroup(replacing: .newItem) {}
      CommandMenu("Engine") {
        Button("Open Outputs Folder") { AppActions.openOutputs() }
          .keyboardShortcut("o", modifiers: [.command, .shift])
        Button("Show Server Log") { AppActions.showLog() }
          .keyboardShortcut("l", modifiers: [.command, .option])
        Button("Restart Engine") { ServerController.shared.restart() }
          .keyboardShortcut("r", modifiers: [.command, .shift])
        Divider()
        Button("Quit KreaImage") { NSApp.terminate(nil) }
      }
    }
  }
}
