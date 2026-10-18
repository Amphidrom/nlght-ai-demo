// Local NDJSON server the orchestrator connects to (BridgeEnvironment in
// among_us/environment.py). Network I/O runs on a background thread; game state
// is only touched on Unity's main thread, which drains Inbox every frame.

using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;

namespace Nlght.AmongUs.Bridge;

public sealed class BridgeServer : IDisposable
{
    private readonly int _port;
    private readonly Action<string> _log;
    private TcpListener? _listener;
    private StreamWriter? _writer;
    private readonly object _writeLock = new();
    private Thread? _thread;
    private volatile bool _running;

    /// <summary>Actions from the orchestrator, consumed on the main thread.</summary>
    public ConcurrentQueue<SemanticAction> Inbox { get; } = new();

    public bool Connected => _writer != null;

    /// <summary>Raised on the network thread when a client connects; reply with Hello.</summary>
    public event Action? ClientConnected;

    public BridgeServer(int port, Action<string> log)
    {
        _port = port;
        _log = log;
    }

    public void Start()
    {
        // Loopback only: the bridge must never be reachable from the network.
        _listener = new TcpListener(IPAddress.Loopback, _port);
        _listener.Start();
        _running = true;
        _thread = new Thread(Serve) { IsBackground = true, Name = "nlght-bridge" };
        _thread.Start();
        _log($"bridge listening on 127.0.0.1:{_port}");
    }

    private void Serve()
    {
        while (_running && _listener != null)
        {
            try
            {
                using var client = _listener.AcceptTcpClient();
                using var stream = client.GetStream();
                using var reader = new StreamReader(stream, new UTF8Encoding(false));
                lock (_writeLock)
                    _writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true, NewLine = "\n" };
                ClientConnected?.Invoke();
                string? line;
                while (_running && (line = reader.ReadLine()) != null)
                    Receive(line);
            }
            catch (Exception ex) when (ex is IOException or SocketException or ObjectDisposedException)
            {
                _log($"bridge connection closed: {ex.Message}");
            }
            finally
            {
                lock (_writeLock)
                    _writer = null;
            }
        }
    }

    private void Receive(string line)
    {
        try
        {
            using var document = JsonDocument.Parse(line);
            var root = document.RootElement;
            if (root.GetProperty("protocol").GetInt32() != Protocol.Version || root.GetProperty("type").GetString() != "action")
            {
                _log($"bridge ignored message: {line}");
                return;
            }
            var action = root.Deserialize<SemanticAction>(Protocol.Json);
            if (action != null)
                Inbox.Enqueue(action);
        }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or InvalidOperationException)
        {
            _log($"bridge malformed message: {ex.Message}");
        }
    }

    public void Send(string type, object body)
    {
        var line = Protocol.Encode(type, body);
        lock (_writeLock)
        {
            try { _writer?.WriteLine(line); }
            catch (IOException ex) { _log($"bridge send failed: {ex.Message}"); }
        }
    }

    public void Dispose()
    {
        _running = false;
        _listener?.Stop();
    }
}
