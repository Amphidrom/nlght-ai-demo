// Protocol v1 — the C# mirror of among_us/protocol.py.
//
// Keep both sides in lockstep: the JSON files in among_us/protocol_fixtures are
// the contract. The Python orchestrator refuses unknown fields, so a field added
// here without being added there fails loudly instead of leaking.

using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Nlght.AmongUs.Bridge;

public static class Protocol
{
    public const int Version = 1;

    public static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never,
        WriteIndented = false,
    };

    /// <summary>One NDJSON line: {"type": ..., "protocol": 1, ...fields}.</summary>
    public static string Encode(string type, object body)
    {
        var fields = JsonSerializer.SerializeToElement(body, Json);
        var message = new Dictionary<string, object?> { ["type"] = type, ["protocol"] = Version };
        foreach (var property in fields.EnumerateObject())
            message[property.Name] = property.Value;
        return JsonSerializer.Serialize(message, Json);
    }
}

public sealed record Hello(string GameVersion, string Source, IReadOnlyList<string> Players);

public sealed record VisiblePlayer(string PlayerId, string Room, bool Alive);

public sealed record VisibleBody(string BodyId, string PlayerId, string Room);

public sealed record ObservedEvent(
    string EventId,
    int Tick,
    string Kind,
    string? Room = null,
    string? Actor = null,
    string? Target = null,
    string? Text = null,
    IReadOnlyList<string>? EvidenceIds = null);

public sealed record LegalAction(string Action, string? Target = null);

public sealed record SelfState(
    string PlayerId,
    string Room,
    string Role,
    bool Alive,
    IReadOnlyList<string> TasksRemaining,
    bool KillReady,
    IReadOnlyList<string> KnownImpostors);

public sealed record Observation(
    string AgentId,
    int Tick,
    string Phase,
    SelfState SelfState,
    IReadOnlyList<VisiblePlayer> VisiblePlayers,
    IReadOnlyList<VisibleBody> VisibleBodies,
    IReadOnlyList<ObservedEvent> Events,
    IReadOnlyList<LegalAction> LegalActions,
    string DecisionReason);

public sealed record DecisionRequest(Observation Observation);

public sealed record SemanticAction(
    string AgentId,
    string Action,
    string? Target,
    string? Text,
    string? Goal,
    IReadOnlyList<string>? EvidenceIds);

public sealed record ActionResult(string AgentId, bool Accepted, string Reason = "");

public sealed record MatchEnd(string Winner, string Reason = "");
