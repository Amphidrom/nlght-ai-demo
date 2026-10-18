// Entry point. Milestone 01 scope: the *local* player is the agent. The bridge
// observes it, reports decision points, and executes semantic actions for it.
// Spawning further, network-synchronised bot players is the open research
// question of milestone 01 (see README.md) and is deliberately not attempted here.

using BepInEx;
using BepInEx.Configuration;
using BepInEx.Unity.IL2CPP;
using HarmonyLib;
using Reactor;

namespace Nlght.AmongUs.Bridge;

[BepInAutoPlugin("ai.nlght.amongus.bridge", "nlght-ai Among Us Bridge")]
[BepInProcess("Among Us.exe")]
[BepInDependency(ReactorPlugin.Id)]
public partial class BridgePlugin : BasePlugin
{
    internal static BridgePlugin Instance { get; private set; } = null!;
    internal BridgeServer Server { get; private set; } = null!;
    internal MatchState Match { get; } = new();
    internal ConfigEntry<string> AgentId { get; private set; } = null!;

    private readonly Harmony _harmony = new("ai.nlght.amongus.bridge");

    public override void Load()
    {
        Instance = this;
        var port = Config.Bind("Bridge", "Port", 47100, "Loopback port the orchestrator connects to (47100-47199).");
        AgentId = Config.Bind("Bridge", "AgentId", "sherlock", "Agent id the local player is reported as.");
        if (port.Value is < 47100 or > 47199)
            port.Value = 47100;

        Server = new BridgeServer(port.Value, message => Log.LogInfo(message));
        Server.ClientConnected += () => Server.Send("hello", new Hello(
            GameVersion: UnityEngine.Application.version,
            Source: "nlght-ai-among-us-bridge",
            Players: new[] { AgentId.Value }));
        Server.Start();
        _harmony.PatchAll();
    }

    public override bool Unload()
    {
        _harmony.UnpatchSelf();
        Server.Dispose();
        return true;
    }
}
