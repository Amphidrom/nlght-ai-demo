// Harmony patches that turn game moments into ticks, perceived events and
// decision requests. TODO(verify on Windows): patched method names and
// signatures against AmongUs.GameLibs for the installed build.

using HarmonyLib;

namespace Nlght.AmongUs.Bridge;

[HarmonyPatch(typeof(PlayerControl), nameof(PlayerControl.FixedUpdate))]
internal static class LocalPlayerTick
{
    private const int FramesPerTick = 25;   // ~0.5 s at the game's 50 Hz physics rate
    private static int _frames;

    public static void Postfix(PlayerControl __instance)
    {
        if (__instance != PlayerControl.LocalPlayer || ShipStatus.Instance == null)
            return;
        var plugin = BridgePlugin.Instance;
        var match = plugin.Match;
        var agentId = plugin.AgentId.Value;

        while (plugin.Server.Inbox.TryDequeue(out var action))
            plugin.Server.Send("action_result", ActionExecutor.Accept(__instance, match, action, agentId));
        ActionExecutor.Step(__instance, match);

        if (++_frames < FramesPerTick)
            return;
        _frames = 0;
        match.NextTick();
        if (plugin.Server.Connected && match.NeedsDecision(out var reason))
        {
            var observation = ObservationCollector.Collect(__instance, match, agentId, reason);
            match.LastPhaseReported = match.Phase;
            // Until the orchestrator answers, the agent keeps its last intent.
            plugin.Server.Send("decision_request", new DecisionRequest(observation));
        }
    }
}

[HarmonyPatch(typeof(PlayerControl), nameof(PlayerControl.MurderPlayer))]
internal static class KillWitness
{
    public static void Postfix(PlayerControl __instance, PlayerControl target)
    {
        var self = PlayerControl.LocalPlayer;
        if (self == null || __instance == self || target == self || !VisibilityFilter.CanSee(self, __instance))
            return;
        var match = BridgePlugin.Instance.Match;
        match.Perceive(new ObservedEvent(
            match.NewEventId(), match.Tick, "KILL_WITNESSED",
            Room: VisibilityFilter.RoomOf(target),
            Actor: __instance.Data.PlayerName,
            Target: target.Data.PlayerName));
    }
}

[HarmonyPatch(typeof(MeetingHud), nameof(MeetingHud.Start))]
internal static class MeetingStarted
{
    public static void Postfix()
    {
        var match = BridgePlugin.Instance.Match;
        match.Phase = "MEETING";
        match.Intent = null;
        match.Perceive(new ObservedEvent(match.NewEventId(), match.Tick, "MEETING_CALLED"));
    }
}

[HarmonyPatch(typeof(ExileController), nameof(ExileController.WrapUp))]
internal static class MeetingEnded
{
    public static void Postfix(ExileController __instance)
    {
        var match = BridgePlugin.Instance.Match;
        var exiled = __instance.exiled;   // TODO(verify): field name of the exiled player's data
        match.Perceive(exiled == null
            ? new ObservedEvent(match.NewEventId(), match.Tick, "NO_EJECTION")
            : new ObservedEvent(match.NewEventId(), match.Tick, "EJECTED", Target: exiled.PlayerName));
        match.Phase = "TASKS";
    }
}
