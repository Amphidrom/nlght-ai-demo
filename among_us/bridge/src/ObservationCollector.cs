// Builds the protocol observation for the local player from live game state.
// TODO(verify on Windows): game API names, see VisibilityFilter.cs.

using System.Collections.Generic;
using System.Linq;

namespace Nlght.AmongUs.Bridge;

internal static class ObservationCollector
{
    public static Observation Collect(PlayerControl self, MatchState match, string agentId, string reason)
    {
        var room = VisibilityFilter.RoomOf(self);
        var isImpostor = self.Data.Role != null && self.Data.Role.IsImpostor;
        var visible = PlayerControl.AllPlayerControls.ToArray()
            .Where(other => match.Phase != "TASKS" || VisibilityFilter.CanSee(self, other))
            .Where(other => other != self && other.Data != null && !other.Data.IsDead)
            .Select(other => new VisiblePlayer(other.Data.PlayerName, VisibilityFilter.RoomOf(other), true))
            .ToList();
        var tasks = self.myTasks.ToArray()
            .Where(t => !t.IsComplete)
            .Select(t => t.TaskType.ToString().ToLowerInvariant())
            .ToList();

        return new Observation(
            AgentId: agentId,
            Tick: match.Tick,
            Phase: match.Phase,
            SelfState: new SelfState(
                PlayerId: agentId,
                Room: room,
                Role: isImpostor ? "IMPOSTOR" : "CREWMATE",
                Alive: !self.Data.IsDead,
                TasksRemaining: tasks,
                KillReady: isImpostor && self.killTimer <= 0f,
                KnownImpostors: VisibilityFilter.KnownImpostors(self)),
            VisiblePlayers: visible,
            VisibleBodies: new List<VisibleBody>(),   // TODO: DeadBody objects within CanSee range.
            Events: match.DrainEvents(),
            LegalActions: LegalActions(self, match, visible, tasks),
            DecisionReason: reason);
    }

    private static List<LegalAction> LegalActions(PlayerControl self, MatchState match, List<VisiblePlayer> visible, List<string> tasks)
    {
        var actions = new List<LegalAction>();
        switch (match.Phase)
        {
            case "MEETING":
                actions.Add(new LegalAction("SAY"));
                actions.Add(new LegalAction("WAIT"));
                return actions;
            case "VOTING":
                actions.AddRange(visible.Select(p => new LegalAction("VOTE", p.PlayerId)));
                actions.Add(new LegalAction("SKIP_VOTE"));
                return actions;
        }
        // Milestone 01 needs MOVE_TO and WAIT only; the rest follow with milestone 04.
        actions.AddRange(ActionExecutor.KnownRooms.Select(r => new LegalAction("MOVE_TO", r)));
        actions.Add(new LegalAction("WAIT"));
        return actions;
    }
}
