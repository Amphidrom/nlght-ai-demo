// Executes semantic actions for the local player on the main thread.
// The model never steers frame by frame: MOVE_TO sets a goal, and this class
// walks there. Milestone 01 acceptance: "a character reacts to an external
// command in the real game" — MOVE_TO and WAIT.
//
// TODO(verify on Windows): game API names, see VisibilityFilter.cs.
// TODO(milestone 04): path finding around walls (straight-line steering only
// works inside a room and along open corridors), DO_TASK, REPORT, KILL,
// CALL_MEETING, SAY (chat), VOTE / SKIP_VOTE (MeetingHud).

using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Nlght.AmongUs.Bridge;

internal static class ActionExecutor
{
    public static IReadOnlyList<string> KnownRooms =>
        ShipStatus.Instance == null
            ? new List<string>()
            : ShipStatus.Instance.AllRooms.Select(r => r.RoomId.ToString().ToLowerInvariant()).Distinct().ToList();

    /// <summary>Validates against the current legal actions, then adopts the intent.</summary>
    public static ActionResult Accept(PlayerControl self, MatchState match, SemanticAction action, string agentId)
    {
        if (action.AgentId != agentId)
            return new ActionResult(action.AgentId, false, "unknown agent");
        if (action.Action == "WAIT")
        {
            match.Intent = null;
            return new ActionResult(agentId, true);
        }
        if (action.Action == "MOVE_TO" && action.Target != null && KnownRooms.Contains(action.Target))
        {
            match.Intent = action;
            return new ActionResult(agentId, true);
        }
        return new ActionResult(agentId, false, $"not supported yet: {action.Action}");
    }

    /// <summary>Called every physics frame while an intent is active.</summary>
    public static void Step(PlayerControl self, MatchState match)
    {
        var intent = match.Intent;
        if (intent?.Action != "MOVE_TO" || intent.Target == null)
            return;
        var room = ShipStatus.Instance.AllRooms
            .FirstOrDefault(r => r.RoomId.ToString().ToLowerInvariant() == intent.Target);
        if (room?.roomArea == null)
        {
            match.Intent = null;
            return;
        }
        Vector2 here = self.GetTruePosition();
        Vector2 goal = room.roomArea.bounds.center;
        if (room.roomArea.OverlapPoint(here))
        {
            self.MyPhysics.SetNormalDirection(Vector2.zero);
            match.Intent = null;   // goal reached → next decision point
            return;
        }
        // TODO(verify): the input-driven movement call the local player uses.
        self.MyPhysics.SetNormalDirection((goal - here).normalized);
    }
}
