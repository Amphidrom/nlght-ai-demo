// The first wall against leaks: decides what the local player can perceive.
// The Python side (among_us/perception.py) re-checks every observation.
//
// TODO(verify on Windows): the game API names below follow the decompiled
// Among Us assemblies as commonly used by Reactor mods; confirm each against
// AmongUs.GameLibs for the installed build.

using System.Linq;
using UnityEngine;

namespace Nlght.AmongUs.Bridge;

internal static class VisibilityFilter
{
    /// <summary>True if the local player could see <paramref name="other"/> right now.</summary>
    public static bool CanSee(PlayerControl self, PlayerControl other)
    {
        if (other == self || other.Data == null || other.Data.IsDead || other.Data.Disconnected)
            return false;
        Vector2 from = self.GetTruePosition();
        Vector2 to = other.GetTruePosition();
        // Light radius shrinks with the lights sabotage, as for a human player.
        float radius = ShipStatus.Instance.CalculateLightRadius(self.Data);
        if (Vector2.Distance(from, to) > radius)
            return false;
        // Walls block sight. TODO(verify): layer mask used by the game's own vision.
        return !PhysicsHelpers.AnythingBetween(from, to, Constants.ShipAndObjectsMask, false);
    }

    /// <summary>The room name used in the protocol (matches among_us/sim/map.py where possible).</summary>
    public static string RoomOf(PlayerControl player)
    {
        var position = player.GetTruePosition();
        var room = ShipStatus.Instance.AllRooms
            .FirstOrDefault(r => r.roomArea != null && r.roomArea.OverlapPoint(position));
        return room == null ? "corridor" : room.RoomId.ToString().ToLowerInvariant();
    }

    /// <summary>
    /// Only the local player's own role leaves the game. Impostor partners are
    /// reported because the game shows them to an impostor; nothing else is.
    /// </summary>
    public static string[] KnownImpostors(PlayerControl self)
    {
        if (self.Data?.Role == null || !self.Data.Role.IsImpostor)
            return System.Array.Empty<string>();
        return PlayerControl.AllPlayerControls.ToArray()
            .Where(p => p != self && p.Data?.Role != null && p.Data.Role.IsImpostor)
            .Select(p => p.Data.PlayerName)
            .ToArray();
    }
}
