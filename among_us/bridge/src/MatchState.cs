// Bridge-side bookkeeping: event ids, the per-agent pending event queue and the
// decision-point rule. Mirrors SimEngine.needs_decision in among_us/sim/engine.py.

using System.Collections.Generic;

namespace Nlght.AmongUs.Bridge;

internal sealed class MatchState
{
    private int _eventCounter;
    private readonly List<ObservedEvent> _pending = new();

    public int Tick { get; private set; }
    public string Phase { get; set; } = "TASKS";
    public SemanticAction? Intent { get; set; }
    public string? LastPhaseReported { get; set; }

    public void NextTick() => Tick++;

    public string NewEventId() => $"g{++_eventCounter:D5}";

    /// <summary>Queue an event the local player perceived. Only VisibilityFilter calls this.</summary>
    public void Perceive(ObservedEvent observed) => _pending.Add(observed);

    public IReadOnlyList<ObservedEvent> DrainEvents()
    {
        var events = _pending.ToArray();
        _pending.Clear();
        return events;
    }

    public bool NeedsDecision(out string reason)
    {
        foreach (var e in _pending)
        {
            if (e.Kind is "KILL_WITNESSED" or "BODY_FOUND" or "MEETING_CALLED")
            {
                reason = $"observed: {e.Kind}";
                return true;
            }
        }
        if (LastPhaseReported != Phase)
        {
            reason = "phase changed";
            return true;
        }
        if (Intent == null)
        {
            reason = "goal reached";
            return true;
        }
        reason = "";
        return false;
    }
}
