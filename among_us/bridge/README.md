# nlght-ai Among Us Bridge (C#, experimental)

A BepInEx 6 (IL2CPP) + Reactor plugin that connects the real Among Us client to
the Python match orchestrator. It speaks **protocol v1** (`among_us/protocol.py`)
as newline-delimited JSON over a loopback TCP socket (`127.0.0.1:47100`).

> **Status: unbuilt and untested.** The skeleton was written on macOS, where the
> IL2CPP Windows build of Among Us cannot be modded. Every game API name is marked
> `TODO(verify)` and must be checked against the generated game assemblies before
> the first build. Do not use this on official servers.

## What it does today (milestone 01 scope)

- The **local player** is the agent (`BepInEx/config/ai.nlght.amongus.bridge.cfg`,
  `AgentId`, default `sherlock`).
- Every ~0.5 s it checks the decision-point rule (`MatchState.NeedsDecision`,
  mirroring `SimEngine.needs_decision`) and sends a `decision_request` with an
  observation filtered by `VisibilityFilter`.
- It executes `MOVE_TO <room>` and `WAIT` (`ActionExecutor`) and answers every
  `action` with an `action_result`.
- It reports kills the local player can see, meetings and ejections as perceived
  events.

Not yet: path finding, tasks, reporting, killing, chat and voting, bodies, and above
all **additional bot players**. Whether bots can be spawned as real, network-
synchronised participants (host-side NPCs, approach A) — or need a private
Impostor server (approach B) — is the open research question of milestone 01.

## Build on Windows

1. Install the Among Us version you test against and note its build number
   (title screen). Pick matching versions for `BepInEx.Unity.IL2CPP`, `Reactor`
   and `AmongUs.GameLibs.Steam` (or `.Epic`) in `AmongUsBridge.csproj`.
2. Install BepInEx 6 IL2CPP into the game folder and start the game once so it
   generates its interop assemblies.
3. Install Reactor into `BepInEx/plugins`.
4. Build: `dotnet build -c Release -p:AmongUsPath="C:\...\Among Us"` — the DLL is
   copied into `BepInEx/plugins`.
5. Verify every `TODO(verify)` against the interop assemblies; fix names first.

## Acceptance test for milestone 01

1. Start the demo with seed `08_among_us.sql` loaded: `uv run python main.py`.
2. Start Among Us with the plugin, host a **local/private** game and enter the map.
3. Start a bridge match:

   ```powershell
   Invoke-RestMethod -Method Post http://localhost:9000/among-us/api/matches `
     -ContentType application/json `
     -Body '{"environment":"bridge","players":{"sherlock":"sherlock"}}'
   ```

4. Pass: the character walks to the room the agent chose, a new
   `decision_request` follows when it arrives, and the match log
   (`GET /among-us/api/matches/{id}/events`) shows each decision and its
   `action_result`.

The contract is `among_us/protocol_fixtures/*.json`: a message the bridge sends must
decode with `among_us.protocol.decode`, which refuses unknown fields.
