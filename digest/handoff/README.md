# Operational handoff notes (local only)

The notes in this directory are the running state of an in-flight experiment
session: which launcher is alive, which process ID is training, which epoch it
reached, and what the next session should pick up. They are git-ignored on
purpose — everything in `.gitignore` under `digest/handoff/*` except this file.

They are excluded rather than committed because they describe *this machine at
one moment*. A process ID or a "do not rerun the launcher" instruction is
actively misleading once merged, and it goes stale within hours rather than
being a durable record of what the project found.

Durable findings do not live here. When a run finishes, its results, audits, and
protocol belong in `digest/` proper, under the experiment's own name — those are
committed, and they are what a reported number is traced back to.

Documents elsewhere in the repository that point at "the current handoff" mean
this directory on the machine that produced the run. If it is empty in your
checkout, that is expected: there is no in-flight session to hand over.

Files kept here in past sessions, for reference:

| File | Covered |
| --- | --- |
| `NEXT-AGENT-HANDOFF.md` | Current session pickup: execution state and pending review |
| `handoff-history-through-20260907.md` | Archived running narrative through 2026-09-07 |
| `handoff-text-queue-through-20260907.md` | Archived English/medical text encoder queue |
