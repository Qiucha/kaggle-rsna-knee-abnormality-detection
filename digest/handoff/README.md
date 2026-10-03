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
| `FEEDBACK-RESPONSE-HANDOFF.md` | The knee CLIP feedback addressed point-by-point: what each of its eight sections still needs, and P0–P3 status. The *actionable* layer over the committed [feedback response](../knee-clip-feedback-response.md), which stays the durable record |
| `handoff-20260913-overfitting-confirmed.md` | **Current session pickup — start here.** Overfitting confirmed and quantified at matched N; the gold-probe estimand is the named next milestone |
| `handoff-20260912-image-block-variants.md` | Image-block depth/language/schedule variants, the loss-redesign and whitening gates, and plane pooling. Superseded as a pickup list, kept as the record of those results |
| `NEXT-AGENT-HANDOFF.md` | Superseded session pickup. **Open it at the "START HERE" block** — that is the single to-do list; the rest is execution state, completed session logs, and live reference on the implementation, artifacts and data limits |
| `handoff-history-through-20260907.md` | Archived running narrative through 2026-09-07 |
| `handoff-text-queue-through-20260907.md` | Archived English/medical text encoder queue |
