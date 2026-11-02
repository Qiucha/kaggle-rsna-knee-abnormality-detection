> **Execution update, 2026-09-06:** User authorized a backward batch and explicitly
> selected Opus 5 low effort. The last 200 rows (indices 4406–4207) are now assigned
> to our bounded run with two workers; this is no longer only a dry-run preparation.
> Two low-effort trials passed and the remaining batch launched persistently.
> See the local handoff notes in `digest/handoff/` and
> `checkpoints/translations/config-8e3bd68-opus5-low/batch-status.json` for live state.
> Eight tests now pass after fixing CLI event-array parsing and adding backward
> scheduling/failure stopping. Historical preparation findings follow below.

# origin/config translation review and parallel execution

Reviewed 2026-09-06. Fetched `origin/config`; it and `FETCH_HEAD` both resolve to
`8e3bd6899d7a7c9e56a8d27110a57f0b711a0a59` (`claude code based translations`).
The working branch and existing uncommitted training changes were preserved.

## Conclusion

**The saved Google corpus has concrete translation defects, so it should not be
adopted unchecked as the English contrastive-training target.** This is a data
quality concern, not an API incompatibility or evidence that every Google
translation is wrong. Claude is a plausible alternative, but fluent wording does
not prove fidelity or better retrieval. No paired Google-versus-Opus quality or
training experiment was run.

The existing completed baseline uses original multilingual reports with E5.
Nothing on `origin/config` automatically changes that baseline. Translation is a
new text-policy experiment, not a requirement for making the existing pipeline run.

## What the branch actually does

[Notebook at the reviewed commit](https://github.com/Qiucha/kaggle-rsna-knee-abnormality-detection/blob/05ff27e8bfcb3688c97e8b24934b9af22941d6ba/notebooks/translation.ipynb):

- Cell 18 uses Google Cloud Translation **v3** `TranslationServiceClient`,
  `global`, `mime_type="text/plain"`, target `en`, and automatic source detection.
  There is no glossary or explicit model revision. Explicit plain text is correct
  for these reports; HTML defaults are not this implementation's problem.
- Cells 20/34 translate serially and rewrite the full CSV after each row. If the
  file exists, they load it and skip translation entirely: an interrupted partial
  file does not automatically resume. Exceptions returned from `translate_text`
  are unpacked as if they were two successful values. These are orchestration
  risks; the committed full CSV has no empty translation cells.
- Cells 36/43 keep source-English reports, then use ftfy, NFKC, zero-width-space
  removal and a few symbol substitutions. The result is in **`Cleaned Translations`**,
  separate from `Translated Report`.
- Cells 52–55 use **Claude Code CLI**, `--model opus`, a fixed musculoskeletal
  translation system prompt, one report per `claude -p`, JSON output, one turn,
  disabled built-in tools and empty strict MCP configuration. Timeout is 180 s,
  with three attempts. Input is **original `Report`**, not Google's translation
  and not a cleaned original. No actual RTF repair is applied before Claude.
- The Sonnet 4.6 comparison earlier in the notebook is illustrative; the batch
  configuration requests the Opus alias. The branch does not contain the configured
  `data/claude_translations.csv`, so completed Opus coverage cannot be inferred.

## Evidence and implications for contrastive training

[Audit counts](config-translation-audit.json) were computed from the pinned
`train_translated_clean.csv`; all 4,407 UID/source-report pairs match our frozen
manifest after its usual outer-whitespace trimming. The saved input snapshot is
`checkpoints/translations/config-8e3bd68/source.csv`.

| Finding | Evidence | Why it matters |
| --- | --- | --- |
| Incorrect language detection and incomplete English | Zero-based row 2419 is visibly Spanish but detected `gn` (Guarani). Output retains `Derrame` and renders the impression heading as `Print`. | A finding remains untranslated, and section selection cannot recognize that heading. |
| Section loss in another suspicious detection | Row 1201 has Spanish headings, is detected `pt`, and the output omits the separately repeated impression section. | Full text may retain the repeated findings, but impression-only processing no longer sees the intended section. This is not evidence of lost unique findings in that example. |
| Awkward anatomical/report terminology | 31 rows contain the whole word `trunk`; 94 contain `neat` in the translation column. The notebook's paired example uses meniscal “trunk” and “neat presentation.” | Terminology may be less aligned with standard English radiology text. Counts are review flags, not 125 adjudicated mistranslations or measured embedding damage. |
| Unicode cleaning is incomplete as semantic repair | 392 rows contain U+200B before the explicit cleaning; zero afterward. The terminology counts remain unchanged. | Cleaning characters does not fix translation meaning. U+200B does not, by itself, establish an encoder crash; test the actual tokenizer. Preserve grade and measurement information. |
| Translation does not remove truncation | With our pinned E5 tokenizer and prefix, 70/441 original validation reports exceed 512 tokens versus 66/441 cleaned translations. | English still needs length/section auditing. Counts are E5-specific, not measurements for a different English encoder. |
| More reports share identical text after translation | 183 source studies belong to exact stripped-text duplicate groups versus 198 after translation; 49 groups in each. | Translation can collapse wording differences. Identical candidate text complicates unique-pair retrieval and creates false-negative pairs. No new metric was computed. |

Only train/validation examples were read for semantic observations. Test source
text participated in mechanical translation inventory/duplicate counts, not model
selection, performance evaluation or a semantic test-set error review.

The contrastive loss treats each report as its image's positive target. A changed
negation, structure, laterality, grade, uncertainty or measurement can make that
supervision wrong. The observed defects justify targeted QC and an alternative
translation corpus; they do not quantify how much baseline performance would change.

The branch's assertion that duplicate reports are harmless when labels agree is
insufficient for this task: there are 12 coarse labels, and unique image/report
retrieval treats another study's identical report as a negative. Templates do not
prove repeated patients, but label agreement does not eliminate false negatives.

## Using the same Claude workflow in parallel

Added `scripts/data/translate_reports_claude.py` with Python standard-library dependencies.
It extracts the original prompt/model/timeout/retry constants from the pinned
notebook using AST literal parsing, without executing notebook cells. Requests
use the same original text, prompt prefix and CLI flags. It does not introduce
Google post-editing, an API migration, or additional preprocessing.

Changes are limited to orchestration: bounded concurrent subprocesses, disjoint
row ranges, one durable JSONL writer per shard, resume by UID + source hash +
contract hash, failure retries, output locks, and conflict checks. It preserves
available model/usage metadata instead of discarding the JSON envelope. Blank
successes are rejected. Checkpoint version/contract changes require a new shard.

Local Claude Code 2.1.261 is installed and reports authenticated Claude subscription
access. No provider call was made. Subscription capacity and whether Opus requests
will succeed have not been tested. Start with one or two workers, then adjust to
observed capacity, rather than launching many independent processes immediately.

Offline setup and example dry runs (the ranges are illustrative, **not assigned**):

```bash
.venv/bin/python scripts/data/translate_reports_claude.py prepare

# Teammate's example range: rows 0–2203
.venv/bin/python scripts/data/translate_reports_claude.py run \
  --start 0 --stop 2204 --workers 2 \
  --output checkpoints/translations/config-8e3bd68/teammate.jsonl

# Our example range: rows 2204–4406
.venv/bin/python scripts/data/translate_reports_claude.py run \
  --start 2204 --stop 4407 --workers 2 \
  --output checkpoints/translations/config-8e3bd68/ours.jsonl
```

These commands only count pending rows. `--execute` launches provider calls.
Use the agreed ranges after checking the teammate's actual completed/reserved
rows; do not assume the illustrative halves avoid their current serial run.
Ranges preserve the pinned CSV's zero-based row ordering. This matching workflow
translates English rows too, just like the notebook; skipping them is a sensible
cost optimization but should be agreed as a protocol change. Final training
export deliberately retains the original English, matching our manifest policy.

`--completed other.jsonl` skips verified successes from another shard. Multiple
machines must have distinct output files and agreed ownership: a file lock on one
machine does not coordinate another machine. Exact translation results are not
guaranteed to match across repeated stochastic calls.

After collecting the runner's complete JSONL shards locally:

```bash
.venv/bin/python scripts/data/translate_reports_claude.py merge \
  --results checkpoints/translations/config-8e3bd68/teammate.jsonl \
            checkpoints/translations/config-8e3bd68/ours.jsonl \
  --output data/processed/claude-translated-v1.csv
```

The merge refuses missing non-English rows, changed UID/source/contract identity,
conflicting translations and an existing output file. It produces the training
loader's `StudyInstanceUID`, `Report`, `Original Language`, `Translated Report`
columns and a provenance sidecar marked **unreviewed**. It retains Google's
source-language detections, which therefore still need QC.

The notebook's existing checkpoint contains only an integer index and
`LLM Translation`. It cannot safely be treated as our verified JSONL: first obtain
the exact source CSV/order used for that checkpoint, verify it against the pinned
source, then explicitly convert by UID/hash. This conversion and teammate ownership
remain pending because their live checkpoint is not in the branch.

## What “same pipeline” still requires agreeing across machines

`opus` is a moving/configurable alias, not a pinned model. Agree on the actual
model ID, CLI version, effort/settings and working-directory context before a
joint run. Our wrapper preserves the notebook's alias to avoid silently changing
it and saves returned `modelUsage`; that metadata is not an enforced cross-host
model match. [Official model configuration](https://code.claude.com/docs/en/model-config).

The CLI can load `CLAUDE.md` separately from its system prompt. Empty tools/MCP
configuration does not ensure identical host context. Standardize that context
with the teammate. Do not blindly add `--bare`: it also changes authentication
and skips OAuth/keychain access. [Official memory documentation](https://code.claude.com/docs/en/memory),
[programmatic usage](https://code.claude.com/docs/en/headless).

Other current provider facts and sources are in
[the primary-source note](translation-api-primary-sources.md). Google terminology
glossaries are another possible remediation, but would also be a changed
translation protocol; documentation does not establish clinical superiority for
either provider.

## Integration and next decision

The manifest builder already supports `translated_english`. Its default reads
`Translated Report`, **not** the branch's `Cleaned Translations` or Claude's
`LLM Translation`; passing either file without adapting that column can silently
use Google text or fail schema validation. The CLI currently does not expose the
alternate translation-column setting, although `ManifestConfig` has one.

Keep the original reports as the source CSV and pass the adapted translation CSV
separately. Do not use the wide translation CSV as the source reports CSV: the
manifest builder treats extra source columns as labels. Build a **new** cohort and
experiment output root; assert exact UID, patient-group, eligibility and split
agreement with v1. Partial translation coverage can exclude rows and change split
assignment, even with the same seed. Never overwrite the frozen baseline manifest.

Before training, review paired Google/Claude samples from train/validation across
languages and the flagged cases, checking negation, anatomy, laterality, grades,
uncertainty, numbers and omissions/additions. Keep providers separate for a
controlled comparison. Complete the existing leakage gate as well. Neither a
new translation nor successful execution alone establishes better representation
learning.

Validation: five focused offline tests pass (request parity, resume identity and
conflicts, failed/blank results, retry/metadata capture, complete merge and English
preservation). Prepared all 4,407 inputs and exercised a 10-row dry run with zero
provider calls. No commit, push, training or test evaluation was performed.
