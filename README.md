# Relay

A batch-inference orchestrator that treats a crash as a normal event. It
resumes in-flight jobs without re-charging you, and re-sends only what's
missing until it can prove 100% coverage.

## Why this is hard

- A crash never double-charges you. A step-0 "resume drain" polls
  in-flight jobs from a previous crashed run and counts their results
  instead of resubmitting.
- Every chunk state transition is persisted atomically (temp file +
  `os.replace`) BEFORE any further side effect, so the tracker can never
  be left half-written.
- After each pass the orchestrator recomputes which order ids are still
  missing from the output CSV (the source of truth) and re-sends only
  those, until it hits 100% coverage or runs out of pass budget.
- OpenAI, Anthropic, Google (Gemini), and a keyless mock all sit behind
  one `Provider` protocol. The core itself is zero-dependency stdlib;
  provider SDKs are opt-in extras.

## See it (60 seconds, no keys, no network)

Run the scripted two-act fault-tolerance demo:

```text
relay demo --seed 7
```

```text
============================================================
relay demo -- fault-tolerance narrative
============================================================
seed=7  records=300  output_dir=<temp dir>

Act 1: chaos run
  injecting: 10% item errors, 10% dropped items,
             30% transient submit failures (simulated 429s)
  watch the orchestrator re-send only what's missing, pass by pass

[pass 1] 300 missing -> 3 chunks
[pass 2] 68 missing -> 1 chunks
[pass 3] 19 missing -> 1 chunks
[pass 4] 4 missing -> 1 chunks
[pass 5] 1 missing -> 1 chunks

Act 1 result: passes=5 resent_items=92 coverage=100.0% converged=True

Act 2: idempotent rerun (same tracker, same output CSV)
  coverage is already 100% -- expect 0 resubmits

Act 2 result: submitted_chunks=0 resent_items=0 coverage=100.0%

============================================================
DEMO PASSED -- fault tolerance verified end to end
============================================================
```

**Act 1** shows the multi-pass coverage loop: 300 records land in 3 chunks,
and despite 10% item errors, 10% dropped items, and 30% simulated 429s on
submit, the orchestrator converges to 100% coverage in 5 passes, resending
only 92 of the 300 original items in total. **Act 2** reruns the same job
against the same tracker and output CSV. Coverage is already 100%, so
zero items are resubmitted. That zero-resubmit rerun is the crash-safety
guarantee in miniature: resuming a finished (or partially finished) run
never re-does work it already did, and never double-charges you.

## Quickstart

```
pip install -e .
```

The core is stdlib-only: no runtime dependencies are required for the
mock provider, the CLI, the tracker, or the demo. Provider SDKs are
opt-in extras: `.[openai]`, `.[anthropic]`, `.[google]`, `.[dev]` (pytest),
`.[all]` (every provider SDK).

```
relay demo
relay run --input data/records.csv --provider mock --run-id 1
relay status --tracker outputs/tracker.json
```

## How it works

Every request's `custom_id` carries a stable `order_id` (`run_{run_id}_item_{order_id}`)
so results always reassemble correctly, regardless of provider ordering or
partial completion. A crash-safe JSON tracker records every chunk's state
through an atomic write pattern (temp file + `os.replace`), and before any
coverage math runs, a step-0 resume drain polls in-flight jobs from a prior
crashed run instead of resubmitting them. Each pass then recomputes which
order ids are still missing from the output CSV (the source of truth)
and re-sends only those, while oversized batches self-correct via
shrink-and-rechunk and transient submit errors are retried with
exponential backoff. Full design, state diagram, and guarantees: see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## CLI reference

| command | purpose |
| --- | --- |
| `relay run` | load a CSV, run the orchestrator against a provider |
| `relay status` | print tracked chunk state from a tracker file |
| `relay cost` | offline token/cost estimate, no network calls |
| `relay demo` | scripted two-act fault-tolerance narrative, no keys |
| `relay smoke` | validate a real provider with a tiny live batch (spends real money) |

<details>
<summary>Full flag reference</summary>

### `relay run`

Load records from a CSV and run the orchestrator against a provider.

| flag | default | meaning |
| --- | --- | --- |
| `--input` | (required) | input CSV, must have an `order_id,text` header |
| `--run-id` | `1` | run identifier (int) |
| `--provider` | `mock` | `mock` \| `openai` \| `anthropic` \| `google` |
| `--model` | adapter default | model name override for real providers |
| `--output-dir` | `outputs` | output directory |
| `--tracker` | `outputs/tracker.json` | tracker JSON path |
| `--poll-interval` | `0.5` | seconds between polls |
| `--seed` | `0` | mock provider seed |
| `--limit-items` | none | cap the number of records loaded |
| `--error-rate` | `0.0` | mock: fraction of items returned as errors |
| `--drop-rate` | `0.0` | mock: fraction of items dropped (partial batch) |
| `--submit-failure-rate` | `0.0` | mock: probability submit raises a transient error |
| `--expire-rate` | `0.0` | mock: probability a job's terminal state is expired |
| `--max-passes` | `5` | max coverage re-send passes per run |
| `--backoff-base` | `0.5` | base seconds for exponential submit-retry backoff |

The rate flags (`--error-rate`, `--drop-rate`, `--submit-failure-rate`,
`--expire-rate`) only affect the mock provider.

### `relay status`

| flag | default | meaning |
| --- | --- | --- |
| `--tracker` | `outputs/tracker.json` | tracker JSON path |

Prints an ASCII table of `chunk / status / job_id` from the tracker file.

### `relay cost`

Offline token/cost estimate for a batch job. No network calls.

| flag | default | meaning |
| --- | --- | --- |
| `--input` | (required) | input CSV, must have an `order_id,text` header |
| `--model` | (required) | model name to price against (see `relay/cost.py`) |
| `--max-tokens` | `64` | assumed output token budget per item |
| `--limit-items` | none | cap the number of records loaded |

### `relay demo`

| flag | default | meaning |
| --- | --- | --- |
| `--seed` | `7` | mock provider seed |
| `--output-dir` | none | output directory (default: a fresh temp dir, cleaned up unless `--keep`) |
| `--keep` | off | keep the (temp) output directory instead of deleting it |

### `relay smoke`

| flag | default | meaning |
| --- | --- | --- |
| `--provider` | (required) | `openai` \| `anthropic` \| `google` (`mock` not accepted) |
| `--model` | adapter default | model name override |
| `--items` | `2` | number of throwaway items to submit (1-10) |
| `--poll-interval` | `30.0` | seconds between polls (real batch APIs are slow) |

</details>

## Plug in a real provider

By default, `relay run --provider mock` and `relay demo` never make a real
network call. Real providers are opt-in and require both the matching SDK
extra and an API key.

1. Install the matching extra: `pip install -e ".[openai]"` (or
   `.[anthropic]` / `.[google]`).
2. Set the matching env var: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or
   `GEMINI_API_KEY` (the google adapter also accepts `GOOGLE_API_KEY`,
   checked first). Copy `.env.example` to `.env` and fill in the key(s) --
   `relay run` auto-loads `.env` with a stdlib-only parser that never
   overrides an already-set environment variable.
3. Run against the real provider:

   ```
   relay run --input data/records.csv --provider openai --run-id 1
   relay run --input data/records.csv --provider anthropic --run-id 1
   relay run --input data/records.csv --provider google --run-id 1
   ```

If the SDK is missing, the CLI exits 2 with a `pip install relay[<extra>]`
hint. If the API key is missing, it exits 2 naming the env var it expected.

### Validating a real provider

`relay smoke --provider <name>` submits a tiny throwaway batch (2 items by
default, capped at 10) through the real Orchestrator against a real
provider, to prove submit/poll/fetch/custom_id round-trip actually works
end to end. It is not a prompt-quality check: an item that comes back with
text that fails to parse still counts as a PASS, because smoke validates
the adapter, not the model's output format.

```
relay smoke --provider openai --model gpt-4o-mini
relay smoke --provider anthropic --model claude-haiku-4-5
relay smoke --provider google --model gemini-3.5-flash
```

Output goes to a fresh temp directory (printed at the start, never
deleted) so you can inspect the raw run CSV and any failures JSON.

The OpenAI adapter has passed this smoke against the live Batch API
(2026-07-10, `gpt-4o-mini`, 2 items):

```
[run1_p1_chunk0] submitted job batch_6a508fac6d2c8190a65579212b1f4aac
[run1_p1_chunk0] in_progress 0/2
[run1_p1_chunk0] ended 2/2
[run1_p1_chunk0] done: 2 ok, 0 parse failures, 0 item errors
coverage         : 100.0%
converged        : True
smoke_01: ok (parsed)
smoke_02: ok (parsed)
SMOKE PASSED
```

The Google adapter has passed the same smoke against the live Gemini
Batch API (2026-07-10, `gemini-3.5-flash`, 2 items, file-based keyed
flow):

```
[run1_p1_chunk0] submitted job batches/2feqkep8u7dx3cl3qojx284ksswdmk14s6av
[run1_p1_chunk0] running 0/0
[run1_p1_chunk0] ended 0/0
[run1_p1_chunk0] done: 2 ok, 0 parse failures, 0 item errors
coverage         : 100.0%
converged        : True
smoke_01: ok (parsed)
smoke_02: ok (parsed)
SMOKE PASSED
```

Two Gemini gotchas the smoke surfaced, now handled by the adapter: the
default model must be a current-generation one (`gemini-2.5-flash` 404s
per-item for API keys created after its new-user cutoff), and Gemini 3.x
thinking tokens count against `max_output_tokens`, so the adapter pins
`thinking_config.thinking_level: MINIMAL` for `gemini-3*` models to keep
small output budgets from coming back as empty text. The Gemini Batch
API also requires a paid-tier key: free-tier keys fail at submit with
`400 FAILED_PRECONDITION`.

## Known limitations

The three real-provider adapters (`relay/providers/openai.py`,
`anthropic.py`, `google.py`) are unit-tested against fake SDK modules
injected into `sys.modules`: request building, status normalization,
result parsing, and error-taxonomy mapping are all covered offline. The
**OpenAI and Google adapters have additionally been validated end to end
against their live Batch APIs** via `relay smoke` (both 2026-07-10: real
batch submitted, polled to terminal, results fetched and parsed, 100%
coverage -- see the transcripts above). The **Anthropic adapter has not
yet been smoked live**; its request/response shapes mirror a working
legacy script, so its residual risk was always the lowest of the three,
but live validation is one `relay smoke --provider anthropic` away.

Note the distinction: the live smoke validates adapter wiring on the
happy path. The fault-tolerance claims (drops, errors, submit failures,
crash recovery) are demonstrated against the mock provider by design --
real providers cannot be made to fail deterministically on demand.

## Project layout

```
relay/
  cli.py                 # run / status / cost / demo subcommands
  demo.py                # scripted two-act fault-tolerance narrative
  cost.py                 # offline token/cost estimator
  env.py                  # stdlib .env loader
  core/
    models.py             # Record, Request, Job, TaskSpec, JobStatus, BatchResult
    orchestrator.py        # multi-pass coverage loop, resume drain, backoff
    tracker.py             # crash-safe JSON chunk state machine
    chunking.py            # provider-limit-aware request splitting
    coverage.py             # present/missing order-id computation from CSV
    parsing.py              # tolerant result-line parsing
  providers/
    base.py                # Provider protocol + error taxonomy
    mock.py                 # keyless provider with injectable chaos
    openai.py, anthropic.py, google.py  # real provider adapters
    registry.py              # provider name -> class lookup

data/
  records.csv              # 300-row sample dataset (order_id,text)
  make_dataset.py           # deterministic regenerator for records.csv

tests/
  test_*.py                 # 150 offline, keyless tests
  conftest.py                # shared fixtures (make_records, make_job, ...)
```

## Running the tests

```
pytest -q
```

150 tests, all offline and keyless: no network access, no API keys, and
no real provider SDKs required (the real-provider tests run against fake
SDK modules injected into `sys.modules`).

## License / status

Feature-complete portfolio project. No license file is currently included;
treat the code as all-rights-reserved unless a LICENSE is added.
