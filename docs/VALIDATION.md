# Relay - Live validation record

Dated transcripts from running `relay smoke` against each real
provider's batch API, plus the provider-specific gotchas those runs
surfaced. The methodology (what the smoke does and does not prove) is
summarized in the [README](../README.md); this file is the receipt.

Each smoke submits a tiny throwaway batch (2 items) through the real
Orchestrator: real batch submitted, polled to a terminal state, results
fetched, every `custom_id` round-tripped, 100% coverage.

## OpenAI

Passed against the live Batch API (2026-07-10, `gpt-4o-mini`, 2 items,
first attempt):

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

## Anthropic

Passed against the live Message Batches API (2026-07-10,
`claude-haiku-4-5`, 2 items, first attempt):

```
[run1_p1_chunk0] submitted job msgbatch_01J1f985HV3NhngLgexXWFY2
[run1_p1_chunk0] in_progress 0/2
[run1_p1_chunk0] ended 2/2
[run1_p1_chunk0] done: 2 ok, 0 parse failures, 0 item errors
coverage         : 100.0%
converged        : True
smoke_01: ok (parsed)
smoke_02: ok (parsed)
SMOKE PASSED
```

## Google (Gemini)

Passed against the live Gemini Batch API (2026-07-10,
`gemini-3.5-flash`, 2 items, file-based keyed flow):

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

### Gemini gotchas the smoke surfaced (now handled by the adapter)

- **Model gating.** The default model must be a current-generation one:
  `gemini-2.5-flash` returns a per-item 404 (`code 5 NOT_FOUND` on every
  item of an otherwise-successful job) for API keys created after its
  new-user cutoff, because the file-based batch flow defers model
  resolution to per-item processing. The adapter default is
  `gemini-3.5-flash`.
- **Thinking tokens.** Gemini 3.x models think by default and thinking
  tokens count against `max_output_tokens`, so small output budgets come
  back as empty text (MAX_TOKENS hit mid-thought). The adapter pins
  `generation_config.thinking_config.thinking_level: MINIMAL` for
  `gemini-3*` models. Note the nesting: the docs' flat
  `thinking_level` is SDK-only sugar that the raw batch-file parser
  rejects with `no such field`.
- **Key tier.** The Gemini Batch API requires a paid-tier key: free-tier
  keys fail at submit with `400 FAILED_PRECONDITION`.
