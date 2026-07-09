# polybatch

Fault-tolerant, resumable batch-inference orchestrator across OpenAI, Anthropic,
and Google, plus a keyless mock provider for local testing. polybatch submits
large batches of inference requests, tracks per-request state through a JSON
tracker so runs can crash-safely resume, splits oversized inputs into
provider-safe chunks, assigns stable order IDs for result reassembly, and
re-sends failed or missing records until full coverage is guaranteed.

Work in progress - see plan.md and PROGRESS.md.
