"""Activity A: Cache-aware scheduling stub.

Activity A (KV Cache-aware Scheduling / Orchestration) is NOT implemented in
this cycle.  This file exists for structural completeness so that the
``vllm_integration`` package has a consistent layout across cycles.

When Activity A is implemented it will subclass or wrap the vLLM v1 scheduler
located at ``vllm.v1.core.sched`` and override the batch-selection logic to
prioritise requests with high cache-hit potential.

Planned integration points (for reference):
  - vllm/v1/core/sched.py         — main scheduler, batch construction
  - vllm/v1/engine/async_engine.py — async request routing / priority queue
  - vllm/v1/executor/             — per-node KV migration routing (multi-node)

No functional code is present in this cycle.
"""

# Intentionally empty — Activity A not in scope for cycle 2026-04-28.
