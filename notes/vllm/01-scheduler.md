# vLLM 01 · How the V1 scheduler decides what runs each step

- **Version:** vLLM V1 engine (2025–2026 `main`). Verify details against the commit you are reading.
- **Source files:** `vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/request.py`
- **Status:** first pass, reading notes

## The one idea to hold on to

The V1 scheduler does not think in "prefill phase" and "decode phase". Every request just has:

- `num_tokens`: tokens it has so far (prompt plus generated output)
- `num_computed_tokens`: tokens whose KV cache already exists

On each engine step the scheduler hands out a **token budget**. For each request it decides how many of the "not yet computed" tokens to process this step. A fresh prompt might get 2,000 tokens. A decoding request gets 1, or 1 + k with speculative decoding. A long prompt can be split across several steps, which is **chunked prefill**. Prefill and decode share the same batch.

That one abstraction explains most of the behaviour below.

## What limits a step

| Knob | What it caps |
|---|---|
| `max_num_batched_tokens` | Total tokens processed in one step (the token budget) |
| `max_num_seqs` | Number of requests in the running batch |
| KV-cache blocks | Whether there is GPU memory to hold the new tokens' KV |

The budget is the main lever. Larger budgets favour throughput, because prompts finish in fewer steps. Smaller budgets protect inter-token latency for requests already decoding, because a big prefill can't hog a step.

## The scheduling loop (simplified)

1. **Running requests first.** Walk the running queue. For each request, work out how many new tokens it needs this step, capped by the remaining budget. Ask the KV-cache manager for enough blocks.
2. **If blocks can't be allocated, preempt.** Under FCFS the request at the back of the running queue is preempted: its blocks are freed and it goes back to waiting. V1 preempts by **recompute**. The KV is dropped and rebuilt later rather than swapped to CPU. Repeat until the current request fits, or it is the one preempted.
3. **Then waiting requests**, if there were no preemptions this step and budget and `max_num_seqs` allow. Before allocating, the scheduler checks the **prefix cache**. Full blocks whose hash matches an existing block are reused, which advances `num_computed_tokens` for free. Only the remaining tokens consume budget.
4. **Emit a `SchedulerOutput`**: which requests run, how many tokens each, and which blocks they use. The model runner executes it as one batch.
5. **After the forward pass,** update each request with its new tokens, check stop conditions, and free blocks of finished requests.

## KV cache in one paragraph

GPU memory left after weights and activations is carved into fixed-size **blocks** (`block_size` tokens each, typically 16). A request's KV lives in a list of blocks that need not be contiguous. That is the PagedAttention idea, and it is why fragmentation stays low. With prefix caching each *full* block is identified by a hash of its tokens chained with the previous block's hash. Two requests sharing a system prompt then share physical blocks. Freed blocks stay reusable until they are evicted (LRU), so a repeated prefix can hit cache even after the first request finished.

## Things that surprised me

- Chunked prefill is not a special mode. It falls out of "give each request *some* of its uncomputed tokens".
- Preemption order matters for tail latency. Under memory pressure the most recently admitted requests get preempted and pay a full recompute.
- Prefix-cache hits reduce compute, not just memory. Cached tokens never enter the budget.

## Open questions (to check in code / benchmarks)

- [ ] How does the priority policy change which request is preempted?
- [ ] What is the TTFT vs. inter-token-latency curve as `max_num_batched_tokens` goes 512 → 2048 → 8192 on one GPU?
- [ ] How are speculative-decoding draft tokens accounted against the budget when they are rejected?
- [ ] At what prefix-cache hit rate does recompute-based preemption become cheap enough not to matter?

## References

- vLLM source: https://github.com/vllm-project/vllm
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024 (chunked prefill)
