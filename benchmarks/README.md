# Benchmarks

Small, reproducible experiments. Each benchmark gets its own folder:

```
benchmarks/
  001-chunked-prefill-budget/
    README.md      # question, setup, results, conclusion
    run.sh         # exact command(s)
    results.csv    # raw numbers
```

## Every benchmark README records

- **Question:** the one thing this experiment answers
- **Hardware:** GPU model, count, driver, CUDA version
- **Software:** engine and version or commit, model and precision
- **Workload:** prompt and output length distribution, request rate, concurrency
- **Metrics:** TTFT (p50/p99), inter-token latency (p50/p99), throughput (tokens/s), GPU memory
- **Result:** table or chart, plus a short conclusion and caveats

## Queue

| # | Question | Status |
|---|---|---|
| 001 | How does `max_num_batched_tokens` trade TTFT against inter-token latency in vLLM? | planned |
| 002 | Throughput gain from prefix caching on a shared-system-prompt workload | planned |
| 003 | vLLM vs. SGLang on the same model, GPU and workload | planned |
| 004 | FP16 vs. FP8 vs. AWQ-INT4: latency, throughput and quality | planned |
