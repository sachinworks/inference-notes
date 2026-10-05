# inference-notes

Working notes on **LLM inference and serving**: how the engines are built, where the time goes, and what actually moves latency, throughput and cost in production.

I read the source of serving engines (vLLM, SGLang, TensorRT-LLM, FlashInfer), run small benchmarks, and write down what I learn. Notes are my own understanding at the time of writing. Engines move fast, so each note records the version or commit it was written against.

## Layout

| Folder | What's inside |
|---|---|
| [`notes/`](notes) | Deep dives on engine internals, one folder per project |
| [`benchmarks/`](benchmarks) | Reproducible experiments: setup, commands, raw numbers, conclusions |
| [`cuda/`](cuda) | Small CUDA / Triton kernels written to understand GPU performance |
| [`log/`](log) | Short daily log: what I read, ran or learned that day |

## Index

### vLLM
- [01 · How the V1 scheduler decides what runs each step](notes/vllm/01-scheduler.md)

### Planned
- vLLM: PagedAttention and the KV-cache block manager
- vLLM: automatic prefix caching (block hashing, eviction)
- SGLang: RadixAttention vs. vLLM prefix caching
- TensorRT-LLM: in-flight batching and FP8 paths
- FlashInfer: attention kernels for decode vs. prefill
- Benchmarks: chunked-prefill token budget vs. TTFT / ITL
- CUDA: tiled matmul from naive to shared memory to tensor cores

## Conventions

- Every note starts with **Version**, **Source files** and **Questions** so it can be re-checked later.
- Benchmarks record hardware, model, engine version and the exact command.
- Corrections are welcome. Open an issue if something is wrong or out of date.

---

Sachin Kr. Rajput · [sachinrajput.dev](https://sachinrajput.dev) · [dev.to](https://dev.to/sachin_krrajput) · [LinkedIn](https://www.linkedin.com/in/skrajput18/)
