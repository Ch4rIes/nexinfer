# NexInfer

A small Python library for decoder-only LLM inference.

The first version keeps the generation loop independent from any tensor runtime.
Backends only need to implement a simple protocol:

- `begin(input_ids)` returns next-token logits after a prompt.
- `step(token_id, state)` consumes the sampled token and returns the next logits.

That shape makes it straightforward to add PyTorch, safetensors, quantized, or
remote backends without rewriting sampling and stop handling.

## What exists now

- backend-agnostic generation engine
- greedy, temperature, top-k, and top-p sampling
- structured results with token usage and logprobs
- streaming chunks
- simple batch APIs
- explicit sequence and decode state
- optional batched backend prefill/decode contract
- FIFO request scheduling
- queued runtime execution
- runtime execution counters
- early KV-cache block-table primitives
- optional Hugging Face tokenizer adapter
- optional PyTorch causal-LM backend scaffold

## Quick start

```python
from nexinfer import GenerationConfig, LLMEngine, VocabularyTokenizer
from nexinfer.backends import BigramBackend

tokenizer = VocabularyTokenizer(["hello", "world", "<eos>"], eos_token="<eos>")
backend = BigramBackend(
    vocab_size=len(tokenizer),
    transitions={
        tokenizer.token_id("hello"): {tokenizer.token_id("world"): 5.0},
        tokenizer.token_id("world"): {tokenizer.eos_token_id: 5.0},
    },
)

engine = LLMEngine(backend, tokenizer)
text = engine.generate(
    "hello",
    GenerationConfig(max_new_tokens=8, stop_token_ids=(tokenizer.eos_token_id,)),
)

print(text)
```

Use `complete` when you need structured metadata:

```python
result = engine.complete("hello")
print(result.text)
print(result.finish_reason)
print(result.generated_token_logprobs)
print(result.usage.total_tokens)
```

Use `complete_batch` for multiple prompts:

```python
results = engine.complete_batch(["hello", "world"])
for result in results:
    print(result.text)
```

The cache module includes early block-table primitives for future paged KV-cache
work:

```python
from nexinfer import KVCacheBlockAllocator

allocator = KVCacheBlockAllocator(block_size=16, max_blocks=1024)
allocation = allocator.allocate("request-1", token_count=33)
print(allocation.block_table)
```

The scheduler module starts with a FIFO queue that can form small execution
batches. The active scheduler mirrors Nano-VLLM's higher-level shape with
separate waiting and running queues plus prefill/decode phases:

```python
from nexinfer import ActiveScheduler, RequestQueue

queue = RequestQueue()
queue.submit("hello")
batch = queue.schedule(max_requests=8)

active_scheduler = ActiveScheduler(max_num_seqs=8, max_num_batched_tokens=2048)
```

For queued execution, wrap an engine in `InferenceRuntime`:

```python
from nexinfer import InferenceRuntime

runtime = InferenceRuntime(
    engine,
    max_batch_size=8,
    max_batch_prompt_tokens=2048,
    decode_strategy="continuous",
)
runtime.submit("hello", request_id="request-1")
completed = runtime.run_once()
all_completed = runtime.run_until_idle()
print(runtime.stats.total_tokens)
```

`continuous` executes one scheduler phase per `run_once`: prefill work is
scheduled before decode work, and unfinished sequences stay in the running queue
until a later decode phase completes them.

## Optional integrations

Install Hugging Face tokenizer support:

```bash
python -m pip install -e ".[transformers]"
```

Install the optional PyTorch backend dependencies:

```bash
python -m pip install -e ".[torch]"
```

Then load compatible components:

```python
from nexinfer import HuggingFaceTokenizer, ModelConfig
from nexinfer.backends import TorchCausalLMBackend

config = ModelConfig("gpt2", device="auto", dtype="auto")
tokenizer = HuggingFaceTokenizer.from_pretrained(config.model_name_or_path)
backend = TorchCausalLMBackend.from_pretrained(config)
```

## Roadmap

Near-term:

- exercise the PyTorch backend against a tiny local model
- expose OpenAI-style completion response fields
- add true batched prefill/decode instead of sequential batch execution
- promote runtime stats into latency-aware metrics

Mid-term:

- connect `DecodeState.cache` to real KV-cache tensors
- connect the active scheduler to KV-cache block allocation and preemption
- replace toy batch methods with real tensor-batched model runner calls
- wire the block allocator into per-sequence KV-cache ownership
- add prefix-cache primitives

Later:

- add HTTP serving and streaming endpoints
- support structured generation constraints
- add quantized loading paths
- support FlashAttention/SDPA backend choices

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```
