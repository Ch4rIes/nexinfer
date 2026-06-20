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
- Nano-VLLM-style batched temperature sampler
- structured results with token usage and logprobs
- streaming chunks
- simple batch APIs
- Nano-VLLM-style `LLM.generate` facade and `SamplingParams`
- Nano-VLLM-style `Config`/`LLMConfig` constructor surface
- Nano-VLLM-style config clamping from `hf_config.max_position_embeddings`
- optional `LLM` execution through `ModelRunner` and `Scheduler`
- Nano-VLLM-style `Sequence` payload for scheduler/model-runner handoff
- explicit sequence and decode state
- optional batched backend prefill/decode contract with scheduled-token metadata
- Nano-VLLM-style model-runner batch preparation
- Nano-VLLM-style model-runner block-table preparation
- Nano-VLLM-style model-runner orchestration over prepared sequence batches
- Nano-VLLM-style `ModelRunner.call` and prepare methods
- Nano-VLLM-style model `compute_logits` runner contract
- Nano-VLLM-style attention context utilities for runner calls
- Nano-VLLM-style safetensors model weight loader
- Nano-VLLM-compatible `Sequence` scheduler facade
- Nano-VLLM-style engine loop over `Scheduler` and `ModelRunner`
- FIFO request scheduling
- queued runtime execution
- runtime execution counters
- Nano-VLLM-style `Block`/`BlockManager` cache facade
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

Use the Nano-VLLM-style facade when you want a familiar list-in/list-out API
backed by the same queued scheduler as `add_request`/`step`:

```python
from nexinfer import LLM, SamplingParams

llm = LLM(
    backend=backend,
    tokenizer=tokenizer,
    max_num_seqs=512,
    max_num_batched_tokens=16384,
    max_model_len=4096,
    kvcache_block_size=256,
)
outputs = llm.generate(["hello"], SamplingParams(max_tokens=8), use_tqdm=False)
print(outputs[0]["text"])
print(outputs[0]["token_ids"])
```

The facade also exposes Nano-VLLM-style queue stepping:

```python
request_id = llm.add_request("hello", SamplingParams(max_tokens=8))
while not llm.is_finished():
    completed, num_tokens = llm.step()
    for seq_id, token_ids in completed:
        print(seq_id, token_ids)
```

Call `llm.exit()` when you are done with a long-lived model, or use `LLM` as a
context manager so backend resources are released automatically.

The cache module includes early block-table primitives for future paged KV-cache
work. `PrefixKVCacheBlockManager` adds the Nano-VLLM-style pieces: ref-counted
blocks, cached-prefix hashes, append reservation, and deallocation:

```python
from nexinfer import BlockManager, KVCacheBlockAllocator, PrefixKVCacheBlockManager

allocator = KVCacheBlockAllocator(block_size=16, max_blocks=1024)
allocation = allocator.allocate("request-1", token_count=33)
print(allocation.block_table)

manager = PrefixKVCacheBlockManager(num_blocks=1024, block_size=16)
manager.allocate("request-1", [1, 2, 3, 4])
manager.hash_blocks("request-1", [1, 2, 3, 4])

sequence_manager = BlockManager(1024, 16)
cached_blocks = sequence_manager.can_allocate(sequence)
sequence_manager.allocate(sequence, cached_blocks)
```

The scheduler module starts with a FIFO queue that can form small execution
batches. The active scheduler mirrors Nano-VLLM's higher-level shape with
separate waiting and running queues plus prefill/decode phases:

```python
from nexinfer import ActiveScheduler, RequestQueue, Scheduler, Sequence

queue = RequestQueue()
queue.submit("hello")
batch = queue.schedule(max_requests=8)

active_scheduler = ActiveScheduler(max_num_seqs=8, max_num_batched_tokens=2048)

scheduler = Scheduler(max_num_seqs=8, max_num_batched_tokens=2048)
scheduler.add(Sequence([1, 2, 3]))
seqs, is_prefill = scheduler.schedule()
```

Creating a `Scheduler` updates `Sequence.block_size` to the scheduler's KV-cache
block size, so sequence block helpers report the same geometry used for
allocation and decode slot mapping.

For direct Nano-VLLM-style runner orchestration, use `NanoLLMEngine` with a
`Scheduler` and `ModelRunner`:

```python
from nexinfer import LLM, ModelRunner, NanoLLMEngine, Sampler, SamplingParams

runner = ModelRunner(model, block_size=16, sampler=Sampler())
nano_engine = NanoLLMEngine(runner, tokenizer, scheduler)
nano_engine.add_request("hello", SamplingParams(max_tokens=8))
outputs, num_tokens = nano_engine.step()
```

`LLM` can also be pointed at a `ModelRunner` directly when you want the public
facade to use the Nano-VLLM scheduler/model-runner loop:

```python
llm = LLM(
    model_runner=runner,
    tokenizer=tokenizer,
    num_kvcache_blocks=1024,
)
```

For queued execution, wrap an engine in `InferenceRuntime`:

```python
from nexinfer import InferenceRuntime, PrefixKVCacheBlockManager

block_manager = PrefixKVCacheBlockManager(num_blocks=1024, block_size=16)

runtime = InferenceRuntime(
    engine,
    max_batch_size=8,
    max_batch_prompt_tokens=2048,
    decode_strategy="continuous",
    block_manager=block_manager,
)
runtime.submit("hello", request_id="request-1")
completed = runtime.run_once()
all_completed = runtime.run_until_idle()
print(runtime.stats.total_tokens)
```

`continuous` executes one scheduler phase per `run_once`: prefill work is
scheduled before decode work, and unfinished sequences stay in the running queue
until a later decode phase completes them. When KV append capacity is exhausted,
the active scheduler can preempt a running sequence, free its blocks, and move it
back to waiting for a later prefill phase. Prompts larger than
`max_batch_prompt_tokens` are admitted through chunked prefill phases; the current
backend path runs model prefill when the final prompt chunk is admitted.
Backend batch inputs carry the scheduled token count, cached-token count, and
block table so future tensor runners can prepare prefill/decode contexts in the
same shape as Nano-VLLM.

```python
from nexinfer import prepare_prefill_batch, prepare_prefill_sequences

prepared = prepare_prefill_batch(prefill_inputs, block_size=16)
print(prepared.input_ids, prepared.positions, prepared.slot_mapping)

prepared_from_sequences = prepare_prefill_sequences(sequences, block_size=16)
```

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

The optional Torch backend records prepared runner contexts for batched prefill
and decode calls, so future paged-attention runners can consume the same
scheduling metadata while the current fallback stays eager and simple.

## Roadmap

Near-term:

- exercise the PyTorch backend against a tiny local model
- expose OpenAI-style completion response fields
- add true batched prefill/decode instead of sequential batch execution
- promote runtime stats into latency-aware metrics

Mid-term:

- connect `DecodeState.cache` to real KV-cache tensors
- replace toy batch methods with real tensor-batched model runner calls

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
