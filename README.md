# NexInfer

A small Python library for decoder-only LLM inference.

The first version keeps the generation loop independent from any tensor runtime.
Backends only need to implement a simple protocol:

- `begin(input_ids)` returns next-token logits after a prompt.
- `step(token_id, state)` consumes the sampled token and returns the next logits.

That shape makes it straightforward to add PyTorch, safetensors, quantized, or
remote backends without rewriting sampling and stop handling.

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

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```
