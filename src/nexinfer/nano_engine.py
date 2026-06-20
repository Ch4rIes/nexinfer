from __future__ import annotations

import atexit
from collections.abc import Sequence
from time import perf_counter
from typing import Any, TypedDict

from nexinfer.errors import ConfigurationError
from nexinfer.sampling_params import SamplingParams
from nexinfer.scheduler import Scheduler
from nexinfer.sequence import Sequence as RunnerSequence


class NanoLLMOutput(TypedDict):
    """Nano-VLLM-style generation output."""

    text: str
    token_ids: list[int]


NanoLLMStepOutput = tuple[list[tuple[int, list[int]]], int]


class NanoLLMEngine:
    """Nano-VLLM-style engine over Scheduler and ModelRunner objects."""

    def __init__(
        self,
        model_runner: Any,
        tokenizer: Any,
        scheduler: Scheduler,
    ) -> None:
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self._closed = False
        self._exit_callback = self.exit
        atexit.register(self._exit_callback)

    def add_request(
        self,
        prompt: str | Sequence[int],
        sampling_params: SamplingParams,
    ) -> int:
        self._ensure_open()
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = [int(token_id) for token_id in prompt]

        seq = RunnerSequence(token_ids, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self) -> NanoLLMStepOutput:
        self._ensure_open()
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0

        num_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs)
            if is_prefill
            else -len(seqs)
        )
        token_ids = self._runner_call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]
        return outputs, num_tokens

    def is_finished(self) -> bool:
        if self._closed:
            return True
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: Sequence[str] | Sequence[Sequence[int]],
        sampling_params: SamplingParams | Sequence[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[NanoLLMOutput]:
        self._ensure_open()
        prompt_list = list(prompts)
        params = self._normalize_sampling_params(sampling_params, len(prompt_list))
        seq_ids = [
            self.add_request(prompt, param)
            for prompt, param in zip(prompt_list, params, strict=True)
        ]

        outputs: dict[int, list[int]] = {}
        prefill_throughput = 0.0
        decode_throughput = 0.0
        progress = _progress_bar(total=len(prompt_list), enabled=use_tqdm)
        try:
            while not self.is_finished():
                start = perf_counter()
                step_outputs, num_tokens = self.step()
                elapsed = max(perf_counter() - start, 1e-12)
                if num_tokens > 0:
                    prefill_throughput = num_tokens / elapsed
                elif num_tokens < 0:
                    decode_throughput = -num_tokens / elapsed
                progress.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
                for seq_id, token_ids in step_outputs:
                    outputs[seq_id] = token_ids
                    progress.update(1)
        finally:
            progress.close()

        return [
            {
                "text": self.tokenizer.decode(outputs[seq_id]),
                "token_ids": outputs[seq_id],
            }
            for seq_id in seq_ids
        ]

    def exit(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            atexit.unregister(self._exit_callback)
        except ValueError:
            pass
        self._runner_exit()

    close = exit

    def _normalize_sampling_params(
        self,
        sampling_params: SamplingParams | Sequence[SamplingParams],
        prompt_count: int,
    ) -> list[SamplingParams]:
        if isinstance(sampling_params, SamplingParams):
            return [sampling_params] * prompt_count

        params = list(sampling_params)
        if len(params) != prompt_count:
            raise ConfigurationError(
                "sampling_params must contain one item per prompt when provided as a list"
            )
        return params

    def _runner_call(self, method_name: str, *args: Any) -> Any:
        call = getattr(self.model_runner, "call", None)
        if callable(call):
            return call(method_name, *args)
        method = getattr(self.model_runner, method_name)
        return method(*args)

    def _runner_exit(self) -> None:
        call = getattr(self.model_runner, "call", None)
        if callable(call):
            call("exit")
            return
        for method_name in ("exit", "close"):
            method = getattr(self.model_runner, method_name, None)
            if callable(method):
                method()
                return

    def _ensure_open(self) -> None:
        if self._closed:
            raise ConfigurationError("NanoLLMEngine is closed")


class _NoopProgress:
    def set_postfix(self, values: dict[str, str]) -> None:
        pass

    def update(self, amount: int) -> None:
        pass

    def close(self) -> None:
        pass


def _progress_bar(*, total: int, enabled: bool) -> Any:
    if not enabled:
        return _NoopProgress()
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NoopProgress()
    return tqdm(
        total=total,
        desc="Generating",
        dynamic_ncols=True,
        disable=not enabled,
    )
