from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Context:
    """Nano-VLLM-style attention context for the current runner call."""

    is_prefill: bool = False
    cu_seqlens_q: Sequence[int] | None = None
    cu_seqlens_k: Sequence[int] | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Sequence[int] | None = None
    context_lens: Sequence[int] | None = None
    block_tables: Sequence[Sequence[int]] | None = None


_CONTEXT = Context()


def get_context() -> Context:
    """Return the current runner context."""

    return _CONTEXT


def set_context(
    is_prefill: bool,
    cu_seqlens_q: Sequence[int] | None = None,
    cu_seqlens_k: Sequence[int] | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: Sequence[int] | None = None,
    context_lens: Sequence[int] | None = None,
    block_tables: Sequence[Sequence[int]] | None = None,
) -> None:
    """Set the active runner context."""

    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )


def reset_context() -> None:
    """Reset the active runner context to its idle state."""

    global _CONTEXT
    _CONTEXT = Context()
