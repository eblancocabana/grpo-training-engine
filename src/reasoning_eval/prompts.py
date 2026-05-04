"""Prompt construction shared by all benchmark tasks."""

from __future__ import annotations

from typing import Any

from src.reasoning_eval.schema import EvalExample


MATH_SUFFIX = (
    "Solve the problem. Show your reasoning, then put only the final answer in "
    "\\boxed{}."
)
MC_SUFFIX = (
    "Answer the multiple-choice question. Show your reasoning, then put only the "
    "final option letter in \\boxed{}."
)
DROP_SUFFIX = (
    "Answer the question from the passage. Show concise reasoning, then put only "
    "the final answer in \\boxed{}."
)
CODE_SUFFIX = (
    "Write a complete Python solution. Return only code in the final answer; do "
    "not include Markdown fences."
)


def _chat_format(tokenizer: Any, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return content


def build_prompt(example: EvalExample, tokenizer: Any | None = None) -> str:
    """Build the canonical prompt for an example."""
    if example.task_type == "multiple_choice":
        choices = "\n".join(
            f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(example.choices)
        )
        content = f"{example.question}\n\nChoices:\n{choices}\n\n{MC_SUFFIX}"
    elif example.task_type == "drop":
        passage = example.metadata.get("passage")
        if passage:
            content = f"Passage:\n{passage}\n\nQuestion:\n{example.question}\n\n{DROP_SUFFIX}"
        else:
            content = f"{example.question}\n\n{DROP_SUFFIX}"
    elif example.task_type == "code":
        content = f"{example.question}\n\n{CODE_SUFFIX}"
    else:
        content = f"{example.question}\n\n{MATH_SUFFIX}"

    return _chat_format(tokenizer, content) if tokenizer is not None else content

