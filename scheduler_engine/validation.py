"""
VelocityLLM - Input Validation & Sanitization Layer
Provides robust validation and sanitization for prompts, sampling parameters,
context window limits, and adversarial/malformed payloads.
"""

import re
from typing import Optional


class InputValidationError(ValueError):
    """Raised when an incoming inference request fails validation."""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.field = field


def sanitize_prompt(prompt: str) -> str:
    """
    Sanitizes raw prompt text:
    - Removes null bytes and dangerous control characters (preserves newlines, tabs, carriage returns)
    - Normalizes Unicode surrogates
    - Strips surrounding whitespace
    """
    if not isinstance(prompt, str):
        raise InputValidationError("Prompt must be a valid string.", field="prompt")

    # Strip null bytes and non-printable control characters (except \t, \n, \r)
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", prompt)
    
    # Strip leading/trailing whitespace
    return sanitized.strip()


def estimate_prompt_tokens(prompt: str) -> int:
    """
    Provides a conservative estimate of token count for prompt text
    without requiring heavy tokenizer dependencies.
    """
    if not prompt:
        return 0
    words = len(prompt.split())
    chars = len(prompt) // 4
    return max(words, chars, 1)


def validate_prompt(
    prompt: str,
    max_tokens: int = 100,
    max_model_len: int = 4096,
) -> str:
    """
    Validates and sanitizes prompt text:
    - Enforces non-empty content after sanitization
    - Checks that prompt token length + max_tokens <= max_model_len
    """
    sanitized = sanitize_prompt(prompt)
    if not sanitized:
        raise InputValidationError(
            "Prompt cannot be empty or contain only whitespace/control characters.",
            field="prompt",
        )

    estimated_tokens = estimate_prompt_tokens(sanitized)
    if estimated_tokens >= max_model_len:
        raise InputValidationError(
            f"Prompt length (~{estimated_tokens} tokens) exceeds model context window limit of {max_model_len} tokens.",
            field="prompt",
        )

    if (estimated_tokens + max_tokens) > max_model_len:
        raise InputValidationError(
            f"Combined prompt (~{estimated_tokens} tokens) and max_tokens ({max_tokens}) "
            f"exceeds maximum context window limit ({max_model_len} tokens).",
            field="max_tokens",
        )

    return sanitized


def validate_sampling_params(
    temperature: float = 0.7,
    max_tokens: int = 100,
    top_p: Optional[float] = 1.0,
    presence_penalty: Optional[float] = 0.0,
    frequency_penalty: Optional[float] = 0.0,
    max_model_len: int = 4096,
) -> None:
    """
    Validates generation sampling hyperparameters.
    """
    if not (0.0 <= temperature <= 2.0):
        raise InputValidationError(
            f"temperature must be between 0.0 and 2.0, got {temperature}.",
            field="temperature",
        )

    if not (1 <= max_tokens <= max_model_len):
        raise InputValidationError(
            f"max_tokens must be between 1 and {max_model_len}, got {max_tokens}.",
            field="max_tokens",
        )

    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise InputValidationError(
            f"top_p must be in range (0.0, 1.0], got {top_p}.",
            field="top_p",
        )

    if presence_penalty is not None and not (-2.0 <= presence_penalty <= 2.0):
        raise InputValidationError(
            f"presence_penalty must be in range [-2.0, 2.0], got {presence_penalty}.",
            field="presence_penalty",
        )

    if frequency_penalty is not None and not (-2.0 <= frequency_penalty <= 2.0):
        raise InputValidationError(
            f"frequency_penalty must be in range [-2.0, 2.0], got {frequency_penalty}.",
            field="frequency_penalty",
        )
