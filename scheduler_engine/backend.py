"""
VelocityLLM - Inference Backend Abstraction
Provides swappable inference execution backends:
1. VLLMBackend: Production async engine utilizing vLLM PagedAttention and CUDA.
2. MockBackend: High-fidelity simulation for deterministic automated testing and portable environments.
"""

import os
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from abc import ABC, abstractmethod
import asyncio
import logging
import random
import time
from typing import AsyncIterator, Dict, Optional, Tuple

from scheduler_engine.types import ServerConfig

logger = logging.getLogger("velocityllm.backend")


class GPUOutOfMemoryError(RuntimeError):
    """Raised when the backend encounters GPU out-of-memory pressure."""
    pass


class InferenceBackend(ABC):
    """Abstract interface for LLM inference execution backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Perform initialization (e.g. engine warmup or model loading)."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_id: str,
    ) -> AsyncIterator[str]:
        """
        Yield generated text tokens iteratively for the given prompt.
        """
        pass

    def handle_oom(self) -> None:
        """Emergency cleanup and cache purging when GPU OOM occurs."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        pass

    @abstractmethod
    def get_gpu_telemetry(self) -> Tuple[int, int, int]:
        """
        Returns (gpu_util_percent, mem_used_mb, mem_total_mb).
        """
        pass


class VLLMBackend(InferenceBackend):
    """
    Production backend integrating directly with vLLM's AsyncLLMEngine.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.engine = None
        self._sampling_params_class = None

    async def initialize(self) -> None:
        try:
            from vllm import SamplingParams  # type: ignore
            from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
            from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore

            self._sampling_params_class = SamplingParams

            logger.info("Initializing vLLM AsyncLLMEngine from model: %s", self.config.model_path)
            engine_args = AsyncEngineArgs(
                model=self.config.model_path,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                max_num_seqs=self.config.max_concurrency,
            )
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("vLLM AsyncLLMEngine initialized successfully.")
        except ImportError as e:
            logger.error("vLLM is not installed in this environment: %s", e)
            raise RuntimeError(
                "vLLM is required for VLLMBackend. Install vLLM or run with --mock flag."
            ) from e

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_id: str,
    ) -> AsyncIterator[str]:
        if not self.engine:
            raise RuntimeError("Backend engine is not initialized.")

        sampling_params = self._sampling_params_class(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        previous_text = ""
        try:
            async for output in self.engine.generate(prompt, sampling_params, request_id):
                if output.outputs:
                    current_text = output.outputs[0].text
                    new_delta = current_text[len(previous_text):]
                    previous_text = current_text
                    if new_delta:
                        yield new_delta
        except Exception as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "cuda error: out of memory" in err_msg:
                self.handle_oom()
                raise GPUOutOfMemoryError(f"CUDA Out of Memory in vLLM engine: {e}") from e
            raise

    def handle_oom(self) -> None:
        """Purge GPU cache upon CUDA OOM event."""
        logger.warning("VLLMBackend: Handling GPU OOM event. Purging CUDA caches.")
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as err:
            logger.warning("VLLMBackend: Error while purging CUDA cache: %s", err)

    async def shutdown(self) -> None:
        logger.info("Shutting down VLLMBackend.")
        self.engine = None

    def get_gpu_telemetry(self) -> Tuple[int, int, int]:
        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 3:
                    return int(parts[0]), int(parts[1]), int(parts[2])
        except Exception:
            pass
        return 0, 0, 8192


class MockBackend(InferenceBackend):
    """
    High-fidelity simulated inference engine for zero-dependency testing,
    continuous integration, and benchmark validation.
    """

    def __init__(
        self,
        config: Optional[ServerConfig] = None,
        tokens_per_second: float = 80.0,
        simulated_ttft_seconds: float = 0.02,
    ):
        self.config = config or ServerConfig(use_mock_backend=True)
        self.tokens_per_second = tokens_per_second
        self.simulated_ttft_seconds = simulated_ttft_seconds
        self.active_inferences = 0
        self.total_generated_tokens = 0
        self.initialized = False

        # Phase 3: Fault-injection and memory pressure testing hooks
        self._simulate_oom: bool = False
        self._custom_mem_used_mb: Optional[int] = None
        self._custom_mem_total_mb: int = 8192

    def inject_oom(self, enable: bool = True) -> None:
        """Inject a simulated CUDA OOM error on subsequent generation."""
        self._simulate_oom = enable

    def set_simulated_memory(self, used_mb: int, total_mb: int = 8192) -> None:
        """Artificially set reported memory for testing soft/hard limits."""
        self._custom_mem_used_mb = used_mb
        self._custom_mem_total_mb = total_mb

    def reset_simulated_memory(self) -> None:
        """Reset simulated memory override back to dynamic calculation."""
        self._custom_mem_used_mb = None

    def handle_oom(self) -> None:
        """Reset simulated OOM trigger and drop simulated memory pressure."""
        logger.warning("MockBackend: Handling simulated GPU OOM event. Purging mock VRAM.")
        self._simulate_oom = False
        self._custom_mem_used_mb = None

    async def initialize(self) -> None:
        self.initialized = True
        logger.info("MockBackend initialized (simulated rate: %.1f tokens/s).", self.tokens_per_second)

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_id: str,
    ) -> AsyncIterator[str]:
        if not self.initialized:
            await self.initialize()

        if self._simulate_oom:
            self.handle_oom()
            raise GPUOutOfMemoryError("Simulated CUDA Out of Memory (OOM) during generation.")

        self.active_inferences += 1
        try:
            # Simulate initial prefill / time-to-first-token delay
            await asyncio.sleep(self.simulated_ttft_seconds)

            # Determine response token length (either max_tokens or natural completion)
            target_tokens = min(max_tokens, random.randint(max(1, max_tokens // 2), max_tokens))
            delay_per_token = 1.0 / max(1.0, self.tokens_per_second)

            sample_vocabulary = [
                "The", " quick", " dynamic", " scheduler", " efficiently",
                " batches", " requests", " on", " the", " GPU",
                " while", " maintaining", " strict", " latency", " SLAs",
                " and", " maximizing", " hardware", " utilization.", " Every",
                " token", " is", " processed", " seamlessly", " with", " continuous",
                " iteration-level", " scheduling", " throughput."
            ]

            for i in range(target_tokens):
                if self._simulate_oom:
                    self.handle_oom()
                    raise GPUOutOfMemoryError("Simulated CUDA Out of Memory (OOM) mid-stream.")
                word = sample_vocabulary[i % len(sample_vocabulary)]
                yield word
                self.total_generated_tokens += 1
                await asyncio.sleep(delay_per_token)
        finally:
            self.active_inferences = max(0, self.active_inferences - 1)

    async def shutdown(self) -> None:
        self.initialized = False
        logger.info("MockBackend shutdown.")

    def get_gpu_telemetry(self) -> Tuple[int, int, int]:
        total_mem = self._custom_mem_total_mb
        if self._custom_mem_used_mb is not None:
            util = min(100, int((self._custom_mem_used_mb / total_mem) * 100))
            return util, self._custom_mem_used_mb, total_mem

        # Synthesize realistic GPU telemetry based on current active concurrency
        simulated_util = min(98, 20 + (self.active_inferences * 8))
        simulated_mem = min(total_mem, 3200 + (self.active_inferences * 160))
        return simulated_util, simulated_mem, total_mem


def create_backend(config: ServerConfig) -> InferenceBackend:
    """Factory to create and return the appropriate backend instance."""
    if config.use_mock_backend:
        return MockBackend(config)

    try:
        import vllm  # noqa: F401  # type: ignore
        return VLLMBackend(config)
    except (ImportError, Exception) as err:
        logger.warning("vLLM unavailable (%s). Falling back to MockBackend.", err)
        return MockBackend(config)
