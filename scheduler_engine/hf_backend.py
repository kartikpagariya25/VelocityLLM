"""
VelocityLLM - Hugging Face Transformers runner.

Iteration-level model runner: every active sequence owns its KV cache and advances one token per
iteration, so sequences join and leave the running set at token granularity. Prefill and decode are
separate forward passes; the scheduler above decides how many sequences are admitted.
"""

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Tuple

from scheduler_engine.backend import GPUOutOfMemoryError, InferenceBackend
from scheduler_engine.types import ServerConfig

logger = logging.getLogger("velocityllm.hf_backend")

_END = object()


@dataclass
class _Sequence:
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    loop: asyncio.AbstractEventLoop
    out: "asyncio.Queue" = field(default_factory=asyncio.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    generated: List[int] = field(default_factory=list)
    cache: object = None
    next_input: object = None
    emitted_text: str = ""

    def push(self, item) -> None:
        self.loop.call_soon_threadsafe(self.out.put_nowait, item)


class HFBackend(InferenceBackend):
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._torch = None
        self._incoming: "queue.Queue[_Sequence]" = queue.Queue()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._token_counts = {}
        self.active_sequences = 0

    def get_prefix_cache_stats(self) -> dict:
        return {"prefix_cache_hit_rate": 0.0, "available": False, "note": "Transformers runner keeps one KV cache per sequence"}

    async def initialize(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self._load)
        self._worker = threading.Thread(target=self._run, name="hf-runner", daemon=True)
        self._worker.start()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info("Loading %s on %s", self.config.model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, dtype=dtype).to(self.device).eval()

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_id: str,
        image: Optional[str] = None,
    ) -> AsyncIterator[str]:
        if self.model is None:
            raise RuntimeError("Backend engine is not initialized.")
        if image:
            raise ValueError("The Transformers runner serves text-only models.")
        seq = _Sequence(request_id, prompt, max_tokens, temperature, asyncio.get_running_loop())
        self._incoming.put(seq)
        try:
            while True:
                item = await seq.out.get()
                if item is _END:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            seq.cancelled.set()

    def pop_token_count(self, request_id: str) -> Optional[int]:
        return self._token_counts.pop(request_id, None)

    def _run(self) -> None:
        active: List[_Sequence] = []
        while not self._stop.is_set():
            try:
                if not active:
                    active.append(self._incoming.get(timeout=0.1))
                while True:
                    active.append(self._incoming.get_nowait())
            except queue.Empty:
                pass
            self.active_sequences = len(active)
            survivors = []
            for seq in active:
                if self._advance(seq):
                    survivors.append(seq)
            active = survivors
            self.active_sequences = len(active)

    def _advance(self, seq: _Sequence) -> bool:
        """Runs one iteration for a sequence. Returns True while it still has tokens to produce."""
        if seq.cancelled.is_set():
            self._finish(seq)
            return False
        try:
            token = self._step(seq)
        except Exception as err:
            if "out of memory" in str(err).lower():
                self.handle_oom()
                err = GPUOutOfMemoryError(f"CUDA Out of Memory in Transformers runner: {err}")
            seq.push(err)
            self._finish(seq, sentinel=False)
            return False
        seq.generated.append(token)
        text = self.tokenizer.decode(seq.generated, skip_special_tokens=True)
        if text.endswith("�"):
            delta = ""
        else:
            delta = text[len(seq.emitted_text):]
            seq.emitted_text = text
        if delta:
            seq.push(delta)
        eos = self.tokenizer.eos_token_id
        if token == eos or len(seq.generated) >= seq.max_tokens:
            self._finish(seq)
            return False
        return True

    def _step(self, seq: _Sequence) -> int:
        torch = self._torch
        with torch.inference_mode():
            if seq.cache is None:
                ids = self.tokenizer(seq.prompt, return_tensors="pt").input_ids
                limit = max(1, self.config.max_model_len - seq.max_tokens)
                ids = ids[:, -limit:].to(self.device)
                out = self.model(input_ids=ids, use_cache=True)
            else:
                out = self.model(input_ids=seq.next_input, past_key_values=seq.cache, use_cache=True)
            seq.cache = out.past_key_values
            logits = out.logits[:, -1, :].float()
            if seq.temperature and seq.temperature > 0:
                probs = torch.softmax(logits / seq.temperature, dim=-1)
                token = torch.multinomial(probs, 1)
            else:
                token = logits.argmax(dim=-1, keepdim=True)
            seq.next_input = token
            return int(token.item())

    def _finish(self, seq: _Sequence, sentinel: bool = True) -> None:
        if seq.generated:
            if len(self._token_counts) > 2000:
                self._token_counts.clear()
            self._token_counts[seq.request_id] = len(seq.generated)
        seq.cache = None
        seq.next_input = None
        if sentinel:
            seq.push(_END)

    def handle_oom(self) -> None:
        logger.warning("HFBackend: handling GPU OOM, purging CUDA caches.")
        try:
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception as err:
            logger.warning("HFBackend: error while purging CUDA cache: %s", err)

    async def shutdown(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self.model = None

    def get_gpu_telemetry(self) -> Tuple[int, int, int]:
        torch = self._torch
        if torch is None or not torch.cuda.is_available():
            return 0, 0, 0
        try:
            free, total = torch.cuda.mem_get_info()
            try:
                util = int(torch.cuda.utilization())
            except Exception:
                util = 0
            return util, (total - free) // (1024 * 1024), total // (1024 * 1024)
        except Exception:
            return 0, 0, 0
