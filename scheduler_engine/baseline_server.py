import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from typing import Any, Optional

try:
    from vllm import SamplingParams  # type: ignore
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
        from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore
    except ImportError:
        from vllm import AsyncEngineArgs, AsyncLLMEngine  # type: ignore
except ImportError:
    SamplingParams = None
    AsyncEngineArgs = None
    AsyncLLMEngine = None

MODEL_PATH = "/home/kartiklin/velocityllm/models/llama-3.2-1b"

engine: Any = None


class _SimulatedOutputItem:
    def __init__(self, text: str):
        self.text = text


class _SimulatedOutput:
    def __init__(self, text: str):
        self.outputs = [_SimulatedOutputItem(text)]


class _SimulatedEngine:
    """Fallback engine used when running in environments without vLLM/CUDA (e.g. Windows)."""
    async def generate(self, prompt: str, sampling_params: Any, request_id: str):
        await asyncio.sleep(0.05)
        yield _SimulatedOutput(f"Baseline simulated completion for: {prompt}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    if AsyncLLMEngine is not None:
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            gpu_memory_utilization=0.80,
            max_model_len=4096,
            max_num_seqs=8,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
    else:
        print("[VelocityLLM] Notice: vLLM not detected in this environment. Operating in baseline simulation mode.")
        engine = _SimulatedEngine()
    yield


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


class GenerateResponse(BaseModel):
    request_id: str
    prompt: str
    response: str
    latency_seconds: float


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    request_id = str(uuid.uuid4())
    if SamplingParams is not None:
        sampling_params = SamplingParams(temperature=0.7, max_tokens=req.max_tokens)
    else:
        sampling_params = {"temperature": 0.7, "max_tokens": req.max_tokens}

    start = time.time()
    final_output = None
    async for output in engine.generate(req.prompt, sampling_params, request_id):
        final_output = output
    elapsed = time.time() - start

    response_text = final_output.outputs[0].text

    return GenerateResponse(
        request_id=request_id,
        prompt=req.prompt,
        response=response_text,
        latency_seconds=elapsed,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
