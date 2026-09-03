import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

MODEL_PATH = "/home/kartiklin/velocityllm/models/llama-3.2-1b"

engine: AsyncLLMEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        max_num_seqs=8,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
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
    sampling_params = SamplingParams(temperature=0.7, max_tokens=req.max_tokens)

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
