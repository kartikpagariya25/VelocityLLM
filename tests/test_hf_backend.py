import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from scheduler_engine.backend import create_backend
from scheduler_engine.hf_backend import HFBackend
from scheduler_engine.types import ServerConfig

CORPUS = ["the scheduler admits requests into the running batch", "tokens stream back as each step finishes"] * 20


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_gpt2")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        CORPUS,
        trainers.BpeTrainer(vocab_size=300, special_tokens=["<eos>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")
    fast.save_pretrained(path)
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=len(fast), n_layer=2, n_embd=32, n_head=2, n_positions=256, eos_token_id=0, bos_token_id=0)
    GPT2LMHeadModel(config).save_pretrained(path)
    return str(path)


@pytest.fixture
def config(model_dir):
    return ServerConfig(model_path=model_dir, runner="hf", max_model_len=200)


async def collect(backend, prompt, max_tokens, request_id, temperature=0.0):
    pieces = []
    async for piece in backend.generate_stream(prompt, max_tokens, temperature, request_id):
        pieces.append(piece)
    return "".join(pieces)


def test_factory_selects_transformers_runner(config):
    assert isinstance(create_backend(config), HFBackend)


@pytest.mark.asyncio
async def test_greedy_output_matches_reference_generate(config):
    backend = HFBackend(config)
    await backend.initialize()
    try:
        prompt = "the scheduler admits"
        text = await collect(backend, prompt, 12, "r1")
        count = backend.pop_token_count("r1")
        ids = backend.tokenizer(prompt, return_tensors="pt").input_ids.to(backend.device)
        with torch.inference_mode():
            ref = backend.model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=0)
        expected = backend.tokenizer.decode(ref[0, ids.shape[1]:], skip_special_tokens=True)
        assert text == expected
        assert 0 < count <= 12
    finally:
        await backend.shutdown()


@pytest.mark.asyncio
async def test_concurrent_sequences_are_interleaved(config):
    backend = HFBackend(config)
    await backend.initialize()
    try:
        order = []

        async def run(name, prompt, n):
            async for _ in backend.generate_stream(prompt, n, 0.0, name):
                order.append(name)

        await asyncio.gather(run("a", "tokens stream", 20), run("b", "the batch", 20), run("c", "requests", 20))
        assert len(set(order[:6])) > 1
    finally:
        await backend.shutdown()


@pytest.mark.asyncio
async def test_max_tokens_is_respected_and_sampling_runs(config):
    backend = HFBackend(config)
    await backend.initialize()
    try:
        await collect(backend, "the scheduler", 5, "s1", temperature=0.8)
        assert backend.pop_token_count("s1") <= 5
    finally:
        await backend.shutdown()


@pytest.mark.asyncio
async def test_cancelled_request_frees_the_runner(config):
    backend = HFBackend(config)
    await backend.initialize()
    try:
        gen = None
        for attempt, prompt in enumerate(["the scheduler", "tokens", "the batch", "requests stream", "each step"]):
            candidate = backend.generate_stream(prompt, 150, 0.8, f"x{attempt}")
            try:
                await candidate.__anext__()
            except StopAsyncIteration:
                continue
            gen = candidate
            break
        assert gen is not None
        await gen.aclose()
        text = await asyncio.wait_for(collect(backend, "tokens stream", 6, "y"), timeout=20)
        assert backend.pop_token_count("y") > 0
        assert isinstance(text, str)
    finally:
        await backend.shutdown()
