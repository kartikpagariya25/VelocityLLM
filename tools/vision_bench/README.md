# Vision benchmark

Tests a vision-language model (VLM) through VelocityLLM: it starts `velocityllm serve --vision` for each policy
(static, dynamic, smart), sends the same image traffic to each, and reports what happened per image size.

Each request carries a different picture, so no cache can hide the cost of reading it. The three image sizes
(224, 448, 896 px = 64, 256, 1024 vision tokens) and their mix match the Arena's "mixed" images.

## Run it

```bash
# simulated engine, no GPU (answers are canned text; speeds are made up)
python tools/vision_bench/vision_check.py --smoke      # 4 pictures with known answers
python tools/vision_bench/vision_check.py              # image traffic on all three policies

# real vision-language model on the GPU
python tools/vision_bench/vision_check.py --real --model-path models/qwen2-vl-2b-awq --smoke
python tools/vision_bench/vision_check.py --real --model-path models/qwen2-vl-2b-awq --rate 2 --duration 20
python tools/vision_bench/vision_check.py --real --model-path models/qwen2-vl-2b-awq --smoke --image my_photo.jpg
```

`--smoke` sends a red, a blue and a green circle and an orange square and asks for the colour or shape. A model that
really looks at the picture answers correctly; the simulated engine cannot, so its text is canned.

It runs from a repository checkout (no `pip install` needed) or from a pip install; it starts the server with `velocityllm serve` when that package is installed and with `python -m scheduler_engine.server` otherwise.

## What it checks

| Column | Meaning |
|---|---|
| On time % | Answered inside the promise, out of ALL users. Refused users count against it |
| 224/448/896px on time %, median s | The same, split by image size: shows what big images cost each policy |
| Vision tokens ok | The vision tokens the server reports add up to what the images cost. Catches accounting bugs |

## Which model fits a 6 GB GPU

This project's vision path builds the Qwen-style prompt (`<|vision_start|><|image_pad|><|vision_end|>`), so it fits the
**Qwen2-VL / Qwen2.5-VL** family. The engine gets 80% of the GPU memory (4.8 GB on a 6 GB card).

- `Qwen/Qwen2-VL-2B-Instruct` in bf16 needs about 4.7 GB for the weights and activations (Qwen's own measurement), so it does not leave room for the cache on a 6 GB card.
- `Qwen/Qwen2-VL-2B-Instruct-AWQ` (4-bit) needs about 2.9 GB in the same measurement, so it should fit. This has not been run on a 6 GB GPU by the authors of this tool.

```bash
hf download Qwen/Qwen2-VL-2B-Instruct-AWQ --local-dir models/qwen2-vl-2b-awq
```

Models with another prompt format (SmolVLM, InternVL, LLaVA ...) need the prompt builder in `scheduler_engine/backend.py` generalised first.

## Notes

- The simulated engine charges 0.4 ms per vision token and has no GPU contention. Use `--real` for real numbers.
- One run is one sample. On a power-capped laptop GPU the same setup can differ a lot between runs: repeat before you compare.
- Found with this tool: the smart policy reported 0 vision tokens in responses and `/stats`; static and dynamic reported them correctly (fixed in the same branch).
