from vllm import LLM, SamplingParams
import time

def main():
    model_path = "/home/kartiklin/velocityllm/models/llama-3.2-1b"

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
    )

    prompts = [
        "What is the capital of France?",
        "What is the capital of Japan?",
    ]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    for output in outputs:
        print("Prompt:", output.prompt)
        print("Response:", output.outputs[0].text)
        print("---")

    print(f"Total generation time (excluding startup): {elapsed:.2f}s")

if __name__ == "__main__":
    main()
