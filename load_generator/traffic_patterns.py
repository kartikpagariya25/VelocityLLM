import random


def generate_arrival_times(pattern, num_requests, duration_seconds, seed=42):
    rng = random.Random(seed)
    arrivals = []

    if pattern == "constant":
        interval = duration_seconds / num_requests
        arrivals = [i * interval for i in range(num_requests)]

    elif pattern == "poisson":
        avg_rate = num_requests / duration_seconds
        t = 0.0
        for _ in range(num_requests):
            inter_arrival = rng.expovariate(avg_rate)
            t += inter_arrival
            arrivals.append(t)

    elif pattern == "burst":
        base_rate = (num_requests * 0.3) / duration_seconds
        burst_rate = (num_requests * 0.7) / (duration_seconds * 0.2)
        t = 0.0
        burst_windows = [(duration_seconds * 0.4, duration_seconds * 0.5),
                          (duration_seconds * 0.8, duration_seconds * 0.9)]
        for _ in range(num_requests):
            in_burst = any(lo <= t <= hi for lo, hi in burst_windows)
            rate = burst_rate if in_burst else base_rate
            inter_arrival = rng.expovariate(rate)
            t += inter_arrival
            arrivals.append(t)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return arrivals


def generate_mixed_prompts(num_requests, seed=42):
    rng = random.Random(seed)
    short_prompts = [
        "What is the capital of France?",
        "Name three programming languages.",
        "What is the boiling point of water?",
        "What is the capital of Italy?",
    ]
    long_prompts = [
        "Write a detailed explanation of how neural networks learn through backpropagation, covering gradient descent and the chain rule.",
        "Describe the historical events that led to the industrial revolution and its long-term economic effects on Europe.",
        "Explain the complete process of photosynthesis in plants, including the light-dependent and light-independent reactions.",
        "Provide a thorough comparison between static batching and continuous batching approaches in LLM inference serving.",
    ]

    requests = []
    for i in range(num_requests):
        is_long = rng.random() < 0.25
        if is_long:
            prompt = rng.choice(long_prompts)
            max_tokens = rng.choice([300, 400, 500])
        else:
            prompt = rng.choice(short_prompts)
            max_tokens = rng.choice([30, 50, 80])
        requests.append({"prompt": prompt, "max_tokens": max_tokens, "is_long": is_long})
    return requests
