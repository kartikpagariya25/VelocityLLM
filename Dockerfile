FROM vllm/vllm-openai:latest

ENV VLLM_USE_V2_MODEL_RUNNER=0
ENV VLLM_USE_FLASHINFER_SAMPLER=0
ENV TMPDIR=/tmp/pip_tmp

WORKDIR /app

RUN pip install pyyaml aiohttp

COPY scheduler_engine/ scheduler_engine/
COPY pytest.ini .
COPY run_tests.py .
COPY tests/ tests/
COPY config.example.yaml .

EXPOSE 8000

ENTRYPOINT []
CMD ["python3", "-m", "scheduler_engine.server", "--config", "config.example.yaml", "--host", "0.0.0.0"]
