"""VelocityLLM: SLA-aware request scheduling for LLM serving.

This package is the installable front door. The implementation lives in the packages shipped with it:
scheduler_engine (server and policies), control_service (benchmark runner) and load_generator (traffic tools).
"""

__version__ = "0.4.0"


def benchmark(*args, **kwargs):
    from velocityllm.bench import benchmark as run

    return run(*args, **kwargs)
